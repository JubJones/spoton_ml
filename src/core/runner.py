import json
import logging
import subprocess
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import mlflow
import numpy as np
import torch
from PIL import Image
from mlflow.models import infer_signature

from src.data.loader import FrameDataLoader
from src.pipelines.detection_pipeline import DetectionPipeline
from src.tracking.strategies import (
    DetectionTrackingStrategy, YoloStrategy, RTDetrStrategy, FasterRCNNStrategy, RfDetrStrategy
)

logger = logging.getLogger(__name__)


# --- Helper Functions---

def log_params_recursive(params_dict: Dict[str, Any], parent_key: str = ""):
    """Recursively logs parameters to the *current active* MLflow run."""
    if not mlflow.active_run():
        logger.warning("Attempted to log parameters outside of an active MLflow run.")
        return
    for key, value in params_dict.items():
        mlflow_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            log_params_recursive(value, mlflow_key)
        elif isinstance(value, list):
            try:
                param_val_str = json.dumps(value)
                if len(param_val_str) > 250:
                    param_val_str = param_val_str[:247] + "..."
                mlflow.log_param(mlflow_key, param_val_str)
            except TypeError:
                # Fallback for non-serializable lists
                mlflow.log_param(mlflow_key, str(value)[:250])
        else:
            # Log scalar values, truncate if too long
            mlflow.log_param(mlflow_key, str(value)[:250])


def log_metrics_dict(metrics: Dict[str, Any]):
    """Logs metrics from a dictionary to the *current active* MLflow run."""
    if not mlflow.active_run():
        logger.warning("Attempted to log metrics outside of an active MLflow run.")
        return
    if not metrics:
        logger.warning("Metrics dictionary is empty. Nothing to log.")
        return

    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, np.number))}
    non_numeric_keys = [k for k in metrics if k not in numeric_metrics]

    if non_numeric_keys:
        logger.warning(f"Skipping non-numeric metrics: {non_numeric_keys}")

    if numeric_metrics:
        logger.info(f"Logging {len(numeric_metrics)} numeric metrics...")
        try:
            mlflow.log_metrics(numeric_metrics)
            logger.info("Numeric metrics logged successfully.")
        except Exception as e:
            logger.error(f"Failed to log metrics batch: {e}. Attempting individual logging.", exc_info=True)
            success_count = 0
            for key, value in numeric_metrics.items():
                try:
                    mlflow.log_metric(key, value)
                    success_count += 1
                except Exception as ind_e:
                    logger.error(f"Failed to log individual metric '{key}': {value}. Error: {ind_e}")
            logger.info(f"Logged {success_count} metrics individually after batch failure.")
    else:
        logger.warning("No numeric metrics found in the provided dictionary.")


def get_sample_data_for_signature(
        config: Dict[str, Any], strategy: DetectionTrackingStrategy, device: torch.device
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Loads a sample frame and runs inference to get sample input/output
    for MLflow signature inference. Returns (sample_input_sig_obj, sample_output_sig_obj).
    The types depend on the model strategy.
    """
    logger.info("Attempting to generate sample input/output for model signature...")
    sample_input_sig = None
    sample_output_sig = None
    try:
        temp_loader = FrameDataLoader(config)
        sample_frame_bgr = None
        if temp_loader.image_filenames and temp_loader.active_camera_ids:
            filename = temp_loader.image_filenames[0]
            cam_id = temp_loader.active_camera_ids[0]
            image_path = temp_loader.camera_image_dirs[cam_id] / filename
            if image_path.is_file():
                img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
                sample_frame_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                logger.info(f"Loaded sample frame: {image_path}")
        del temp_loader

        if sample_frame_bgr is None:
            logger.warning("Could not load a sample frame for signature generation.")
            return None, None

        inference_batch = None
        if isinstance(strategy, (YoloStrategy, RTDetrStrategy)):
            sample_input_sig = sample_frame_bgr
            inference_batch = sample_frame_bgr
            logger.info("Using numpy frame (BGR HWC) as input signature object for Ultralytics.")
        elif isinstance(strategy, FasterRCNNStrategy):
            sample_input_tensor = strategy.get_sample_input_tensor(sample_frame_bgr)  # Gets CHW tensor
            if sample_input_tensor is not None:
                sample_input_sig = sample_input_tensor.cpu().numpy()  # Use numpy representation (NCHW or CHW)
                inference_batch = [sample_input_tensor.to(device)]  # Model expects list of tensors
                logger.info("Using numpy tensor (CHW) as input signature object for FasterRCNN.")
            else:
                logger.warning("Could not get sample tensor for FasterRCNN signature.")
                return None, None
        elif isinstance(strategy, RfDetrStrategy):
            # RFDETR predict takes PIL Image
            img_rgb_pil = Image.fromarray(cv2.cvtColor(sample_frame_bgr, cv2.COLOR_BGR2RGB))
            sample_input_sig = img_rgb_pil
            inference_batch = img_rgb_pil
            logger.info("Using PIL Image (RGB) as input signature object for RFDETR.")
        else:
            # Fallback: Generic tensor
            sample_input_tensor = strategy.get_sample_input_tensor(sample_frame_bgr)
            if sample_input_tensor is not None:
                sample_input_sig = sample_input_tensor.cpu().numpy()
                inference_batch = [sample_input_tensor.to(device)]
                logger.info("Using generic numpy tensor (CHW) as input signature object.")
            else:
                logger.warning("Could not get sample tensor for generic signature.")
                return None, None

        model_object = strategy.get_model()
        if model_object is None:
            logger.warning("Cannot get sample output: Model object not available from strategy.")
            return sample_input_sig, None

        if inference_batch is None:
            logger.error("Failed to prepare input batch for inference.")
            return sample_input_sig, None

        with torch.no_grad():
            if isinstance(strategy, (YoloStrategy, RTDetrStrategy)):
                results = model_object.predict(inference_batch, device=device, verbose=False)
                if results and results[0].boxes:
                    # Output signature: bounding boxes (e.g., xyxy)
                    if hasattr(results[0].boxes, 'xyxy') and results[0].boxes.xyxy is not None:
                        sample_output_sig = results[0].boxes.xyxy.cpu().numpy()
                    elif hasattr(results[0].boxes, 'data') and results[0].boxes.data is not None:  # Fallback
                        sample_output_sig = results[0].boxes.data.cpu().numpy()[:,
                                            :4]  # Assuming first 4 are box coords
            elif isinstance(strategy, FasterRCNNStrategy) and isinstance(model_object, torch.nn.Module):
                predictions = model_object(inference_batch)
                if isinstance(predictions, list) and len(predictions) > 0 and 'boxes' in predictions[0]:
                    sample_output_sig = predictions[0]['boxes'].cpu().numpy()
            elif isinstance(strategy, RfDetrStrategy) and hasattr(model_object, 'predict'):
                detections = model_object.predict(inference_batch)  # Pass PIL image
                if detections and hasattr(detections, 'xyxy'):
                    out = detections.xyxy
                    sample_output_sig = out.cpu().numpy() if isinstance(out, torch.Tensor) else out
            elif isinstance(model_object, torch.nn.Module):  # Generic torch module fallback
                predictions = model_object(inference_batch)  # Assume list of tensors in
                if isinstance(predictions, torch.Tensor):
                    sample_output_sig = predictions.cpu().numpy()
                # Add more specific handling if needed for other model types
            else:
                logger.warning(
                    f"Cannot determine how to run inference for model type {type(model_object)} to get sample output.")

        if sample_input_sig is not None and sample_output_sig is not None:
            logger.info(
                f"Successfully generated sample input (type: {type(sample_input_sig)}) and output (type: {type(sample_output_sig)})")
        elif sample_input_sig is not None:
            logger.warning("Generated sample input, but failed to generate sample output.")
        else:
            logger.warning("Failed to generate sample input and/or output.")

        return sample_input_sig, sample_output_sig

    except Exception as e:
        logger.error(f"Error during sample data generation for signature: {e}", exc_info=True)
        return None, None


def log_git_info():
    """Logs Git commit hash and status to the current MLflow run."""
    if not mlflow.active_run(): return
    try:
        project_root = Path(__file__).parent.parent.parent.resolve()
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project_root).strip().decode('utf-8')
        mlflow.set_tag("git_commit_hash", commit)
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=project_root).strip().decode('utf-8')
        git_status = "dirty" if status else "clean"
        mlflow.set_tag("git_status", git_status)
        logger.info(f"Logged Git info: Commit={commit[:7]}, Status={git_status}")
        if status:
            try:
                diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=project_root).strip().decode('utf-8',
                                                                                                         errors='ignore')
                if diff:
                    mlflow.log_text(diff, artifact_file="code/git_diff.diff")
                    logger.info("Logged git diff as artifact.")
            except Exception as diff_err:
                logger.warning(f"Could not log git diff: {diff_err}")
    except Exception as git_err:
        logger.warning(f"Could not get git commit hash or status: {git_err}")
        mlflow.set_tag("git_status", "unknown")


# --- Core Runner Function ---
def run_single_experiment(
        run_config: Dict[str, Any],
        base_device_preference: str,  # e.g., "auto", "cuda:0"
        seed: int,
        config_file_path: str,
        log_file_path: Optional[str] = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Executes a single MLflow run based on the provided configuration.
    """
    active_run = mlflow.active_run()
    if not active_run:
        logger.critical("run_single_experiment called without an active MLflow run!")
        return "FAILED", None
    run_id = active_run.info.run_id
    logger.info(f"--- Starting Execution Logic for Run ID: {run_id} ---")

    run_status = "FAILED"
    metrics = None
    pipeline: Optional[DetectionPipeline] = None
    actual_device: Optional[torch.device] = None

    try:
        # 1. Determine Actual Device (Applying overrides)
        model_type = run_config.get("model", {}).get("type", "").lower()
        model_name = run_config.get("model", {}).get("model_name", model_type)
        requested_device = torch.device(base_device_preference)
        actual_device = requested_device
        device_override_reason = "None"

        # --- Device Compatibility Checks ---
        if model_type == "fasterrcnn":
            if requested_device.type not in ['cuda', 'mps']:
                logger.warning(f"[{model_name}] FasterRCNN requested on '{requested_device.type}'. Forcing to CPU.")
                actual_device = torch.device('cpu')
                device_override_reason = f"FasterRCNN forced to CPU from {requested_device.type}"
            elif requested_device.type == 'mps':
                actual_device = torch.device('mps')
                logger.info(f"[{model_name}] Using MPS for FasterRCNN as requested.")

        elif actual_device.type == 'mps':
            if model_type in ['rfdetr']:
                logger.warning(f"[{model_name}] Model type '{model_type}' has known issues on MPS. Forcing to CPU.")
                actual_device = torch.device('cpu')
                device_override_reason = f"{model_type} forced to CPU due to MPS issues"

        logger.info(
            f"[{model_name}] Device Selection: Requested='{base_device_preference}', Actual='{actual_device}', Reason='{device_override_reason}'")

        # 2. Log Parameters & Tags (including device info)
        logger.info(f"[{model_name}] Logging parameters...")
        log_params_recursive(run_config)
        mlflow.log_param("environment.seed", seed)
        mlflow.log_param("environment.requested_device", base_device_preference)
        mlflow.log_param("environment.actual_device_used", str(actual_device))
        mlflow.log_param("environment.device_override_reason", device_override_reason)

        logger.info(f"[{model_name}] Logging tags...")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset", run_config.get('data', {}).get('selected_environment', 'unknown'))
        env_data = run_config.get('data', {}).get(run_config.get('data', {}).get('selected_environment', ''), {})
        mlflow.set_tag("scene_id", env_data.get('scene_id', 'unknown'))
        log_git_info()

        # 3. Log Configuration Artifact
        if config_file_path and Path(config_file_path).is_file():
            mlflow.log_artifact(config_file_path, artifact_path="config")
            logger.info(f"Logged configuration artifact: {Path(config_file_path).name}")
        else:
            logger.warning(f"Could not find config file to log: {config_file_path}")

        # Log requirements.txt (common artifact) - check if it's already logged by parent? Maybe log always.
        req_path = Path(__file__).parent.parent.parent / "requirements.txt"
        if req_path.is_file():
            mlflow.log_artifact(str(req_path), artifact_path="code")
        else:
            logger.warning(f"requirements.txt not found at {req_path}")

        # 4. Initialize Pipeline
        logger.info(f"[{model_name}] Initializing detection pipeline...")
        pipeline = DetectionPipeline(run_config, actual_device)
        if not pipeline.initialize_components():
            raise RuntimeError(f"Pipeline initialization failed for model {model_name}")
        logger.info(f"[{model_name}] Pipeline initialized successfully.")

        # 5. Generate Signature & Log Model/Examples (if possible)
        signature = None
        sample_input_data, sample_output_data = None, None
        if pipeline.detection_strategy:
            sample_input_data, sample_output_data = get_sample_data_for_signature(
                run_config, pipeline.detection_strategy, actual_device
            )
            if sample_input_data is not None and sample_output_data is not None:
                try:
                    signature = infer_signature(sample_input_data, sample_output_data)
                    logger.info(f"[{model_name}] Model signature inferred successfully.")
                    # Log signature as dict artifact
                    mlflow.log_dict(signature.to_dict(), "signature.json")
                except Exception as infer_err:
                    logger.warning(f"[{model_name}] Could not infer model signature: {infer_err}", exc_info=True)
                    signature = None
            else:
                logger.warning(f"[{model_name}] Failed to generate sample data for signature.")

            # Log input example artifact (handle different types)
            if sample_input_data is not None:
                example_dir = Path("./mlflow_temp_examples")  # Use a temporary local dir
                example_dir.mkdir(exist_ok=True)
                try:
                    if isinstance(sample_input_data, np.ndarray):
                        example_path = example_dir / f"input_example_{run_id}.npy"
                        np.save(example_path, sample_input_data)
                        mlflow.log_artifact(str(example_path), artifact_path="examples")
                        logger.info(f"[{model_name}] Logged numpy input example artifact.")
                        example_path.unlink()
                    elif isinstance(sample_input_data, Image.Image):
                        example_path = example_dir / f"input_example_{run_id}.png"
                        sample_input_data.save(example_path)
                        mlflow.log_artifact(str(example_path), artifact_path="examples")
                        logger.info(f"[{model_name}] Logged PIL input example artifact.")
                        example_path.unlink()
                    # Add other types if needed (e.g., torch tensor, although numpy is common)
                except Exception as ex_log_err:
                    logger.warning(f"Failed to log input example artifact: {ex_log_err}")
                finally:
                    if example_dir.exists():
                        try:
                            example_dir.rmdir()
                        except OSError:
                            pass  # Ignore if not empty (e.g. failed unlink)

            # --- Model Logging ---
            logger.info(
                f"[{model_name}] Generic model artifact logging is currently disabled. Signature/example logged if available.")
            # model_obj = pipeline.detection_strategy.get_model()
            # if model_obj:
            #    mlflow.log_model(model_obj, artifact_path="model_untyped", signature=signature) # Example

        else:
            logger.warning(f"[{model_name}] Detection strategy not available, cannot generate signature or log model.")

        # 6. Run Pipeline Processing
        logger.info(f"[{model_name}] Starting pipeline processing...")
        pipeline_success, calculated_metrics, active_cameras, num_frames_processed = pipeline.run()

        # 7. Log Pipeline Info (Actual cameras/frames)
        if active_cameras:
            mlflow.log_param("data.actual_cameras_used", ",".join(active_cameras))
        if num_frames_processed is not None:
            # Check if 'unique_frame_indices_processed' exists in metrics, otherwise use num_frames_processed
            unique_indices = calculated_metrics.get('unique_frame_indices_processed', num_frames_processed)
            mlflow.log_param("data.actual_frames_processed_count", unique_indices)

        # 8. Log Metrics
        if calculated_metrics:
            logger.info(f"[{model_name}] Logging metrics...")
            log_metrics_dict(calculated_metrics)
            metrics = calculated_metrics  # Store for return value

            # Log metrics dictionary as JSON artifact
            try:
                metrics_path = Path(f"./run_{run_id}_metrics.json")  # Temp local file
                with open(metrics_path, 'w') as f:
                    metrics_serializable = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in
                                            calculated_metrics.items()}
                    json.dump(metrics_serializable, f, indent=4)
                mlflow.log_artifact(str(metrics_path), artifact_path="results")
                metrics_path.unlink()  # Clean up temp file
                logger.info("Metrics dictionary logged as JSON artifact.")
            except Exception as json_err:
                logger.warning(f"Could not log metrics dictionary as JSON: {json_err}")
        else:
            logger.warning(f"[{model_name}] No metrics were calculated by the pipeline.")

        # 9. Determine Final Run Status
        if pipeline_success:
            if metrics:  # Successful run with metrics
                run_status = "FINISHED"
                mlflow.set_tag("run_outcome", "Success")
                logger.info(f"[{model_name}] Run finished successfully with metrics.")
            else:  # Pipeline reported success, but no metrics? (Shouldn't happen with current pipeline)
                run_status = "FINISHED"  # Still finished
                mlflow.set_tag("run_outcome", "Success (No Metrics)")
                logger.warning(f"[{model_name}] Run finished successfully but no metrics were returned.")
        else:
            if metrics:  # Failed run but partial metrics exist
                run_status = "FAILED"
                mlflow.set_tag("run_outcome", "Failed (Partial Metrics)")
                logger.error(f"[{model_name}] Pipeline execution failed, but partial metrics were calculated.")
            else:  # Failed run, no metrics
                run_status = "FAILED"
                mlflow.set_tag("run_outcome", "Failed Execution")
                logger.error(f"[{model_name}] Pipeline execution failed. No metrics generated.")


    except KeyboardInterrupt:
        logger.warning(f"[{model_name}] Run interrupted by user (KeyboardInterrupt).")
        run_status = "KILLED"
        mlflow.set_tag("run_outcome", "Killed by user")
        # The exception will propagate up and be handled by the calling script's finally block.
        raise  # Re-raise KeyboardInterrupt

    except Exception as e:
        logger.critical(f"[{model_name}] An uncaught error occurred during the run: {e}", exc_info=True)
        run_status = "FAILED"
        mlflow.set_tag("run_outcome", "Crashed")
        # Log error details if possible
        try:
            mlflow.log_text(
                f"Error Type: {type(e).__name__}\nError Message: {e}\n\nTraceback:\n{traceback.format_exc()}",
                "error_log.txt")
        except Exception:
            pass  # Avoid nested errors during logging

    finally:
        # 10. Log the main experiment log file (if provided and run didn't crash before tag setting)
        if log_file_path and Path(log_file_path).exists() and run_status != "FAILED" and run_status != "KILLED":
            pass

        logger.info(f"--- Finished Execution Logic for Run ID: {run_id} (Attempted Status: {run_status}) ---")

    return run_status, metrics
