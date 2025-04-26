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

# --- Local Imports Need Correct Path Handling ---
try:
    from src.utils.device_utils import get_selected_device
    # Detection Imports
    from src.data.loader import FrameDataLoader
    from src.pipelines.detection_pipeline import DetectionPipeline
    from src.tracking.strategies import (
        DetectionTrackingStrategy, YoloStrategy, RTDetrStrategy, FasterRCNNStrategy, RfDetrStrategy
    )
    # Re-ID Imports
    from src.data.reid_dataset_loader import ReidDatasetLoader, ReidCropInfo
    from src.reid.strategies import ReIDStrategy, \
        get_reid_strategy_from_run_config
    from src.evaluation.reid_metrics import compute_reid_metrics
    from src.pipelines.reid_pipeline import ReidPipeline
except ImportError:
    import sys

    if str(Path(__file__).parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent.parent))
    # Detection Imports
    from utils.device_utils import get_selected_device
    from data.loader import FrameDataLoader
    from pipelines.detection_pipeline import DetectionPipeline
    from tracking.strategies import (
        DetectionTrackingStrategy, YoloStrategy, RTDetrStrategy, FasterRCNNStrategy, RfDetrStrategy
    )
    # Re-ID Imports
    from data.reid_dataset_loader import ReidDatasetLoader, ReidCropInfo  # Kept for reference, used by pipeline
    from reid.strategies import ReIDStrategy, get_reid_strategy_from_run_config  # Kept for reference, used by pipeline
    from evaluation.reid_metrics import compute_reid_metrics  # Kept for reference, used by pipeline
    from pipelines.reid_pipeline import ReidPipeline  # Import the new Re-ID pipeline
# --- End Local Import Handling ---

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


# --- Helper Functions (log_params_recursive, log_metrics_dict, log_git_info) ---
def log_params_recursive(params_dict: Dict[str, Any], parent_key: str = ""):
    """Recursively logs parameters to the *current active* MLflow run."""
    if not mlflow.active_run():
        logger.warning("Attempted to log parameters outside of an active MLflow run.")
        return
    for key, value in params_dict.items():
        mlflow_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            # Avoid logging entire large sub-dictionaries like detailed model lists from parent
            if key == 'models_to_run' and parent_key == '':
                mlflow.log_param(mlflow_key + ".count", len(value))
                continue
            # Avoid logging injected run config back
            if key == '_run_config':
                continue
            log_params_recursive(value, mlflow_key)
        elif isinstance(value, list):
            try:
                param_val_str = json.dumps(value)
                # MLflow Param limit is 500 chars now (increased from 250)
                if len(param_val_str) > 500:
                    param_val_str = param_val_str[:497] + "..."
                mlflow.log_param(mlflow_key, param_val_str)
            except TypeError:
                # Fallback for non-serializable lists
                mlflow.log_param(mlflow_key, str(value)[:500])
        else:
            # Log scalar values, truncate if too long
            mlflow.log_param(mlflow_key, str(value)[:500])


def log_metrics_dict(metrics: Dict[str, Any], is_reid: bool = False):
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
        log_type = "Re-ID" if is_reid else "Detection"
        logger.info(f"Logging {len(numeric_metrics)} numeric {log_type} metrics...")
        try:
            mlflow.log_metrics(numeric_metrics)
            logger.info(f"Numeric {log_type} metrics logged successfully.")
        except Exception as e:
            logger.error(f"Failed to log {log_type} metrics batch: {e}. Attempting individual logging.", exc_info=False)
            success_count = 0
            for key, value in numeric_metrics.items():
                try:
                    mlflow.log_metric(key, value)
                    success_count += 1
                except Exception as ind_e:
                    logger.error(f"Failed to log individual {log_type} metric '{key}': {value}. Error: {ind_e}")
            logger.info(f"Logged {success_count} {log_type} metrics individually after batch failure.")
    else:
        logger.warning("No numeric metrics found in the provided dictionary.")


def log_git_info():
    """Logs Git commit hash and status to the current MLflow run."""
    if not mlflow.active_run(): return
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
                                         stderr=subprocess.STDOUT).strip().decode('utf-8')
        mlflow.set_tag("git_commit_hash", commit)
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT,
                                         stderr=subprocess.STDOUT).strip().decode('utf-8')
        git_status = "dirty" if status else "clean"
        mlflow.set_tag("git_status", git_status)
        logger.info(f"Logged Git info: Commit={commit[:7]}, Status={git_status}")
        if status:
            try:
                diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=PROJECT_ROOT,
                                               stderr=subprocess.STDOUT).strip().decode('utf-8', errors='ignore')
                if diff:
                    max_diff_len = 100 * 1024
                    if len(diff) > max_diff_len:
                        diff = diff[:max_diff_len] + "\n... (diff truncated)"
                    mlflow.log_text(diff, artifact_file="code/git_diff.diff")
                    logger.info("Logged git diff as artifact.")
            except Exception as diff_err:
                logger.warning(f"Could not log git diff: {diff_err}")
    except subprocess.CalledProcessError as git_err:
        logger.warning(f"Could not get git info (git command failed): {git_err}")
        mlflow.set_tag("git_status", "unknown (git error)")
    except FileNotFoundError:
        logger.warning("Could not get git info ('git' command not found).")
        mlflow.set_tag("git_status", "unknown (git not found)")
    except Exception as git_err:
        logger.warning(f"Could not get git commit hash or status (unexpected error): {git_err}")
        mlflow.set_tag("git_status", "unknown (error)")


def get_sample_data_for_signature(
        config: Dict[str, Any], strategy: DetectionTrackingStrategy, device: torch.device
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Loads a sample frame and runs inference to get sample input/output
    for MLflow signature inference (Detection Models).
    """
    logger.info("Attempting to generate sample input/output for DETECTION model signature...")
    sample_input_sig, sample_output_sig = None, None
    try:
        temp_loader = FrameDataLoader(config)
        sample_frame_bgr = None
        if temp_loader.image_filenames and temp_loader.active_camera_ids:
            filename = temp_loader.image_filenames[0];
            cam_id = temp_loader.active_camera_ids[0]
            image_path = temp_loader.camera_image_dirs[cam_id] / filename
            if image_path.is_file():
                img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
                sample_frame_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if sample_frame_bgr is not None:
                    logger.info(f"Loaded sample frame for detection signature: {image_path}")
                else:
                    logger.warning(f"Failed to decode sample frame: {image_path}")
            else:
                logger.warning(f"Sample frame file not found: {image_path}")
        del temp_loader

        if sample_frame_bgr is None:
            logger.warning("Could not load a sample frame for detection signature generation.")
            return None, None

        inference_batch = None
        if isinstance(strategy, (YoloStrategy, RTDetrStrategy)):
            sample_input_sig = sample_frame_bgr
            inference_batch = sample_frame_bgr
            logger.info("Using numpy frame (BGR HWC) as input signature object for Ultralytics.")
        elif isinstance(strategy, FasterRCNNStrategy):
            sample_input_tensor = strategy.get_sample_input_tensor(sample_frame_bgr)
            if sample_input_tensor is not None:
                sample_input_sig = sample_input_tensor.cpu().numpy()
                inference_batch = [sample_input_tensor.to(device)]
                logger.info("Using numpy tensor (CHW) as input signature object for FasterRCNN.")
            else:
                logger.warning("Could not get sample tensor for FasterRCNN signature.")
                return None, None
        elif isinstance(strategy, RfDetrStrategy):
            img_rgb = cv2.cvtColor(sample_frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            sample_input_sig = img_pil
            inference_batch = img_pil
            logger.info("Using PIL Image (RGB) as input signature object for RFDETR.")
        else:
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
                    if hasattr(results[0].boxes, 'xyxy') and results[0].boxes.xyxy is not None:
                        sample_output_sig = results[0].boxes.xyxy.cpu().numpy()
                    elif hasattr(results[0].boxes, 'data') and results[0].boxes.data is not None:
                        sample_output_sig = results[0].boxes.data.cpu().numpy()[:, :4]
            elif isinstance(strategy, FasterRCNNStrategy) and isinstance(model_object, torch.nn.Module):
                model_object.to(device);
                predictions = model_object(inference_batch)
                if isinstance(predictions, list) and len(predictions) > 0 and 'boxes' in predictions[
                    0]: sample_output_sig = predictions[0]['boxes'].cpu().numpy()
            elif isinstance(strategy, RfDetrStrategy) and hasattr(model_object, 'predict'):
                detections = model_object.predict(inference_batch)
                if detections and hasattr(detections,
                                          'xyxy'): out = detections.xyxy
                sample_output_sig = out.cpu().numpy() if isinstance(
                    out, torch.Tensor) else out
            elif isinstance(model_object, torch.nn.Module):
                model_object.to(device);
                predictions = model_object(inference_batch)
                if isinstance(predictions, torch.Tensor): sample_output_sig = predictions.cpu().numpy()
            else:
                logger.warning(
                    f"Cannot determine how to run inference for model type {type(model_object)} to get sample output.")

        if sample_input_sig is not None and sample_output_sig is not None:
            logger.info(
                f"Successfully generated sample input (type: {type(sample_input_sig)}) and output (type: {type(sample_output_sig)}) for detection.")
        elif sample_input_sig is not None:
            logger.warning("Generated sample input, but failed to generate sample output for detection.")
        else:
            logger.warning("Failed to generate sample input and/or output for detection.")
        return sample_input_sig, sample_output_sig
    except Exception as e:
        logger.error(f"Error during sample data generation for detection signature: {e}", exc_info=True)
        return None, None


# --- Detection Runner (run_single_experiment) ---
def run_single_experiment(
        run_config: Dict[str, Any],
        base_device_preference: str,
        seed: int,
        config_file_path: str,
        log_file_path: Optional[str] = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Executes a single DETECTION MLflow run based on the provided configuration."""
    active_run = mlflow.active_run();
    if not active_run: logger.critical(
        "run_single_experiment called without an active MLflow run!")
    return "FAILED", None
    run_id = active_run.info.run_id;
    logger.info(f"--- Starting Detection Execution Logic for Run ID: {run_id} ---")
    run_status = "FAILED";
    metrics = None;
    pipeline: Optional[DetectionPipeline] = None;
    actual_device: Optional[torch.device] = None
    try:
        resolved_initial_device = get_selected_device(base_device_preference);
        actual_device = resolved_initial_device;
        device_override_reason = "None"
        model_type = run_config.get("model", {}).get("type", "").lower();
        model_name = run_config.get("model", {}).get("model_name", model_type)
        if model_type == "fasterrcnn":
            if resolved_initial_device.type not in ['cuda', 'mps']: logger.warning(
                f"[{model_name}] FasterRCNN requested/resolved to '{resolved_initial_device.type}'. Forcing to CPU.")
            actual_device = torch.device(
                'cpu')
            device_override_reason = f"FasterRCNN forced to CPU from {resolved_initial_device.type}"
        elif resolved_initial_device.type == 'mps':
            if model_type in ['rfdetr']: logger.warning(
                f"[{model_name}] Model type '{model_type}' has known issues on MPS. Forcing to CPU.")
            actual_device = torch.device(
                'cpu')
            device_override_reason = f"{model_type} forced to CPU due to MPS issues"
        logger.info(
            f"[{model_name}] Device Selection: Requested='{base_device_preference}', Resolved='{resolved_initial_device}', Actual Used='{actual_device}', Override Reason='{device_override_reason}'")
        logger.info(f"[{model_name}] Logging detection parameters...");
        log_params_recursive(run_config);
        mlflow.log_param("environment.seed", seed);
        mlflow.log_param("environment.requested_device", base_device_preference);
        mlflow.log_param("environment.resolved_initial_device", str(resolved_initial_device));
        mlflow.log_param("environment.actual_device_used", str(actual_device));
        mlflow.log_param("environment.device_override_reason", device_override_reason)
        logger.info(f"[{model_name}] Logging detection tags...");
        mlflow.set_tag("model_name", model_name);
        mlflow.set_tag("model_type", model_type);
        mlflow.set_tag("dataset", run_config.get('data', {}).get('selected_environment', 'unknown'));
        env_data = run_config.get('data', {}).get(run_config.get('data', {}).get('selected_environment', ''), {});
        mlflow.set_tag("scene_id", env_data.get('scene_id', 'unknown'));
        log_git_info()
        if config_file_path and Path(config_file_path).is_file(): mlflow.log_artifact(config_file_path,
                                                                                      artifact_path="config")
        req_path = Path(__file__).parent.parent.parent / "requirements.txt";
        if req_path.is_file(): mlflow.log_artifact(str(req_path), artifact_path="code")
        logger.info(f"[{model_name}] Initializing detection pipeline on device '{actual_device}'...");
        pipeline = DetectionPipeline(run_config, actual_device)
        if not pipeline.initialize_components(): raise RuntimeError(
            f"Detection pipeline initialization failed for model {model_name}")
        logger.info(f"[{model_name}] Detection Pipeline initialized successfully.")
        signature = None;
        sample_input_data, sample_output_data = None, None
        if pipeline.detection_strategy:
            sample_input_data, sample_output_data = get_sample_data_for_signature(run_config,
                                                                                  pipeline.detection_strategy,
                                                                                  actual_device)
            if sample_input_data is not None and sample_output_data is not None:
                try:
                    signature = infer_signature(sample_input_data, sample_output_data);
                    logger.info(
                        f"[{model_name}] Detection signature inferred.");
                    mlflow.log_dict(signature.to_dict(),
                                    "signature.json")
                except Exception as infer_err:
                    logger.warning(f"[{model_name}] Could not infer detection signature: {infer_err}",
                                   exc_info=False);
                    signature = None
            else:
                logger.warning(f"[{model_name}] Failed to generate sample data for detection signature.")
            if sample_input_data is not None:
                try:
                    if isinstance(sample_input_data, np.ndarray):
                        np.save(f"input_example_{run_id}.npy", sample_input_data);
                        mlflow.log_artifact(
                            f"input_example_{run_id}.npy", artifact_path="examples");
                        Path(
                            f"input_example_{run_id}.npy").unlink()
                    elif isinstance(sample_input_data, Image.Image):
                        sample_input_data.save(f"input_example_{run_id}.png");
                        mlflow.log_artifact(
                            f"input_example_{run_id}.png", artifact_path="examples");
                        Path(
                            f"input_example_{run_id}.png").unlink()
                except Exception as ex_log_err:
                    logger.warning(f"Failed to log detection input example artifact: {ex_log_err}")
        else:
            logger.warning(f"[{model_name}] Detection strategy not available, cannot generate signature.")
        logger.info(f"[{model_name}] Starting detection pipeline processing...");
        pipeline_success, calculated_metrics, active_cameras, num_frames_processed = pipeline.run()
        if active_cameras: mlflow.log_param("data.actual_cameras_used", ",".join(active_cameras))
        if num_frames_processed is not None and calculated_metrics is not None:
            unique_indices = calculated_metrics.get('unique_frame_indices_processed', num_frames_processed);
            mlflow.log_param("data.actual_frames_processed_count", unique_indices)
        if calculated_metrics:
            logger.info(f"[{model_name}] Logging detection metrics...")
            log_metrics_dict(calculated_metrics)
            metrics = calculated_metrics
            try:
                metrics_path = Path(f"./run_{run_id}_detection_metrics.json")
                with open(metrics_path, 'w') as f:
                    metrics_serializable = {k: (v.item() if isinstance(v, (np.generic, np.number)) else v) for k, v in
                                            calculated_metrics.items()}
                    for k, v in metrics_serializable.items():
                        if isinstance(v, torch.Tensor): metrics_serializable[
                            k] = v.item() if v.numel() == 1 else v.cpu().cpu().tolist()
                    json.dump(metrics_serializable, f, indent=4)
                mlflow.log_artifact(str(metrics_path), artifact_path="results")
                metrics_path.unlink()
            except Exception as json_err:
                logger.warning(f"Could not log detection metrics dictionary as JSON artifact: {json_err}")
        else:
            logger.warning(f"[{model_name}] No detection metrics were calculated.")
        if pipeline_success:
            run_status = "FINISHED"
            mlflow.set_tag("run_outcome", "Success" if metrics else "Success (No Metrics)")
        else:
            run_status = "FAILED"
            mlflow.set_tag("run_outcome",
                           "Failed (Partial Metrics)" if metrics else "Failed Execution")
    except KeyboardInterrupt:
        logger.warning(
            f"[{model_name}] Detection run interrupted by user (KeyboardInterrupt).");
        run_status = "KILLED";
        mlflow.set_tag(
            "run_outcome", "Killed by user");
        raise
    except Exception as e:
        logger.critical(f"[{model_name}] An uncaught error occurred during the detection run: {e}", exc_info=True);
        run_status = "FAILED";
        mlflow.set_tag("run_outcome", "Crashed")
        try:
            mlflow.log_text(
                f"Error Type: {type(e).__name__}\nError Message: {e}\n\nTraceback:\n{traceback.format_exc()}",
                "error_log.txt")
        except Exception as log_err:
            logger.error(f"Failed to log detection error details to MLflow: {log_err}")
    finally:
        logger.info(f"--- Finished Detection Execution Logic for Run ID: {run_id} (Attempted Status: {run_status}) ---")
    return run_status, metrics


# --- Re-ID Runner (Refactored to use ReidPipeline) ---
def run_single_reid_experiment(
        run_config: Dict[str, Any],
        base_device_preference: str,
        seed: int,
        config_file_path: str,
        log_file_path: Optional[str] = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Executes a single Re-ID MLflow run using the ReidPipeline."""
    active_run = mlflow.active_run()
    if not active_run:
        logger.critical("run_single_reid_experiment called without active MLflow run!")
        return "FAILED", None
    run_id = active_run.info.run_id
    logger.info(f"--- Starting Re-ID Execution Logic for Run ID: {run_id} (Using Pipeline) ---")

    run_status = "FAILED"
    metrics: Optional[Dict[str, Any]] = None
    pipeline: Optional[ReidPipeline] = None
    actual_device: Optional[torch.device] = None
    model_name_tag = run_config.get("model", {}).get("model_type", "unknown_reid")  # Default tag

    try:
        # 1. Determine Device
        resolved_initial_device = get_selected_device(base_device_preference)
        actual_device = resolved_initial_device
        model_config = run_config.get("model", {})
        model_type = model_config.get("model_type", "unknown_reid")
        weights_file = Path(model_config.get("weights_path", "")).stem
        model_name_tag = f"{model_type}_{weights_file}" if weights_file else model_type  # Update tag
        logger.info(f"[{model_name_tag}] Re-ID Device: Requested='{base_device_preference}', Used='{actual_device}'")

        # 2. Log Parameters & Tags
        logger.info(f"[{model_name_tag}] Logging Re-ID parameters...")
        log_params_recursive(run_config)
        mlflow.log_param("environment.seed", seed)
        mlflow.log_param("environment.actual_device_used", str(actual_device))
        logger.info(f"[{model_name_tag}] Logging Re-ID tags...")
        mlflow.set_tag("model_name", model_name_tag)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("weights_file", weights_file if weights_file else "Default")
        mlflow.set_tag("feature_dim", model_config.get("feature_dim", "unknown"))
        mlflow.set_tag("input_size", str(model_config.get("input_size", "unknown")))
        selected_env = run_config.get('data', {}).get('selected_environment', 'unknown')
        env_data = run_config.get('data', {}).get(selected_env, {})
        mlflow.set_tag("dataset_split", run_config.get('data', {}).get('split_type', 'unknown'))
        mlflow.set_tag("scene_id", env_data.get('scene_id', 'unknown'))
        mlflow.set_tag("scene_annotation", env_data.get('scene_annotation_file', 'unknown'))
        if 'camera_ids' in env_data: mlflow.set_tag("camera_ids_loaded", ",".join(sorted(env_data['camera_ids'])))
        log_git_info()

        # 3. Log Config/Requirements
        if config_file_path and Path(config_file_path).is_file():
            mlflow.log_artifact(config_file_path, artifact_path="config")
        req_path = PROJECT_ROOT / "requirements.txt"
        if req_path.is_file():
            mlflow.log_artifact(str(req_path), artifact_path="code")

        # 4. Initialize and Run Re-ID Pipeline
        logger.info(f"[{model_name_tag}] Initializing and running Re-ID pipeline...")
        pipeline = ReidPipeline(run_config, actual_device, PROJECT_ROOT)
        pipeline_success, calculated_metrics, eval_counts = pipeline.run()
        metrics = calculated_metrics

        # Log evaluation counts (e.g., num_crops, num_features, query/gallery size)
        if eval_counts:
            logger.info(f"[{model_name_tag}] Logging evaluation counts...")
            # Log counts as parameters for easier comparison in MLflow UI
            for k, v in eval_counts.items():
                mlflow.log_param(f"eval_counts.{k}", v)

        # 5. Log Metrics
        if metrics:
            logger.info(f"[{model_name_tag}] Logging Re-ID metrics...")
            log_metrics_dict(metrics, is_reid=True)
            try:  # Log metrics dict as JSON artifact
                metrics_path = Path(f"./run_{run_id}_reid_metrics.json")
                with open(metrics_path, 'w') as f:
                    # Handle potential numpy types during serialization
                    metrics_serializable = {k: (v.item() if isinstance(v, (np.generic, np.number)) else v) for k, v in
                                            metrics.items()}
                    json.dump(metrics_serializable, f, indent=4)
                mlflow.log_artifact(str(metrics_path), artifact_path="results")
                metrics_path.unlink()
            except Exception as json_err:
                logger.warning(f"Could not log Re-ID metrics dictionary as JSON artifact: {json_err}")
        else:
            logger.warning(f"[{model_name_tag}] No Re-ID metrics were calculated by the pipeline.")

        # 6. Determine Final Status
        if pipeline_success:
            run_status = "FINISHED"
            mlflow.set_tag("run_outcome", "Success" if metrics else "Success (No Metrics)")
        else:
            run_status = "FAILED"
            mlflow.set_tag("run_outcome", "Failed (Partial Metrics)" if metrics else "Failed Execution")

    except KeyboardInterrupt:
        logger.warning(f"[{model_name_tag}] Re-ID run interrupted by user (KeyboardInterrupt).")
        run_status = "KILLED"
        mlflow.set_tag("run_outcome", "Killed by user")
        raise  # Re-raise to allow outer handler to catch it
    except Exception as e:
        logger.critical(f"[{model_name_tag}] An uncaught error occurred during the Re-ID run: {e}", exc_info=True)
        run_status = "FAILED"
        mlflow.set_tag("run_outcome", "Crashed")
        try:
            # Log error details to MLflow artifact
            mlflow.log_text(
                f"Error Type: {type(e).__name__}\nError Message: {e}\n\nTraceback:\n{traceback.format_exc()}",
                "error_log.txt"
            )
        except Exception as log_err:
            logger.error(f"Failed to log Re-ID error details to MLflow: {log_err}")
    finally:
        logger.info(
            f"--- Finished Re-ID Execution Logic for Run ID: {run_id} [{model_name_tag}] (Attempted Status: {run_status}) ---")

    return run_status, metrics
