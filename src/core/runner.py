import json
import logging
import random
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import cv2
import mlflow
import numpy as np
import torch
from PIL import Image
from mlflow.models import infer_signature
from tqdm import tqdm

# --- Local Imports Need Correct Path Handling ---
try:
    from src.utils.device_utils import get_selected_device
    from src.data.loader import FrameDataLoader
    from src.pipelines.detection_pipeline import DetectionPipeline
    from src.tracking.strategies import (
        DetectionTrackingStrategy, YoloStrategy, RTDetrStrategy, FasterRCNNStrategy, RfDetrStrategy
    )
    from src.data.reid_dataset_loader import ReidDatasetLoader, ReidCropInfo
    from src.reid.strategies import ReIDStrategy, get_reid_strategy_from_run_config
    from src.evaluation.reid_metrics import compute_reid_metrics
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
    from data.reid_dataset_loader import ReidDatasetLoader, ReidCropInfo
    from reid.strategies import ReIDStrategy, get_reid_strategy_from_run_config
    from evaluation.reid_metrics import compute_reid_metrics
# --- End Local Import Handling ---

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


# --- Helper Functions ---
def log_params_recursive(params_dict: Dict[str, Any], parent_key: str = ""):
    """Recursively logs parameters to the *current active* MLflow run."""
    if not mlflow.active_run():
        # logger.warning("Attempted to log parameters outside of an active MLflow run.")
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
        # project_root = Path(__file__).parent.parent.parent.resolve() # Already defined globally
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
                    # Limit diff size for MLflow text artifact
                    max_diff_len = 100 * 1024  # 100 KB limit? Check MLflow docs
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


# --- Detection Signature Helper ---
def get_sample_data_for_signature(
        config: Dict[str, Any], strategy: DetectionTrackingStrategy, device: torch.device
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Loads a sample frame and runs inference to get sample input/output
    for MLflow signature inference (Detection Models).
    """
    # This function is for DETECTION models and remains unchanged from previous version.
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
                                          'xyxy'): out = detections.xyxy; sample_output_sig = out.cpu().numpy() if isinstance(
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


# --- Detection Runner ---
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
        "run_single_experiment called without an active MLflow run!"); return "FAILED", None
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
                f"[{model_name}] FasterRCNN requested/resolved to '{resolved_initial_device.type}'. Forcing to CPU."); actual_device = torch.device(
                'cpu'); device_override_reason = f"FasterRCNN forced to CPU from {resolved_initial_device.type}"
        elif resolved_initial_device.type == 'mps':
            if model_type in ['rfdetr']: logger.warning(
                f"[{model_name}] Model type '{model_type}' has known issues on MPS. Forcing to CPU."); actual_device = torch.device(
                'cpu'); device_override_reason = f"{model_type} forced to CPU due to MPS issues"
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
                    signature = infer_signature(sample_input_data, sample_output_data); logger.info(
                        f"[{model_name}] Detection signature inferred."); mlflow.log_dict(signature.to_dict(),
                                                                                          "signature.json")
                except Exception as infer_err:
                    logger.warning(f"[{model_name}] Could not infer detection signature: {infer_err}",
                                   exc_info=False); signature = None
            else:
                logger.warning(f"[{model_name}] Failed to generate sample data for detection signature.")
            if sample_input_data is not None:
                try:
                    if isinstance(sample_input_data, np.ndarray):
                        np.save(f"input_example_{run_id}.npy", sample_input_data); mlflow.log_artifact(
                            f"input_example_{run_id}.npy", artifact_path="examples"); Path(
                            f"input_example_{run_id}.npy").unlink()
                    elif isinstance(sample_input_data, Image.Image):
                        sample_input_data.save(f"input_example_{run_id}.png"); mlflow.log_artifact(
                            f"input_example_{run_id}.png", artifact_path="examples"); Path(
                            f"input_example_{run_id}.png").unlink()
                except Exception as ex_log_err:
                    logger.warning(f"Failed to log detection input example artifact: {ex_log_err}")
        else:
            logger.warning(f"[{model_name}] Detection strategy not available, cannot generate signature.")
        logger.info(f"[{model_name}] Starting detection pipeline processing...");
        pipeline_success, calculated_metrics, active_cameras, num_frames_processed = pipeline.run()
        if active_cameras: mlflow.log_param("data.actual_cameras_used", ",".join(active_cameras))
        if num_frames_processed is not None and calculated_metrics is not None: unique_indices = calculated_metrics.get(
            'unique_frame_indices_processed', num_frames_processed); mlflow.log_param(
            "data.actual_frames_processed_count", unique_indices)
        if calculated_metrics:
            logger.info(f"[{model_name}] Logging detection metrics...");
            log_metrics_dict(calculated_metrics);
            metrics = calculated_metrics
            try:
                metrics_path = Path(f"./run_{run_id}_detection_metrics.json")
                with open(metrics_path, 'w') as f:
                    metrics_serializable = {k: (v.item() if isinstance(v, (np.generic, np.number)) else v) for k, v in
                                            calculated_metrics.items()}
                    for k, v in metrics_serializable.items():
                        if isinstance(v, torch.Tensor): metrics_serializable[
                            k] = v.item() if v.numel() == 1 else v.cpu().tolist()
                    json.dump(metrics_serializable, f, indent=4)
                mlflow.log_artifact(str(metrics_path), artifact_path="results");
                metrics_path.unlink()
            except Exception as json_err:
                logger.warning(f"Could not log detection metrics dictionary as JSON artifact: {json_err}")
        else:
            logger.warning(f"[{model_name}] No detection metrics were calculated.")
        if pipeline_success:
            run_status = "FINISHED"; mlflow.set_tag("run_outcome", "Success" if metrics else "Success (No Metrics)")
        else:
            run_status = "FAILED"; mlflow.set_tag("run_outcome",
                                                  "Failed (Partial Metrics)" if metrics else "Failed Execution")
    except KeyboardInterrupt:
        logger.warning(
            f"[{model_name}] Detection run interrupted by user (KeyboardInterrupt)."); run_status = "KILLED"; mlflow.set_tag(
            "run_outcome", "Killed by user"); raise
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


# --- Re-ID Runner (Standard Query/Gallery Evaluation) ---
def run_single_reid_experiment(
        run_config: Dict[str, Any],
        base_device_preference: str,
        seed: int,
        config_file_path: str,
        log_file_path: Optional[str] = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Executes a single Re-ID MLflow run using standard Query/Gallery evaluation."""
    active_run = mlflow.active_run()
    if not active_run: logger.critical(
        "run_single_reid_experiment called without active MLflow run!"); return "FAILED", None
    run_id = active_run.info.run_id
    logger.info(f"--- Starting Re-ID Execution Logic for Run ID: {run_id} (Query/Gallery) ---")

    run_status = "FAILED";
    metrics = None;
    reid_strategy: Optional[ReIDStrategy] = None
    actual_device: Optional[torch.device] = None;
    data_loader: Optional[ReidDatasetLoader] = None
    # Store features as list of tuples: (feature_vector, global_id, camera_id, frame_index)
    extracted_features_list: List[Tuple[np.ndarray, int, str, int]] = []

    try:
        # 1. Determine Device
        resolved_initial_device = get_selected_device(base_device_preference);
        actual_device = resolved_initial_device
        model_config = run_config.get("model", {});
        model_type = model_config.get("model_type", "unknown_reid")
        weights_file = Path(model_config.get("weights_path", "")).stem
        model_name_tag = f"{model_type}_{weights_file}" if weights_file else model_type
        logger.info(f"[{model_name_tag}] Re-ID Device: Requested='{base_device_preference}', Used='{actual_device}'")

        # 2. Log Parameters & Tags
        logger.info(f"[{model_name_tag}] Logging Re-ID parameters...");
        log_params_recursive(run_config)
        mlflow.log_param("environment.seed", seed);
        mlflow.log_param("environment.actual_device_used", str(actual_device))
        logger.info(f"[{model_name_tag}] Logging Re-ID tags...");
        mlflow.set_tag("model_name", model_name_tag);
        mlflow.set_tag("model_type", model_type);
        mlflow.set_tag("weights_file", weights_file if weights_file else "Default")
        mlflow.set_tag("feature_dim", model_config.get("feature_dim", "unknown"));
        mlflow.set_tag("input_size", str(model_config.get("input_size", "unknown")))
        selected_env = run_config.get('data', {}).get('selected_environment', 'unknown')
        env_data = run_config.get('data', {}).get(selected_env, {})
        mlflow.set_tag("dataset_split", run_config.get('data', {}).get('split_type', 'unknown'))
        mlflow.set_tag("scene_id", env_data.get('scene_id', 'unknown'))
        mlflow.set_tag("scene_annotation", env_data.get('scene_annotation_file', 'unknown'))
        if 'camera_ids' in env_data: mlflow.set_tag("camera_ids_loaded", ",".join(sorted(env_data['camera_ids'])))
        log_git_info()

        # 3. Log Config/Requirements
        if config_file_path and Path(config_file_path).is_file(): mlflow.log_artifact(config_file_path,
                                                                                      artifact_path="config")
        req_path = PROJECT_ROOT / "requirements.txt";
        if req_path.is_file(): mlflow.log_artifact(str(req_path), artifact_path="code")

        # 4. Initialize Data Loader
        logger.info(f"[{model_name_tag}] Initializing Re-ID dataset loader...")
        try:
            data_loader = ReidDatasetLoader(run_config)
            all_crop_data: List[ReidCropInfo] = data_loader.get_data()
            if not all_crop_data: raise ValueError("Re-ID data loader returned no crop info.")
            logger.info(f"Found {len(all_crop_data)} valid crops for {model_name_tag}.")
            mlflow.log_param("data.total_gt_crops_found", len(all_crop_data))
            unique_ids = len(set(c.instance_id for c in all_crop_data))
            mlflow.log_param("data.unique_person_ids_found", unique_ids)
        except Exception as data_err:
            logger.critical(f"[{model_name_tag}] Failed to load data: {data_err}", exc_info=True); raise

        # 5. Initialize Re-ID Strategy
        logger.info(f"[{model_name_tag}] Initializing Re-ID strategy...")
        try:
            # Pass the full run_config so strategy can get weights_base_dir
            reid_strategy = get_reid_strategy_from_run_config(run_config, actual_device, PROJECT_ROOT)
            if reid_strategy is None: raise RuntimeError("Failed to get ReID strategy.")
            logger.info(f"Re-ID strategy '{reid_strategy.__class__.__name__}' initialized.")
        except Exception as strategy_err:
            logger.critical(f"[{model_name_tag}] Failed to init strategy: {strategy_err}", exc_info=True); raise

        # 6. Feature Extraction
        logger.info(f"[{model_name_tag}] Starting feature extraction for {len(all_crop_data)} crops...")
        data_grouped_by_frame: Dict[str, List[ReidCropInfo]] = defaultdict(list)
        for crop_info in all_crop_data: data_grouped_by_frame[crop_info.frame_path].append(crop_info)

        extraction_start_time = time.time()
        with tqdm(total=len(data_grouped_by_frame), desc=f"Extracting Features ({model_name_tag})") as pbar:
            for frame_path_str, crops_in_frame in data_grouped_by_frame.items():
                pbar.set_postfix({"Frame": Path(frame_path_str).name});
                frame_path = Path(frame_path_str)
                if not frame_path.is_file(): logger.warning(f"Frame file not found: {frame_path}"); pbar.update(
                    1); continue
                try:
                    frame_bytes = np.fromfile(str(frame_path), dtype=np.uint8);
                    frame_bgr = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
                    if frame_bgr is None or frame_bgr.size == 0: logger.warning(
                        f"Failed decode frame: {frame_path}"); pbar.update(1); continue
                    bboxes_xyxy = np.array([c.bbox_xyxy for c in crops_in_frame], dtype=np.float32)
                    # Ensure bounding boxes are valid before passing to strategy
                    valid_bbox_indices = [i for i, bbox in enumerate(bboxes_xyxy) if
                                          bbox[2] > bbox[0] and bbox[3] > bbox[1]]
                    if not valid_bbox_indices: continue  # Skip if no valid boxes in this frame's group

                    valid_bboxes_xyxy = bboxes_xyxy[valid_bbox_indices]
                    valid_crops_in_frame = [crops_in_frame[i] for i in valid_bbox_indices]

                    features_dict: Dict[int, np.ndarray] = reid_strategy.extract_features(frame_bgr, valid_bboxes_xyxy)

                    if features_dict:
                        # Map features back using the index within the *valid* subset
                        for valid_idx, crop_info in enumerate(valid_crops_in_frame):
                            feature = features_dict.get(valid_idx)  # Index corresponds to valid_bboxes_xyxy
                            if feature is not None and isinstance(feature,
                                                                  np.ndarray) and feature.size > 0 and crop_info.frame_index is not None:
                                extracted_features_list.append(
                                    (feature, crop_info.instance_id, crop_info.camera_id, crop_info.frame_index))
                            # else: logger.debug(f"Feature not extracted for valid_idx {valid_idx} in frame {frame_path.name}") # Too verbose

                except Exception as frame_proc_err:
                    logger.error(f"Error processing frame {frame_path}: {frame_proc_err}", exc_info=True)
                finally:
                    pbar.update(1)

        extraction_time = time.time() - extraction_start_time
        logger.info(f"[{model_name_tag}] Feature extraction completed in {extraction_time:.2f}s.")
        logger.info(f"Extracted {len(extracted_features_list)} features.")
        mlflow.log_metric("perf_feature_extraction_time_sec", round(extraction_time, 2))
        mlflow.log_metric("perf_total_features_extracted", len(extracted_features_list))

        if not extracted_features_list: logger.error(
            f"[{model_name_tag}] No features extracted."); run_status = "FAILED"; mlflow.set_tag("run_outcome",
                                                                                                 "Failed (No Features)"); return run_status, metrics

        # 7. Re-ID Evaluation (Standard Query/Gallery)
        logger.info(f"[{model_name_tag}] Preparing standard Query/Gallery sets...")
        # Get query/gallery camera sets from the specific environment config
        env_specific_config = run_config.get("data", {}).get(selected_env, {})
        query_cams = set(env_specific_config.get("query_cameras", []))
        gallery_cams = set(env_specific_config.get("gallery_cameras", []))
        distance_metric = run_config.get("evaluation", {}).get("distance_metric", "cosine")

        if not query_cams or not gallery_cams:
            logger.error("Query or Gallery cameras not defined in config for selected environment. Cannot evaluate.");
            run_status = "FAILED";
            mlflow.set_tag("run_outcome", "Failed (Config Error)");
            return run_status, metrics
        if not query_cams.isdisjoint(gallery_cams):
            logger.warning(
                f"Query cameras {query_cams} and gallery cameras {gallery_cams} overlap. Standard ReID typically uses disjoint sets.")
            # Evaluation logic should still work correctly due to same-cam filtering in metrics.

        query_data: List[Tuple[np.ndarray, int, str]] = []  # (feature, global_id, camera_id)
        gallery_data: List[Tuple[np.ndarray, int, str]] = []  # (feature, global_id, camera_id)
        all_gids = sorted(list(set(item[1] for item in extracted_features_list)))  # Get all unique person IDs

        logger.info(f"Found {len(all_gids)} unique person IDs. Selecting queries...")
        # --- Query Selection: Pick one instance per person ID from query cameras ---
        # Group features by person ID first
        features_by_gid = defaultdict(list)
        for feature, gid, cid, fidx in extracted_features_list:
            if cid in query_cams:
                features_by_gid[gid].append((feature, cid))  # Store feature and its camera

        selected_query_count = 0
        for gid in all_gids:
            if gid in features_by_gid:
                # Randomly pick one instance from the available query cameras for this ID
                chosen_query = random.choice(features_by_gid[gid])
                query_data.append((chosen_query[0], gid, chosen_query[1]))  # (feature, gid, query_cam_id)
                selected_query_count += 1

        # --- Gallery Selection: All instances from gallery cameras ---
        # Include distractors (IDs not in query set) if they appear in gallery cams
        query_ids_set = set(item[1] for item in query_data)
        for feature, gid, cid, fidx in extracted_features_list:
            if cid in gallery_cams:
                # Add if it's from a gallery camera. The metric function will handle
                # filtering out same-ID, same-camera results during comparison.
                gallery_data.append((feature, gid, cid))

        logger.info(
            f"Prepared {len(query_data)} queries (one per ID found in query cams: {selected_query_count}) and {len(gallery_data)} gallery items.")
        mlflow.log_param("eval_query_count", len(query_data))
        mlflow.log_param("eval_gallery_count", len(gallery_data))
        mlflow.log_param("eval_query_cameras", ",".join(sorted(list(query_cams))))
        mlflow.log_param("eval_gallery_cameras", ",".join(sorted(list(gallery_cams))))

        if not query_data or not gallery_data: logger.error(
            f"[{model_name_tag}] Query or Gallery set is empty."); run_status = "FAILED"; mlflow.set_tag("run_outcome",
                                                                                                         "Failed (Empty Query/Gallery)"); return run_status, metrics

        # Calculate Re-ID metrics
        query_features_np = np.array([item[0] for item in query_data])
        query_gids_np = np.array([item[1] for item in query_data])
        query_cids_list = [item[2] for item in query_data]
        gallery_features_np = np.array([item[0] for item in gallery_data])
        gallery_gids_np = np.array([item[1] for item in gallery_data])
        gallery_cids_list = [item[2] for item in gallery_data]

        try:
            metrics = compute_reid_metrics(query_features_np, query_gids_np, query_cids_list, gallery_features_np,
                                           gallery_gids_np, gallery_cids_list, distance_metric)
            logger.info(f"[{model_name_tag}] Re-ID Metrics: {metrics}")
        except Exception as eval_err:
            logger.error(f"[{model_name_tag}] Failed compute metrics: {eval_err}", exc_info=True); metrics = None

        # 8. Log Metrics
        if metrics:
            log_metrics_dict(metrics, is_reid=True)
            try:  # Log metrics dict as JSON
                metrics_path = Path(f"./run_{run_id}_reid_metrics.json")
                with open(metrics_path, 'w') as f:
                    metrics_serializable = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in
                                            metrics.items()}; json.dump(metrics_serializable, f, indent=4)
                mlflow.log_artifact(str(metrics_path), artifact_path="results");
                metrics_path.unlink()
            except Exception as json_err:
                logger.warning(f"Could not log Re-ID metrics dict: {json_err}")
        else:
            logger.error(f"[{model_name_tag}] No Re-ID metrics calculated.")

        # 9. Final Status
        if metrics:
            run_status = "FINISHED"; mlflow.set_tag("run_outcome", "Success")
        else:
            run_status = "FAILED"; mlflow.set_tag("run_outcome", "Failed (Metric Error)")

    except KeyboardInterrupt:
        logger.warning(f"[{model_name_tag}] Re-ID run interrupted."); run_status = "KILLED"; mlflow.set_tag(
            "run_outcome", "Killed"); raise
    except Exception as e:
        model_name_tag = run_config.get("model", {}).get("model_type", "unknown_reid");
        logger.critical(f"[{model_name_tag}] Uncaught error: {e}", exc_info=True);
        run_status = "FAILED";
        mlflow.set_tag("run_outcome", "Crashed")
        try:
            mlflow.log_text(
                f"Error Type: {type(e).__name__}\nError Message: {e}\n\nTraceback:\n{traceback.format_exc()}",
                "error_log.txt")
        except Exception as log_err:
            logger.error(f"Failed to log error details: {log_err}")
    finally:
        model_name_tag = run_config.get("model", {}).get("model_type", "unknown_reid"); logger.info(
            f"--- Finished Re-ID Execution Logic for Run ID: {run_id} [{model_name_tag}] (Status: {run_status}) ---")

    return run_status, metrics
