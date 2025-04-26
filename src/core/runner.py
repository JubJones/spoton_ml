import json
import logging
import subprocess
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import cv2
import mlflow
import numpy as np
import torch
from PIL import Image
from mlflow.models import infer_signature

# --- Local Imports Need Correct Path Handling ---
try:
    from src.utils.device_utils import get_selected_device
    from src.reid.strategies import get_reid_device_specifier_string
    # Detection Imports (kept for run_single_experiment)
    from src.data.loader import FrameDataLoader
    from src.pipelines.detection_pipeline import DetectionPipeline
    from src.tracking.strategies import (
        DetectionTrackingStrategy, YoloStrategy, RTDetrStrategy, FasterRCNNStrategy, RfDetrStrategy
    )
    # Re-ID Imports (kept for run_single_reid_experiment)
    from src.data.reid_dataset_loader import ReidDatasetLoader, ReidCropInfo
    from src.reid.strategies import ReIDStrategy, get_reid_strategy_from_run_config
    from src.evaluation.reid_metrics import compute_reid_metrics
    from src.pipelines.reid_pipeline import ReidPipeline
    # Tracking+ReID Imports (New)
    from src.pipelines.tracking_reid_pipeline import TrackingReidPipeline, TrackingResultSummary

except ImportError:
    import sys

    if str(Path(__file__).parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent.parent))
    # Re-attempt imports after potentially fixing path
    from utils.device_utils import get_selected_device
    from reid.strategies import get_reid_device_specifier_string
    # Detection Imports
    from data.loader import FrameDataLoader
    from pipelines.detection_pipeline import DetectionPipeline
    from tracking.strategies import (
        DetectionTrackingStrategy, YoloStrategy, RTDetrStrategy, FasterRCNNStrategy, RfDetrStrategy
    )
    # Re-ID Imports
    from data.reid_dataset_loader import ReidDatasetLoader, ReidCropInfo
    from src.reid.strategies import ReIDStrategy, get_reid_strategy_from_run_config
    from evaluation.reid_metrics import compute_reid_metrics
    from pipelines.reid_pipeline import ReidPipeline
    # Tracking+ReID Imports
    from pipelines.tracking_reid_pipeline import TrackingReidPipeline, TrackingResultSummary
# --- End Local Import Handling ---

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


# --- Helper Functions (log_params_recursive, log_metrics_dict, log_git_info) ---
# (These functions remain unchanged from the original)
def log_params_recursive(params_dict: Dict[str, Any], parent_key: str = ""):
    """Recursively logs parameters to the *current active* MLflow run."""
    if not mlflow.active_run():
        logger.warning("Attempted to log parameters outside of an active MLflow run.")
        return
    for key, value in params_dict.items():
        mlflow_key = f"{parent_key}.{key}" if parent_key else key
        # Prevent logging huge lists/dicts - log count or summary instead
        if isinstance(value, dict):
            if key in ['models_to_run', 'trackers_to_run', 'reid_models_to_associate'] and parent_key == '':
                mlflow.log_param(mlflow_key + ".count", len(value))
                continue
            # Avoid logging injected run config back or potentially large nested data structures
            if key in ['_run_config', 'campus', 'factory']:
                continue
            if len(str(value)) > 500: # Heuristic for large dicts
                 mlflow.log_param(f"{mlflow_key}.keys", list(value.keys())[:10]) # Log some keys
                 continue
            log_params_recursive(value, mlflow_key)
        elif isinstance(value, list):
             try:
                 # Log count if list is large
                 if len(value) > 20:
                     mlflow.log_param(mlflow_key + ".count", len(value))
                     param_val_str = json.dumps(value[:5] + ["..."]) # Log first few items
                 else:
                    param_val_str = json.dumps(value)

                 if len(param_val_str) > 500:
                     param_val_str = param_val_str[:497] + "..."
                 mlflow.log_param(mlflow_key, param_val_str)
             except TypeError:
                 mlflow.log_param(mlflow_key, str(value)[:500]) # Fallback
        else:
            # Log scalar values, truncate if too long
            mlflow.log_param(mlflow_key, str(value)[:500])


def log_metrics_dict(metrics: Dict[str, Any], prefix: str = "eval"):
    """Logs metrics from a dictionary to the *current active* MLflow run, adding a prefix."""
    if not mlflow.active_run():
        logger.warning("Attempted to log metrics outside of an active MLflow run.")
        return
    if not metrics:
        logger.warning("Metrics dictionary is empty. Nothing to log.")
        return

    # Filter only numeric types that MLflow can handle directly
    numeric_metrics = {
        f"{prefix}_{k.replace('-', '_').replace('.', '_')}": v for k, v in metrics.items()
        if isinstance(v, (int, float, np.number)) and np.isfinite(v)
    }
    non_numeric_keys = [k for k in metrics if k not in [nk.split('_', 1)[1] for nk in numeric_metrics.keys()]] # Rough check

    if non_numeric_keys:
        # Log non-numeric items as params if they are simple types
        for key in non_numeric_keys:
            value = metrics[key]
            if isinstance(value, (str, bool)):
                 mlflow_key = f"{prefix}_{key.replace('-', '_').replace('.', '_')}"
                 mlflow.log_param(mlflow_key, str(value)[:500])
            else:
                 logger.debug(f"Skipping non-numeric/non-simple metric/param: {key} (type: {type(value)})")


    if numeric_metrics:
        log_type = prefix.upper()
        logger.info(f"Logging {len(numeric_metrics)} numeric {log_type} metrics...")
        try:
            # Log metrics in batches for efficiency
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
        logger.warning(f"No numeric metrics found in the provided dictionary for prefix '{prefix}'.")


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
                # Log diff only if status is dirty
                diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=PROJECT_ROOT,
                                               stderr=subprocess.STDOUT).strip().decode('utf-8', errors='ignore')
                if diff:
                    max_diff_len = 100 * 1024 # Limit diff size artifact
                    if len(diff) > max_diff_len:
                        diff = diff[:max_diff_len] + "\n... (diff truncated)"
                    mlflow.log_text(diff, artifact_file="code/git_diff.diff")
                    logger.info("Logged git diff as artifact (due to dirty status).")
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


# --- Detection Runner (run_single_experiment - Unchanged) ---
def run_single_experiment(
        run_config: Dict[str, Any],
        base_device_preference: str,
        seed: int,
        config_file_path: str,
        log_file_path: Optional[str] = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Executes a single DETECTION MLflow run based on the provided configuration."""
    active_run = mlflow.active_run()
    if not active_run:
        logger.critical("run_single_experiment called without an active MLflow run!")
        return "FAILED", None
    run_id = active_run.info.run_id
    logger.info(f"--- Starting Detection Execution Logic for Run ID: {run_id} ---")
    run_status = "FAILED"
    metrics = None
    pipeline: Optional[DetectionPipeline] = None
    actual_device: Optional[torch.device] = None
    model_name_tag = run_config.get("model", {}).get("model_name", "unknown_det")

    try:
        # --- Device Selection ---
        resolved_initial_device = get_selected_device(base_device_preference)
        actual_device = resolved_initial_device # Start with resolved
        device_override_reason = "None"
        model_type = run_config.get("model", {}).get("type", "").lower()

        # Apply device overrides if necessary (example: FasterRCNN on CPU)
        if model_type == "fasterrcnn":
            if resolved_initial_device.type != 'cpu': # Simpler: If not CPU, force CPU
                 logger.warning(f"[{model_name_tag}] FasterRCNN requested/resolved to '{resolved_initial_device.type}'. Forcing to CPU.")
                 actual_device = torch.device('cpu')
                 device_override_reason = f"FasterRCNN forced to CPU from {resolved_initial_device.type}"
        elif resolved_initial_device.type == 'mps':
             # Example: RFDETR might have MPS issues
             if model_type in ['rfdetr']:
                 logger.warning(f"[{model_name_tag}] Model type '{model_type}' may have issues on MPS. Forcing to CPU.")
                 actual_device = torch.device('cpu')
                 device_override_reason = f"{model_type} forced to CPU due to potential MPS issues"

        logger.info(f"[{model_name_tag}] Device Selection: Requested='{base_device_preference}', Resolved='{resolved_initial_device}', Actual Used='{actual_device}', Override Reason='{device_override_reason}'")

        # --- MLflow Logging: Parameters & Tags ---
        logger.info(f"[{model_name_tag}] Logging detection parameters...")
        log_params_recursive(run_config)
        mlflow.log_param("environment.seed", seed)
        mlflow.log_param("environment.requested_device", base_device_preference)
        mlflow.log_param("environment.resolved_initial_device", str(resolved_initial_device))
        mlflow.log_param("environment.actual_device_used", str(actual_device))
        mlflow.log_param("environment.device_override_reason", device_override_reason)

        logger.info(f"[{model_name_tag}] Logging detection tags...")
        mlflow.set_tag("model_name", model_name_tag)
        mlflow.set_tag("model_type", model_type)
        selected_env = run_config.get('data', {}).get('selected_environment', 'unknown')
        mlflow.set_tag("dataset", selected_env)
        env_data = run_config.get('data', {}).get(selected_env, {})
        mlflow.set_tag("scene_id", env_data.get('scene_id', 'unknown'))
        log_git_info()

        # Log Config/Requirements Artifacts
        if config_file_path and Path(config_file_path).is_file():
            mlflow.log_artifact(config_file_path, artifact_path="config")
        req_path = PROJECT_ROOT / "requirements.txt"
        if req_path.is_file():
            mlflow.log_artifact(str(req_path), artifact_path="code")

        # --- Pipeline Initialization & Execution ---
        logger.info(f"[{model_name_tag}] Initializing detection pipeline on device '{actual_device}'...")
        pipeline = DetectionPipeline(run_config, actual_device)
        if not pipeline.initialize_components():
            raise RuntimeError(f"Detection pipeline initialization failed for model {model_name_tag}")
        logger.info(f"[{model_name_tag}] Detection Pipeline initialized successfully.")

        # --- MLflow Signature (Optional) ---
        # Note: get_sample_data_for_signature might need refinement for robustness
        signature = None
        try:
            if pipeline.detection_strategy:
                 logger.info(f"[{model_name_tag}] Attempting to generate detection model signature...")
                 sample_input_data, sample_output_data = get_sample_data_for_signature(run_config, pipeline.detection_strategy, actual_device)
                 if sample_input_data is not None and sample_output_data is not None:
                     signature = infer_signature(sample_input_data, sample_output_data)
                     logger.info(f"[{model_name_tag}] Detection signature inferred.")
                     mlflow.log_dict(signature.to_dict(), "signature.json")
                 else: logger.warning(f"[{model_name_tag}] Failed to generate sample data for detection signature.")
                 # Log input example if available
                 if sample_input_data is not None:
                    example_dir = Path("./mlflow_examples")
                    example_dir.mkdir(exist_ok=True)
                    try:
                        if isinstance(sample_input_data, np.ndarray):
                            ex_path = example_dir / f"input_example_{run_id}.npy"
                            np.save(ex_path, sample_input_data)
                        elif isinstance(sample_input_data, Image.Image):
                            ex_path = example_dir / f"input_example_{run_id}.png"
                            sample_input_data.save(ex_path)
                        else: ex_path = None

                        if ex_path:
                             mlflow.log_artifact(str(ex_path.resolve()), artifact_path="examples")
                             ex_path.unlink() # Clean up local file
                    except Exception as ex_log_err: logger.warning(f"Failed to log detection input example artifact: {ex_log_err}")

            else: logger.warning(f"[{model_name_tag}] Detection strategy not available, cannot generate signature.")
        except Exception as sig_err: logger.warning(f"[{model_name_tag}] Error during signature generation: {sig_err}", exc_info=False)


        # --- Run Pipeline ---
        logger.info(f"[{model_name_tag}] Starting detection pipeline processing...")
        pipeline_success, calculated_metrics, active_cameras, _ = pipeline.run() # num_frames logged internally

        # --- Log Metrics & Results ---
        if active_cameras: mlflow.log_param("data.actual_cameras_used", ",".join(sorted(active_cameras)))

        if calculated_metrics:
            logger.info(f"[{model_name_tag}] Logging detection metrics...")
            log_metrics_dict(calculated_metrics, prefix="detection") # Use prefix
            metrics = calculated_metrics # Store for return
            try:
                # Log metrics dictionary as JSON artifact
                metrics_path = Path(f"./run_{run_id}_detection_metrics.json")
                with open(metrics_path, 'w') as f:
                    metrics_serializable = {}
                    for k, v in calculated_metrics.items():
                         # Attempt conversion for numpy/torch types
                         if isinstance(v, (np.generic, np.number)): metrics_serializable[k] = v.item()
                         elif isinstance(v, torch.Tensor): metrics_serializable[k] = v.item() if v.numel() == 1 else v.cpu().tolist()
                         elif isinstance(v, (int, float, str, bool)): metrics_serializable[k] = v
                         else: metrics_serializable[k] = str(v) # Fallback string conversion
                    json.dump(metrics_serializable, f, indent=4)
                mlflow.log_artifact(str(metrics_path), artifact_path="results")
                metrics_path.unlink() # Clean up local file
            except Exception as json_err:
                logger.warning(f"Could not log detection metrics dictionary as JSON artifact: {json_err}")
        else:
            logger.warning(f"[{model_name_tag}] No detection metrics were calculated.")

        # --- Final Status ---
        if pipeline_success:
            run_status = "FINISHED"
            mlflow.set_tag("run_outcome", "Success" if metrics else "Success (No Metrics)")
        else:
            run_status = "FAILED"
            mlflow.set_tag("run_outcome", "Failed (Partial Metrics)" if metrics else "Failed Execution")

    except KeyboardInterrupt:
        logger.warning(f"[{model_name_tag}] Detection run interrupted by user (KeyboardInterrupt).")
        run_status = "KILLED"
        mlflow.set_tag("run_outcome", "Killed by user")
        raise # Re-raise to allow outer handler to catch it
    except Exception as e:
        logger.critical(f"[{model_name_tag}] An uncaught error occurred during the detection run: {e}", exc_info=True)
        run_status = "FAILED"
        mlflow.set_tag("run_outcome", "Crashed")
        try: # Log error details to MLflow artifact
            mlflow.log_text(
                f"Error Type: {type(e).__name__}\nError Message: {e}\n\nTraceback:\n{traceback.format_exc()}",
                "error_log.txt"
            )
        except Exception as log_err: logger.error(f"Failed to log detection error details to MLflow: {log_err}")
    finally:
        logger.info(f"--- Finished Detection Execution Logic for Run ID: {run_id} [{model_name_tag}] (Attempted Status: {run_status}) ---")

    return run_status, metrics


# --- Re-ID Runner (run_single_reid_experiment - Unchanged) ---
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
    logger.info(f"--- Starting Re-ID Execution Logic for Run ID: {run_id} (Using ReidPipeline) ---")

    run_status = "FAILED"
    metrics: Optional[Dict[str, Any]] = None
    pipeline: Optional[ReidPipeline] = None
    actual_device: Optional[torch.device] = None
    model_name_tag = "unknown_reid" # Default tag

    try:
        # --- Device Selection ---
        resolved_initial_device = get_selected_device(base_device_preference)
        actual_device = resolved_initial_device # Start with resolved device
        # Note: ReID models might have different device compatibilities, handled in strategy/pipeline
        model_config = run_config.get("model", {})
        model_type = model_config.get("model_type", "unknown_reid")
        weights_file = Path(model_config.get("weights_path", "")).stem
        model_name_tag = f"{model_type}_{weights_file}" if weights_file else model_type # Update tag
        logger.info(f"[{model_name_tag}] Re-ID Device: Requested='{base_device_preference}', Used='{actual_device}'")

        # --- MLflow Logging: Parameters & Tags ---
        logger.info(f"[{model_name_tag}] Logging Re-ID parameters...")
        log_params_recursive(run_config)
        mlflow.log_param("environment.seed", seed)
        mlflow.log_param("environment.actual_device_used", str(actual_device)) # Log the device used by pipeline

        logger.info(f"[{model_name_tag}] Logging Re-ID tags...")
        mlflow.set_tag("model_name", model_name_tag)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("weights_file", weights_file if weights_file else "Default/None")
        mlflow.set_tag("feature_dim", model_config.get("feature_dim", "unknown"))
        mlflow.set_tag("input_size", str(model_config.get("input_size", "unknown")))
        selected_env = run_config.get('data', {}).get('selected_environment', 'unknown')
        env_data = run_config.get('data', {}).get(selected_env, {})
        mlflow.set_tag("dataset_split", run_config.get('data', {}).get('split_type', 'unknown'))
        mlflow.set_tag("scene_id", env_data.get('scene_id', 'unknown'))
        mlflow.set_tag("scene_annotation", env_data.get('scene_annotation_file', 'unknown'))
        if 'camera_ids' in env_data: mlflow.set_tag("camera_ids_loaded", ",".join(sorted(env_data['camera_ids'])))
        log_git_info()

        # Log Config/Requirements Artifacts
        if config_file_path and Path(config_file_path).is_file():
            mlflow.log_artifact(config_file_path, artifact_path="config")
        req_path = PROJECT_ROOT / "requirements.txt"
        if req_path.is_file():
            mlflow.log_artifact(str(req_path), artifact_path="code")

        # --- Pipeline Initialization & Execution ---
        logger.info(f"[{model_name_tag}] Initializing and running Re-ID pipeline...")
        # Pass project root for potential weight path resolution inside pipeline/strategy
        pipeline = ReidPipeline(run_config, actual_device, PROJECT_ROOT)
        pipeline_success, calculated_metrics, eval_counts = pipeline.run()
        metrics = calculated_metrics # Store for return

        # --- Log Metrics & Results ---
        # Log evaluation counts (e.g., num_crops, query/gallery size) as params
        if eval_counts:
            logger.info(f"[{model_name_tag}] Logging evaluation counts as params...")
            for k, v in eval_counts.items():
                 # Ensure keys are MLflow compatible
                 mlflow_key = f"eval_counts.{k.replace('-', '_').replace('.', '_')}"
                 mlflow.log_param(mlflow_key, v)

        if metrics:
            logger.info(f"[{model_name_tag}] Logging Re-ID metrics...")
            log_metrics_dict(metrics, prefix="reid") # Use prefix
            try:  # Log metrics dict as JSON artifact
                metrics_path = Path(f"./run_{run_id}_reid_metrics.json")
                with open(metrics_path, 'w') as f:
                    # Handle potential numpy/torch types during serialization
                    metrics_serializable = {}
                    for k, v in metrics.items():
                         if isinstance(v, (np.generic, np.number)): metrics_serializable[k] = v.item()
                         elif isinstance(v, torch.Tensor): metrics_serializable[k] = v.item() if v.numel() == 1 else v.cpu().tolist()
                         elif isinstance(v, (int, float, str, bool)): metrics_serializable[k] = v
                         else: metrics_serializable[k] = str(v)
                    json.dump(metrics_serializable, f, indent=4)
                mlflow.log_artifact(str(metrics_path), artifact_path="results")
                metrics_path.unlink() # Clean up local file
            except Exception as json_err:
                logger.warning(f"Could not log Re-ID metrics dictionary as JSON artifact: {json_err}")
        else:
            logger.warning(f"[{model_name_tag}] No Re-ID metrics were calculated by the pipeline.")

        # --- Final Status ---
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
        try: # Log error details to MLflow artifact
            mlflow.log_text(
                f"Error Type: {type(e).__name__}\nError Message: {e}\n\nTraceback:\n{traceback.format_exc()}",
                "error_log.txt"
            )
        except Exception as log_err: logger.error(f"Failed to log Re-ID error details to MLflow: {log_err}")
    finally:
        logger.info(f"--- Finished Re-ID Execution Logic for Run ID: {run_id} [{model_name_tag}] (Attempted Status: {run_status}) ---")

    return run_status, metrics


# --- Tracking + Re-ID Runner (New Function) ---
def run_single_tracking_reid_experiment(
        run_config: Dict[str, Any],
        base_device_preference: str,
        seed: int,
        config_file_path: str,
        log_file_path: Optional[str] = None
) -> Tuple[str, Optional[TrackingResultSummary]]:
    """Executes a single Tracking+Re-ID MLflow run using the TrackingReidPipeline."""
    active_run = mlflow.active_run()
    if not active_run:
        logger.critical("run_single_tracking_reid_experiment called without active MLflow run!")
        return "FAILED", None
    run_id = active_run.info.run_id
    logger.info(f"--- Starting Tracking+ReID Execution Logic for Run ID: {run_id} (Using TrackingReidPipeline) ---")

    run_status = "FAILED"
    summary_metrics: Optional[TrackingResultSummary] = None
    pipeline: Optional[TrackingReidPipeline] = None
    actual_device: Optional[torch.device] = None

    # Determine tags early for logging context
    tracker_config = run_config.get("tracker", {})
    reid_config = run_config.get("reid_model", {})
    tracker_type = tracker_config.get("type", "unknown_tracker")
    reid_model_type = reid_config.get("model_type", "default_reid")
    reid_weights_file = Path(reid_config.get("weights_path", "")).stem
    reid_name_tag = f"{reid_model_type}_{reid_weights_file}" if reid_weights_file else reid_model_type
    run_name_tag = f"Trk:{tracker_type}_ReID:{reid_name_tag}"

    try:
        # --- Device Selection ---
        # BoxMOT trackers handle device internally based on specifier string ('0', 'mps', 'cpu')
        # We resolve the preference here and pass the appropriate string/device object later
        resolved_initial_device = get_selected_device(base_device_preference)
        actual_device = resolved_initial_device # Pipeline might override based on tracker needs
        logger.info(f"[{run_name_tag}] Device Preference: Requested='{base_device_preference}', Resolved='{actual_device}'")
        # Actual device used by BoxMOT might differ, log it from pipeline if possible

        # --- MLflow Logging: Parameters & Tags ---
        logger.info(f"[{run_name_tag}] Logging Tracking+ReID parameters...")
        # Log relevant parts of the config
        log_params_recursive({"tracker": tracker_config, "reid_model": reid_config, "data": run_config.get("data", {}), "environment": {"seed": seed, "device_pref": base_device_preference}})
        mlflow.log_param("environment.seed", seed)
        mlflow.log_param("environment.requested_device", base_device_preference)
        mlflow.log_param("environment.resolved_initial_device", str(resolved_initial_device))
        # Actual device used will be logged by pipeline if possible

        logger.info(f"[{run_name_tag}] Logging Tracking+ReID tags...")
        mlflow.set_tag("tracker_type", tracker_type)
        mlflow.set_tag("reid_model_type", reid_model_type)
        mlflow.set_tag("reid_weights_file", reid_weights_file if reid_weights_file else "Default/None")
        selected_env = run_config.get('data', {}).get('selected_environment', 'unknown')
        mlflow.set_tag("dataset", selected_env)
        env_data = run_config.get('data', {}).get(selected_env, {})
        mlflow.set_tag("scene_id", env_data.get('scene_id', 'unknown'))
        if 'camera_ids' in env_data: mlflow.set_tag("camera_ids_loaded", ",".join(sorted(env_data['camera_ids'])))
        log_git_info()

        # Log Config/Requirements Artifacts
        if config_file_path and Path(config_file_path).is_file():
            mlflow.log_artifact(config_file_path, artifact_path="config")
        req_path = PROJECT_ROOT / "requirements.txt"
        if req_path.is_file():
            mlflow.log_artifact(str(req_path), artifact_path="code")

        # --- Pipeline Initialization & Execution ---
        logger.info(f"[{run_name_tag}] Initializing and running Tracking+ReID pipeline...")
        pipeline = TrackingReidPipeline(run_config, actual_device, PROJECT_ROOT)
        pipeline_success, pipeline_summary, tracker_device = pipeline.run()
        summary_metrics = pipeline_summary # Store for return

        # Log the actual device used by the tracker if reported
        if tracker_device:
             mlflow.log_param("environment.actual_tracker_device", str(tracker_device))

        # --- Log Metrics & Results ---
        if summary_metrics:
            logger.info(f"[{run_name_tag}] Logging tracking summary metrics...")
            log_metrics_dict(summary_metrics, prefix="tracking") # Use prefix
            try:  # Log summary dict as JSON artifact
                summary_path = Path(f"./run_{run_id}_tracking_summary.json")
                with open(summary_path, 'w') as f:
                     # Ensure types are JSON serializable
                     serializable_summary = {k: (v.item() if isinstance(v, (np.generic, np.number)) else v) for k, v in summary_metrics.items()}
                     for k, v in serializable_summary.items(): # Handle torch tensors if any sneak in
                         if isinstance(v, torch.Tensor): serializable_summary[k] = v.item() if v.numel() == 1 else v.cpu().tolist()
                     json.dump(serializable_summary, f, indent=4)
                mlflow.log_artifact(str(summary_path), artifact_path="results")
                summary_path.unlink() # Clean up local file
            except Exception as json_err:
                logger.warning(f"Could not log tracking summary dictionary as JSON artifact: {json_err}")
        else:
            logger.warning(f"[{run_name_tag}] No tracking summary metrics were calculated by the pipeline.")

        # --- Final Status ---
        if pipeline_success:
            run_status = "FINISHED"
            mlflow.set_tag("run_outcome", "Success" if summary_metrics else "Success (No Metrics)")
        else:
            run_status = "FAILED"
            mlflow.set_tag("run_outcome", "Failed (Partial Metrics)" if summary_metrics else "Failed Execution")

    except KeyboardInterrupt:
        logger.warning(f"[{run_name_tag}] Tracking+ReID run interrupted by user (KeyboardInterrupt).")
        run_status = "KILLED"
        mlflow.set_tag("run_outcome", "Killed by user")
        raise  # Re-raise to allow outer handler to catch it
    except Exception as e:
        logger.critical(f"[{run_name_tag}] An uncaught error occurred during the Tracking+ReID run: {e}", exc_info=True)
        run_status = "FAILED"
        mlflow.set_tag("run_outcome", "Crashed")
        try: # Log error details to MLflow artifact
            mlflow.log_text(
                f"Error Type: {type(e).__name__}\nError Message: {e}\n\nTraceback:\n{traceback.format_exc()}",
                "error_log.txt"
            )
        except Exception as log_err: logger.error(f"Failed to log Tracking+ReID error details to MLflow: {log_err}")
    finally:
        logger.info(f"--- Finished Tracking+ReID Execution Logic for Run ID: {run_id} [{run_name_tag}] (Attempted Status: {run_status}) ---")

    return run_status, summary_metrics


# --- Sample Data Generation (Detection - Unchanged but potentially needed helper) ---
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
        temp_loader = FrameDataLoader(config) # Use detection loader for this
        sample_frame_bgr = None
        if temp_loader.image_filenames and temp_loader.active_camera_ids:
            # Get the first frame from the first active camera
            filename = temp_loader.image_filenames[0]
            cam_id = temp_loader.active_camera_ids[0]
            image_path = temp_loader.camera_image_dirs[cam_id] / filename
            if image_path.is_file():
                img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
                sample_frame_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if sample_frame_bgr is not None: logger.info(f"Loaded sample frame for detection signature: {image_path}")
                else: logger.warning(f"Failed to decode sample frame: {image_path}")
            else: logger.warning(f"Sample frame file not found: {image_path}")
        else: logger.warning("No frames or active cameras found by temp loader.")
        del temp_loader # Clean up loader

        if sample_frame_bgr is None:
            logger.warning("Could not load a sample frame for detection signature generation.")
            return None, None

        # --- Prepare Input based on Strategy ---
        inference_batch = None
        if isinstance(strategy, (YoloStrategy, RTDetrStrategy)):
            # Ultralytics models often take numpy/PIL directly
            sample_input_sig = sample_frame_bgr
            inference_batch = sample_frame_bgr # Predict uses this directly
            logger.info("Using numpy frame (BGR HWC) as input signature object for Ultralytics.")
        elif isinstance(strategy, FasterRCNNStrategy):
            # FasterRCNN needs a specific tensor input
            sample_input_tensor = strategy.get_sample_input_tensor(sample_frame_bgr)
            if sample_input_tensor is not None:
                sample_input_sig = sample_input_tensor.cpu().numpy()
                inference_batch = [sample_input_tensor.to(device)] # Model expects a list of tensors
                logger.info("Using numpy tensor (CHW) as input signature object for FasterRCNN.")
            else:
                logger.warning("Could not get sample tensor for FasterRCNN signature.")
                return None, None
        elif isinstance(strategy, RfDetrStrategy):
             # RFDETR predict likely takes PIL
             img_rgb = cv2.cvtColor(sample_frame_bgr, cv2.COLOR_BGR2RGB)
             img_pil = Image.fromarray(img_rgb)
             sample_input_sig = img_pil
             inference_batch = img_pil # Predict uses this
             logger.info("Using PIL Image (RGB) as input signature object for RFDETR.")
        else:
            # Generic fallback (assuming tensor input)
            sample_input_tensor = strategy.get_sample_input_tensor(sample_frame_bgr)
            if sample_input_tensor is not None:
                 sample_input_sig = sample_input_tensor.cpu().numpy()
                 # Assume model takes list of tensors like FasterRCNN if it's a torch Module
                 inference_batch = [sample_input_tensor.to(device)] if isinstance(strategy.get_model(), torch.nn.Module) else sample_input_tensor.to(device)
                 logger.info(f"Using generic numpy tensor (shape: {sample_input_sig.shape}) as input signature object.")
            else:
                 logger.warning("Could not get sample tensor for generic signature.")
                 return None, None

        # --- Run Inference for Output ---
        model_object = strategy.get_model()
        if model_object is None:
            logger.warning("Cannot get sample output: Model object not available from strategy.")
            return sample_input_sig, None
        if inference_batch is None:
             logger.error("Failed to prepare input batch for inference.")
             return sample_input_sig, None

        with torch.no_grad():
            if isinstance(strategy, (YoloStrategy, RTDetrStrategy)) and hasattr(model_object, 'predict'):
                results = model_object.predict(inference_batch, device=device, verbose=False)
                # Extract bounding boxes (xyxy) as sample output
                if results and results[0].boxes:
                    if hasattr(results[0].boxes, 'xyxy') and results[0].boxes.xyxy is not None:
                        sample_output_sig = results[0].boxes.xyxy.cpu().numpy()
                    elif hasattr(results[0].boxes, 'data') and results[0].boxes.data is not None: # Fallback if xyxy missing
                        sample_output_sig = results[0].boxes.data.cpu().numpy()[:, :4] # Assume first 4 are bbox coords

            elif isinstance(strategy, FasterRCNNStrategy) and isinstance(model_object, torch.nn.Module):
                model_object.to(device) # Ensure model is on correct device
                predictions = model_object(inference_batch) # Input is list of tensors
                # Output is list of dicts, get boxes from first dict
                if isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], dict) and 'boxes' in predictions[0]:
                     sample_output_sig = predictions[0]['boxes'].cpu().numpy()

            elif isinstance(strategy, RfDetrStrategy) and hasattr(model_object, 'predict'):
                 # Assuming predict returns a detection object with .xyxy
                 detections = model_object.predict(inference_batch) # Input is PIL
                 if detections and hasattr(detections, 'xyxy'):
                      out = detections.xyxy
                      sample_output_sig = out.cpu().numpy() if isinstance(out, torch.Tensor) else out # Handle tensor or numpy output

            # Generic fallback for torch.nn.Module (if not covered above)
            elif isinstance(model_object, torch.nn.Module):
                 model_object.to(device)
                 predictions = model_object(inference_batch) # Assume list or tensor input
                 # Try to guess output format (tensor or list of tensors/dicts) - highly unreliable
                 if isinstance(predictions, torch.Tensor): sample_output_sig = predictions.cpu().numpy()
                 elif isinstance(predictions, list) and len(predictions)>0 and isinstance(predictions[0], torch.Tensor): sample_output_sig = predictions[0].cpu().numpy()
                 elif isinstance(predictions, list) and len(predictions)>0 and isinstance(predictions[0], dict) and 'boxes' in predictions[0]: sample_output_sig = predictions[0]['boxes'].cpu().numpy()
                 else: logger.warning(f"Cannot determine output format for generic torch module signature: {type(predictions)}")

            else:
                logger.warning(f"Cannot determine how to run inference for model type {type(model_object)} to get sample output.")

        # --- Log Outcome ---
        if sample_input_sig is not None and sample_output_sig is not None:
            input_type = type(sample_input_sig).__name__
            output_type= type(sample_output_sig).__name__
            input_shape = getattr(sample_input_sig, 'shape', 'N/A')
            output_shape= getattr(sample_output_sig, 'shape', 'N/A')
            logger.info(f"Successfully generated sample input ({input_type}, shape {input_shape}) and output ({output_type}, shape {output_shape}) for detection.")
        elif sample_input_sig is not None:
            logger.warning("Generated sample input, but failed to generate sample output for detection.")
        else:
            logger.warning("Failed to generate sample input and/or output for detection.")

        return sample_input_sig, sample_output_sig

    except Exception as e:
        logger.error(f"Error during sample data generation for detection signature: {e}", exc_info=True)
        return None, None # Return None for both on error