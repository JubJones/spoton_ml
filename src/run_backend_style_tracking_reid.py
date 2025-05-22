# File: jubjones-spoton_ml/src/run_backend_style_tracking_reid.py
"""
Orchestrator script for running Tracking + Re-ID using adapted backend logic.
This script sets up MLflow, loads configuration, instantiates and runs
the BackendStyleTrackingReidPipeline, and logs results.
This version aligns with the MLflow runner patterns used in the spoton_ml project.
"""
import logging
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Local Imports ---
try:
    from src.utils.config_loader import load_config
    from src.utils.reproducibility import set_seed
    from src.utils.logging_utils import setup_logging
    from src.utils.mlflow_utils import setup_mlflow_experiment
    from src.utils.device_utils import get_selected_device
    from src.core.runner import log_params_recursive, log_metrics_dict, log_git_info
    from src.pipelines.backend_style_tracking_reid_pipeline import BackendStyleTrackingReIDPipeline
except ImportError as e:
    print(f"Error importing local modules in {Path(__file__).name}: {e}")
    print(f"PYTHONPATH: {sys.path}")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file_path = setup_logging(log_prefix="backend_style_tracking_reid", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    """
    Main function to orchestrate the backend-style tracking and Re-ID evaluation.
    """
    logger.info("--- Starting Backend-Style Tracking + Re-ID Run (MLOps Aligned) ---")
    config_path_str = "configs/backend_style_tracking_reid_config.yaml"
    overall_status = "FAILED" # Default status
    run_id: Optional[str] = None

    # 1. Load Configuration
    config = load_config(config_path_str)
    if not config:
        logger.critical(f"Failed to load configuration from {config_path_str}. Exiting.")
        sys.exit(1)

    # 2. Setup MLflow Experiment
    experiment_id = setup_mlflow_experiment(config, default_experiment_name="Default BackendStyle TrackingReID")
    if not experiment_id:
        logger.critical("MLflow experiment setup failed. Exiting.")
        sys.exit(1)

    # 3. Set Global Seed & Determine Device Preference
    seed = config.get("environment", {}).get("seed", int(time.time()))
    set_seed(seed)
    logger.info(f"Global random seed set to: {seed}")

    base_device_preference = config.get("environment", {}).get("device", "auto")
    preferred_device = get_selected_device(base_device_preference)
    logger.info(f"Preferred device for run: {preferred_device}")

    run_name_from_config = config.get("parent_run_name", f"backend_tracking_reid_{int(time.time())}")

    try:
        # 4. Start MLflow Run
        with mlflow.start_run(run_name=run_name_from_config, experiment_id=experiment_id) as current_run:
            run_id = current_run.info.run_id
            logger.info(f"--- MLflow Run Started ---")
            logger.info(f"Run Name: {run_name_from_config}")
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Experiment ID: {experiment_id}")
            logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

            # Log parameters and tags
            log_params_recursive(config) # Log entire config
            mlflow.log_param("environment.seed_used", seed)
            mlflow.log_param("environment.preferred_device_config", base_device_preference)
            mlflow.log_param("environment.actual_preferred_device_resolved", str(preferred_device))
            log_git_info()
            mlflow.set_tag("run_type", "backend_style_tracking_reid")
            
            # Log config file as artifact
            cfg_path_abs = (PROJECT_ROOT / config_path_str).resolve()
            if cfg_path_abs.is_file():
                 mlflow.log_artifact(str(cfg_path_abs), artifact_path="config")
            else:
                 logger.warning(f"Config file not found at {cfg_path_abs} for artifact logging.")
            
            # Log requirements.txt
            req_path = PROJECT_ROOT / "requirements.txt"
            if req_path.is_file():
                mlflow.log_artifact(str(req_path), artifact_path="code")

            # 5. Instantiate and Run the Pipeline
            logger.info("Instantiating BackendStyleTrackingReidPipeline...")
            pipeline = BackendStyleTrackingReIDPipeline(
                config=config,
                device=preferred_device, # Pass the resolved preferred device
                project_root=PROJECT_ROOT
            )
            
            logger.info("Running BackendStyleTrackingReidPipeline...")
            pipeline_success, result_summary = pipeline.run()

            # Log actual devices used by trackers (if pipeline collected them)
            if hasattr(pipeline, 'actual_tracker_devices') and pipeline.actual_tracker_devices:
                for cam_id_log, dev_log in pipeline.actual_tracker_devices.items():
                    mlflow.set_tag(f"actual_device_cam_{cam_id_log}", str(dev_log))
                # Log a summary tag if all devices were the same
                unique_devices_used = set(str(d) for d in pipeline.actual_tracker_devices.values())
                if len(unique_devices_used) == 1:
                    mlflow.set_tag("actual_device_all_trackers", unique_devices_used.pop())
                elif len(unique_devices_used) > 1:
                    mlflow.set_tag("actual_device_all_trackers", "mixed")


            # Log Metrics
            if result_summary:
                logger.info("Logging metrics from pipeline summary...")
                log_metrics_dict(result_summary, prefix="eval") # Using "eval" prefix for consistency
            else:
                logger.warning("Pipeline returned no result summary to log.")
            
            if pipeline_success:
                overall_status = "FINISHED"
                mlflow.set_tag("run_outcome", "Success")
            else:
                overall_status = "FAILED"
                mlflow.set_tag("run_outcome", "Pipeline Failed")

    except KeyboardInterrupt:
        logger.warning("Run interrupted by user (KeyboardInterrupt).")
        overall_status = "KILLED"
        if run_id and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
            mlflow.set_tag("run_outcome", "Killed by user")
        # Note: mlflow.end_run() will be called in finally
    except Exception as e:
        logger.critical(f"An uncaught error occurred during the run: {e}", exc_info=True)
        overall_status = "FAILED"
        if run_id and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
            mlflow.set_tag("run_outcome", "Crashed - Outer Script")
            try:
                # Log traceback to MLflow
                error_log_content = f"Error Type: {type(e).__name__}\nError Message: {e}\n\nTraceback:\n{traceback.format_exc()}"
                mlflow.log_text(error_log_content, "error_log.txt")
            except Exception as log_err:
                logger.error(f"Failed to log error details to MLflow: {log_err}")
        # Note: mlflow.end_run() will be called in finally
    finally:
        logger.info(f"--- Finalizing Run (Script Status: {overall_status}) ---")
        
        # Log the main script log file as an artifact
        if run_id and log_file_path.exists():
            try:
                # Ensure all log handlers are flushed before attempting to log the file
                for handler in logging.getLogger().handlers:
                    handler.flush()
                
                client = MlflowClient()
                # Check if run is still active before logging artifacts
                # This can prevent errors if the run was terminated by an exception within the `with` block
                current_run_info = client.get_run(run_id)
                if current_run_info.info.status in ["RUNNING", "SCHEDULED"]:
                     client.log_artifact(run_id, str(log_file_path), artifact_path="run_logs")
                     logger.info(f"Main run log file '{log_file_path.name}' logged as artifact to MLflow run {run_id}.")
                else:
                     logger.warning(f"Skipping script log artifact logging as run {run_id} is already terminated ({current_run_info.info.status}).")

            except Exception as log_artifact_err:
                logger.warning(f"Could not log main run log file artifact '{log_file_path.name}': {log_artifact_err}")

        # Ensure MLflow run is properly terminated
        active_run_obj = mlflow.active_run()
        if active_run_obj and active_run_obj.info.run_id == run_id:
            logger.info(f"Ensuring MLflow run {run_id} is terminated with status '{overall_status}'.")
            mlflow.end_run(status=overall_status)
        elif run_id: # If run exists but isn't active (e.g., error outside context manager)
            try:
                logger.warning(f"Attempting to terminate MLflow run {run_id} externally with status '{overall_status}'.")
                client = MlflowClient()
                current_run_info = client.get_run(run_id)
                if current_run_info.info.status in ["RUNNING", "SCHEDULED"]:
                    client.set_terminated(run_id, status=overall_status)
                else:
                    logger.info(f"MLflow run {run_id} already terminated with status: {current_run_info.info.status}.")
            except Exception as term_err:
                logger.error(f"Failed to terminate MLflow run {run_id} externally: {term_err}")

    logger.info(f"--- Backend-Style Tracking + Re-ID Run Completed (Final Overall Status: {overall_status}) ---")
    sys.exit(0 if overall_status == "FINISHED" else 1)

if __name__ == "__main__":
    main()