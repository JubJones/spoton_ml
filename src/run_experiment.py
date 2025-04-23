import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

import dagshub
import mlflow
import torch
from dotenv import load_dotenv

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"PROJECT_ROOT added to sys.path: {PROJECT_ROOT}")

# --- Local Imports ---
try:
    from src.utils.config_loader import load_config
    from src.tracking.device_utils import get_selected_device
    from src.pipelines.detection_pipeline import DetectionPipeline
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print(f"PYTHONPATH: {sys.path}")
    print("Ensure you are running this script from the project root or that src is in PYTHONPATH.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = PROJECT_ROOT / "experiment.log"
if log_file.exists():
    try:
        open(log_file, 'w').close()
    except OSError as e:
        print(f"Warning: Could not clear log file {log_file}: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler(sys.stdout)  # Also print logs to console
    ]
)
logger = logging.getLogger(__name__)  # Use root logger or specific names

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress general future warnings


# --- Helper Functions ---

def setup_mlflow(config: Dict[str, Any]) -> Optional[str]:
    """Initializes MLflow connection (Dagshub or local) and sets experiment."""
    mlflow_config = config.get("mlflow", {})
    dotenv_path = PROJECT_ROOT / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logger.info("Loaded environment variables from .env file.")
    else:
        logger.info(".env file not found, relying on environment or defaults.")

    try:
        repo_owner = "Jwizzed"
        repo_name = "spoton_ml"
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        logger.info(f"Dagshub initialized successfully for {repo_owner}/{repo_name}.")
        tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"MLflow tracking URI automatically set by Dagshub: {tracking_uri}")
    except Exception as dag_err:
        logger.warning(f"Dagshub initialization failed: {dag_err}. Attempting manual MLflow URI setup.")
        # Fallback to environment variable or local
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set from environment variable: {tracking_uri}")
        else:
            logger.warning("MLFLOW_TRACKING_URI not set and Dagshub init failed. Using local tracking.")
            local_mlruns = PROJECT_ROOT / "mlruns"
            local_mlruns.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{local_mlruns.resolve()}")
            logger.info(f"MLflow tracking URI set to local: {mlflow.get_tracking_uri()}")

    experiment_name = mlflow_config.get("experiment_name", "Default Experiment")
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: '{experiment_name}'")

    try:
        # Ensure the experiment exists
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.info(f"Experiment '{experiment_name}' not found. Creating...")
            experiment_id = client.create_experiment(experiment_name)
            logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
        elif experiment.lifecycle_stage != 'active':
            logger.error(f"Experiment '{experiment_name}' exists but is deleted or archived.")
            return None
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment ID: {experiment_id}")
        return experiment_id
    except Exception as client_err:
        logger.error(f"Failed to connect to MLflow tracking server or get/create experiment: {client_err}",
                     exc_info=True)
        return None


def log_params_recursive(params_dict: Dict[str, Any], parent_key: str = ""):
    """Recursively logs parameters to MLflow, handling nested dictionaries and lists."""
    for key, value in params_dict.items():
        mlflow_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            log_params_recursive(value, mlflow_key)
        elif isinstance(value, list):
            # Attempt to log list as JSON, fallback to string
            try:
                mlflow.log_param(mlflow_key, json.dumps(value))
            except TypeError:
                # Truncate long lists/complex objects represented as strings
                mlflow.log_param(mlflow_key, str(value)[:250])
        else:
            # Truncate potentially long string values
            mlflow.log_param(mlflow_key, str(value)[:250])


def log_metrics_dict(metrics: Dict[str, Any]):
    """Logs metrics from a dictionary to MLflow."""
    if not metrics:
        logger.warning("Metrics dictionary is empty. Nothing to log.")
        return
    logger.info(f"Logging {len(metrics)} metrics to MLflow...")
    # Use log_metrics (plural) for potentially better performance with many metrics
    try:
        mlflow.log_metrics(metrics)
        logger.info("Metrics logged successfully.")
    except Exception as e:
        logger.error(f"Failed to log metrics batch: {e}. Attempting individual logging.", exc_info=True)
        # Fallback to individual logging if batch fails
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, value)
            except Exception as ind_e:
                logger.error(f"Failed to log individual metric '{key}': {value}. Error: {ind_e}")


# --- Main Execution Orchestrator ---
def main():
    """Main function to orchestrate the experiment run."""
    logger.info("--- Starting Experiment Orchestration ---")
    config_path_str = "configs/experiment_config.yaml"
    run_status = "FAILED"  # Default status
    run_id = None

    # 1. Load Configuration
    config = load_config(config_path_str)
    if not config:
        logger.critical("Failed to load configuration. Exiting.")
        sys.exit(1)

    # 2. Setup MLflow
    experiment_id = setup_mlflow(config)
    if not experiment_id:
        logger.critical("MLflow setup failed. Exiting.")
        sys.exit(1)

    # 3. Determine Device
    requested_device_name = config.get("environment", {}).get("device", "auto")
    requested_device = get_selected_device(requested_device_name)

    # Handle potential FasterRCNN CPU override explicitly here
    model_type = config.get("model", {}).get("type", "").lower()
    actual_device = requested_device
    device_override_reason = "None"
    if model_type == "fasterrcnn" and requested_device.type not in ['cuda', 'mps']:  # Check if not GPU
        logger.warning(
            f"FasterRCNN selected but device is '{requested_device.type}'. Forcing to CPU for compatibility.")
        actual_device = torch.device('cpu')
        device_override_reason = f"FasterRCNN forced to CPU from {requested_device.type}"
    elif requested_device.type == 'mps' and model_type not in ['yolo',
                                                               'rtdetr']:  # Example if other models don't support MPS well
        logger.warning(
            f"Model type '{model_type}' may have limited support on MPS device. Using {requested_device.type}.")
        actual_device = torch.device('cpu')
        device_override_reason = f"Model {model_type} potentially incompatible with MPS, using {actual_device.type}"
        pass

    run_name = config.get("run_name", "unnamed_run")

    try:
        # 4. Start MLflow Run
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            logger.info(f"Starting MLflow run: {run_name} (ID: {run_id})")

            # 5. Log Parameters and Config
            logger.info("Logging parameters...")
            log_params_recursive(config)
            mlflow.log_param("environment.requested_device", str(requested_device))
            mlflow.log_param("environment.actual_device_used", str(actual_device))
            mlflow.log_param("environment.device_override_reason", device_override_reason)

            config_artifact_path = PROJECT_ROOT / config_path_str
            if config_artifact_path.is_file():
                mlflow.log_artifact(str(config_artifact_path), artifact_path="config")
                logger.info("Configuration artifact logged.")
            else:
                logger.warning(f"Could not find config file to log as artifact: {config_artifact_path}")

            # 6. Initialize and Run the Pipeline
            logger.info("Initializing detection pipeline...")
            pipeline = DetectionPipeline(config, actual_device)
            pipeline_success, metrics, active_cameras, num_frames = pipeline.run()

            # 7. Log Pipeline Info (even if failed)
            if active_cameras is not None:
                mlflow.log_param("data.actual_cameras_used", ",".join(active_cameras))
            if num_frames is not None:
                mlflow.log_param("data.actual_frame_indices_processed", num_frames)

            # 8. Log Metrics (if successful)
            if pipeline_success and metrics:
                logger.info("Pipeline finished successfully. Logging metrics.")
                log_metrics_dict(metrics)
                run_status = "FINISHED"
            elif metrics:  # Log metrics even if pipeline failed but metrics were calculated (partial run)
                logger.warning(
                    "Pipeline did not finish successfully, but partial metrics were calculated. Logging metrics.")
                log_metrics_dict(metrics)
                run_status = "FAILED"  # Mark as FAILED despite partial metrics
            else:
                logger.error("Pipeline execution failed. No metrics to log.")
                run_status = "FAILED"


    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user (KeyboardInterrupt). Terminating run.")
        run_status = "KILLED"
        # MLflow context manager handles ending the run on exception, but we ensure status is KILLED
    except Exception as e:
        logger.critical(f"An uncaught error occurred during the experiment run: {e}", exc_info=True)
        run_status = "FAILED"  # Ensure status is FAILED on any other exception
    finally:
        # 9. End MLflow Run (robustly)
        if run_id:
            active_mlflow_run = mlflow.active_run()
            if active_mlflow_run and active_mlflow_run.info.run_id == run_id:
                # If inside the 'with' block and it exited normally or via known exception
                mlflow.end_run(status=run_status)
                logger.info(f"MLflow run {run_id} ended with status: {run_status}")
            else:
                # If 'with' block failed very early or we are outside of it unexpectedly
                logger.warning(
                    f"Attempting to terminate run {run_id} outside of active context. Final status: {run_status}")
                try:
                    mlflow.tracking.MlflowClient().set_terminated(run_id, status=run_status)
                    logger.info(f"MLflow run {run_id} explicitly terminated with status: {run_status}")
                except Exception as client_err:
                    logger.error(f"Could not explicitly set final status for run {run_id}: {client_err}")
        else:
            logger.warning("No run ID was generated. Cannot terminate MLflow run.")

    logger.info(f"--- Experiment Orchestration Completed (Final Status: {run_status}) ---")
    if run_status != "FINISHED":
        sys.exit(1)  # Exit with error code if run wasn't successful


if __name__ == "__main__":
    main()
