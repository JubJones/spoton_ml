import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"PROJECT_ROOT added to sys.path: {PROJECT_ROOT}")

# --- Local Imports ---
try:
    from src.utils.config_loader import load_config
    from src.utils.reproducibility import set_seed
    from src.utils.logging_utils import setup_logging
    from src.utils.mlflow_utils import setup_mlflow_experiment
    from src.utils.device_utils import get_selected_device
    from src.training.runner import run_single_training_job
except ImportError as e:
    print(f"Error importing local modules in run_training_fasterrcnn.py: {e}")
    print("Please ensure all modules exist and PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="train_fasterrcnn", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")


# --- Main Execution ---
def main():
    """
    Runs a single Faster R-CNN training job based on the configuration file.
    """
    logger.info("--- Starting Faster R-CNN Training Run ---")
    config_path_str = "configs/fasterrcnn_training_config.yaml"
    final_status = "FAILED"
    run_id = None

    # 1. Load Configuration
    config = load_config(config_path_str)
    if not config:
        logger.critical(
            f"Failed to load configuration from {config_path_str}. Exiting."
        )
        sys.exit(1)

    # --- Extract the single training job config ---
    models_to_train = config.get("models_to_train")
    if (
        not models_to_train
        or not isinstance(models_to_train, list)
        or len(models_to_train) == 0
    ):
        logger.critical(
            f"Config {config_path_str} must contain a list 'models_to_train' with at least one entry."
        )
        sys.exit(1)
    if len(models_to_train) > 1:
        logger.warning(
            f"Config contains {len(models_to_train)} entries in 'models_to_train'. Using only the first one."
        )

    job_config_entry = models_to_train[0]
    if "model" not in job_config_entry or "training" not in job_config_entry:
        logger.critical(
            "The first entry in 'models_to_train' must contain 'model' and 'training' keys."
        )
        sys.exit(1)

    # --- Construct the config needed by run_single_training_job ---
    single_run_config = {
        "environment": config.get("environment", {}),
        "data": config.get("data", {}),
        "mlflow": config.get(
            "mlflow", {}
        ),  # Pass mlflow config for experiment name etc.
        "model": job_config_entry["model"],
        "training": job_config_entry["training"],
        # Add run name if specified at top level, otherwise runner uses model tag
        "run_name": config.get(
            "run_name", job_config_entry["model"].get("name_tag", "fasterrcnn_run")
        ),
    }

    # 2. Setup MLflow Experiment
    experiment_id = setup_mlflow_experiment(
        config, default_experiment_name="Default FasterRCNN Training"
    )
    if not experiment_id:
        logger.critical("MLflow experiment setup failed. Exiting.")
        sys.exit(1)

    # 3. Set Seed & Determine Device
    seed = config.get("environment", {}).get("seed", int(time.time()))
    set_seed(seed)
    logger.info(f"Global random seed set to: {seed}")
    base_device_preference = config.get("environment", {}).get("device", "auto")
    resolved_device = get_selected_device(base_device_preference)
    logger.info(f"Resolved base device: {resolved_device}")

    try:
        # 4. Start ONE MLflow Run
        run_name = single_run_config["run_name"]
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            logger.info(f"--- MLflow Run Started ---")
            logger.info(f"Run Name: {run_name}")
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Experiment ID: {experiment_id}")
            logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

            # Log the original config file path as an artifact
            orig_conf_path = PROJECT_ROOT / config_path_str
            if orig_conf_path.is_file():
                mlflow.log_artifact(str(orig_conf_path), artifact_path="config")

            # 5. Execute the single training job using the runner function
            run_status, _ = run_single_training_job(
                run_config=single_run_config,
                device=resolved_device,
                project_root=PROJECT_ROOT,
            )
            final_status = run_status

    except KeyboardInterrupt:
        logger.warning("Training run interrupted by user (KeyboardInterrupt).")
        final_status = "KILLED"
        if run_id:
            try:
                MlflowClient().set_terminated(run_id, status="KILLED")
            except Exception as term_err:
                logger.warning(
                    f"Could not terminate run {run_id} after KILLED: {term_err}"
                )
    except Exception as e:
        logger.critical(f"An uncaught error occurred: {e}", exc_info=True)
        final_status = "FAILED"
        if run_id:
            try:
                client = MlflowClient()
                client.set_tag(run_id, "run_outcome", "Crashed - Outer")
                client.set_terminated(run_id, status="FAILED")
            except Exception as term_err:
                logger.warning(
                    f"Could not terminate run {run_id} after CRASHED: {term_err}"
                )
    finally:
        logger.info(f"--- Finalizing Run (Final Status: {final_status}) ---")
        # Log the main script log file to the run if it exists
        if run_id and log_file.exists():
            try:
                for handler in logging.getLogger().handlers:
                    handler.flush()
                client = MlflowClient()
                client.log_artifact(run_id, str(log_file), artifact_path="logs")
                logger.info(
                    f"Main training log file '{log_file.name}' logged as artifact to run {run_id}."
                )
            except Exception as log_artifact_err:
                logger.warning(
                    f"Could not log main training log file artifact '{log_file}': {log_artifact_err}"
                )

        # Ensure run termination status is set correctly
        active_run = mlflow.active_run()
        if active_run and active_run.info.run_id == run_id:
            logger.info(
                f"Ensuring MLflow run {run_id} is terminated with status {final_status}."
            )
            mlflow.end_run(status=final_status)
        elif run_id:
            try:
                logger.warning(
                    f"Attempting to terminate run {run_id} outside active context with status {final_status}."
                )
                MlflowClient().set_terminated(run_id, status=final_status)
            except Exception as term_err:
                logger.error(f"Failed to terminate run {run_id} forcefully: {term_err}")

    logger.info(f"--- Faster R-CNN Training Run Completed (Status: {final_status}) ---")
    exit_code = 0 if final_status == "FINISHED" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
