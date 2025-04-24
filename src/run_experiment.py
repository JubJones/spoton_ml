import logging
import sys
import time
import warnings
from pathlib import Path

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
    from src.core.runner import run_single_experiment
except ImportError as e:
    print(f"Error importing local modules: {e}\nPlease ensure PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="experiment", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Main Execution ---
def main():
    logger.info("--- Starting Single Experiment Run ---")
    config_path_str = "configs/single_run_config.yaml"
    final_status = "FAILED"
    run_id = None

    # 1. Load Configuration
    config = load_config(config_path_str)
    if not config:
        logger.critical(f"Failed to load configuration from {config_path_str}. Exiting.")
        sys.exit(1)

    # 2. Setup MLflow Experiment using the utility function
    experiment_id = setup_mlflow_experiment(config, default_experiment_name="Default Single Runs")
    if not experiment_id:
        logger.critical("MLflow experiment setup failed. Exiting.")
        sys.exit(1)

    # 3. Set Seed
    seed = config.get("environment", {}).get("seed", int(time.time()))
    set_seed(seed)
    logger.info(f"Using random seed: {seed}")

    # 4. Determine Base Device Preference
    base_device_preference = config.get("environment", {}).get("device", "auto")
    # The actual device used will be determined within the runner

    run_name = config.get("run_name", f"single_run_{int(time.time())}")

    try:
        # 5. Start MLflow Run context
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            logger.info(f"--- MLflow Run Started ---")
            logger.info(f"Run Name: {run_name}")
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Experiment ID: {experiment_id}")
            logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

            # 6. Execute the core experiment logic using the runner
            run_status, _ = run_single_experiment(  # metrics not needed directly here
                run_config=config,
                base_device_preference=base_device_preference,
                seed=seed,
                config_file_path=str(PROJECT_ROOT / config_path_str),
                log_file_path=None  # Runner doesn't need to log this script's log
            )
            final_status = run_status  # Store status from runner

    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user (KeyboardInterrupt).")
        final_status = "KILLED"
        if run_id:
            try:
                MlflowClient().set_terminated(run_id, status="KILLED")
            except Exception as term_err:
                logger.warning(f"Could not terminate run {run_id} after KILLED: {term_err}")
    except Exception as e:
        logger.critical(f"An uncaught error occurred outside the runner: {e}", exc_info=True)
        final_status = "FAILED"
        if run_id:
            try:
                client = MlflowClient()
                client.set_tag(run_id, "run_outcome", "Crashed - Outer")
                client.set_terminated(run_id, status="FAILED")
            except Exception as term_err:
                logger.warning(f"Could not terminate run {run_id} after CRASHED: {term_err}")
    finally:
        logger.info(f"--- Finalizing Run (Final Status: {final_status}) ---")
        # Log the main script log file to the run if it exists
        if run_id and log_file.exists():
            try:
                for handler in logging.getLogger().handlers: handler.flush()
                client = MlflowClient()
                client.log_artifact(run_id, str(log_file), artifact_path="logs")
                logger.info(f"Main experiment log file '{log_file.name}' logged as artifact to run {run_id}.")
            except Exception as log_artifact_err:
                logger.warning(f"Could not log main experiment log file artifact '{log_file}': {log_artifact_err}")

        # Ensure run termination status is set correctly
        active_run = mlflow.active_run()
        if active_run and active_run.info.run_id == run_id:
            logger.info(f"Ensuring MLflow run {run_id} is terminated with status {final_status}.")
            mlflow.end_run(status=final_status)
        elif run_id:
            try:
                logger.warning(f"Attempting to terminate run {run_id} outside active context.")
                MlflowClient().set_terminated(run_id, status=final_status)
            except Exception as term_err:
                logger.error(f"Failed to terminate run {run_id} forcefully: {term_err}")

    logger.info(f"--- Single Experiment Run Completed (Status: {final_status}) ---")
    if final_status != "FINISHED":
        sys.exit(1)


if __name__ == "__main__":
    main()
