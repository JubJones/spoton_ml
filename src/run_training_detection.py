import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

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
    # Import the runner function for single jobs
    from src.training.runner import run_single_training_job
    # Import core runner helpers only if needed (like log_git_info)
    from src.core.runner import log_git_info
except ImportError as e:
    print(f"Error importing local modules in run_training_detection_comparison.py: {e}")
    print("Please ensure all modules exist and PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="training_comparison", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='torch.utils.data')


# --- Main Comparison Orchestrator ---
def main():
    """
    Runs a comparison of different detection model training configurations.
    Each configuration is trained as a nested MLflow run.
    """
    logger.info("--- Starting Detection Model Training Comparison Run ---")
    config_path_str = "configs/training_detection_config.yaml"
    overall_status = "SUCCESS"
    parent_run_id = None

    # 1. Load Comparison Configuration
    comparison_config = load_config(config_path_str)
    if not comparison_config or "models_to_train" not in comparison_config:
        logger.critical(f"Failed to load valid comparison config from {config_path_str}. "
                        f"Ensure 'models_to_train' list is present. Exiting.")
        sys.exit(1)

    models_to_train: List[Dict[str, Any]] = comparison_config.get("models_to_train", [])
    if not models_to_train:
        logger.critical("No models found in 'models_to_train' list in config. Exiting.")
        sys.exit(1)

    # 2. Setup MLflow Experiment
    experiment_id = setup_mlflow_experiment(comparison_config,
                                            default_experiment_name="Default Detection Training Comparison")
    if not experiment_id:
        logger.critical("MLflow experiment setup failed. Exiting.")
        sys.exit(1)
    logger.info(f"Training comparison runs will be logged under Experiment ID: {experiment_id}")

    # 3. Set Global Seed & Determine Device
    seed = comparison_config.get("environment", {}).get("seed", int(time.time()))
    set_seed(seed)
    logger.info(f"Global random seed set to: {seed}")
    base_device_preference = comparison_config.get("environment", {}).get("device", "auto")
    # Resolve device once for the parent, child jobs might use it or override
    resolved_device = get_selected_device(base_device_preference)
    logger.info(f"Resolved base device: {resolved_device}")

    parent_run_name = comparison_config.get("parent_run_name", f"training_comparison_{int(time.time())}")
    child_run_statuses = {}
    run_counter = 0

    try:
        # 4. Start Parent MLflow Run
        with mlflow.start_run(run_name=parent_run_name, experiment_id=experiment_id) as parent_run:
            parent_run_id = parent_run.info.run_id
            logger.info(f"--- Parent Training Comparison MLflow Run Started ---")
            logger.info(f"Parent Run Name: {parent_run_name}")
            logger.info(f"Parent Run ID: {parent_run_id}")

            # Log parent parameters
            mlflow.log_param("parent.seed", seed)
            mlflow.log_param("parent.base_device_preference", base_device_preference)
            mlflow.log_param("parent.resolved_base_device", str(resolved_device))
            mlflow.log_param("parent.num_models_compared", len(models_to_train))
            # Log relevant data params to parent
            mlflow.log_param("parent.data.base_path", comparison_config.get("data", {}).get("base_path"))
            mlflow.log_param("parent.data.scenes_included", str([s.get('scene_id') for s in
                                                                 comparison_config.get("data", {}).get(
                                                                     "scenes_to_include", [])]))
            mlflow.log_param("parent.data.use_subset", comparison_config.get("data", {}).get("use_data_subset"))
            mlflow.log_param("parent.data.subset_fraction",
                             comparison_config.get("data", {}).get("data_subset_fraction"))
            mlflow.log_param("parent.data.val_split_ratio", comparison_config.get("data", {}).get("val_split_ratio"))

            # Log comparison config and requirements to parent run
            comp_conf_path = PROJECT_ROOT / config_path_str
            if comp_conf_path.is_file():
                mlflow.log_artifact(str(comp_conf_path), artifact_path="config")
            req_path = PROJECT_ROOT / "requirements.txt"
            if req_path.is_file():
                mlflow.log_artifact(str(req_path), artifact_path="code")

            log_git_info()  # Log git info for parent
            mlflow.set_tag("run_type", "training_comparison_parent")

            # 5. Loop Through Training Configurations and Start Nested Child Runs
            for training_job_config in models_to_train:
                run_counter += 1
                if "model" not in training_job_config or "training" not in training_job_config:
                    logger.warning(
                        f"Skipping invalid entry in 'models_to_train': Missing 'model' or 'training' key. Config: {training_job_config}")
                    continue

                model_type = training_job_config.get("model", {}).get("type", "unknown")
                model_tag = training_job_config.get("model", {}).get("name_tag", model_type)
                child_run_name = f"Train_{model_tag}_{run_counter}"
                logger.info(
                    f"\n--- Starting Nested Child Training Run ({run_counter}/{len(models_to_train)}): {child_run_name} ---")

                # Construct full config for the child run by merging global and specific parts
                child_run_config = {
                    "environment": comparison_config.get("environment", {}),
                    "data": comparison_config.get("data", {}),  # Pass the full data config
                    "mlflow": comparison_config.get("mlflow", {}),
                    # Pass mlflow config (though child uses parent's exp)
                    "model": training_job_config["model"],
                    "training": training_job_config["training"],
                    # Add run name for potential use within the child run
                    "run_name": child_run_name
                }

                child_status = "FAILED"
                child_run_id_for_status: Optional[str] = None
                try:
                    # Start nested run
                    with mlflow.start_run(run_name=child_run_name, experiment_id=experiment_id,
                                          nested=True) as child_run:
                        child_run_id = child_run.info.run_id
                        child_run_id_for_status = child_run_id
                        logger.info(f"Child Run '{child_run_name}' Started (ID: {child_run_id})")
                        mlflow.set_tag("parent_run_id", parent_run_id)
                        mlflow.set_tag("run_type", "training_comparison_child")
                        mlflow.set_tag("model_name_tag", model_tag)  # Use the tag from config

                        # Execute the single training job using the runner function
                        # Pass the merged config and the resolved device
                        child_status, _ = run_single_training_job(
                            run_config=child_run_config,
                            device=resolved_device,  # Use the device resolved for the parent
                            project_root=PROJECT_ROOT
                        )
                        child_run_statuses[child_run_id_for_status] = {"name": child_run_name, "status": child_status}
                        logger.info(
                            f"Child Run '{child_run_name}' (ID: {child_run_id}) finished with status: {child_status}")

                        if child_status != "FINISHED":
                            overall_status = "PARTIAL_FAILURE"

                except KeyboardInterrupt:
                    logger.warning(f"Comparison interrupted during child run for {child_run_name}.")
                    if child_run_id_for_status: child_run_statuses[child_run_id_for_status] = {"name": child_run_name,
                                                                                               "status": "KILLED"}
                    overall_status = "KILLED"
                    raise  # Propagate interruption

                except Exception as child_err:
                    logger.critical(f"Unhandled error during child training run for {child_run_name}: {child_err}",
                                    exc_info=True)
                    if child_run_id_for_status: child_run_statuses[child_run_id_for_status] = {"name": child_run_name,
                                                                                               "status": "CRASHED"}
                    overall_status = "PARTIAL_FAILURE"
                    # Mark MLflow run if possible
                    try:
                        active_child = mlflow.active_run()
                        if active_child and active_child.info.run_id == child_run_id_for_status:
                            mlflow.set_tag("run_outcome", "Crashed - Outer Loop")
                            mlflow.end_run("FAILED")
                    except Exception:
                        pass  # Ignore errors during error handling

            # Log summary of child statuses
            mlflow.log_dict(child_run_statuses, "child_training_run_summary.json")
            mlflow.set_tag("overall_status", overall_status)

    except KeyboardInterrupt:
        logger.warning("Training Comparison run interrupted by user (KeyboardInterrupt).")
        overall_status = "KILLED"
        if parent_run_id:
            try:
                MlflowClient().set_terminated(parent_run_id, status="KILLED")
            except Exception as term_err:
                logger.warning(f"Could not terminate parent run {parent_run_id} after KILLED: {term_err}")

    except Exception as e:
        logger.critical(f"An uncaught error occurred during training comparison orchestration: {e}", exc_info=True)
        overall_status = "FAILED"
        if parent_run_id:
            try:
                client = MlflowClient();
                client.set_tag(parent_run_id, "overall_status", "CRASHED");
                client.set_terminated(parent_run_id, status="FAILED")
            except Exception as term_err:
                logger.warning(f"Could not terminate parent run {parent_run_id} after CRASHED: {term_err}")
    finally:
        logger.info(f"--- Finalizing Training Comparison Run (Overall Status: {overall_status}) ---")

        # Log the main comparison log file to the PARENT run
        if parent_run_id and log_file.exists():
            try:
                for handler in logging.getLogger().handlers: handler.flush()
                client = MlflowClient();
                client.log_artifact(parent_run_id, str(log_file), artifact_path="comparison_logs")
                logger.info(
                    f"Training Comparison log file '{log_file.name}' logged as artifact to parent run {parent_run_id}.")
            except Exception as log_artifact_err:
                logger.warning(
                    f"Could not log comparison log file artifact '{log_file}' to parent run {parent_run_id}: {log_artifact_err}")

        # Ensure parent run termination status is set correctly
        active_parent_run = mlflow.active_run()
        final_parent_status = "FINISHED" if overall_status in ["SUCCESS", "PARTIAL_FAILURE"] else overall_status
        if active_parent_run and active_parent_run.info.run_id == parent_run_id:
            logger.info(f"Ensuring parent MLflow run {parent_run_id} is terminated with status {final_parent_status}.")
            mlflow.end_run(status=final_parent_status)
        elif parent_run_id:
            try:
                logger.warning(f"Attempting to terminate parent run {parent_run_id} outside active context.")
                MlflowClient().set_terminated(parent_run_id, status=final_parent_status)
            except Exception as term_err:
                logger.error(f"Failed to terminate parent run {parent_run_id} forcefully: {term_err}")

    logger.info(f"--- Detection Model Training Comparison Run Completed (Overall Status: {overall_status}) ---")
    exit_code = 0
    if overall_status != "SUCCESS":
        logger.error("One or more child training runs failed, were killed, or crashed.")
        exit_code = 1
    elif overall_status == "PARTIAL_FAILURE":
        logger.warning("Comparison orchestration completed, but one or more child training runs failed.")
        exit_code = 1  # Treat partial failure as an error exit code
    else:
        logger.info("All training comparison child runs completed successfully.")
        exit_code = 0

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
