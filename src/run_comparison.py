import copy
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List

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
    from src.core.runner import run_single_experiment, log_git_info
except ImportError as e:
    print(f"Error importing local modules: {e}\nPlease ensure PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="comparison", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Main Comparison Orchestrator ---
def main():
    logger.info("--- Starting Model Comparison Run ---")
    config_path_str = "configs/comparison_run_config.yaml"
    overall_status = "SUCCESS"
    parent_run_id = None

    # 1. Load Comparison Configuration
    comparison_config = load_config(config_path_str)
    if not comparison_config or "models_to_run" not in comparison_config:
        logger.critical(f"Failed to load valid comparison config from {config_path_str}. Exiting.")
        sys.exit(1)
    models_to_run: List[Dict[str, Any]] = comparison_config.get("models_to_run", [])
    if not models_to_run:
        logger.critical("No models found in 'models_to_run' list in config. Exiting.")
        sys.exit(1)

    # 2. Setup MLflow Experiment using the utility function
    experiment_id = setup_mlflow_experiment(comparison_config, default_experiment_name="Default Comparison Runs")
    if not experiment_id:
        logger.critical("MLflow experiment setup failed. Exiting.")
        sys.exit(1)
    logger.info(f"Comparison runs will be logged under Experiment ID: {experiment_id}")

    # 3. Set Global Seed
    seed = comparison_config.get("environment", {}).get("seed", int(time.time()))
    set_seed(seed)
    logger.info(f"Global random seed set to: {seed}")

    # 4. Determine Base Device Preference
    base_device_preference = comparison_config.get("environment", {}).get("device", "auto")

    parent_run_name = comparison_config.get("parent_run_name", f"comparison_{int(time.time())}")
    child_run_statuses = {}

    try:
        # 5. Start Parent MLflow Run
        with mlflow.start_run(run_name=parent_run_name, experiment_id=experiment_id) as parent_run:
            parent_run_id = parent_run.info.run_id
            logger.info(f"--- Parent MLflow Run Started ---")
            logger.info(f"Parent Run Name: {parent_run_name}")
            logger.info(f"Parent Run ID: {parent_run_id}")

            # Log parent parameters
            mlflow.log_param("parent.seed", seed)
            mlflow.log_param("parent.base_device_preference", base_device_preference)
            mlflow.log_param("parent.num_models_compared", len(models_to_run))

            # Log comparison config and requirements to parent run
            comp_conf_path = PROJECT_ROOT / config_path_str
            if comp_conf_path.is_file():
                mlflow.log_artifact(str(comp_conf_path), artifact_path="config")
            req_path = PROJECT_ROOT / "requirements.txt"
            if req_path.is_file():
                mlflow.log_artifact(str(req_path), artifact_path="code")

            # Log parent git info
            log_git_info()  # Use helper from core.runner now
            mlflow.set_tag("run_type", "comparison_parent")

            # 6. Loop Through Models and Start Nested Child Runs
            for model_variation_config in models_to_run:
                if "model" not in model_variation_config:
                    logger.warning(f"Skipping invalid entry: Missing 'model' key. Config: {model_variation_config}")
                    continue

                model_specific_config = model_variation_config["model"]
                model_name = model_specific_config.get("model_name", model_specific_config.get("type", "unknown"))
                child_run_name = f"child_{model_name}"
                logger.info(f"\n--- Starting Nested Child Run for: {model_name} ---")

                # Construct full config for the child run
                child_run_config = copy.deepcopy(comparison_config)
                del child_run_config["models_to_run"]
                del child_run_config["parent_run_name"]
                child_run_config["model"] = model_specific_config

                child_status = "FAILED"
                child_run_id_for_status = None
                try:
                    # Start nested run
                    with mlflow.start_run(run_name=child_run_name, experiment_id=experiment_id,
                                          nested=True) as child_run:
                        child_run_id = child_run.info.run_id
                        child_run_id_for_status = child_run_id  # Store for status dict key
                        logger.info(f"Child Run '{model_name}' Started (ID: {child_run_id})")
                        mlflow.set_tag("parent_run_id", parent_run_id)
                        mlflow.set_tag("run_type", "comparison_child")

                        # Execute core logic
                        child_status, _ = run_single_experiment(
                            run_config=child_run_config,
                            base_device_preference=base_device_preference,
                            seed=seed,
                            config_file_path=str(PROJECT_ROOT / config_path_str),
                            log_file_path=None
                        )
                        child_run_statuses[child_run_id_for_status] = {"name": model_name, "status": child_status}
                        logger.info(
                            f"Child Run '{model_name}' (ID: {child_run_id}) finished with status: {child_status}")

                        if child_status != "FINISHED":
                            overall_status = "PARTIAL_FAILURE"

                except KeyboardInterrupt:
                    logger.warning(f"Comparison interrupted during child run for {model_name}.")
                    if child_run_id_for_status: child_run_statuses[child_run_id_for_status] = {"name": model_name,
                                                                                               "status": "KILLED"}
                    overall_status = "KILLED"
                    raise

                except Exception as child_err:
                    logger.critical(f"Unhandled error during child run for {model_name}: {child_err}", exc_info=True)
                    if child_run_id_for_status: child_run_statuses[child_run_id_for_status] = {"name": model_name,
                                                                                               "status": "CRASHED"}
                    overall_status = "PARTIAL_FAILURE"
                    # Mark MLflow run if possible
                    try:
                        active_child = mlflow.active_run()
                        if active_child and active_child.info.run_id == child_run_id_for_status:
                            mlflow.set_tag("run_outcome", "Crashed - Outer Loop")
                            mlflow.end_run("FAILED")
                    except Exception:
                        pass

            # Log summary of child statuses (using run IDs as keys)
            mlflow.log_dict(child_run_statuses, "child_run_summary.json")
            mlflow.set_tag("overall_status", overall_status)


    except KeyboardInterrupt:
        logger.warning("Comparison run interrupted by user (KeyboardInterrupt).")
        overall_status = "KILLED"
        if parent_run_id:
            try:
                MlflowClient().set_terminated(parent_run_id, status="KILLED")
            except Exception as term_err:
                logger.warning(f"Could not terminate parent run {parent_run_id} after KILLED: {term_err}")

    except Exception as e:
        logger.critical(f"An uncaught error occurred during comparison orchestration: {e}", exc_info=True)
        overall_status = "FAILED"
        if parent_run_id:
            try:
                client = MlflowClient()
                client.set_tag(parent_run_id, "overall_status", "CRASHED")
                client.set_terminated(parent_run_id, status="FAILED")
            except Exception as term_err:
                logger.warning(f"Could not terminate parent run {parent_run_id} after CRASHED: {term_err}")
    finally:
        logger.info(f"--- Finalizing Comparison Run (Overall Status: {overall_status}) ---")

        # Log the main comparison log file to the PARENT run
        if parent_run_id and log_file.exists():
            try:
                for handler in logging.getLogger().handlers: handler.flush()
                client = MlflowClient()
                client.log_artifact(parent_run_id, str(log_file), artifact_path="comparison_logs")
                logger.info(f"Comparison log file '{log_file.name}' logged as artifact to parent run {parent_run_id}.")
            except Exception as log_artifact_err:
                logger.warning(
                    f"Could not log comparison log file artifact '{log_file}' to parent run {parent_run_id}: {log_artifact_err}")

        # Ensure parent run termination status is set correctly
        active_parent_run = mlflow.active_run()
        if active_parent_run and active_parent_run.info.run_id == parent_run_id:
            final_parent_status = "FINISHED" if overall_status in ["SUCCESS", "PARTIAL_FAILURE"] else overall_status
            logger.info(f"Ensuring parent MLflow run {parent_run_id} is terminated with status {final_parent_status}.")
            mlflow.end_run(status=final_parent_status)
        elif parent_run_id:
            try:
                logger.warning(f"Attempting to terminate parent run {parent_run_id} outside active context.")
                final_parent_status = "FINISHED" if overall_status in ["SUCCESS", "PARTIAL_FAILURE"] else overall_status
                MlflowClient().set_terminated(parent_run_id, status=final_parent_status)
            except Exception as term_err:
                logger.error(f"Failed to terminate parent run {parent_run_id} forcefully: {term_err}")

    logger.info(f"--- Model Comparison Run Completed (Overall Status: {overall_status}) ---")
    if overall_status != "SUCCESS":
        logger.error("One or more child runs failed, were killed, or crashed.")
        sys.exit(1)
    else:
        # Note: SUCCESS means orchestration completed. PARTIAL_FAILURE means orchestration ok, but some child failed.
        if overall_status == "PARTIAL_FAILURE":
            logger.warning("Comparison orchestration completed, but one or more child runs failed.")
            sys.exit(1)  # Exit with error code even if orchestration finished
        else:
            logger.info("All comparison child runs completed successfully.")
            sys.exit(0)


if __name__ == "__main__":
    main()
