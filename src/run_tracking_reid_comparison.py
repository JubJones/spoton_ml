import copy
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
    from src.core.runner import run_single_tracking_reid_experiment, log_git_info
except ImportError as e:
    print(f"Error importing local modules: {e}\nPlease ensure PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="tracking_reid_comparison", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- Main Comparison Orchestrator ---
def main():
    """
    Runs a comparison experiment for different BoxMOT trackers combined with Re-ID models.
    Uses ground truth bounding boxes as input for the trackers.
    """
    logger.info("--- Starting Tracking + Re-ID Comparison Run ---")
    config_path_str = "configs/tracking_reid_comparison_config.yaml"
    overall_status = "SUCCESS"
    parent_run_id = None

    # 1. Load Comparison Configuration
    comparison_config = load_config(config_path_str)
    if not comparison_config or "trackers_to_run" not in comparison_config or "reid_models_to_associate" not in comparison_config:
        logger.critical(f"Failed to load valid tracking+ReID config from {config_path_str}. "
                        f"Ensure 'trackers_to_run' and 'reid_models_to_associate' are present. Exiting.")
        sys.exit(1)

    trackers_to_run: List[str] = comparison_config.get("trackers_to_run", [])
    reid_models_to_associate: List[Dict[str, Any]] = comparison_config.get("reid_models_to_associate", [])

    if not trackers_to_run:
        logger.critical("No trackers found in 'trackers_to_run' list in config. Exiting.")
        sys.exit(1)
    if not reid_models_to_associate:
        logger.warning("No Re-ID models found in 'reid_models_to_associate' list. Trackers might use defaults or fail.")
        # Allow running even without specific ReID models if trackers support it (e.g., OCSort, ByteTrack might run without)
        # reid_models_to_associate.append({}) # Add dummy entry if run must have reid model

    # 2. Setup MLflow Experiment
    experiment_id = setup_mlflow_experiment(comparison_config, default_experiment_name="Default TrackingReID Runs")
    if not experiment_id:
        logger.critical("MLflow experiment setup failed. Exiting.")
        sys.exit(1)
    logger.info(f"Tracking+ReID comparison runs will be logged under Experiment ID: {experiment_id}")

    # 3. Set Global Seed
    seed = comparison_config.get("environment", {}).get("seed", int(time.time()))
    set_seed(seed)
    logger.info(f"Global random seed set to: {seed}")

    # 4. Determine Base Device Preference
    base_device_preference = comparison_config.get("environment", {}).get("device", "auto")

    parent_run_name = comparison_config.get("parent_run_name", f"tracking_reid_comparison_{int(time.time())}")
    child_run_statuses = {}
    total_combinations = len(trackers_to_run) * len(reid_models_to_associate)
    run_counter = 0

    try:
        # 5. Start Parent MLflow Run
        with mlflow.start_run(run_name=parent_run_name, experiment_id=experiment_id) as parent_run:
            parent_run_id = parent_run.info.run_id
            logger.info(f"--- Parent Tracking+ReID MLflow Run Started ---")
            logger.info(f"Parent Run Name: {parent_run_name}")
            logger.info(f"Parent Run ID: {parent_run_id}")

            # Log parent parameters
            mlflow.log_param("parent.seed", seed)
            mlflow.log_param("parent.base_device_preference", base_device_preference)
            mlflow.log_param("parent.num_trackers_compared", len(trackers_to_run))
            mlflow.log_param("parent.num_reid_models_compared", len(reid_models_to_associate))
            mlflow.log_param("parent.total_combinations", total_combinations)

            # Log comparison config and requirements to parent run
            comp_conf_path = PROJECT_ROOT / config_path_str
            if comp_conf_path.is_file():
                mlflow.log_artifact(str(comp_conf_path), artifact_path="config")
            req_path = PROJECT_ROOT / "requirements.txt"
            if req_path.is_file():
                mlflow.log_artifact(str(req_path), artifact_path="code")

            # Log parent git info
            log_git_info()
            mlflow.set_tag("run_type", "tracking_reid_comparison_parent")

            # 6. Loop Through Combinations of Trackers and Re-ID Models
            for tracker_type in trackers_to_run:
                for reid_model_config in reid_models_to_associate:
                    run_counter += 1
                    reid_model_type = reid_model_config.get("model_type", "default")
                    reid_weights_file = Path(reid_model_config.get("weights_path", "")).stem
                    reid_name = f"{reid_model_type}_{reid_weights_file}" if reid_weights_file else reid_model_type

                    combination_name = f"tracker_{tracker_type}_reid_{reid_name}"
                    child_run_name = f"{combination_name}"
                    logger.info(
                        f"\n--- Starting Nested Child Run ({run_counter}/{total_combinations}): {combination_name} ---")

                    # Construct full config for the child run
                    child_run_config = copy.deepcopy(comparison_config)
                    # Remove lists of models/trackers, parent name
                    del child_run_config["trackers_to_run"]
                    del child_run_config["reid_models_to_associate"]
                    del child_run_config["parent_run_name"]
                    # Add specific tracker and reid model config for this child
                    child_run_config["tracker"] = {"type": tracker_type}
                    child_run_config["reid_model"] = reid_model_config  # Pass the reid model dict

                    child_status = "FAILED"
                    child_run_id_for_status: Optional[str] = None
                    try:
                        # Start nested run
                        with mlflow.start_run(run_name=child_run_name, experiment_id=experiment_id,
                                              nested=True) as child_run:
                            child_run_id = child_run.info.run_id
                            child_run_id_for_status = child_run_id
                            logger.info(f"Child Run '{combination_name}' Started (ID: {child_run_id})")
                            mlflow.set_tag("parent_run_id", parent_run_id)
                            mlflow.set_tag("run_type", "tracking_reid_comparison_child")
                            mlflow.set_tag("tracker_type", tracker_type)
                            mlflow.set_tag("reid_model_type", reid_model_type)
                            mlflow.set_tag("reid_weights_file",
                                           reid_weights_file if reid_weights_file else "Default/None")

                            # Execute core tracking+ReID logic using the new runner
                            child_status, _ = run_single_tracking_reid_experiment(  # Metrics not needed directly here
                                run_config=child_run_config,
                                base_device_preference=base_device_preference,
                                seed=seed,
                                config_file_path=str(PROJECT_ROOT / config_path_str),
                                log_file_path=None
                            )
                            child_run_statuses[child_run_id_for_status] = {"name": combination_name,
                                                                           "status": child_status}
                            logger.info(
                                f"Child Run '{combination_name}' (ID: {child_run_id}) finished with status: {child_status}")

                            if child_status != "FINISHED":
                                overall_status = "PARTIAL_FAILURE"

                    except KeyboardInterrupt:
                        logger.warning(f"Comparison interrupted during child run for {combination_name}.")
                        if child_run_id_for_status: child_run_statuses[child_run_id_for_status] = {
                            "name": combination_name, "status": "KILLED"}
                        overall_status = "KILLED"
                        raise  # Propagate interruption

                    except Exception as child_err:
                        logger.critical(f"Unhandled error during child run for {combination_name}: {child_err}",
                                        exc_info=True)
                        if child_run_id_for_status: child_run_statuses[child_run_id_for_status] = {
                            "name": combination_name, "status": "CRASHED"}
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
            mlflow.log_dict(child_run_statuses, "child_tracking_reid_run_summary.json")
            mlflow.set_tag("overall_status", overall_status)

    except KeyboardInterrupt:
        logger.warning("Tracking+ReID Comparison run interrupted by user (KeyboardInterrupt).")
        overall_status = "KILLED"
        if parent_run_id:
            try:
                MlflowClient().set_terminated(parent_run_id, status="KILLED")
            except Exception as term_err:
                logger.warning(f"Could not terminate parent run {parent_run_id} after KILLED: {term_err}")

    except Exception as e:
        logger.critical(f"An uncaught error occurred during Tracking+ReID comparison orchestration: {e}", exc_info=True)
        overall_status = "FAILED"
        if parent_run_id:
            try:
                client = MlflowClient();
                client.set_tag(parent_run_id, "overall_status", "CRASHED");
                client.set_terminated(parent_run_id, status="FAILED")
            except Exception as term_err:
                logger.warning(f"Could not terminate parent run {parent_run_id} after CRASHED: {term_err}")
    finally:
        logger.info(f"--- Finalizing Tracking+ReID Comparison Run (Overall Status: {overall_status}) ---")

        # Log the main comparison log file to the PARENT run
        if parent_run_id and log_file.exists():
            try:
                for handler in logging.getLogger().handlers: handler.flush()
                client = MlflowClient();
                client.log_artifact(parent_run_id, str(log_file), artifact_path="tracking_reid_comparison_logs")
                logger.info(
                    f"Tracking+ReID Comparison log file '{log_file.name}' logged as artifact to parent run {parent_run_id}.")
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

    logger.info(f"--- Tracking+ReID Model Comparison Run Completed (Overall Status: {overall_status}) ---")
    exit_code = 0
    if overall_status != "SUCCESS":
        logger.error("One or more child Tracking+ReID runs failed, were killed, or crashed.")
        exit_code = 1
    elif overall_status == "PARTIAL_FAILURE":
        logger.warning("Comparison orchestration completed, but one or more child Tracking+ReID runs failed.")
        exit_code = 1  # Treat partial failure as an error exit code
    else:
        logger.info("All Tracking+ReID comparison child runs completed successfully.")
        exit_code = 0

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
