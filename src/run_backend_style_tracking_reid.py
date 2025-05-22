"""
Orchestrator script for running Tracking + Re-ID using adapted backend logic.
Compares different Re-ID association/similarity methods.
"""
import copy
import logging
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils.config_loader import load_config
    from src.utils.reproducibility import set_seed
    from src.utils.logging_utils import setup_logging
    from src.utils.mlflow_utils import setup_mlflow_experiment
    from src.utils.device_utils import get_selected_device
    from src.core.runner import log_params_recursive, log_metrics_dict, log_git_info
    from src.pipelines.backend_style_tracking_reid_pipeline import BackendStyleTrackingReidPipeline, TrackingReidResultSummary
except ImportError as e:
    print(f"Error importing local modules in {Path(__file__).name}: {e}")
    sys.exit(1)

log_file_path = setup_logging(log_prefix="backend_reid_compare", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    logger.info("--- Starting Backend-Style Re-ID Association Method Comparison Run ---")
    config_path_str = "configs/backend_style_tracking_reid_config.yaml"
    overall_status = "SUCCESS" # Parent run status
    parent_run_id: Optional[str] = None

    config = load_config(config_path_str)
    if not config:
        logger.critical(f"Failed to load configuration from {config_path_str}. Exiting.")
        sys.exit(1)

    association_methods = config.get("reid_association_methods_to_compare", ["cosine"]) # Default to cosine
    if not association_methods:
        logger.critical("'reid_association_methods_to_compare' list is empty in config. Exiting.")
        sys.exit(1)

    experiment_id = setup_mlflow_experiment(config, default_experiment_name="Default BackendStyle ReID Method Comparison")
    if not experiment_id:
        logger.critical("MLflow experiment setup failed. Exiting.")
        sys.exit(1)

    seed = config.get("environment", {}).get("seed", int(time.time()))
    set_seed(seed)
    logger.info(f"Global random seed set to: {seed}")

    base_device_preference = config.get("environment", {}).get("device", "auto")
    preferred_device = get_selected_device(base_device_preference)
    logger.info(f"Preferred device for runs: {preferred_device}")

    parent_run_name = config.get("parent_run_name", f"reid_assoc_compare_{int(time.time())}")
    child_run_statuses: Dict[str, Dict[str, Any]] = {} # Child Run ID -> {name, status}

    try:
        with mlflow.start_run(run_name=parent_run_name, experiment_id=experiment_id) as parent_run_obj:
            parent_run_id = parent_run_obj.info.run_id
            logger.info(f"--- Parent MLflow Run Started: {parent_run_name} (ID: {parent_run_id}) ---")
            
            log_params_recursive(config)
            mlflow.log_param("parent.seed", seed)
            mlflow.log_param("parent.base_device_preference", base_device_preference)
            mlflow.log_param("parent.num_methods_compared", len(association_methods))
            log_git_info()
            mlflow.set_tag("run_type", "reid_method_comparison_parent")
            
            cfg_path_abs = (PROJECT_ROOT / config_path_str).resolve()
            if cfg_path_abs.is_file(): mlflow.log_artifact(str(cfg_path_abs), artifact_path="config")
            req_path = PROJECT_ROOT / "requirements.txt"
            if req_path.is_file(): mlflow.log_artifact(str(req_path), artifact_path="code")

            for idx, sim_method in enumerate(association_methods):
                child_run_name = f"method_{sim_method}"
                logger.info(f"\n--- Starting Nested Child Run ({idx+1}/{len(association_methods)}): {child_run_name} ---")
                
                # Create a copy of the config for this child run if modifications are needed
                # For this case, the similarity_method is passed to the pipeline directly
                
                child_status = "FAILED"
                child_run_id_for_status: Optional[str] = None
                try:
                    with mlflow.start_run(run_name=child_run_name, experiment_id=experiment_id, nested=True) as child_run_obj:
                        child_run_id = child_run_obj.info.run_id
                        child_run_id_for_status = child_run_id
                        logger.info(f"Child Run '{child_run_name}' Started (ID: {child_run_id})")
                        mlflow.set_tag("parent_run_id", parent_run_id)
                        mlflow.set_tag("run_type", "reid_method_comparison_child")
                        mlflow.set_tag("similarity_method_tested", sim_method)
                        # Log config again to child for self-containment, or just specific parts
                        mlflow.log_dict(config.get("reid_params",{}), "config/reid_params.json")
                        mlflow.log_param("child_similarity_method", sim_method)


                        logger.info(f"Instantiating BackendStyleTrackingReidPipeline for method: {sim_method}...")
                        pipeline = BackendStyleTrackingReidPipeline(
                            config=config, 
                            device=preferred_device,
                            project_root=PROJECT_ROOT,
                            similarity_method=sim_method # Pass the current method
                        )
                        
                        logger.info(f"Running pipeline for method: {sim_method}...")
                        pipeline_success, result_summary = pipeline.run()

                        if hasattr(pipeline, 'actual_tracker_devices') and pipeline.actual_tracker_devices:
                            for cam_id_log, dev_log_val in pipeline.actual_tracker_devices.items():
                                mlflow.set_tag(f"actual_device_cam_{cam_id_log}", str(dev_log_val))
                            unique_devices_used_str = {str(d) for d in pipeline.actual_tracker_devices.values()}
                            mlflow.set_tag("actual_device_all_trackers", 
                                           unique_devices_used_str.pop() if len(unique_devices_used_str) == 1 else "mixed")

                        if result_summary:
                            logger.info("Logging metrics from pipeline summary...")
                            log_metrics_dict(result_summary, prefix="eval")
                        
                        if pipeline_success:
                            child_status = "FINISHED"
                            mlflow.set_tag("run_outcome", "Success")
                        else:
                            child_status = "FAILED"
                            mlflow.set_tag("run_outcome", "Pipeline Failed")
                            overall_status = "PARTIAL_FAILURE" # Mark parent if any child fails
                            
                        child_run_statuses[child_run_id_for_status] = {"name": child_run_name, "status": child_status, "method": sim_method}

                except KeyboardInterrupt:
                    logger.warning(f"Comparison interrupted during child run for {child_run_name}.")
                    if child_run_id_for_status: child_run_statuses[child_run_id_for_status] = {"name": child_run_name, "status": "KILLED", "method": sim_method}
                    overall_status = "KILLED"
                    raise
                except Exception as child_err:
                    logger.critical(f"Unhandled error during child run for {child_run_name}: {child_err}", exc_info=True)
                    if child_run_id_for_status: child_run_statuses[child_run_id_for_status] = {"name": child_run_name, "status": "CRASHED", "method": sim_method}
                    overall_status = "PARTIAL_FAILURE"
                    try:
                        active_child_run = mlflow.active_run()
                        if active_child_run and active_child_run.info.run_id == child_run_id_for_status:
                            mlflow.set_tag("run_outcome", "Crashed - Child Loop")
                            mlflow.end_run("FAILED")
                    except Exception: pass
            
            mlflow.log_dict(child_run_statuses, "child_run_summary.json")
            mlflow.set_tag("overall_status", overall_status)

    except KeyboardInterrupt:
        logger.warning("Parent comparison run interrupted by user (KeyboardInterrupt).")
        overall_status = "KILLED"
    except Exception as e:
        logger.critical(f"An uncaught error occurred during comparison orchestration: {e}", exc_info=True)
        overall_status = "FAILED"
    finally:
        logger.info(f"--- Finalizing Parent Comparison Run (Overall Status: {overall_status}) ---")
        if parent_run_id and log_file_path.exists():
            try:
                for handler in logging.getLogger().handlers: handler.flush()
                client = MlflowClient()
                active_parent_run_info = client.get_run(parent_run_id)
                if active_parent_run_info.info.status in ["RUNNING", "SCHEDULED"]:
                     client.log_artifact(parent_run_id, str(log_file_path), artifact_path="comparison_run_logs")
            except Exception as log_artifact_err:
                logger.warning(f"Could not log parent run log artifact: {log_artifact_err}")

        active_parent_run_obj = mlflow.active_run()
        if active_parent_run_obj and active_parent_run_obj.info.run_id == parent_run_id:
            final_parent_status_mlflow = "FINISHED" if overall_status in ["SUCCESS", "PARTIAL_FAILURE"] else overall_status
            mlflow.end_run(status=final_parent_status_mlflow)
        elif parent_run_id:
            try:
                client = MlflowClient()
                active_parent_run_info = client.get_run(parent_run_id)
                if active_parent_run_info.info.status in ["RUNNING", "SCHEDULED"]:
                    final_parent_status_mlflow = "FINISHED" if overall_status in ["SUCCESS", "PARTIAL_FAILURE"] else overall_status
                    client.set_terminated(parent_run_id, status=final_parent_status_mlflow)
            except Exception as term_err:
                logger.error(f"Failed to terminate parent run {parent_run_id} externally: {term_err}")
        
    logger.info(f"--- Backend-Style Re-ID Association Method Comparison Completed (Overall Status: {overall_status}) ---")
    sys.exit(0 if overall_status == "SUCCESS" else 1)

if __name__ == "__main__":
    main()