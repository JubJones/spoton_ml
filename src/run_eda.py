# src/run_eda.py

import logging
import sys
import time
import traceback
from typing import Dict, Any, List, Tuple, Optional # Keep necessary types
import warnings
from pathlib import Path
import shutil

import mlflow
import numpy as np # Keep numpy if still needed for log_to_mlflow

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"PROJECT_ROOT added to sys.path: {PROJECT_ROOT}")

# --- Local Imports ---
try:
    from src.utils.config_loader import load_config
    from src.utils.logging_utils import setup_logging
    from src.utils.mlflow_utils import setup_mlflow_experiment
    from src.core.runner import log_git_info
    from src.pipelines.eda_pipeline import EDAPipeline, EDAResults # Import the pipeline and results type
except ImportError as e:
    print(f"Error importing local modules: {e}\nPlease ensure PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="eda", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning)


# --- MLflow Logging Function ---
def log_to_mlflow(
    stats: Dict[str, Any],
    quality_report: Dict[str, Any],
    dist_plot_paths: List[Path],
    comparison_plot_paths: List[Path],
    summary_path: Optional[Path], # Made Optional
    prep_summary_path: Optional[Path], # Add prep summary path
    config_path: str
):
    """Logs EDA results to the active MLflow run."""
    logger.info("Logging EDA results to MLflow...")

    # Log Config
    try:
        mlflow.log_artifact(config_path, artifact_path="config")
    except Exception as e: logger.error(f"Failed to log config artifact: {e}")

    # Log Parameters (Counts)
    params_to_log = [
        'total_scenes', 'total_cameras', 'total_frame_indices_discovered',
        'total_annotations', 'total_unique_obj_ids'
    ]
    for key in params_to_log:
        if key in stats:
            try: mlflow.log_param(key, stats[key])
            except Exception as e: logger.warning(f"Failed to log param '{key}': {e}")

    # Log Metrics (Statistics & Quality Checks)
    metrics_to_log = {**stats, **quality_report}
    for key, value in metrics_to_log.items():
        # Log only simple numeric metrics directly
        if key not in params_to_log and isinstance(value, (int, float, np.number)) and np.isfinite(value):
            try:
                mlflow.log_metric(key, float(value))
            except Exception as e: logger.warning(f"Failed to log metric '{key}': {e}")
        # Log other non-numeric/complex items as parameters (truncating if needed)
        elif key not in params_to_log:
             try:
                 param_value_str = str(value)
                 if len(param_value_str) > 500: # MLflow limit
                     param_value_str = param_value_str[:497] + "..."
                 mlflow.log_param(key, param_value_str)
             except Exception as param_log_err:
                  logger.warning(f"Could not log non-metric item as param '{key}': {param_log_err}", exc_info=False)


    # --- Log Artifacts ---
    # Log Distribution Plots
    if dist_plot_paths:
        try:
            dist_plot_dir = dist_plot_paths[0].parent
            mlflow.log_artifacts(str(dist_plot_dir), artifact_path="eda_plots/distributions")
            logger.info(f"Logged {len(dist_plot_paths)} distribution plots.")
        except Exception as e:
            logger.error(f"Failed to log distribution plot artifacts from {dist_plot_dir}: {e}")

    # Log Comparison Plots
    if comparison_plot_paths:
        try:
            comp_plot_dir = comparison_plot_paths[0].parent
            mlflow.log_artifacts(str(comp_plot_dir), artifact_path="eda_plots/preprocessing_comparison")
            logger.info(f"Logged {len(comparison_plot_paths)} comparison plots.")
        except Exception as e:
            logger.error(f"Failed to log comparison plot artifacts from {comp_plot_dir}: {e}")

    # Log Main Summary Report
    if summary_path and summary_path.exists():
        try:
            mlflow.log_artifact(str(summary_path), artifact_path="summary")
            logger.info(f"Logged main summary report artifact.")
        except Exception as e:
            logger.error(f"Failed to log main summary artifact: {e}")

    # Log Preprocessing Summary Report
    if prep_summary_path and prep_summary_path.exists():
        try:
            mlflow.log_artifact(str(prep_summary_path), artifact_path="summary") # Log to same dir
            logger.info(f"Logged preprocessing summary artifact.")
        except Exception as e:
            logger.error(f"Failed to log preprocessing summary artifact: {e}")

    logger.info("MLflow logging complete.")


# --- Main Execution ---
def main():
    """Main function to orchestrate the EDA process using EDAPipeline."""
    logger.info("--- Starting MTMMC Dataset EDA Run (Orchestrator) ---")
    config_path_str = "configs/eda_config.yaml"
    final_status = "FAILED"
    run_id = None
    temp_output_dir = None
    eda_results: EDAResults = {} # Store results from pipeline

    try:
        # 1. Load Configuration
        config = load_config(config_path_str)
        if not config:
            logger.critical(f"Failed to load configuration from {config_path_str}. Exiting.")
            sys.exit(1)
        config_path_abs = (PROJECT_ROOT / config_path_str).resolve()

        temp_output_dir = PROJECT_ROOT / config.get("output_dir", "eda_artifacts_temp")
        # Clean up existing temp dir if it exists
        if temp_output_dir.exists():
            logger.warning(f"Removing existing temporary output directory: {temp_output_dir}")
            shutil.rmtree(temp_output_dir)
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using temporary output directory: {temp_output_dir}")

        # 2. Setup MLflow
        experiment_id = setup_mlflow_experiment(config, default_experiment_name="Default MTMMC EDA")
        if not experiment_id:
            logger.critical("MLflow experiment setup failed. Exiting.")
            sys.exit(1)
        run_name = f"{config.get('mlflow', {}).get('run_name_prefix', 'eda_run')}_{time.strftime('%Y%m%d_%H%M%S')}"


        # 3. Start MLflow Run
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            logger.info(f"--- MLflow Run Started ---")
            logger.info(f"Run Name: {run_name}")
            logger.info(f"Run ID: {run_id}")
            log_git_info() # Log git info if available

            # 4. Instantiate and Run the EDA Pipeline
            logger.info("--- Instantiating and Running EDAPipeline ---")
            pipeline = EDAPipeline(config, temp_output_dir)
            pipeline_success, eda_results = pipeline.run()

            # 5. Log Results to MLflow (using results from pipeline)
            logger.info("--- Logging Results to MLflow ---")
            log_to_mlflow(
                stats=eda_results.get("stats", {}),
                quality_report=eda_results.get("quality_report", {}),
                dist_plot_paths=eda_results.get("dist_plot_paths", []),
                comparison_plot_paths=eda_results.get("comparison_plot_paths", []),
                summary_path=eda_results.get("summary_path"),
                prep_summary_path=eda_results.get("prep_summary_path"), # Get prep summary path
                config_path=str(config_path_abs)
            )

            if pipeline_success:
                final_status = "FINISHED"
                mlflow.set_tag("run_outcome", "Success")
            else:
                final_status = "FAILED" # Pipeline itself reported failure
                mlflow.set_tag("run_outcome", "Pipeline Failed")


    except KeyboardInterrupt:
        logger.warning("EDA run interrupted by user (KeyboardInterrupt).")
        final_status = "KILLED"
        if run_id and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
            mlflow.set_tag("run_outcome", "Killed by user")
            mlflow.end_run(status=final_status)
        elif run_id:
            try: mlflow.tracking.MlflowClient().set_terminated(run_id, status=final_status)
            except Exception: logger.warning(f"Could not terminate run {run_id} externally.")
    except Exception as e:
        # Catch errors outside the pipeline run (e.g., config load, MLflow setup, pipeline instantiation)
        logger.critical(f"An uncaught error occurred during the EDA orchestration: {e}", exc_info=True)
        final_status = "FAILED"
        if run_id and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
             mlflow.set_tag("run_outcome", "Orchestrator Crashed")
             try: mlflow.log_text(f"Orchestrator Error: {type(e).__name__}\n{e}\n{traceback.format_exc()}", "error_log.txt")
             except Exception: pass
             mlflow.end_run(status=final_status)
        elif run_id:
             try:
                 client = mlflow.tracking.MlflowClient()
                 client.set_tag(run_id, "run_outcome", "Orchestrator Crashed")
                 client.set_terminated(run_id, status=final_status)
             except Exception: logger.warning(f"Could not terminate run {run_id} externally after orchestrator crash.")

    finally:
        logger.info(f"--- Finalizing EDA Run (Final Status: {final_status}) ---")
        # Log the main script log file to the run if it exists
        if run_id and log_file.exists():
            try:
                # Ensure logs are flushed
                for handler in logging.getLogger().handlers: handler.flush()
                client = mlflow.tracking.MlflowClient()
                client.log_artifact(run_id, str(log_file), artifact_path="logs")
                logger.info(f"Main EDA log file '{log_file.name}' logged as artifact to run {run_id}.")
            except Exception as log_artifact_err:
                logger.warning(f"Could not log main EDA log file artifact '{log_file}': {log_artifact_err}")

        # Ensure run termination status is set correctly if run wasn't ended above
        active_run = mlflow.active_run()
        if active_run and active_run.info.run_id == run_id:
            logger.info(f"Ensuring MLflow run {run_id} is terminated with status {final_status}.")
            mlflow.end_run(status=final_status)
        elif run_id: # If run exists but isn't active (e.g., due to error outside context)
            try:
                logger.warning(f"Attempting to terminate run {run_id} outside active context with status {final_status}.")
                mlflow.tracking.MlflowClient().set_terminated(run_id, status=final_status)
            except Exception as term_err:
                logger.error(f"Failed to terminate run {run_id} forcefully: {term_err}")

        # Clean up temporary directory ONLY IF THE RUN WAS SUCCESSFUL OR FAILED (NOT KILLED)
        if final_status != "KILLED" and temp_output_dir and temp_output_dir.exists():
             try:
                 shutil.rmtree(temp_output_dir)
                 logger.info(f"Cleaned up temporary output directory: {temp_output_dir}")
             except Exception as e:
                 logger.warning(f"Failed to clean up temporary output directory {temp_output_dir}: {e}")
        elif temp_output_dir and temp_output_dir.exists():
            logger.warning(f"Run status was '{final_status}'. Preserving temporary artifacts in {temp_output_dir}")


    logger.info(f"--- EDA Run Completed (Status: {final_status}) ---")
    sys.exit(0 if final_status == "FINISHED" else 1)


if __name__ == "__main__":
    # Ensure matplotlib backend is suitable for non-interactive use if needed
    import matplotlib
    matplotlib.use('Agg') # Use Agg backend which doesn't require a GUI
    main()