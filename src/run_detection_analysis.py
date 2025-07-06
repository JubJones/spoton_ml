"""
Main entry point for running the detection model performance analysis pipeline.

This script orchestrates the process of loading a trained model, running it on a
dataset, evaluating its performance on a frame-by-frame basis, and saving
visual examples of the best and worst performing frames for qualitative review.
"""
import logging
import sys
from pathlib import Path

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Local Imports ---
try:
    from src.utils.config_loader import load_config
    from src.utils.reproducibility import set_seed
    from src.utils.logging_utils import setup_logging
    from src.utils.device_utils import get_selected_device
    from src.pipelines.detection_analysis_pipeline import run_analysis
except ImportError as e:
    print(f"Error importing local modules in run_detection_analysis.py: {e}")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="detection_analysis", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the detection analysis pipeline.
    """
    logger.info("--- Starting Detection Performance Analysis Run ---")
    config_path_str = "configs/detection_analysis_config.yaml"
    final_status = "FAILED"

    # 1. Load Configuration
    config = load_config(config_path_str)
    if not config:
        logger.critical(f"Failed to load configuration from {config_path_str}. Exiting.")
        sys.exit(1)

    # 2. Validate Configuration
    has_local_path = config.get("local_model_path")
    has_mlflow_id = config.get("mlflow_run_id") and config["mlflow_run_id"] != "CHANGE_ME_RUN_ID"

    if not has_local_path and not has_mlflow_id:
        logger.critical(
            "Configuration error: Either 'local_model_path' or 'mlflow_run_id' must be set "
            f"in {config_path_str}."
        )
        sys.exit(1)

    # 3. Set Seed & Determine Device
    seed = config.get("environment", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Global random seed set to: {seed}")
    device_preference = config.get("environment", {}).get("device", "auto")
    resolved_device = get_selected_device(device_preference)
    logger.info(f"Resolved analysis device: {resolved_device}")

    try:
        # 4. Execute the analysis pipeline
        run_analysis(
            config=config,
            device=resolved_device,
            project_root=PROJECT_ROOT,
        )
        final_status = "FINISHED"

    except Exception as e:
        logger.critical(f"An uncaught error occurred during analysis: {e}", exc_info=True)
        final_status = "FAILED"

    finally:
        logger.info(f"--- Finalizing Analysis Run (Final Status: {final_status}) ---")

    logger.info(f"--- Detection Performance Analysis Completed (Status: {final_status}) ---")
    exit_code = 0 if final_status == "FINISHED" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 