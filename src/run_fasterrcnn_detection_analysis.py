"""
FasterRCNN Detection Analysis Runner

This script runs the per-scenario/camera detection analysis
using trained FasterRCNN models.

Usage:
    python src/run_fasterrcnn_detection_analysis.py

The script will:
1. Load the trained FasterRCNN model from the specified checkpoint
2. Analyze performance across all scene/camera combinations
3. Collect 1 failure image per person ID with dual-color visualization
4. Generate comprehensive reports and performance mapping
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
    from src.pipelines.phase1_detection_analysis import run_phase1_analysis
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="fasterrcnn_detection_analysis", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)


def main():
    """Main function to run FasterRCNN detection analysis."""
    
    logger.info("=== Starting FasterRCNN Detection Analysis ===")
    config_path_str = "configs/fasterrcnn_detection_analysis_config.yaml"
    final_status = "FAILED"

    try:
        # 1. Load Configuration
        config = load_config(config_path_str)
        if not config:
            logger.critical(f"Failed to load configuration from {config_path_str}")
            sys.exit(1)

        # 2. Validate Configuration
        checkpoint_path = config.get("local_model_path")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            logger.critical(f"Model checkpoint not found at: {checkpoint_path}")
            logger.critical("Please ensure the checkpoint path is correct in the config")
            sys.exit(1)

        # 3. Set Seed & Device
        seed = config.get("environment", {}).get("seed", 42)
        set_seed(seed)
        logger.info(f"Random seed set to: {seed}")
        
        device_preference = config.get("environment", {}).get("device", "cpu")
        device = get_selected_device(device_preference)
        logger.info(f"Using device: {device}")
        
        # Force CPU for FasterRCNN if needed
        if device.type != "cpu":
            logger.info("Forcing CPU for FasterRCNN compatibility")
            device = get_selected_device("cpu")

        # 4. Update config for Phase 1 analysis
        config["analysis"]["output_dir"] = "outputs/phase1_detection_analysis"
        
        # 5. Run Phase 1 Analysis
        logger.info("Starting Phase 1 analysis pipeline...")
        run_phase1_analysis(config, device)
        
        final_status = "COMPLETED"
        logger.info("Phase 1 analysis completed successfully!")
        
    except Exception as e:
        logger.critical(f"Phase 1 analysis failed: {e}", exc_info=True)
        final_status = "FAILED"
        
    finally:
        logger.info(f"=== Phase 1 Analysis Complete (Status: {final_status}) ===")
        
        if final_status == "COMPLETED":
            output_dir = Path(config.get("analysis", {}).get("output_dir", "outputs/phase1_detection_analysis"))
            logger.info(f"Results saved to: {output_dir}")
            logger.info("Generated outputs:")
            logger.info("  - failure_images/: Dual-color failure visualizations")
            logger.info("  - reports/: HTML reports and environment analysis")
            logger.info("  - statistics/: Performance matrices and CSV data")


if __name__ == "__main__":
    main()