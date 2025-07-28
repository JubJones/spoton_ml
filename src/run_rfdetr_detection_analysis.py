"""
RF-DETR Detection Analysis Runner

This script runs the per-scenario/camera detection analysis
using trained RF-DETR models.

Usage:
    python src/run_rfdetr_detection_analysis.py

The script will:
1. Load the trained RF-DETR model from the specified checkpoint
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
log_file = setup_logging(log_prefix="rfdetr_detection_analysis", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)


def validate_rfdetr_checkpoint(checkpoint_path: str) -> bool:
    """
    Validate that the checkpoint is a valid RF-DETR model checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        
    Returns:
        bool: True if valid RF-DETR checkpoint, False otherwise
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return False
            
        # Check if it's a .pth file
        if checkpoint_path.suffix != '.pth':
            logger.error(f"Expected .pth file, got: {checkpoint_path.suffix}")
            return False
            
        # Try to load and check if it contains RF-DETR model structure
        import torch
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check for RF-DETR model structure indicators
            if isinstance(checkpoint, dict):
                # Look for RF-DETR specific keys
                if 'model' in checkpoint or 'model_state_dict' in checkpoint:
                    logger.info(f"Valid RF-DETR checkpoint found: {checkpoint_path}")
                    return True
                else:
                    logger.warning(f"Checkpoint structure may not be RF-DETR format: {checkpoint_path}")
                    return True  # Still attempt to use it
            else:
                logger.warning(f"Unexpected checkpoint format: {type(checkpoint)}")
                return False
                
        except Exception as load_error:
            logger.error(f"Failed to load checkpoint: {load_error}")
            return False
            
    except Exception as e:
        logger.error(f"Error validating checkpoint: {e}")
        return False


def main():
    """Main function to run RF-DETR detection analysis."""
    
    logger.info("=== Starting RF-DETR Detection Analysis ===")
    config_path_str = "configs/rfdetr_detection_analysis_config.yaml"
    final_status = "FAILED"

    try:
        # 1. Load Configuration
        config = load_config(config_path_str)
        if not config:
            logger.critical(f"Failed to load configuration from {config_path_str}")
            sys.exit(1)

        # 2. Validate Configuration
        checkpoint_path = config.get("local_model_path")
        if checkpoint_path and checkpoint_path.strip():
            # User specified a checkpoint path, validate it
            if not validate_rfdetr_checkpoint(checkpoint_path):
                logger.critical(f"Invalid or missing RF-DETR checkpoint: {checkpoint_path}")
                logger.critical("Please ensure the checkpoint path is correct and the model is trained")
                sys.exit(1)
            logger.info(f"Using trained RF-DETR checkpoint: {checkpoint_path}")
        else:
            # No checkpoint specified, use pre-trained model
            logger.info("No checkpoint path specified, using pre-trained RF-DETR model")
            logger.info("For better analysis results, train your own RF-DETR model first")

        # 3. Set Seed & Device
        seed = config.get("environment", {}).get("seed", 42)
        set_seed(seed)
        logger.info(f"Random seed set to: {seed}")
        
        device_preference = config.get("environment", {}).get("device", "auto")
        device = get_selected_device(device_preference)
        logger.info(f"Using device: {device}")
        
        # RF-DETR can use GPU if available
        logger.info(f"RF-DETR will use device: {device}")

        # 4. Update config for RF-DETR analysis
        config["analysis"]["output_dir"] = "outputs/rfdetr_detection_analysis"
        
        # Ensure model type is set to rfdetr
        if "model" not in config:
            config["model"] = {}
        config["model"]["type"] = "rfdetr"
        
        # Set RF-DETR specific parameters if not already set
        if "size" not in config["model"]:
            config["model"]["size"] = "base"  # Default to base model
        if "num_classes" not in config["model"]:
            config["model"]["num_classes"] = 2  # person + background
        
        # 5. Run RF-DETR Analysis
        logger.info("Starting RF-DETR analysis pipeline...")
        run_phase1_analysis(config, device)
        
        final_status = "COMPLETED"
        logger.info("RF-DETR analysis completed successfully!")
        
    except Exception as e:
        logger.critical(f"RF-DETR analysis failed: {e}", exc_info=True)
        final_status = "FAILED"
        
    finally:
        logger.info(f"=== RF-DETR Analysis Complete (Status: {final_status}) ===")
        
        if final_status == "COMPLETED":
            output_dir = Path(config.get("analysis", {}).get("output_dir", "outputs/rfdetr_detection_analysis"))
            logger.info(f"Results saved to: {output_dir}")
            logger.info("Generated outputs:")
            logger.info("  - failure_images/: Dual-color failure visualizations")
            logger.info("  - reports/: HTML reports and environment analysis")
            logger.info("  - statistics/: Performance matrices and CSV data")
            logger.info("  - rf_detr_analysis_summary.json: Analysis summary")


if __name__ == "__main__":
    main()