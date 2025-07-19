"""
Enhanced Detection Analysis Runner - Phase 1 Implementation.

This script runs the comprehensive Phase 1 enhanced detection analysis pipeline
implementing all components from ANALYSIS_PLANNING.md:

Phase 1: Enhanced Detection Analysis
1.1 Multi-dimensional failure detection and classification
1.2 Cross-model detection comparison 
1.3 Advanced detection metrics and automated reporting

Usage:
    python src/run_enhanced_detection_analysis.py

Configuration:
    Update configs/enhanced_detection_analysis_config.yaml with your settings.
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
    from src.pipelines.enhanced_detection_analysis_pipeline import run_enhanced_analysis
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="enhanced_detection_analysis", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the enhanced detection analysis pipeline.
    """
    logger.info("🚀 Starting Enhanced Detection Analysis - Phase 1 Implementation")
    config_path_str = "configs/enhanced_detection_analysis_config.yaml"
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
        # 4. Execute the enhanced analysis pipeline
        logger.info("🔥 Launching Enhanced Detection Analysis Pipeline...")
        logger.info("📋 Phase 1 Components:")
        logger.info("   ✓ Multi-dimensional failure detection and classification")
        logger.info("   ✓ Temporal failure pattern analysis") 
        logger.info("   ✓ Enhanced failure visualization system")
        logger.info("   ✓ Statistical failure analysis and reporting")
        logger.info("   ✓ Cross-model detection comparison matrix")
        logger.info("   ✓ Ensemble opportunities analysis")
        logger.info("   ✓ Advanced detection metrics collection")
        logger.info("   ✓ Automated report generation system")
        logger.info("=" * 60)
        
        generated_files = run_enhanced_analysis(
            config=config,
            device=resolved_device,
            project_root=PROJECT_ROOT,
        )
        
        # Log success with file summary
        logger.info("🎉 ENHANCED DETECTION ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("📊 Generated Analysis Files:")
        
        for file_type, file_path in generated_files.items():
            logger.info(f"   📄 {file_type.replace('_', ' ').title()}: {file_path}")
        
        logger.info("=" * 60)
        logger.info("🎯 Key Outputs:")
        logger.info("   📈 Interactive HTML Report with visualizations")
        logger.info("   📊 Executive Dashboard with performance metrics")
        logger.info("   🔍 Detailed failure analysis with scene context")
        logger.info("   📈 Cross-model comparison (if enabled)")
        logger.info("   🚨 Automated performance alerts")
        logger.info("   📋 Comprehensive JSON reports")
        
        final_status = "FINISHED"

    except Exception as e:
        logger.critical(f"Enhanced analysis pipeline failed: {e}", exc_info=True)
        final_status = "FAILED"

    finally:
        logger.info(f"--- Enhanced Detection Analysis Completed (Status: {final_status}) ---")

    exit_code = 0 if final_status == "FINISHED" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()