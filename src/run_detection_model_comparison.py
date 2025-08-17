"""
Detection Model Comparison Runner - RF-DETR vs FasterRCNN

This script runs a comprehensive comparison between RF-DETR and FasterRCNN
detection models on the same validation dataset.

Usage:
    python src/run_detection_model_comparison.py

The script will:
1. Load both trained RF-DETR and FasterRCNN models from checkpoints
2. Evaluate both models on the same validation dataset
3. Calculate comprehensive performance metrics for both models
4. Perform statistical analysis and significance testing
5. Generate detailed comparison visualizations and reports
6. Provide recommendations based on performance analysis

Features:
- Unified evaluation framework for both models
- Comprehensive performance metrics (mAP, precision, recall, speed)
- Per-scene and per-camera analysis
- Statistical significance testing
- Detailed HTML reports and executive summary
- MLflow experiment tracking
- Multiple export formats (JSON, CSV, HTML)
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

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
    from src.utils.mlflow_utils import setup_mlflow_experiment
    from src.pipelines.detection_model_comparison_pipeline import DetectionModelComparisonPipeline
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure all modules exist and PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="detection_model_comparison", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

def validate_configuration(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate configuration and check for required model checkpoints.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict with validation results
    """
    validation_results = {
        "config_valid": True,
        "rfdetr_model_available": False,
        "fasterrcnn_model_available": False,
        "dataset_config_valid": False
    }
    
    # Check model configurations
    models_config = config.get("models", {})
    
    # Validate RF-DETR configuration
    rfdetr_config = models_config.get("rfdetr", {})
    if rfdetr_config:
        checkpoint_path = rfdetr_config.get("trained_model_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            validation_results["rfdetr_model_available"] = True
            logger.info(f"RF-DETR checkpoint found: {checkpoint_path}")
        else:
            logger.warning(f"RF-DETR checkpoint not found: {checkpoint_path}")
            logger.warning("Will use pre-trained RF-DETR weights")
    
    # Validate FasterRCNN configuration  
    fasterrcnn_config = models_config.get("fasterrcnn", {})
    if fasterrcnn_config:
        checkpoint_path = fasterrcnn_config.get("trained_model_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            validation_results["fasterrcnn_model_available"] = True
            logger.info(f"FasterRCNN checkpoint found: {checkpoint_path}")
        else:
            logger.warning(f"FasterRCNN checkpoint not found: {checkpoint_path}")
            logger.warning("Will use pre-trained FasterRCNN weights")
    
    # Validate dataset configuration
    data_config = config.get("data", {})
    base_path = data_config.get("base_path")
    if base_path and Path(base_path).exists():
        validation_results["dataset_config_valid"] = True
        logger.info(f"Dataset path validated: {base_path}")
    else:
        logger.error(f"Dataset path not found: {base_path}")
        logger.error("Please update base_path in the configuration file")
        validation_results["dataset_config_valid"] = False
        validation_results["config_valid"] = False
    
    # Check scenes configuration
    scenes_to_include = data_config.get("scenes_to_include", [])
    if not scenes_to_include:
        logger.warning("No scenes configured - using default scenes")
    
    return validation_results

def setup_mlflow_tracking(config: Dict[str, Any]) -> Optional[str]:
    """
    Set up MLflow experiment tracking.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MLflow experiment ID or None if setup fails
    """
    try:
        mlflow_config = config.get("mlflow", {})
        experiment_name = mlflow_config.get("experiment_name", "Detection Model Comparison - RF-DETR vs FasterRCNN")
        
        experiment_id = setup_mlflow_experiment(
            config=config,
            default_experiment_name=experiment_name
        )
        
        if experiment_id:
            logger.info(f"MLflow experiment setup successful: {experiment_name}")
            logger.info(f"Experiment ID: {experiment_id}")
        else:
            logger.warning("MLflow experiment setup failed")
        
        return experiment_id
        
    except Exception as e:
        logger.warning(f"Failed to setup MLflow tracking: {e}")
        return None

def run_comparison_analysis(config: Dict[str, Any], device) -> Dict[str, Any]:
    """
    Run the complete detection model comparison analysis.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device to use
        
    Returns:
        Dict with complete comparison results
    """
    logger.info("Starting detection model comparison analysis...")
    
    # Initialize comparison pipeline
    pipeline = DetectionModelComparisonPipeline(config, device)
    
    # Run complete comparison
    start_time = time.time()
    comparison_results = pipeline.run_complete_comparison()
    execution_time = time.time() - start_time
    
    logger.info(f"Comparison analysis completed in {execution_time:.2f} seconds")
    
    # Add execution metadata
    comparison_results["execution_metadata"] = {
        "total_execution_time_seconds": execution_time,
        "device_used": str(device),
        "dataset_size": len(pipeline.evaluator.dataset)
    }
    
    return comparison_results

def print_comparison_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of the comparison results to console.
    
    Args:
        results: Comparison results dictionary
    """
    logger.info("=" * 60)
    logger.info("DETECTION MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    # Extract metrics
    rfdetr_metrics = results.get("rfdetr_metrics", {})
    fasterrcnn_metrics = results.get("fasterrcnn_metrics", {})
    comparison_results = results.get("comparison_results", {})
    
    # Performance comparison
    logger.info("PERFORMANCE METRICS:")
    logger.info("-" * 30)
    logger.info(f"RF-DETR:")
    logger.info(f"  mAP@0.5: {rfdetr_metrics.get('map_50', 0):.4f}")
    logger.info(f"  Precision: {rfdetr_metrics.get('precision', 0):.4f}")
    logger.info(f"  Recall: {rfdetr_metrics.get('recall', 0):.4f}")
    logger.info(f"  F1-Score: {rfdetr_metrics.get('f1_score', 0):.4f}")
    logger.info(f"  Avg Inference: {rfdetr_metrics.get('avg_inference_time_ms', 0):.2f}ms")
    
    logger.info(f"\nFasterRCNN:")
    logger.info(f"  mAP@0.5: {fasterrcnn_metrics.get('map_50', 0):.4f}")
    logger.info(f"  Precision: {fasterrcnn_metrics.get('precision', 0):.4f}")
    logger.info(f"  Recall: {fasterrcnn_metrics.get('recall', 0):.4f}")
    logger.info(f"  F1-Score: {fasterrcnn_metrics.get('f1_score', 0):.4f}")
    logger.info(f"  Avg Inference: {fasterrcnn_metrics.get('avg_inference_time_ms', 0):.2f}ms")
    
    # Key findings
    logger.info("\nKEY FINDINGS:")
    logger.info("-" * 20)
    logger.info(f"Best Overall Model: {comparison_results.get('better_model', 'Unknown')}")
    logger.info(f"Performance Gap: {comparison_results.get('performance_gap', 0):.4f} mAP@0.5")
    logger.info(f"Significant Difference: {'Yes' if comparison_results.get('significant_difference', False) else 'No'}")
    
    speed_diff = comparison_results.get('speed_difference_ms', 0)
    if speed_diff > 0:
        logger.info(f"Speed: FasterRCNN is {abs(speed_diff):.1f}ms faster")
    elif speed_diff < 0:
        logger.info(f"Speed: RF-DETR is {abs(speed_diff):.1f}ms faster")
    else:
        logger.info("Speed: Models have similar inference times")
    
    # Execution info
    exec_metadata = results.get("execution_metadata", {})
    logger.info(f"\nExecution Time: {exec_metadata.get('total_execution_time_seconds', 0):.2f} seconds")
    logger.info(f"Dataset Size: {exec_metadata.get('dataset_size', 0)} samples")
    
    logger.info("=" * 60)

def main():
    """Main function to run detection model comparison."""
    
    logger.info("=== Starting Detection Model Comparison (RF-DETR vs FasterRCNN) ===")
    config_path_str = "configs/detection_model_comparison_config.yaml"
    final_status = "FAILED"
    
    try:
        # 1. Load Configuration
        logger.info(f"Loading configuration from: {config_path_str}")
        config = load_config(config_path_str)
        if not config:
            logger.critical(f"Failed to load configuration from {config_path_str}")
            logger.critical("Please ensure the configuration file exists and is valid YAML")
            sys.exit(1)
        
        # 2. Validate Configuration
        logger.info("Validating configuration...")
        validation_results = validate_configuration(config)
        
        if not validation_results["config_valid"]:
            logger.critical("Configuration validation failed")
            sys.exit(1)
        
        if not validation_results["dataset_config_valid"]:
            logger.critical("Dataset configuration is invalid")
            logger.critical("Please check the base_path in your configuration")
            sys.exit(1)
        
        # Warn about missing model checkpoints
        if not validation_results["rfdetr_model_available"]:
            logger.warning("RF-DETR trained checkpoint not available - using pre-trained weights")
        
        if not validation_results["fasterrcnn_model_available"]:
            logger.warning("FasterRCNN trained checkpoint not available - using pre-trained weights")
        
        if not (validation_results["rfdetr_model_available"] or validation_results["fasterrcnn_model_available"]):
            logger.warning("No trained checkpoints found - comparing pre-trained models only")
        
        # 3. Set Seed & Device
        seed = config.get("environment", {}).get("seed", 42)
        set_seed(seed)
        logger.info(f"Random seed set to: {seed}")
        
        device_preference = config.get("environment", {}).get("device", "auto")
        device = get_selected_device(device_preference)
        logger.info(f"Using device: {device}")
        
        # 4. Setup MLflow Tracking
        logger.info("Setting up MLflow experiment tracking...")
        experiment_id = setup_mlflow_tracking(config)
        
        # 5. Run Comparison Analysis
        logger.info("Starting comprehensive model comparison...")
        logger.info("This may take several minutes depending on dataset size...")
        
        comparison_results = run_comparison_analysis(config, device)
        
        # 6. Print Summary
        print_comparison_summary(comparison_results)
        
        # 7. Log completion
        output_dir = Path(config.get("comparison", {}).get("output_dir", "outputs/detection_model_comparison"))
        logger.info(f"Results saved to: {output_dir}")
        logger.info("Generated outputs:")
        logger.info("  * visualizations/: Performance comparison charts")
        logger.info("  * reports/: Executive summary and technical reports") 
        logger.info("  * comparison_data.json: Raw comparison data")
        logger.info("  * performance_metrics.csv: Tabulated metrics")
        
        if experiment_id:
            logger.info(f"  * MLflow Experiment ID: {experiment_id}")
        
        final_status = "COMPLETED"
        logger.info("Detection model comparison completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Comparison interrupted by user (Ctrl+C)")
        final_status = "INTERRUPTED"
        
    except Exception as e:
        logger.critical(f"Detection model comparison failed: {e}", exc_info=True)
        final_status = "FAILED"
        
    finally:
        logger.info(f"=== Detection Model Comparison Complete (Status: {final_status}) ===")
        
        if final_status == "COMPLETED":
            logger.info("\nNext Steps:")
            logger.info("1. Review the executive summary in reports/executive_summary.txt")
            logger.info("2. Examine detailed analysis in reports/technical_report.html")
            logger.info("3. Check visualizations for performance insights")
            logger.info("4. Use findings to select the best model for your use case")
        elif final_status == "FAILED":
            logger.info("\nTroubleshooting:")
            logger.info("1. Verify dataset path in configuration")
            logger.info("2. Check model checkpoint paths")
            logger.info("3. Ensure sufficient disk space and memory")
            logger.info("4. Review error logs above for specific issues")

if __name__ == "__main__":
    main()