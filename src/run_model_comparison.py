"""
Model Comparison Runner - Compare Initial vs Trained Detection Models

This script compares the performance of detection models (FasterRCNN or RF-DETR)
between their initial pre-trained weights and trained weights.

Usage:
    python src/run_model_comparison.py

The script will:
1. Load both initial (pre-trained) and trained model weights
2. Run detection on the validation dataset
3. Calculate performance metrics (mAP, precision, recall, etc.)
4. Generate detailed comparison reports and visualizations
5. Save results for further analysis

Supports both FasterRCNN and RF-DETR models with unified interface.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json

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
    from src.pipelines.model_comparison_pipeline import ModelComparisonPipeline
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="model_comparison", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

@dataclass
class ModelComparisonResult:
    """Results from model comparison analysis."""
    model_type: str
    initial_metrics: Dict[str, float]
    trained_metrics: Dict[str, float]
    improvement_metrics: Dict[str, float]
    comparison_summary: Dict[str, Any]
    timestamp: str
    config_used: Dict[str, Any]

def validate_model_checkpoints(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate that model checkpoints exist and are accessible.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict with validation results for each model path
    """
    validation_results = {}
    
    # Check trained model checkpoint
    trained_path = config.get("trained_model_path")
    if trained_path and trained_path.strip():
        trained_path = Path(trained_path)
        validation_results["trained_model"] = trained_path.exists()
        if not validation_results["trained_model"]:
            logger.warning(f"Trained model checkpoint not found: {trained_path}")
    else:
        validation_results["trained_model"] = False
        logger.warning("No trained model path specified")
    
    # Check if we can load initial model (pre-trained weights)
    model_type = config.get("model", {}).get("type", "").lower()
    validation_results["initial_model"] = model_type in ["fasterrcnn", "rfdetr"]
    
    if not validation_results["initial_model"]:
        logger.error(f"Unsupported model type: {model_type}")
    
    return validation_results

def setup_mlflow_tracking(config: Dict[str, Any]) -> str:
    """
    Set up MLflow tracking for the comparison experiment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MLflow run ID
    """
    experiment_name = config.get("mlflow", {}).get("experiment_name", "Model Comparison")
    
    # Setup MLflow experiment
    run_id = setup_mlflow_experiment(
        experiment_name=experiment_name,
        run_name=f"comparison_{config.get('model', {}).get('type', 'unknown')}_{int(time.time())}",
        description="Comparison between initial and trained model performance"
    )
    
    return run_id

def run_model_comparison(config: Dict[str, Any], device) -> ModelComparisonResult:
    """
    Run the complete model comparison pipeline.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device to use
        
    Returns:
        ModelComparisonResult with all comparison metrics
    """
    logger.info("Starting model comparison pipeline...")
    
    # Initialize comparison pipeline
    pipeline = ModelComparisonPipeline(config, device)
    
    # Run comparison
    comparison_result = pipeline.run_complete_comparison()
    
    # Create result object
    result = ModelComparisonResult(
        model_type=config.get("model", {}).get("type", "unknown"),
        initial_metrics=comparison_result["initial_metrics"],
        trained_metrics=comparison_result["trained_metrics"],
        improvement_metrics=comparison_result["improvement_metrics"],
        comparison_summary=comparison_result["summary"],
        timestamp=comparison_result["timestamp"],
        config_used=config
    )
    
    logger.info("Model comparison pipeline completed successfully")
    return result

def save_comparison_results(result: ModelComparisonResult, output_dir: Path) -> None:
    """
    Save comparison results to files.
    
    Args:
        result: ModelComparisonResult object
        output_dir: Directory to save results
    """
    logger.info(f"Saving comparison results to: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    results_file = output_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    
    # Save summary report
    summary_file = output_dir / "comparison_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Model Comparison Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Model Type: {result.model_type}\n")
        f.write(f"Timestamp: {result.timestamp}\n\n")
        
        f.write(f"Initial Model Performance:\n")
        for metric, value in result.initial_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write(f"\nTrained Model Performance:\n")
        for metric, value in result.trained_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write(f"\nImprovement Metrics:\n")
        for metric, value in result.improvement_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write(f"\nComparison Summary:\n")
        for key, value in result.comparison_summary.items():
            f.write(f"  {key}: {value}\n")
    
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Summary saved to: {summary_file}")

def main():
    """Main function to run model comparison."""
    
    logger.info("=== Starting Model Comparison ===")
    config_path_str = "configs/model_comparison_config.yaml"
    final_status = "FAILED"
    
    try:
        # 1. Load Configuration
        config = load_config(config_path_str)
        if not config:
            logger.critical(f"Failed to load configuration from {config_path_str}")
            sys.exit(1)
        
        # 2. Validate Configuration
        validation_results = validate_model_checkpoints(config)
        
        if not validation_results.get("initial_model", False):
            logger.critical("Invalid model configuration")
            sys.exit(1)
        
        if not validation_results.get("trained_model", False):
            logger.warning("No trained model found - will use pre-trained weights only")
            logger.info("This will compare pre-trained weights against themselves")
        
        # 3. Set Seed & Device
        seed = config.get("environment", {}).get("seed", 42)
        set_seed(seed)
        logger.info(f"Random seed set to: {seed}")
        
        device_preference = config.get("environment", {}).get("device", "auto")
        device = get_selected_device(device_preference)
        logger.info(f"Using device: {device}")
        
        # 4. Setup MLflow Tracking
        run_id = setup_mlflow_tracking(config)
        logger.info(f"MLflow run ID: {run_id}")
        
        # 5. Run Model Comparison
        logger.info("Running model comparison...")
        comparison_result = run_model_comparison(config, device)
        
        # 6. Save Results
        output_dir = Path(config.get("comparison", {}).get("output_dir", "outputs/model_comparison"))
        save_comparison_results(comparison_result, output_dir)
        
        # 7. Log Summary
        logger.info("=== Comparison Summary ===")
        logger.info(f"Model Type: {comparison_result.model_type}")
        logger.info(f"Initial mAP: {comparison_result.initial_metrics.get('map_50', 0):.4f}")
        logger.info(f"Trained mAP: {comparison_result.trained_metrics.get('map_50', 0):.4f}")
        logger.info(f"Improvement: {comparison_result.improvement_metrics.get('map_50_improvement', 0):.4f}")
        
        final_status = "COMPLETED"
        logger.info("Model comparison completed successfully!")
        
    except Exception as e:
        logger.critical(f"Model comparison failed: {e}", exc_info=True)
        final_status = "FAILED"
        
    finally:
        logger.info(f"=== Model Comparison Complete (Status: {final_status}) ===")
        
        if final_status == "COMPLETED":
            output_dir = Path(config.get("comparison", {}).get("output_dir", "outputs/model_comparison"))
            logger.info(f"Results saved to: {output_dir}")
            logger.info("Generated outputs:")
            logger.info("  - comparison_results.json: Detailed comparison data")
            logger.info("  - comparison_summary.txt: Human-readable summary")
            logger.info("  - visualizations/: Performance comparison charts")
            logger.info("  - reports/: Detailed analysis reports")

if __name__ == "__main__":
    main()