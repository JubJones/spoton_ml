import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

# Project Setup
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local Imports
try:
    from src.utils.config_loader import load_config
    from src.utils.reproducibility import set_seed
    from src.utils.logging_utils import setup_logging
    from src.utils.mlflow_utils import setup_mlflow_experiment
    from src.utils.device_utils import get_device
    from src.pipelines.tracking_failure_analysis_pipeline import TrackingFailureAnalysisPipeline
    from src.components.evaluation.enhanced_tracking_metrics import (
        calculate_enhanced_tracking_metrics, 
        analyze_tracker_specific_performance
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Setup logging
log_file = setup_logging(log_prefix="tracking_failure_analysis", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_single_tracker_analysis(
    config: Dict[str, Any],
    tracker_type: str,
    reid_config: Dict[str, Any],
    device_preference: str = "auto"
) -> Dict[str, Any]:
    """
    Run tracking failure analysis for a single tracker configuration.
    
    Args:
        config: Base configuration dictionary
        tracker_type: Type of tracker to analyze
        reid_config: Re-ID model configuration
        device_preference: Device preference string
    
    Returns:
        Dictionary containing analysis results
    """
    run_name = f"TrackingAnalysis_{tracker_type}_{reid_config.get('model_type', 'default')}"
    logger.info(f"[{run_name}] Starting tracking failure analysis...")
    
    try:
        # Setup device
        device = get_device(device_preference)
        logger.info(f"[{run_name}] Using device: {device}")
        
        # Create analysis configuration
        analysis_config = config.copy()
        analysis_config["tracker"] = {"type": tracker_type}
        analysis_config["reid_model"] = reid_config
        
        # Initialize and run analysis pipeline
        pipeline = TrackingFailureAnalysisPipeline(
            config=analysis_config,
            device=device,
            project_root=PROJECT_ROOT
        )
        
        # Run complete analysis
        success, results = pipeline.run_complete_analysis()
        
        if success:
            logger.info(f"[{run_name}] Analysis completed successfully")
            
            # Log key metrics to MLflow
            mlflow.log_param("tracker_type", tracker_type)
            mlflow.log_param("reid_model_type", reid_config.get("model_type", "default"))
            mlflow.log_param("reid_weights", reid_config.get("weights_path", "default"))
            
            # Log quality metrics
            if "quality_metrics" in results and results["quality_metrics"]:
                for metric_name, metric_value in results["quality_metrics"].items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(f"quality_{metric_name}", metric_value)
            
            # Log failure counts
            if "failure_counts" in results:
                for failure_type, count in results["failure_counts"].items():
                    mlflow.log_metric(f"failure_{failure_type}", count)
            
            # Log analysis report as artifact
            if "report_path" in results:
                try:
                    report_path = Path(results["report_path"])
                    if report_path.exists():
                        mlflow.log_artifacts(str(report_path), artifact_path="tracking_analysis")
                except Exception as e:
                    logger.warning(f"Failed to log analysis artifacts: {e}")
            
            return {
                "success": True,
                "tracker_type": tracker_type,
                "reid_config": reid_config,
                "results": results
            }
        else:
            logger.error(f"[{run_name}] Analysis failed")
            return {
                "success": False,
                "tracker_type": tracker_type,
                "reid_config": reid_config,
                "error": "Analysis pipeline failed"
            }
            
    except Exception as e:
        logger.error(f"[{run_name}] Error during analysis: {e}", exc_info=True)
        return {
            "success": False,
            "tracker_type": tracker_type,
            "reid_config": reid_config,
            "error": str(e)
        }


def run_comparative_tracking_analysis(config_path: str) -> bool:
    """
    Run comparative tracking failure analysis across multiple trackers and Re-ID models.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Success status
    """
    logger.info("=== Starting Comparative Tracking Failure Analysis ===")
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        logger.error(f"Failed to load configuration from {config_path}")
        return False
    
    # Extract tracker and Re-ID configurations
    trackers_to_analyze = config.get("trackers_to_analyze", [])
    reid_models = config.get("reid_models_to_associate", [])
    
    if not trackers_to_analyze:
        logger.error("No trackers specified for analysis")
        return False
    
    if not reid_models:
        logger.warning("No Re-ID models specified, using default configuration")
        reid_models = [{"model_type": "default", "weights_path": None}]
    
    # Setup MLflow experiment
    experiment_id = setup_mlflow_experiment(config, default_experiment_name="Tracking Failure Analysis")
    if not experiment_id:
        logger.error("Failed to setup MLflow experiment")
        return False
    
    # Set global seed
    seed = config.get("environment", {}).get("seed", int(time.time()))
    set_seed(seed)
    
    # Device preference
    device_preference = config.get("environment", {}).get("device", "auto")
    
    parent_run_name = config.get("parent_run_name", f"tracking_analysis_comparison_{int(time.time())}")
    analysis_results = {}
    overall_status = "SUCCESS"
    
    try:
        with mlflow.start_run(run_name=parent_run_name, experiment_id=experiment_id) as parent_run:
            parent_run_id = parent_run.info.run_id
            logger.info(f"Parent run started: {parent_run_id}")
            
            # Log parent parameters
            mlflow.log_param("analysis_type", "tracking_failure_analysis")
            mlflow.log_param("num_trackers", len(trackers_to_analyze))
            mlflow.log_param("num_reid_models", len(reid_models))
            mlflow.log_param("seed", seed)
            mlflow.log_param("device_preference", device_preference)
            
            # Log configuration
            config_file_path = PROJECT_ROOT / config_path
            if config_file_path.exists():
                mlflow.log_artifact(str(config_file_path), artifact_path="config")
            
            total_combinations = len(trackers_to_analyze) * len(reid_models)
            completed_runs = 0
            
            # Run analysis for each tracker-ReID combination
            for tracker_type in trackers_to_analyze:
                for reid_config in reid_models:
                    combination_name = f"{tracker_type}_{reid_config.get('model_type', 'default')}"
                    logger.info(f"\n--- Analyzing combination: {combination_name} ---")
                    
                    try:
                        with mlflow.start_run(
                            run_name=f"analysis_{combination_name}",
                            experiment_id=experiment_id,
                            nested=True
                        ) as child_run:
                            child_run_id = child_run.info.run_id
                            mlflow.set_tag("parent_run_id", parent_run_id)
                            mlflow.set_tag("analysis_type", "single_tracker_analysis")
                            
                            # Run single tracker analysis
                            result = run_single_tracker_analysis(
                                config=config,
                                tracker_type=tracker_type,
                                reid_config=reid_config,
                                device_preference=device_preference
                            )
                            
                            analysis_results[combination_name] = result
                            completed_runs += 1
                            
                            if result["success"]:
                                mlflow.set_tag("analysis_status", "SUCCESS")
                                logger.info(f"Analysis completed for {combination_name}")
                            else:
                                mlflow.set_tag("analysis_status", "FAILED")
                                mlflow.log_param("error_message", result.get("error", "Unknown error"))
                                overall_status = "PARTIAL_FAILURE"
                                logger.error(f"Analysis failed for {combination_name}: {result.get('error')}")
                    
                    except Exception as e:
                        logger.error(f"Error during analysis for {combination_name}: {e}", exc_info=True)
                        analysis_results[combination_name] = {
                            "success": False,
                            "tracker_type": tracker_type,
                            "reid_config": reid_config,
                            "error": str(e)
                        }
                        overall_status = "PARTIAL_FAILURE"
            
            # Generate comparative analysis
            logger.info("Generating comparative analysis...")
            successful_results = {
                name: result for name, result in analysis_results.items() 
                if result["success"]
            }
            
            if successful_results:
                comparative_analysis = generate_comparative_analysis(successful_results)
                
                # Log comparative metrics
                if "best_performers" in comparative_analysis:
                    for metric, performers in comparative_analysis["best_performers"].items():
                        best_tracker = performers["best"]["tracker"]
                        best_value = performers["best"]["value"]
                        mlflow.log_metric(f"best_{metric}_tracker", hash(best_tracker) % 1000)  # Hash for numeric logging
                        mlflow.log_metric(f"best_{metric}_value", best_value)
                
                # Save comparative analysis
                analysis_output_path = PROJECT_ROOT / "analysis_outputs" / "comparative_tracking_analysis.json"
                analysis_output_path.parent.mkdir(parents=True, exist_ok=True)
                
                import json
                with open(analysis_output_path, 'w') as f:
                    json.dump(comparative_analysis, f, indent=2, default=str)
                
                mlflow.log_artifact(str(analysis_output_path), artifact_path="comparative_analysis")
                logger.info(f"Comparative analysis saved to {analysis_output_path}")
            
            # Log summary
            mlflow.log_metric("total_combinations_analyzed", total_combinations)
            mlflow.log_metric("successful_analyses", len(successful_results))
            mlflow.log_metric("failed_analyses", total_combinations - len(successful_results))
            mlflow.set_tag("overall_status", overall_status)
            
            # Log main log file
            if log_file.exists():
                mlflow.log_artifact(str(log_file), artifact_path="logs")
            
            logger.info(f"=== Comparative Analysis Completed ===")
            logger.info(f"Total combinations: {total_combinations}")
            logger.info(f"Successful analyses: {len(successful_results)}")
            logger.info(f"Overall status: {overall_status}")
            
            return overall_status in ["SUCCESS", "PARTIAL_FAILURE"]
    
    except Exception as e:
        logger.error(f"Error during comparative analysis: {e}", exc_info=True)
        return False


def generate_comparative_analysis(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate comparative analysis from multiple tracker results.
    
    Args:
        results: Dictionary of successful analysis results
    
    Returns:
        Comparative analysis dictionary
    """
    logger.info("Generating comparative analysis across trackers...")
    
    comparative_analysis = {
        "summary": {
            "total_trackers_analyzed": len(results),
            "analysis_timestamp": time.time()
        },
        "performance_comparison": {},
        "best_performers": {},
        "failure_patterns": {},
        "recommendations": []
    }
    
    # Extract key metrics for comparison
    metrics_to_compare = [
        "mota", "motp", "idf1", "id_switches", "fragmentations",
        "false_positives", "false_negatives", "trajectory_consistency_score"
    ]
    
    for metric in metrics_to_compare:
        metric_values = {}
        
        for combination_name, result in results.items():
            if "results" in result and "quality_metrics" in result["results"]:
                quality_metrics = result["results"]["quality_metrics"]
                if metric in quality_metrics:
                    metric_values[combination_name] = quality_metrics[metric]
            elif "results" in result and "failure_counts" in result["results"]:
                failure_counts = result["results"]["failure_counts"]
                if metric in failure_counts:
                    metric_values[combination_name] = failure_counts[metric]
        
        if metric_values:
            comparative_analysis["performance_comparison"][metric] = metric_values
            
            # Find best and worst performers
            if metric in ["id_switches", "fragmentations", "false_positives", "false_negatives"]:
                # Lower is better
                best = min(metric_values.items(), key=lambda x: x[1])
                worst = max(metric_values.items(), key=lambda x: x[1])
            else:
                # Higher is better
                best = max(metric_values.items(), key=lambda x: x[1])
                worst = min(metric_values.items(), key=lambda x: x[1])
            
            comparative_analysis["best_performers"][metric] = {
                "best": {"tracker": best[0], "value": best[1]},
                "worst": {"tracker": worst[0], "value": worst[1]}
            }
    
    # Generate failure pattern analysis
    for combination_name, result in results.items():
        if "results" in result and "failure_counts" in result["results"]:
            failure_counts = result["results"]["failure_counts"]
            total_failures = sum(v for k, v in failure_counts.items() if k != "analysis_completed")
            comparative_analysis["failure_patterns"][combination_name] = {
                "total_failures": total_failures,
                "failure_breakdown": failure_counts
            }
    
    # Generate recommendations
    recommendations = []
    
    if "idf1" in comparative_analysis["best_performers"]:
        best_idf1 = comparative_analysis["best_performers"]["idf1"]["best"]["tracker"]
        recommendations.append(f"Best for identity preservation: {best_idf1}")
    
    if "mota" in comparative_analysis["best_performers"]:
        best_mota = comparative_analysis["best_performers"]["mota"]["best"]["tracker"]
        recommendations.append(f"Best for overall tracking accuracy: {best_mota}")
    
    if "id_switches" in comparative_analysis["best_performers"]:
        best_id_switches = comparative_analysis["best_performers"]["id_switches"]["best"]["tracker"]
        recommendations.append(f"Most stable identity tracking: {best_id_switches}")
    
    comparative_analysis["recommendations"] = recommendations
    
    return comparative_analysis


def main():
    """Main entry point for tracking failure analysis"""
    logger.info("Starting Tracking Failure Analysis...")
    
    config_path = "configs/tracking_failure_analysis_config.yaml"
    
    try:
        success = run_comparative_tracking_analysis(config_path)
        
        if success:
            logger.info("Tracking failure analysis completed successfully")
            sys.exit(0)
        else:
            logger.error("Tracking failure analysis failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()