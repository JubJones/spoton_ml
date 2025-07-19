"""
Enhanced Detection Analysis Pipeline - Phase 1 Integration.

This pipeline integrates all Phase 1 components:
1. Enhanced failure detection and analysis
2. Cross-model comparison
3. Advanced metrics collection
4. Automated report generation

Main entry point for comprehensive detection analysis.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import torch
import cv2
import numpy as np
from tqdm import tqdm

from src.utils.mlflow_utils import download_best_model_checkpoint, setup_mlflow_experiment
from src.components.training.runner import get_fasterrcnn_model, get_transform
from src.components.data.training_dataset import MTMMCDetectionDataset
from src.analysis.enhanced_detection_analysis import EnhancedDetectionAnalyzer, SceneAnalyzer
from src.analysis.cross_model_comparison import CrossModelComparisonSystem
from src.analysis.advanced_metrics import AdvancedMetricsCollector
from src.analysis.automated_reporting import AutomatedReportingSystem

logger = logging.getLogger(__name__)


class EnhancedDetectionAnalysisPipeline:
    """Main pipeline for comprehensive detection analysis."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device, project_root: Path):
        self.config = config
        self.device = device
        self.project_root = project_root
        
        # Setup output directory
        self.output_dir = Path(config["analysis"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analysis components
        self.enhanced_analyzer = EnhancedDetectionAnalyzer(
            config, self.output_dir / "enhanced_analysis"
        )
        self.metrics_collector = AdvancedMetricsCollector(
            self.output_dir / "metrics"
        )
        self.reporting_system = AutomatedReportingSystem(
            self.output_dir / "reports"
        )
        
        # Cross-model comparison (if multiple models configured)
        self.cross_model_system = None
        if self._has_multiple_models():
            model_configs = self._prepare_model_configs()
            self.cross_model_system = CrossModelComparisonSystem(
                model_configs, device, self.output_dir / "cross_model"
            )
        
        # Analysis state
        self.analysis_start_time = None
        self.total_frames_processed = 0
        
    def _has_multiple_models(self) -> bool:
        """Check if multiple models are configured for comparison."""
        # This would need to be configured in the config file
        return self.config.get("cross_model_analysis", {}).get("enabled", False)
    
    def _prepare_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Prepare model configurations for cross-model comparison."""
        # Example configuration - would be loaded from config file
        default_configs = {
            "fasterrcnn": {
                "type": "fasterrcnn",
                "weights_path": "DEFAULT",
                "confidence_threshold": 0.5,
                "person_class_id": 1  # COCO person class
            }
            # Additional models would be added here based on config
        }
        
        return self.config.get("cross_model_analysis", {}).get("models", default_configs)
    
    def run_comprehensive_analysis(self) -> Dict[str, Path]:
        """Run complete enhanced detection analysis pipeline."""
        logger.info("Starting Enhanced Detection Analysis Pipeline - Phase 1")
        self.analysis_start_time = datetime.now()
        
        try:
            # 1. Load model and dataset
            model, dataset = self._load_model_and_dataset()
            if not model or not dataset:
                raise RuntimeError("Failed to load model or dataset")
            
            # 2. Run frame-by-frame analysis
            self._run_frame_analysis(model, dataset)
            
            # 3. Run cross-model comparison (if enabled)
            cross_model_results = None
            if self.cross_model_system:
                logger.info("Running cross-model comparison analysis...")
                cross_model_results = self.cross_model_system.analyze_dataset(
                    dataset, max_frames=self.config.get("analysis", {}).get("max_frames_comparison", 100)
                )
            
            # 4. Generate comprehensive reports
            analysis_results = self._compile_analysis_results(cross_model_results)
            
            # 5. Generate all reports and visualizations
            generated_files = self._generate_all_reports(analysis_results)
            
            analysis_duration = (datetime.now() - self.analysis_start_time).total_seconds()
            logger.info(f"Enhanced Detection Analysis Pipeline completed in {analysis_duration:.2f} seconds")
            logger.info(f"Generated {len(generated_files)} analysis files")
            
            return generated_files
            
        except Exception as e:
            logger.error(f"Error in enhanced detection analysis pipeline: {e}", exc_info=True)
            raise
    
    def _load_model_and_dataset(self) -> Tuple[Optional[torch.nn.Module], Optional[MTMMCDetectionDataset]]:
        """Load model and dataset for analysis."""
        model_path = None
        
        # Determine model path: local file OR MLflow download
        local_path_str = self.config.get("local_model_path")
        if local_path_str:
            logger.info(f"Using local model path: {local_path_str}")
            candidate_path = self.project_root / local_path_str
            if candidate_path.is_file():
                model_path = candidate_path
            else:
                logger.error(f"Local model file not found: {candidate_path}")
                return None, None
        else:
            logger.info("Downloading model from MLflow...")
            setup_mlflow_experiment(self.config, "Analysis")
            run_id = self.config.get("mlflow_run_id")
            if not run_id:
                logger.error("No model path or MLflow run ID provided")
                return None, None
            
            with tempfile.TemporaryDirectory() as tmpdir:
                downloaded_path = download_best_model_checkpoint(run_id, Path(tmpdir))
                if not downloaded_path:
                    logger.error(f"Could not download model for run_id {run_id}")
                    return None, None
                model_path = downloaded_path
        
        if not model_path:
            logger.error("Could not determine valid model path")
            return None, None
        
        # Load model
        logger.info("Loading model architecture...")
        model = get_fasterrcnn_model(self.config)
        logger.info(f"Loading model weights from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict_key = 'model_state_dict'
        if state_dict_key not in checkpoint:
            logger.warning(f"'{state_dict_key}' not in checkpoint. Using checkpoint as state_dict.")
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint[state_dict_key])
        
        model.to(self.device)
        model.eval()
        logger.info("Model loaded successfully")
        
        # Load dataset
        logger.info("Loading validation dataset...")
        val_transforms = get_transform(train=False, config=self.config)
        dataset = MTMMCDetectionDataset(
            config=self.config,
            mode='val',
            transforms=val_transforms
        )
        
        if len(dataset) == 0:
            logger.error("Validation dataset is empty")
            return None, None
        
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        return model, dataset
    
    def _run_frame_analysis(self, model: torch.nn.Module, dataset: MTMMCDetectionDataset):
        """Run enhanced frame-by-frame analysis."""
        logger.info("Starting frame-by-frame enhanced analysis...")
        
        # Get analysis parameters
        analysis_config = self.config.get("analysis", {})
        max_frames = analysis_config.get("max_frames", -1)
        sample_rate = analysis_config.get("sample_rate", 1)  # Process every Nth frame
        
        # Discover all scene/camera pairs
        all_scene_camera_pairs = set()
        for i in range(len(dataset)):
            info = dataset.get_sample_info(i)
            if info:
                all_scene_camera_pairs.add((info['scene_id'], info['camera_id']))
        
        sorted_pairs = sorted(list(all_scene_camera_pairs))
        logger.info(f"Found {len(sorted_pairs)} unique scene/camera pairs")
        
        # Process each camera
        total_processed = 0
        for scene_id, camera_id in sorted_pairs:
            camera_processed = self._analyze_camera_enhanced(
                scene_id, camera_id, model, dataset, sample_rate, max_frames
            )
            total_processed += camera_processed
            
            if max_frames > 0 and total_processed >= max_frames:
                logger.info(f"Reached maximum frames limit ({max_frames})")
                break
        
        self.total_frames_processed = total_processed
        logger.info(f"Completed frame analysis. Processed {total_processed} frames total.")
    
    def _analyze_camera_enhanced(self, scene_id: str, camera_id: str, 
                               model: torch.nn.Module, dataset: MTMMCDetectionDataset,
                               sample_rate: int = 1, max_frames: int = -1) -> int:
        """Run enhanced analysis for a single camera."""
        logger.info(f"Analyzing camera {scene_id}/{camera_id} with enhanced pipeline")
        
        # Get indices for this camera
        camera_indices = []
        for i in range(len(dataset)):
            info = dataset.get_sample_info(i)
            if info and info['scene_id'] == scene_id and info['camera_id'] == camera_id:
                camera_indices.append(i)
        
        if not camera_indices:
            logger.warning(f"No data found for {scene_id}/{camera_id}")
            return 0
        
        # Apply sampling
        sampled_indices = camera_indices[::sample_rate]
        if max_frames > 0:
            sampled_indices = sampled_indices[:max_frames]
        
        logger.info(f"Processing {len(sampled_indices)} frames for {camera_id}")
        
        processed_count = 0
        for idx in tqdm(sampled_indices, desc=f"Processing {camera_id}"):
            try:
                # Get sample data
                image_tensor, target = dataset[idx]
                sample_info = dataset.get_sample_info(idx)
                
                if not sample_info:
                    continue
                
                # Convert tensor to numpy for analysis
                if isinstance(image_tensor, torch.Tensor):
                    image_np = image_tensor.permute(1, 2, 0).numpy()
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    continue
                
                # Run model inference with timing
                start_time = datetime.now()
                with torch.no_grad():
                    predictions = model([image_tensor.to(self.device)])[0]
                inference_time = (datetime.now() - start_time).total_seconds()
                
                # Convert predictions to CPU
                pred_cpu = {k: v.cpu() for k, v in predictions.items()}
                target_cpu = {k: v.cpu() for k, v in target.items()}
                
                # Run enhanced analysis
                frame_failures = self.enhanced_analyzer.analyze_frame(
                    frame_idx=idx,
                    image=image_bgr,
                    predictions=pred_cpu,
                    targets=target_cpu,
                    scene_id=scene_id,
                    camera_id=camera_id,
                    timestamp=None  # Could extract from sample_info if available
                )
                
                # Add to advanced metrics collector
                timing_data = {
                    'inference_time': inference_time,
                    'preprocessing_time': 0.0,  # Not measured separately
                    'postprocessing_time': 0.0  # Not measured separately
                }
                
                self.metrics_collector.add_detection_results(
                    frame=image_bgr,
                    frame_idx=idx,
                    scene_id=scene_id,
                    camera_id=camera_id,
                    ground_truth=target_cpu,
                    predictions=pred_cpu,
                    model_name=self.config.get("model", {}).get("type", "unknown"),
                    timing_data=timing_data
                )
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing frame {idx}: {e}")
                continue
        
        logger.info(f"Completed analysis for {camera_id}. Processed {processed_count} frames.")
        return processed_count
    
    def _compile_analysis_results(self, cross_model_results=None) -> Dict[str, Any]:
        """Compile all analysis results into a unified structure."""
        logger.info("Compiling comprehensive analysis results...")
        
        # Get enhanced analysis results
        enhanced_report = self.enhanced_analyzer.generate_comprehensive_report()
        
        # Get advanced metrics
        model_name = self.config.get("model", {}).get("type", "primary_model")
        advanced_metrics = self.metrics_collector.generate_comprehensive_metrics(model_name)
        
        # Compile unified results
        results = {
            'analysis_metadata': {
                'pipeline_version': 'Phase 1 - Enhanced Detection Analysis',
                'timestamp': datetime.now().isoformat(),
                'analysis_duration': (datetime.now() - self.analysis_start_time).total_seconds() if self.analysis_start_time else 0,
                'total_frames': self.total_frames_processed,
                'primary_model': model_name
            },
            'enhanced_analysis': {
                'total_failures': len(self.enhanced_analyzer.failures),
                'generated_files': {str(k): str(v) for k, v in enhanced_report.items()}
            },
            'advanced_metrics': advanced_metrics,
            'models': {
                model_name: self._extract_model_summary_metrics(advanced_metrics)
            },
            'dataset_info': {
                'total_samples': self.total_frames_processed,
                'scenes_analyzed': len(set(f.scene_id for f in self.enhanced_analyzer.failures)),
                'cameras_analyzed': len(set(f.camera_id for f in self.enhanced_analyzer.failures))
            }
        }
        
        # Add cross-model results if available
        if cross_model_results and self.cross_model_system:
            cross_model_report = self.cross_model_system.generate_comprehensive_report(cross_model_results)
            results['cross_model_analysis'] = {
                'comparison_matrix': cross_model_results,
                'generated_files': {str(k): str(v) for k, v in cross_model_report.items()}
            }
            
            # Update models dict with cross-model results
            for model_name, metrics in cross_model_results.individual_metrics.items():
                results['models'][model_name] = {
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'average_processing_time': metrics.average_processing_time,
                    'success_rate': metrics.success_rate,
                    'total_frames': metrics.total_frames,
                    'source': 'cross_model_comparison'
                }
        
        return results
    
    def _extract_model_summary_metrics(self, advanced_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary metrics for a model."""
        # Extract key metrics from advanced metrics structure
        map_metrics = advanced_metrics.get('advanced_map', {})
        confidence_analysis = advanced_metrics.get('confidence_analysis', {})
        performance_profile = advanced_metrics.get('performance_profile', {})
        
        return {
            'precision': 0.0,  # Would calculate from advanced metrics
            'recall': 0.0,     # Would calculate from advanced metrics  
            'f1_score': map_metrics.get('overall_map_50', 0.0),  # Using mAP as proxy
            'average_processing_time': performance_profile.get('mean_inference_time', 0.0),
            'success_rate': 1.0,  # Assume 100% unless we track failures
            'total_frames': self.total_frames_processed,
            'confidence_threshold': confidence_analysis.get('optimal_threshold', 0.5),
            'source': 'advanced_metrics'
        }
    
    def _generate_all_reports(self, analysis_results: Dict[str, Any]) -> Dict[str, Path]:
        """Generate all reports and visualizations."""
        logger.info("Generating comprehensive reports and visualizations...")
        
        all_generated_files = {}
        
        # 1. Generate visualization plots from advanced metrics
        model_name = self.config.get("model", {}).get("type", "primary_model")
        viz_plots = self.metrics_collector.generate_visualization_plots(model_name)
        
        # 2. Save advanced metrics report
        metrics_report = self.metrics_collector.save_comprehensive_report(model_name)
        all_generated_files['advanced_metrics_report'] = metrics_report
        
        # 3. Generate automated reports with all visualizations
        all_visualization_paths = []
        
        # Add enhanced analysis visualizations
        enhanced_files = analysis_results.get('enhanced_analysis', {}).get('generated_files', {})
        for file_path_str in enhanced_files.values():
            file_path = Path(file_path_str)
            if file_path.exists() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                all_visualization_paths.append(file_path)
        
        # Add advanced metrics visualizations
        all_visualization_paths.extend(viz_plots)
        
        # Add cross-model visualizations if available
        if 'cross_model_analysis' in analysis_results:
            cross_model_files = analysis_results['cross_model_analysis'].get('generated_files', {})
            for file_path_str in cross_model_files.values():
                file_path = Path(file_path_str)
                if file_path.exists() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    all_visualization_paths.append(file_path)
        
        # Generate comprehensive automated report
        automated_reports = self.reporting_system.generate_comprehensive_report(
            analysis_results, all_visualization_paths
        )
        all_generated_files.update(automated_reports)
        
        # 4. Create final summary file
        summary_file = self._create_final_summary(analysis_results, all_generated_files)
        all_generated_files['final_summary'] = summary_file
        
        return all_generated_files
    
    def _create_final_summary(self, analysis_results: Dict[str, Any], 
                            generated_files: Dict[str, Path]) -> Path:
        """Create final summary of all analysis."""
        summary_data = {
            'pipeline_summary': {
                'version': 'Enhanced Detection Analysis Pipeline - Phase 1',
                'completion_timestamp': datetime.now().isoformat(),
                'total_analysis_duration': analysis_results['analysis_metadata']['analysis_duration'],
                'total_frames_processed': analysis_results['analysis_metadata']['total_frames'],
                'components_executed': []
            },
            'key_findings': {
                'total_detection_failures': analysis_results.get('enhanced_analysis', {}).get('total_failures', 0),
                'models_analyzed': list(analysis_results.get('models', {}).keys()),
                'best_performing_model': None,
                'primary_recommendations': []
            },
            'generated_artifacts': {
                'total_files_generated': len(generated_files),
                'file_manifest': {str(k): str(v) for k, v in generated_files.items()}
            },
            'next_steps': [
                "Review generated HTML report for detailed analysis",
                "Examine failure visualizations for pattern identification", 
                "Consider implementing recommended model improvements",
                "Schedule regular monitoring using automated reporting system"
            ]
        }
        
        # Determine best performing model
        models = analysis_results.get('models', {})
        if models:
            best_model = max(models.items(), key=lambda x: x[1].get('f1_score', 0))
            summary_data['key_findings']['best_performing_model'] = best_model[0]
        
        # Add component execution info
        components = ['Enhanced Failure Analysis', 'Advanced Metrics Collection', 'Automated Reporting']
        if 'cross_model_analysis' in analysis_results:
            components.append('Cross-Model Comparison')
        summary_data['pipeline_summary']['components_executed'] = components
        
        # Save summary
        summary_path = self.output_dir / 'PIPELINE_SUMMARY.json'
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        return summary_path


def run_enhanced_analysis(config: Dict[str, Any], device: torch.device, project_root: Path):
    """Main entry point for enhanced detection analysis pipeline."""
    logger.info("=" * 80)
    logger.info("ENHANCED DETECTION ANALYSIS PIPELINE - PHASE 1")
    logger.info("=" * 80)
    
    try:
        # Initialize and run pipeline
        pipeline = EnhancedDetectionAnalysisPipeline(config, device, project_root)
        generated_files = pipeline.run_comprehensive_analysis()
        
        # Log completion summary
        logger.info("=" * 80)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total files generated: {len(generated_files)}")
        
        # Print key output files
        key_files = ['html_report', 'final_summary', 'advanced_metrics_report']
        for key in key_files:
            if key in generated_files:
                logger.info(f"📄 {key.replace('_', ' ').title()}: {generated_files[key]}")
        
        logger.info("=" * 80)
        logger.info("🎉 Enhanced Detection Analysis Pipeline - Phase 1 Complete!")
        logger.info("=" * 80)
        
        return generated_files
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("PIPELINE EXECUTION FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)
        raise