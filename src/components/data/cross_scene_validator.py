"""
Cross-Scene Validator for RF-DETR Training
Comprehensive validation across different surveillance scenes
"""
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path

from .scene_analyzer import SceneCharacteristics
from .scene_balancer import SceneBalancer

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for cross-scene validation."""
    
    # Validation frequency
    validation_frequency: int = 500  # Steps between validations
    scene_validation_frequency: int = 1000  # Steps between per-scene validations
    
    # Metrics to track
    primary_metrics: List[str] = field(default_factory=lambda: ['mAP', 'precision', 'recall'])
    secondary_metrics: List[str] = field(default_factory=lambda: ['F1', 'inference_time'])
    
    # Scene-specific analysis
    scene_analysis_enabled: bool = True
    difficulty_correlation: bool = True
    size_specific_metrics: bool = True
    
    # Performance thresholds
    min_acceptable_map: float = 0.3
    max_inference_time_ms: float = 100.0
    scene_variance_threshold: float = 0.15  # Maximum acceptable variance across scenes
    
    # Early stopping based on cross-scene performance
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 10  # Validations without improvement
    early_stopping_metric: str = 'mAP'
    
    # Reporting
    detailed_reports: bool = True
    save_per_scene_results: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.validation_frequency <= 0:
            raise ValueError(f"validation_frequency must be positive, got {self.validation_frequency}")
        if self.early_stopping_metric not in self.primary_metrics + self.secondary_metrics:
            raise ValueError(f"early_stopping_metric must be in tracked metrics")


@dataclass
class SceneValidationResult:
    """Results from validating on a specific scene."""
    
    scene_id: str
    difficulty_score: float
    crowd_level: str
    
    # Core metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Detailed analysis
    size_specific_results: Dict[str, Dict[str, float]] = field(default_factory=dict)  # small, medium, large
    confidence_analysis: Dict[str, Any] = field(default_factory=dict)
    failure_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Performance characteristics
    inference_times: List[float] = field(default_factory=list)
    memory_usage: Optional[float] = None
    
    # Comparison with scene characteristics
    expected_difficulty: Optional[float] = None
    performance_difficulty_ratio: Optional[float] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        return {
            'scene_id': self.scene_id,
            'difficulty': self.difficulty_score,
            'crowd_level': self.crowd_level,
            'metrics': self.metrics.copy(),
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0.0,
            'performance_difficulty_ratio': self.performance_difficulty_ratio
        }


@dataclass
class CrossSceneValidationResult:
    """Comprehensive cross-scene validation results."""
    
    step: int
    epoch: int
    timestamp: float
    
    # Overall metrics
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Per-scene results
    scene_results: Dict[str, SceneValidationResult] = field(default_factory=dict)
    
    # Cross-scene analysis
    scene_variance: Dict[str, float] = field(default_factory=dict)  # Variance of metrics across scenes
    difficulty_correlation: Dict[str, float] = field(default_factory=dict)  # Correlation with difficulty
    
    # Performance distribution
    performance_distribution: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary."""
        return {
            'step': self.step,
            'epoch': self.epoch,
            'overall_metrics': self.overall_metrics.copy(),
            'scene_count': len(self.scene_results),
            'scene_variance': self.scene_variance.copy(),
            'difficulty_correlation': self.difficulty_correlation.copy(),
            'recommendations': self.recommendations.copy()
        }


class CrossSceneValidator:
    """
    Advanced cross-scene validator for RF-DETR training.
    Provides comprehensive validation across surveillance scenes with detailed analysis.
    """
    
    def __init__(
        self,
        scene_balancer: SceneBalancer,
        config: ValidationConfig,
        save_dir: Optional[Path] = None
    ):
        """
        Initialize cross-scene validator.
        
        Args:
            scene_balancer: Configured scene balancer with scene splits
            config: Validation configuration
            save_dir: Directory for saving validation results
        """
        self.scene_balancer = scene_balancer
        self.config = config
        self.save_dir = save_dir or Path("cross_scene_validation")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation state
        self.validation_history = []
        self.scene_performance_history = defaultdict(list)
        self.best_performance = {}
        self.patience_counter = 0
        
        # Scene information
        self.training_scenes = scene_balancer.get_training_scenes()
        self.validation_scenes = scene_balancer.get_validation_scenes()
        self.scene_characteristics = scene_balancer.scene_analyzer.scene_characteristics
        
        logger.info(f"Initialized Cross-Scene Validator:")
        logger.info(f"  Validation scenes: {len(self.validation_scenes)}")
        logger.info(f"  Training scenes: {len(self.training_scenes)}")
        logger.info(f"  Primary metrics: {config.primary_metrics}")
        logger.info(f"  Save directory: {self.save_dir}")
    
    def validate_cross_scene(
        self,
        model: torch.nn.Module,
        validation_fn: Callable,
        step: int,
        epoch: int,
        validation_data: Dict[str, Any],  # Scene-specific validation data
        **kwargs
    ) -> CrossSceneValidationResult:
        """
        Perform comprehensive cross-scene validation.
        
        Args:
            model: Model to validate
            validation_fn: Function that validates model on specific scene data
            step: Current training step
            epoch: Current epoch
            validation_data: Dictionary mapping scene_id to validation dataset
            **kwargs: Additional arguments for validation function
            
        Returns:
            Cross-scene validation results
        """
        logger.info(f"Starting cross-scene validation at step {step}, epoch {epoch}")
        
        result = CrossSceneValidationResult(
            step=step,
            epoch=epoch,
            timestamp=torch.cuda.Event(enable_timing=True).record() if torch.cuda.is_available() else 0.0
        )
        
        # Validate on each scene
        scene_metrics = {}
        all_inference_times = []
        
        model.eval()
        with torch.no_grad():
            for scene_id in self.validation_scenes:
                if scene_id not in validation_data:
                    logger.warning(f"No validation data for scene {scene_id}")
                    continue
                
                scene_result = self._validate_single_scene(
                    model, validation_fn, scene_id, validation_data[scene_id], **kwargs
                )
                
                result.scene_results[scene_id] = scene_result
                scene_metrics[scene_id] = scene_result.metrics
                all_inference_times.extend(scene_result.inference_times)
                
                # Store in scene performance history
                self.scene_performance_history[scene_id].append({
                    'step': step,
                    'metrics': scene_result.metrics.copy()
                })
        
        # Calculate overall metrics
        result.overall_metrics = self._calculate_overall_metrics(scene_metrics)
        result.overall_metrics['avg_inference_time'] = np.mean(all_inference_times) if all_inference_times else 0.0
        
        # Analyze cross-scene performance
        result = self._analyze_cross_scene_performance(result)
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        
        # Store validation result
        self.validation_history.append(result)
        
        # Check early stopping
        if self.config.early_stopping_enabled:
            self._check_early_stopping(result)
        
        # Save results if configured
        if self.config.save_per_scene_results:
            self._save_validation_results(result)
        
        logger.info(f"Cross-scene validation completed:")
        logger.info(f"  Overall mAP: {result.overall_metrics.get('mAP', 0):.4f}")
        logger.info(f"  Scene variance: {result.scene_variance.get('mAP', 0):.4f}")
        logger.info(f"  Difficulty correlation: {result.difficulty_correlation.get('mAP', 0):.3f}")
        
        return result
    
    def _validate_single_scene(
        self,
        model: torch.nn.Module,
        validation_fn: Callable,
        scene_id: str,
        scene_data: Any,
        **kwargs
    ) -> SceneValidationResult:
        """Validate model on a single scene."""
        
        scene_characteristics = self.scene_characteristics.get(scene_id)
        if not scene_characteristics:
            logger.warning(f"No characteristics found for scene {scene_id}")
            scene_characteristics = SceneCharacteristics(scene_id=scene_id)
        
        result = SceneValidationResult(
            scene_id=scene_id,
            difficulty_score=scene_characteristics.difficulty_score,
            crowd_level=scene_characteristics.crowd_density_level
        )
        
        # Measure inference time
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Run validation function
        validation_results = validation_fn(model, scene_data, **kwargs)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)
            result.inference_times.append(inference_time)
        
        # Extract metrics
        if isinstance(validation_results, dict):
            for metric in self.config.primary_metrics + self.config.secondary_metrics:
                if metric in validation_results:
                    result.metrics[metric] = float(validation_results[metric])
        
        # Detailed analysis if enabled
        if self.config.scene_analysis_enabled:
            result = self._perform_detailed_scene_analysis(result, validation_results, scene_characteristics)
        
        # Calculate performance-difficulty ratio
        if result.metrics.get('mAP') and result.difficulty_score > 0:
            # Higher ratio means better performance relative to difficulty
            result.performance_difficulty_ratio = result.metrics['mAP'] / result.difficulty_score
        
        return result
    
    def _perform_detailed_scene_analysis(
        self,
        result: SceneValidationResult,
        validation_results: Dict[str, Any],
        scene_characteristics: SceneCharacteristics
    ) -> SceneValidationResult:
        """Perform detailed analysis of scene-specific performance."""
        
        # Size-specific analysis
        if self.config.size_specific_metrics and 'size_specific_results' in validation_results:
            result.size_specific_results = validation_results['size_specific_results']
        
        # Confidence analysis
        if 'predictions' in validation_results:
            result.confidence_analysis = self._analyze_prediction_confidence(
                validation_results['predictions']
            )
        
        # Failure analysis
        if 'failures' in validation_results:
            result.failure_analysis = self._analyze_failures(
                validation_results['failures'], scene_characteristics
            )
        
        return result
    
    def _analyze_prediction_confidence(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prediction confidence distribution."""
        
        confidences = []
        for pred in predictions:
            if 'confidence' in pred or 'score' in pred:
                conf = pred.get('confidence', pred.get('score', 0))
                confidences.append(conf)
        
        if not confidences:
            return {}
        
        return {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'confidence_distribution': np.histogram(confidences, bins=10)[0].tolist()
        }
    
    def _analyze_failures(
        self,
        failures: List[Dict[str, Any]],
        scene_characteristics: SceneCharacteristics
    ) -> Dict[str, Any]:
        """Analyze failure patterns in relation to scene characteristics."""
        
        analysis = {
            'total_failures': len(failures),
            'failure_types': defaultdict(int),
            'failure_patterns': {}
        }
        
        for failure in failures:
            failure_type = failure.get('type', 'unknown')
            analysis['failure_types'][failure_type] += 1
        
        # Correlate failures with scene characteristics
        if scene_characteristics.crowd_density_level == 'high' and len(failures) > 0:
            analysis['failure_patterns']['high_crowd_correlation'] = True
        
        if scene_characteristics.occlusion_level > 0.5 and len(failures) > 0:
            analysis['failure_patterns']['occlusion_correlation'] = True
        
        return dict(analysis)
    
    def _calculate_overall_metrics(self, scene_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate overall metrics across all scenes."""
        
        if not scene_metrics:
            return {}
        
        overall = {}
        
        # For each metric, calculate mean across scenes
        all_metrics = set()
        for metrics in scene_metrics.values():
            all_metrics.update(metrics.keys())
        
        for metric in all_metrics:
            values = [
                metrics[metric] for metrics in scene_metrics.values()
                if metric in metrics
            ]
            
            if values:
                overall[metric] = np.mean(values)
                overall[f'{metric}_std'] = np.std(values)
                overall[f'{metric}_min'] = min(values)
                overall[f'{metric}_max'] = max(values)
        
        return overall
    
    def _analyze_cross_scene_performance(self, result: CrossSceneValidationResult) -> CrossSceneValidationResult:
        """Analyze performance patterns across scenes."""
        
        if not result.scene_results:
            return result
        
        # Calculate variance across scenes for each metric
        scene_metrics = {}
        for scene_id, scene_result in result.scene_results.items():
            scene_metrics[scene_id] = scene_result.metrics
        
        for metric in self.config.primary_metrics + self.config.secondary_metrics:
            values = [
                metrics.get(metric, 0) for metrics in scene_metrics.values()
                if metric in metrics
            ]
            
            if values and len(values) > 1:
                result.scene_variance[metric] = np.var(values)
        
        # Calculate correlation with scene difficulty
        if self.config.difficulty_correlation:
            for metric in self.config.primary_metrics:
                metric_values = []
                difficulty_values = []
                
                for scene_id, scene_result in result.scene_results.items():
                    if metric in scene_result.metrics:
                        metric_values.append(scene_result.metrics[metric])
                        difficulty_values.append(scene_result.difficulty_score)
                
                if len(metric_values) > 2:
                    # Calculate Pearson correlation
                    correlation = np.corrcoef(metric_values, difficulty_values)[0, 1]
                    if not np.isnan(correlation):
                        result.difficulty_correlation[metric] = correlation
        
        # Performance distribution analysis
        result.performance_distribution = self._analyze_performance_distribution(result)
        
        return result
    
    def _analyze_performance_distribution(self, result: CrossSceneValidationResult) -> Dict[str, Any]:
        """Analyze distribution of performance across scenes."""
        
        distribution = {}
        
        for metric in self.config.primary_metrics:
            values = [
                scene_result.metrics.get(metric, 0)
                for scene_result in result.scene_results.values()
                if metric in scene_result.metrics
            ]
            
            if values:
                distribution[metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'iqr': np.percentile(values, 75) - np.percentile(values, 25),
                    'outliers_low': len([v for v in values if v < np.percentile(values, 25) - 1.5 * (np.percentile(values, 75) - np.percentile(values, 25))]),
                    'outliers_high': len([v for v in values if v > np.percentile(values, 75) + 1.5 * (np.percentile(values, 75) - np.percentile(values, 25))])
                }
        
        return distribution
    
    def _generate_recommendations(self, result: CrossSceneValidationResult) -> Dict[str, Any]:
        """Generate training recommendations based on cross-scene performance."""
        
        recommendations = {
            'training_adjustments': [],
            'data_augmentation': [],
            'model_architecture': [],
            'priority_scenes': []
        }
        
        # Check if performance meets thresholds
        overall_map = result.overall_metrics.get('mAP', 0)
        
        if overall_map < self.config.min_acceptable_map:
            recommendations['training_adjustments'].append(
                f"Overall mAP ({overall_map:.3f}) below threshold ({self.config.min_acceptable_map}). "
                "Consider increasing training duration or adjusting learning rate."
            )
        
        # Check scene variance
        map_variance = result.scene_variance.get('mAP', 0)
        
        if map_variance > self.config.scene_variance_threshold:
            recommendations['training_adjustments'].append(
                f"High variance across scenes ({map_variance:.3f}). "
                "Consider scene-specific training or increased regularization."
            )
        
        # Check difficulty correlation
        map_difficulty_corr = result.difficulty_correlation.get('mAP', 0)
        
        if map_difficulty_corr < -0.3:  # Strong negative correlation expected
            recommendations['training_adjustments'].append(
                "Weak correlation between performance and scene difficulty. "
                "Model may need more challenging training data."
            )
        
        # Identify underperforming scenes
        if result.scene_results:
            scene_performances = [
                (scene_id, scene_result.metrics.get('mAP', 0))
                for scene_id, scene_result in result.scene_results.items()
            ]
            
            scene_performances.sort(key=lambda x: x[1])  # Sort by performance
            
            # Bottom 25% of scenes
            worst_count = max(1, len(scene_performances) // 4)
            worst_scenes = scene_performances[:worst_count]
            
            for scene_id, performance in worst_scenes:
                scene_char = self.scene_characteristics.get(scene_id)
                if scene_char:
                    recommendations['priority_scenes'].append({
                        'scene_id': scene_id,
                        'performance': performance,
                        'difficulty': scene_char.difficulty_score,
                        'crowd_level': scene_char.crowd_density_level,
                        'suggestion': 'Increase sampling weight or specific augmentation'
                    })
        
        # Inference time recommendations
        avg_inference_time = result.overall_metrics.get('avg_inference_time', 0)
        if avg_inference_time > self.config.max_inference_time_ms:
            recommendations['model_architecture'].append(
                f"Inference time ({avg_inference_time:.1f}ms) exceeds threshold. "
                "Consider model pruning or knowledge distillation."
            )
        
        return recommendations
    
    def _check_early_stopping(self, result: CrossSceneValidationResult):
        """Check if early stopping criteria are met."""
        
        metric = self.config.early_stopping_metric
        current_performance = result.overall_metrics.get(metric, 0)
        
        # Check if this is the best performance so far
        best_performance = self.best_performance.get(metric, -float('inf'))
        
        if current_performance > best_performance:
            self.best_performance[metric] = current_performance
            self.patience_counter = 0
            logger.info(f"New best {metric}: {current_performance:.4f}")
        else:
            self.patience_counter += 1
            logger.info(f"No improvement in {metric}: {self.patience_counter}/{self.config.early_stopping_patience}")
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early."""
        
        return (self.config.early_stopping_enabled and 
                self.patience_counter >= self.config.early_stopping_patience)
    
    def _save_validation_results(self, result: CrossSceneValidationResult):
        """Save validation results to disk."""
        
        # Save comprehensive results
        results_file = self.save_dir / f"validation_step_{result.step}.json"
        
        save_data = {
            'summary': result.get_summary(),
            'scene_results': {
                scene_id: scene_result.get_summary()
                for scene_id, scene_result in result.scene_results.items()
            },
            'detailed_analysis': {
                'performance_distribution': result.performance_distribution,
                'recommendations': result.recommendations
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        # Save scene performance trends
        trends_file = self.save_dir / "scene_performance_trends.json"
        with open(trends_file, 'w') as f:
            json.dump(dict(self.scene_performance_history), f, indent=2, default=str)
        
        logger.debug(f"Saved validation results to {results_file}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        
        if not self.validation_history:
            return {}
        
        latest_result = self.validation_history[-1]
        
        # Calculate trends
        trends = {}
        if len(self.validation_history) > 1:
            for metric in self.config.primary_metrics:
                recent_values = [
                    result.overall_metrics.get(metric, 0)
                    for result in self.validation_history[-5:]  # Last 5 validations
                    if metric in result.overall_metrics
                ]
                
                if len(recent_values) > 1:
                    trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                    trends[metric] = trend
        
        summary = {
            'total_validations': len(self.validation_history),
            'latest_step': latest_result.step,
            'latest_performance': latest_result.overall_metrics,
            'best_performance': self.best_performance,
            'trends': trends,
            'early_stopping': {
                'enabled': self.config.early_stopping_enabled,
                'patience_counter': self.patience_counter,
                'should_stop': self.should_stop_early()
            },
            'scene_coverage': {
                'validation_scenes': len(self.validation_scenes),
                'training_scenes': len(self.training_scenes)
            }
        }
        
        return summary


def create_cross_scene_validator(
    scene_balancer: SceneBalancer,
    validation_frequency: int = 500,
    primary_metrics: List[str] = None,
    save_dir: Optional[Path] = None,
    **kwargs
) -> CrossSceneValidator:
    """
    Convenience function to create cross-scene validator.
    
    Args:
        scene_balancer: Configured scene balancer
        validation_frequency: Steps between validations
        primary_metrics: Primary metrics to track
        save_dir: Directory for saving results
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured cross-scene validator
    """
    
    if primary_metrics is None:
        primary_metrics = ['mAP', 'precision', 'recall']
    
    config = ValidationConfig(
        validation_frequency=validation_frequency,
        primary_metrics=primary_metrics,
        **kwargs
    )
    
    return CrossSceneValidator(scene_balancer, config, save_dir)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Cross-Scene Validator with mock data")
    
    # Mock components (would use real ones in practice)
    from .scene_analyzer import create_scene_analyzer, SceneCharacteristics
    from .scene_balancer import create_scene_balancer
    
    # Create mock scene analyzer
    analyzer = create_scene_analyzer()
    analyzer.scene_characteristics = {
        'scene_01': SceneCharacteristics(scene_id='scene_01', difficulty_score=0.2),
        'scene_02': SceneCharacteristics(scene_id='scene_02', difficulty_score=0.5),
        'scene_03': SceneCharacteristics(scene_id='scene_03', difficulty_score=0.8)
    }
    
    # Create scene balancer
    balancer = create_scene_balancer(analyzer)
    
    # Create cross-scene validator
    validator = create_cross_scene_validator(
        balancer,
        validation_frequency=100,
        primary_metrics=['mAP', 'precision', 'recall']
    )
    
    # Mock validation function
    def mock_validation_fn(model, scene_data, **kwargs):
        # Simulate scene-dependent performance
        scene_id = scene_data.get('scene_id', 'unknown')
        difficulty = analyzer.scene_characteristics.get(scene_id, SceneCharacteristics(scene_id)).difficulty_score
        
        # Performance inversely related to difficulty
        base_map = 0.8 - difficulty * 0.3 + np.random.normal(0, 0.05)
        
        return {
            'mAP': max(0.1, base_map),
            'precision': max(0.1, base_map + 0.05),
            'recall': max(0.1, base_map - 0.03),
            'inference_time': 50 + difficulty * 30
        }
    
    # Mock model
    class MockModel(torch.nn.Module):
        def forward(self, x):
            return x
    
    model = MockModel()
    
    # Mock validation data
    validation_data = {
        scene_id: {'scene_id': scene_id}
        for scene_id in validator.validation_scenes
    }
    
    # Run cross-scene validation
    result = validator.validate_cross_scene(
        model=model,
        validation_fn=mock_validation_fn,
        step=100,
        epoch=5,
        validation_data=validation_data
    )
    
    print(f"\nCross-Scene Validation Results:")
    print(f"  Overall mAP: {result.overall_metrics.get('mAP', 0):.4f}")
    print(f"  Scene count: {len(result.scene_results)}")
    print(f"  mAP variance: {result.scene_variance.get('mAP', 0):.4f}")
    print(f"  Difficulty correlation: {result.difficulty_correlation.get('mAP', 0):.3f}")
    
    # Get validation summary
    summary = validator.get_validation_summary()
    print(f"\nValidation Summary:")
    print(f"  Total validations: {summary['total_validations']}")
    print(f"  Best mAP: {summary['best_performance'].get('mAP', 0):.4f}")
    
    print("✅ Cross-Scene Validator validation completed")