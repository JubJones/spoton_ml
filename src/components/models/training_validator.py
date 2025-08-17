"""
Advanced Training Validation Framework for RF-DETR
Multi-modal validation with comprehensive analysis and performance tracking
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from .surveillance_detector import SurveillanceMetrics, CrowdAwareDetector

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for advanced training validation."""
    
    # Validation scheduling
    validation_frequency: int = 500  # Steps between validations
    full_validation_frequency: int = 2000  # Steps between comprehensive validations
    quick_validation_samples: int = 100  # Samples for quick validation
    
    # Multi-modal validation
    scene_specific_validation: bool = True
    scale_specific_validation: bool = True
    temporal_validation: bool = False  # For video sequences
    
    # Performance tracking
    performance_history_size: int = 100
    trend_analysis_window: int = 20
    performance_regression_threshold: float = 0.05
    
    # Comparative validation
    baseline_comparison: bool = True
    cross_architecture_comparison: bool = False
    ensemble_validation: bool = False
    
    # Resource monitoring
    memory_profiling: bool = True
    inference_timing: bool = True
    batch_size_optimization: bool = True
    
    # Early stopping and adaptation
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = 'mAP'
    adaptive_validation: bool = True
    
    # Output and reporting
    detailed_reports: bool = True
    visualization_enabled: bool = True
    export_predictions: bool = False
    save_frequency: int = 5  # Save every N validations


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    
    # Basic information
    step: int
    epoch: int
    timestamp: float
    validation_type: str = "full"  # "quick", "full", "comprehensive"
    
    # Core metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Scene-specific results
    scene_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Scale-specific results
    scale_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Performance characteristics
    inference_stats: Dict[str, float] = field(default_factory=dict)
    memory_stats: Dict[str, float] = field(default_factory=dict)
    
    # Model analysis
    model_stats: Dict[str, float] = field(default_factory=dict)
    gradient_stats: Dict[str, float] = field(default_factory=dict)
    
    # Comparative analysis
    baseline_comparison: Optional[Dict[str, float]] = None
    improvement_analysis: Optional[Dict[str, float]] = None
    
    # Detailed analysis
    failure_analysis: Dict[str, Any] = field(default_factory=dict)
    success_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'step': self.step,
            'epoch': self.epoch,
            'validation_type': self.validation_type,
            'core_metrics': self.metrics,
            'performance': {
                'avg_inference_time': self.inference_stats.get('avg_time', 0),
                'memory_usage': self.memory_stats.get('peak_memory', 0)
            },
            'scene_performance': {
                scene: metrics.get('mAP', 0)
                for scene, metrics in self.scene_metrics.items()
            }
        }


class PerformanceTracker:
    """Performance tracking and trend analysis."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.history = deque(maxlen=config.performance_history_size)
        self.best_performance = {}
        self.performance_trends = {}
    
    def update(self, result: ValidationResult):
        """Update performance tracking with new result."""
        self.history.append(result)
        
        # Update best performance
        for metric, value in result.metrics.items():
            if metric not in self.best_performance or value > self.best_performance[metric]:
                self.best_performance[metric] = value
        
        # Update trends
        if len(self.history) >= self.config.trend_analysis_window:
            self._update_trends()
    
    def _update_trends(self):
        """Update performance trend analysis."""
        window_size = self.config.trend_analysis_window
        recent_results = list(self.history)[-window_size:]
        
        for metric in self.best_performance.keys():
            values = [r.metrics.get(metric, 0) for r in recent_results if metric in r.metrics]
            
            if len(values) >= 2:
                # Linear regression for trend
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                slope = coeffs[0]
                
                # Trend classification
                if abs(slope) < 1e-6:
                    trend = "stable"
                elif slope > 0:
                    trend = "improving"
                else:
                    trend = "declining"
                
                self.performance_trends[metric] = {
                    'slope': float(slope),
                    'trend': trend,
                    'recent_change': values[-1] - values[0],
                    'volatility': float(np.std(values))
                }
    
    def detect_regression(self) -> Dict[str, bool]:
        """Detect performance regression."""
        regressions = {}
        
        if len(self.history) < 5:
            return regressions
        
        recent_results = list(self.history)[-5:]  # Last 5 validations
        
        for metric in self.best_performance.keys():
            recent_values = [r.metrics.get(metric, 0) for r in recent_results if metric in r.metrics]
            
            if len(recent_values) >= 3:
                best_value = self.best_performance[metric]
                recent_avg = np.mean(recent_values)
                
                # Check if recent performance is significantly worse
                regression_threshold = best_value * self.config.performance_regression_threshold
                regressions[metric] = (best_value - recent_avg) > regression_threshold
        
        return regressions
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.history:
            return {}
        
        latest = self.history[-1]
        
        return {
            'total_validations': len(self.history),
            'latest_step': latest.step,
            'best_performance': self.best_performance.copy(),
            'latest_performance': latest.metrics.copy(),
            'trends': self.performance_trends.copy(),
            'regressions': self.detect_regression()
        }


class ModelAnalyzer:
    """Advanced model analysis and diagnostics."""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_model_state(
        self,
        model: nn.Module,
        step: int
    ) -> Dict[str, Any]:
        """Analyze current model state."""
        
        analysis = {
            'step': step,
            'parameter_stats': self._analyze_parameters(model),
            'gradient_stats': self._analyze_gradients(model),
            'activation_stats': self._analyze_activations(model),
            'memory_usage': self._get_memory_usage()
        }
        
        return analysis
    
    def _analyze_parameters(self, model: nn.Module) -> Dict[str, float]:
        """Analyze model parameters."""
        param_stats = {
            'total_params': 0,
            'trainable_params': 0,
            'param_norm': 0.0,
            'param_mean': 0.0,
            'param_std': 0.0,
            'zero_params': 0,
            'inf_params': 0,
            'nan_params': 0
        }
        
        all_params = []
        
        for param in model.parameters():
            param_stats['total_params'] += param.numel()
            
            if param.requires_grad:
                param_stats['trainable_params'] += param.numel()
            
            # Collect parameter values
            param_data = param.data.flatten()
            all_params.extend(param_data.cpu().numpy())
            
            # Check for problematic values
            param_stats['zero_params'] += (param_data == 0).sum().item()
            param_stats['inf_params'] += torch.isinf(param_data).sum().item()
            param_stats['nan_params'] += torch.isnan(param_data).sum().item()
        
        if all_params:
            all_params = np.array(all_params)
            param_stats['param_norm'] = float(np.linalg.norm(all_params))
            param_stats['param_mean'] = float(np.mean(all_params))
            param_stats['param_std'] = float(np.std(all_params))
        
        return param_stats
    
    def _analyze_gradients(self, model: nn.Module) -> Dict[str, float]:
        """Analyze model gradients."""
        grad_stats = {
            'grad_norm': 0.0,
            'grad_mean': 0.0,
            'grad_std': 0.0,
            'zero_grads': 0,
            'inf_grads': 0,
            'nan_grads': 0,
            'params_with_grads': 0,
            'max_grad': 0.0,
            'min_grad': 0.0
        }
        
        all_grads = []
        
        for param in model.parameters():
            if param.grad is not None:
                grad_stats['params_with_grads'] += 1
                
                grad_data = param.grad.data.flatten()
                all_grads.extend(grad_data.cpu().numpy())
                
                # Check for problematic gradients
                grad_stats['zero_grads'] += (grad_data == 0).sum().item()
                grad_stats['inf_grads'] += torch.isinf(grad_data).sum().item()
                grad_stats['nan_grads'] += torch.isnan(grad_data).sum().item()
        
        if all_grads:
            all_grads = np.array(all_grads)
            grad_stats['grad_norm'] = float(np.linalg.norm(all_grads))
            grad_stats['grad_mean'] = float(np.mean(all_grads))
            grad_stats['grad_std'] = float(np.std(all_grads))
            grad_stats['max_grad'] = float(np.max(all_grads))
            grad_stats['min_grad'] = float(np.min(all_grads))
        
        return grad_stats
    
    def _analyze_activations(self, model: nn.Module) -> Dict[str, float]:
        """Analyze model activations (placeholder for future implementation)."""
        # This would require hooking into model layers during forward pass
        return {
            'activation_analysis': 'not_implemented'
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_stats = {}
        
        if torch.cuda.is_available():
            memory_stats['cuda_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_stats['cuda_reserved'] = torch.cuda.memory_reserved() / 1024**2   # MB
            memory_stats['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return memory_stats


class ValidationEngine:
    """Advanced validation engine with multi-modal analysis."""
    
    def __init__(
        self,
        config: ValidationConfig,
        detector: CrowdAwareDetector,
        save_dir: Optional[Path] = None
    ):
        self.config = config
        self.detector = detector
        self.save_dir = save_dir or Path("validation_results")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.performance_tracker = PerformanceTracker(config)
        self.model_analyzer = ModelAnalyzer()
        self.metrics_calculator = SurveillanceMetrics()
        
        # Validation state
        self.validation_count = 0
        self.baseline_results = None
        
        logger.info(f"Validation Engine initialized:")
        logger.info(f"  Validation frequency: {config.validation_frequency}")
        logger.info(f"  Scene-specific validation: {config.scene_specific_validation}")
        logger.info(f"  Performance tracking: enabled")
    
    def validate(
        self,
        model: nn.Module,
        validation_loader: torch.utils.data.DataLoader,
        step: int,
        epoch: int,
        validation_type: str = "full",
        scene_loaders: Optional[Dict[str, torch.utils.data.DataLoader]] = None,
        baseline_model: Optional[nn.Module] = None
    ) -> ValidationResult:
        """
        Comprehensive validation with multi-modal analysis.
        
        Args:
            model: Model to validate
            validation_loader: Main validation data loader
            step: Current training step
            epoch: Current epoch
            validation_type: Type of validation ("quick", "full", "comprehensive")
            scene_loaders: Optional scene-specific data loaders
            baseline_model: Optional baseline model for comparison
            
        Returns:
            Comprehensive validation result
        """
        
        logger.info(f"Starting {validation_type} validation at step {step}, epoch {epoch}")
        start_time = time.time()
        
        result = ValidationResult(
            step=step,
            epoch=epoch,
            timestamp=start_time,
            validation_type=validation_type
        )
        
        model.eval()
        
        with torch.no_grad():
            # Core validation
            result.metrics = self._validate_core(model, validation_loader, validation_type)
            
            # Scene-specific validation
            if self.config.scene_specific_validation and scene_loaders:
                result.scene_metrics = self._validate_scenes(model, scene_loaders)
            
            # Scale-specific validation
            if self.config.scale_specific_validation:
                result.scale_metrics = self._validate_scales(model, validation_loader)
            
            # Performance profiling
            if self.config.inference_timing or self.config.memory_profiling:
                result.inference_stats, result.memory_stats = self._profile_performance(
                    model, validation_loader
                )
            
            # Model analysis
            result.model_stats = self.model_analyzer.analyze_model_state(model, step)
            
            # Baseline comparison
            if self.config.baseline_comparison and baseline_model:
                result.baseline_comparison = self._compare_with_baseline(
                    model, baseline_model, validation_loader
                )
            
            # Failure analysis for comprehensive validation
            if validation_type == "comprehensive":
                result.failure_analysis = self._analyze_failures(model, validation_loader)
        
        # Update performance tracking
        self.performance_tracker.update(result)
        
        # Generate improvement recommendations
        result.improvement_analysis = self._analyze_improvements(result)
        
        # Save results
        if self.validation_count % self.config.save_frequency == 0:
            self._save_validation_result(result)
        
        self.validation_count += 1
        
        validation_time = time.time() - start_time
        logger.info(f"Validation completed in {validation_time:.2f}s")
        logger.info(f"  mAP: {result.metrics.get('mAP', 0):.4f}")
        logger.info(f"  Avg inference time: {result.inference_stats.get('avg_time', 0):.3f}ms")
        
        return result
    
    def _validate_core(
        self,
        model: nn.Module,
        validation_loader: torch.utils.data.DataLoader,
        validation_type: str
    ) -> Dict[str, float]:
        """Core validation logic."""
        
        # Reset metrics calculator
        self.metrics_calculator.reset()
        
        # Determine number of batches to process
        if validation_type == "quick":
            max_batches = min(len(validation_loader), self.config.quick_validation_samples // validation_loader.batch_size)
        else:
            max_batches = len(validation_loader)
        
        processed_batches = 0
        
        for batch_idx, batch in enumerate(validation_loader):
            if processed_batches >= max_batches:
                break
            
            images = batch['images']
            targets = batch.get('targets', batch.get('annotations', []))
            
            # Model inference
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                predictions = model(images)
            
            # Post-process predictions
            image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
            detection_results = self.detector(predictions, image_sizes)
            
            # Update metrics
            if targets:
                self.metrics_calculator.update(detection_results, targets)
            
            processed_batches += 1
        
        # Compute final metrics
        metrics = self.metrics_calculator.compute()
        
        return metrics
    
    def _validate_scenes(
        self,
        model: nn.Module,
        scene_loaders: Dict[str, torch.utils.data.DataLoader]
    ) -> Dict[str, Dict[str, float]]:
        """Scene-specific validation."""
        
        scene_results = {}
        
        for scene_id, loader in scene_loaders.items():
            scene_metrics = SurveillanceMetrics()
            
            for batch in loader:
                images = batch['images']
                targets = batch.get('targets', batch.get('annotations', []))
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    predictions = model(images)
                
                image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
                detection_results = self.detector(predictions, image_sizes)
                
                if targets:
                    scene_metrics.update(detection_results, targets)
            
            scene_results[scene_id] = scene_metrics.compute()
        
        return scene_results
    
    def _validate_scales(
        self,
        model: nn.Module,
        validation_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Dict[str, float]]:
        """Scale-specific validation."""
        
        # Group detections by scale
        scale_groups = {
            'small': {'predictions': [], 'targets': []},
            'medium': {'predictions': [], 'targets': []},
            'large': {'predictions': [], 'targets': []}
        }
        
        for batch in validation_loader:
            images = batch['images']
            targets = batch.get('targets', batch.get('annotations', []))
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                predictions = model(images)
            
            image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
            detection_results = self.detector(predictions, image_sizes)
            
            # Group by scale based on ground truth box sizes
            for i, (pred, target) in enumerate(zip(detection_results, targets)):
                if 'boxes' in target and len(target['boxes']) > 0:
                    gt_boxes = target['boxes']
                    
                    # Calculate box areas
                    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                    
                    # Classify by scale (thresholds can be adjusted)
                    for j, area in enumerate(areas):
                        if area < 32**2:
                            scale = 'small'
                        elif area < 96**2:
                            scale = 'medium'
                        else:
                            scale = 'large'
                        
                        # Create single-detection versions for this scale
                        single_pred = {
                            'boxes': pred['boxes'],
                            'scores': pred['scores'],
                            'labels': pred['labels']
                        }
                        
                        single_target = {
                            'boxes': gt_boxes[j:j+1],
                            'labels': target['labels'][j:j+1] if 'labels' in target else torch.tensor([1])
                        }
                        
                        scale_groups[scale]['predictions'].append(single_pred)
                        scale_groups[scale]['targets'].append(single_target)
        
        # Compute metrics for each scale
        scale_results = {}
        
        for scale, data in scale_groups.items():
            if data['predictions'] and data['targets']:
                scale_metrics = SurveillanceMetrics()
                scale_metrics.update(data['predictions'], data['targets'])
                scale_results[scale] = scale_metrics.compute()
            else:
                scale_results[scale] = {}
        
        return scale_results
    
    def _profile_performance(
        self,
        model: nn.Module,
        validation_loader: torch.utils.data.DataLoader
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Profile model performance."""
        
        inference_times = []
        memory_stats = []
        
        # Profile on a subset of batches
        num_profile_batches = min(10, len(validation_loader))
        
        for batch_idx, batch in enumerate(validation_loader):
            if batch_idx >= num_profile_batches:
                break
            
            images = batch['images']
            
            # Memory before inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
            
            # Time inference
            start_time = time.time()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                predictions = model(images)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Memory after inference
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                memory_stats.append(mem_after - mem_before)
            
            # Calculate per-image inference time
            batch_time = (end_time - start_time) * 1000  # Convert to ms
            per_image_time = batch_time / len(images)
            inference_times.append(per_image_time)
        
        # Aggregate statistics
        inference_stats = {
            'avg_time': np.mean(inference_times) if inference_times else 0,
            'std_time': np.std(inference_times) if inference_times else 0,
            'min_time': np.min(inference_times) if inference_times else 0,
            'max_time': np.max(inference_times) if inference_times else 0,
            'fps': 1000 / np.mean(inference_times) if inference_times and np.mean(inference_times) > 0 else 0
        }
        
        memory_stats_dict = {}
        if memory_stats:
            memory_stats_dict = {
                'avg_memory': np.mean(memory_stats) / 1024**2,  # MB
                'peak_memory': np.max(memory_stats) / 1024**2,  # MB
                'min_memory': np.min(memory_stats) / 1024**2    # MB
            }
        
        return inference_stats, memory_stats_dict
    
    def _compare_with_baseline(
        self,
        model: nn.Module,
        baseline_model: nn.Module,
        validation_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Compare current model with baseline."""
        
        # Get baseline metrics if not cached
        if self.baseline_results is None:
            baseline_metrics = SurveillanceMetrics()
            
            baseline_model.eval()
            with torch.no_grad():
                for batch in validation_loader:
                    images = batch['images']
                    targets = batch.get('targets', batch.get('annotations', []))
                    
                    predictions = baseline_model(images)
                    image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
                    detection_results = self.detector(predictions, image_sizes)
                    
                    if targets:
                        baseline_metrics.update(detection_results, targets)
            
            self.baseline_results = baseline_metrics.compute()
        
        # Current model metrics
        current_metrics = self._validate_core(model, validation_loader, "full")
        
        # Calculate improvements
        comparison = {}
        for metric in current_metrics:
            if metric in self.baseline_results:
                improvement = current_metrics[metric] - self.baseline_results[metric]
                relative_improvement = improvement / max(self.baseline_results[metric], 1e-8)
                
                comparison[f'{metric}_improvement'] = improvement
                comparison[f'{metric}_relative_improvement'] = relative_improvement
        
        return comparison
    
    def _analyze_failures(
        self,
        model: nn.Module,
        validation_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Analyze model failures for improvement insights."""
        
        failure_analysis = {
            'false_positives': [],
            'false_negatives': [],
            'low_confidence_detections': [],
            'failure_patterns': {}
        }
        
        # Analyze a subset of validation data
        analyzed_batches = 0
        max_analysis_batches = 5
        
        for batch in validation_loader:
            if analyzed_batches >= max_analysis_batches:
                break
            
            images = batch['images']
            targets = batch.get('targets', batch.get('annotations', []))
            
            if not targets:
                continue
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                predictions = model(images)
            
            image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
            detection_results = self.detector(predictions, image_sizes)
            
            # Analyze each image in batch
            for pred, target in zip(detection_results, targets):
                # Analyze false positives and negatives
                self._analyze_detection_errors(pred, target, failure_analysis)
            
            analyzed_batches += 1
        
        # Aggregate failure patterns
        failure_analysis['failure_patterns'] = self._aggregate_failure_patterns(failure_analysis)
        
        return failure_analysis
    
    def _analyze_detection_errors(
        self,
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        failure_analysis: Dict[str, Any]
    ):
        """Analyze detection errors for a single image."""
        
        pred_boxes = prediction.get('boxes', torch.empty(0, 4))
        pred_scores = prediction.get('scores', torch.empty(0))
        pred_labels = prediction.get('labels', torch.empty(0))
        
        gt_boxes = target.get('boxes', torch.empty(0, 4))
        gt_labels = target.get('labels', torch.empty(0))
        
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            return
        
        # Calculate IoU matrix
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            iou_matrix = self._calculate_iou_matrix(pred_boxes, gt_boxes)
            
            # Find matches (IoU > 0.5)
            max_ious, matched_gt = iou_matrix.max(dim=1)
            matched_pred_mask = max_ious > 0.5
            
            # False positives (predictions without good matches)
            fp_mask = ~matched_pred_mask
            if fp_mask.any():
                fp_boxes = pred_boxes[fp_mask]
                fp_scores = pred_scores[fp_mask]
                
                for i, (box, score) in enumerate(zip(fp_boxes, fp_scores)):
                    failure_analysis['false_positives'].append({
                        'box': box.tolist(),
                        'score': float(score),
                        'area': float((box[2] - box[0]) * (box[3] - box[1]))
                    })
            
            # False negatives (ground truth without matches)
            matched_gt_indices = matched_gt[matched_pred_mask]
            unmatched_gt_mask = torch.ones(len(gt_boxes), dtype=torch.bool)
            if len(matched_gt_indices) > 0:
                unmatched_gt_mask[matched_gt_indices] = False
            
            if unmatched_gt_mask.any():
                fn_boxes = gt_boxes[unmatched_gt_mask]
                
                for box in fn_boxes:
                    failure_analysis['false_negatives'].append({
                        'box': box.tolist(),
                        'area': float((box[2] - box[0]) * (box[3] - box[1]))
                    })
        
        # Low confidence detections
        low_conf_mask = pred_scores < 0.7  # Arbitrary threshold
        if low_conf_mask.any():
            low_conf_boxes = pred_boxes[low_conf_mask]
            low_conf_scores = pred_scores[low_conf_mask]
            
            for box, score in zip(low_conf_boxes, low_conf_scores):
                failure_analysis['low_confidence_detections'].append({
                    'box': box.tolist(),
                    'score': float(score),
                    'area': float((box[2] - box[0]) * (box[3] - box[1]))
                })
    
    def _calculate_iou_matrix(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """Calculate IoU matrix between two sets of boxes."""
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        intersection = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - intersection
        iou = intersection / (union + 1e-8)
        
        return iou
    
    def _aggregate_failure_patterns(self, failure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate failure patterns for insights."""
        
        patterns = {}
        
        # False positive patterns
        fp_areas = [fp['area'] for fp in failure_analysis['false_positives']]
        if fp_areas:
            patterns['fp_size_distribution'] = {
                'small': sum(1 for area in fp_areas if area < 32**2) / len(fp_areas),
                'medium': sum(1 for area in fp_areas if 32**2 <= area < 96**2) / len(fp_areas),
                'large': sum(1 for area in fp_areas if area >= 96**2) / len(fp_areas)
            }
        
        # False negative patterns
        fn_areas = [fn['area'] for fn in failure_analysis['false_negatives']]
        if fn_areas:
            patterns['fn_size_distribution'] = {
                'small': sum(1 for area in fn_areas if area < 32**2) / len(fn_areas),
                'medium': sum(1 for area in fn_areas if 32**2 <= area < 96**2) / len(fn_areas),
                'large': sum(1 for area in fn_areas if area >= 96**2) / len(fn_areas)
            }
        
        return patterns
    
    def _analyze_improvements(self, result: ValidationResult) -> Dict[str, Any]:
        """Analyze potential improvements based on validation results."""
        
        improvements = {
            'recommendations': [],
            'priority_areas': [],
            'training_adjustments': []
        }
        
        # Check performance trends
        performance_summary = self.performance_tracker.get_performance_summary()
        
        if 'trends' in performance_summary:
            for metric, trend_info in performance_summary['trends'].items():
                if trend_info['trend'] == 'declining':
                    improvements['recommendations'].append(
                        f"{metric} is declining (slope: {trend_info['slope']:.4f}). "
                        "Consider reducing learning rate or adjusting regularization."
                    )
                elif trend_info['volatility'] > 0.1:
                    improvements['recommendations'].append(
                        f"{metric} shows high volatility ({trend_info['volatility']:.3f}). "
                        "Consider increasing batch size or adding gradient smoothing."
                    )
        
        # Check for regressions
        regressions = performance_summary.get('regressions', {})
        for metric, has_regressed in regressions.items():
            if has_regressed:
                improvements['priority_areas'].append(f"Address {metric} regression")
        
        # Inference time recommendations
        avg_inference_time = result.inference_stats.get('avg_time', 0)
        if avg_inference_time > 100:  # ms
            improvements['training_adjustments'].append(
                "Inference time is high. Consider model pruning or knowledge distillation."
            )
        
        # Memory usage recommendations
        peak_memory = result.memory_stats.get('peak_memory', 0)
        if peak_memory > 8000:  # MB
            improvements['training_adjustments'].append(
                "High memory usage detected. Consider reducing batch size or using gradient checkpointing."
            )
        
        return improvements
    
    def _save_validation_result(self, result: ValidationResult):
        """Save validation result to disk."""
        
        # Save main result
        result_file = self.save_dir / f"validation_step_{result.step}.json"
        
        save_data = {
            'summary': result.get_summary(),
            'detailed_metrics': result.metrics,
            'scene_metrics': result.scene_metrics,
            'scale_metrics': result.scale_metrics,
            'performance_stats': {
                'inference': result.inference_stats,
                'memory': result.memory_stats
            },
            'analysis': {
                'model_stats': result.model_stats,
                'improvement_analysis': result.improvement_analysis
            }
        }
        
        with open(result_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        # Save performance trends
        performance_summary = self.performance_tracker.get_performance_summary()
        trends_file = self.save_dir / "performance_trends.json"
        
        with open(trends_file, 'w') as f:
            json.dump(performance_summary, f, indent=2, default=str)
        
        logger.debug(f"Saved validation results to {result_file}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        
        performance_summary = self.performance_tracker.get_performance_summary()
        
        summary = {
            'total_validations': self.validation_count,
            'performance_tracking': performance_summary,
            'config': self.config.__dict__
        }
        
        return summary
    
    def should_stop_early(self) -> bool:
        """Check if early stopping criteria are met."""
        
        if not self.config.early_stopping_enabled:
            return False
        
        # Check if we have enough validations
        if len(self.performance_tracker.history) < self.config.early_stopping_patience:
            return False
        
        # Get recent performance for early stopping metric
        metric = self.config.early_stopping_metric
        recent_results = list(self.performance_tracker.history)[-self.config.early_stopping_patience:]
        
        recent_values = [r.metrics.get(metric, 0) for r in recent_results if metric in r.metrics]
        
        if len(recent_values) < self.config.early_stopping_patience:
            return False
        
        # Check if performance has plateaued
        best_value = max(recent_values)
        current_value = recent_values[-1]
        
        # If current performance is significantly worse than best recent performance
        threshold = 0.01  # 1% threshold
        return (best_value - current_value) / max(best_value, 1e-8) > threshold


def create_validation_engine(
    detector: CrowdAwareDetector,
    validation_frequency: int = 500,
    scene_specific_validation: bool = True,
    performance_tracking: bool = True,
    save_dir: Optional[Path] = None,
    **kwargs
) -> ValidationEngine:
    """
    Create advanced validation engine.
    
    Args:
        detector: Surveillance detector for post-processing
        validation_frequency: Steps between validations
        scene_specific_validation: Enable scene-specific analysis
        performance_tracking: Enable performance trend tracking
        save_dir: Directory for saving validation results
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured validation engine
    """
    
    config = ValidationConfig(
        validation_frequency=validation_frequency,
        scene_specific_validation=scene_specific_validation,
        **kwargs
    )
    
    return ValidationEngine(config, detector, save_dir)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Advanced Training Validation Framework")
    
    from .surveillance_detector import create_surveillance_detector
    
    # Create detector
    detector = create_surveillance_detector()
    
    # Create validation engine
    engine = create_validation_engine(
        detector=detector,
        validation_frequency=100,
        scene_specific_validation=True,
        performance_tracking=True
    )
    
    print(f"✅ Validation engine created")
    
    # Mock model
    class MockRFDETR(nn.Module):
        def forward(self, x):
            batch_size = len(x)
            return {
                'pred_logits': torch.randn(batch_size, 100, 91),
                'pred_boxes': torch.rand(batch_size, 100, 4)
            }
    
    model = MockRFDETR()
    
    # Mock data loader
    class MockDataset:
        def __init__(self, size=10):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'images': [torch.randn(3, 512, 512)],
                'targets': [{
                    'boxes': torch.tensor([[100, 100, 200, 300]], dtype=torch.float),
                    'labels': torch.tensor([1], dtype=torch.long)
                }]
            }
    
    from torch.utils.data import DataLoader
    mock_loader = DataLoader(MockDataset(), batch_size=2, collate_fn=lambda x: {
        'images': [item['images'][0] for item in x],
        'targets': [item['targets'][0] for item in x]
    })
    
    # Test validation
    try:
        result = engine.validate(
            model=model,
            validation_loader=mock_loader,
            step=100,
            epoch=5,
            validation_type="full"
        )
        
        print(f"✅ Validation completed successfully")
        print(f"  Validation type: {result.validation_type}")
        print(f"  Metrics computed: {len(result.metrics)}")
        print(f"  Performance stats: {len(result.inference_stats)}")
        
        # Test validation summary
        summary = engine.get_validation_summary()
        print(f"✅ Validation summary generated")
        print(f"  Total validations: {summary['total_validations']}")
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Advanced Training Validation Framework testing completed")