"""
Advanced Gradient Handling for RF-DETR Training
Includes gradient clipping, monitoring, and stability mechanisms
"""
import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
import numpy as np
from collections import deque
import math

logger = logging.getLogger(__name__)


@dataclass
class GradientClipConfig:
    """Configuration for gradient clipping strategies."""
    
    # Gradient clipping
    enabled: bool = True
    clip_method: str = "norm"  # "norm", "value", "adaptive"
    max_norm: float = 1.0
    norm_type: float = 2.0
    
    # Adaptive clipping
    adaptive_percentile: float = 95.0
    adaptive_window_size: int = 100
    adaptive_min_norm: float = 0.1
    adaptive_max_norm: float = 10.0
    
    # Gradient monitoring
    monitor_gradients: bool = True
    log_frequency: int = 100
    detect_anomalies: bool = True
    
    # Stability thresholds
    explosion_threshold: float = 100.0
    vanishing_threshold: float = 1e-7
    
    def __post_init__(self):
        """Validate configuration."""
        if self.clip_method not in ["norm", "value", "adaptive"]:
            raise ValueError(f"clip_method must be 'norm', 'value', or 'adaptive', got {self.clip_method}")
        if not 0 < self.max_norm <= 100:
            raise ValueError(f"max_norm must be in (0, 100], got {self.max_norm}")


class GradientStatistics:
    """Track gradient statistics for monitoring and adaptive clipping."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.grad_norms = deque(maxlen=window_size)
        self.grad_stds = deque(maxlen=window_size)
        self.param_grad_norms = {}
        self.step_count = 0
    
    def update(self, model: nn.Module, grad_norm: float):
        """Update gradient statistics."""
        
        self.step_count += 1
        self.grad_norms.append(grad_norm)
        
        # Calculate gradient standard deviation across parameters
        grad_values = []
        param_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.norm().item()
                param_norms[name] = param_grad_norm
                grad_values.extend(param.grad.flatten().cpu().numpy())
        
        if grad_values:
            grad_std = np.std(grad_values)
            self.grad_stds.append(grad_std)
        
        # Store parameter-wise gradient norms
        for name, norm in param_norms.items():
            if name not in self.param_grad_norms:
                self.param_grad_norms[name] = deque(maxlen=self.window_size)
            self.param_grad_norms[name].append(norm)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current gradient statistics."""
        
        stats = {}
        
        if self.grad_norms:
            stats.update({
                'grad_norm_mean': np.mean(self.grad_norms),
                'grad_norm_std': np.std(self.grad_norms),
                'grad_norm_min': np.min(self.grad_norms),
                'grad_norm_max': np.max(self.grad_norms),
                'grad_norm_current': self.grad_norms[-1],
                'grad_norm_percentile_95': np.percentile(self.grad_norms, 95)
            })
        
        if self.grad_stds:
            stats.update({
                'grad_std_mean': np.mean(self.grad_stds),
                'grad_std_current': self.grad_stds[-1]
            })
        
        stats['step_count'] = self.step_count
        return stats
    
    def get_adaptive_clip_value(self, percentile: float = 95.0, min_val: float = 0.1, max_val: float = 10.0) -> float:
        """Calculate adaptive gradient clipping value."""
        
        if len(self.grad_norms) < 10:  # Need minimum samples
            return max_val
        
        clip_value = np.percentile(self.grad_norms, percentile)
        return max(min_val, min(clip_value, max_val))
    
    def detect_gradient_anomalies(self, explosion_threshold: float = 100.0, vanishing_threshold: float = 1e-7) -> Dict[str, bool]:
        """Detect gradient anomalies."""
        
        anomalies = {
            'gradient_explosion': False,
            'gradient_vanishing': False,
            'gradient_nan': False,
            'gradient_inf': False
        }
        
        if self.grad_norms:
            current_norm = self.grad_norms[-1]
            
            if current_norm > explosion_threshold:
                anomalies['gradient_explosion'] = True
            if current_norm < vanishing_threshold:
                anomalies['gradient_vanishing'] = True
            if math.isnan(current_norm):
                anomalies['gradient_nan'] = True
            if math.isinf(current_norm):
                anomalies['gradient_inf'] = True
        
        return anomalies


class GradientHandler:
    """
    Advanced gradient handling with clipping, monitoring, and stability mechanisms.
    Designed for stable RF-DETR training with comprehensive gradient analysis.
    """
    
    def __init__(self, model: nn.Module, config: GradientClipConfig):
        """
        Initialize gradient handler.
        
        Args:
            model: PyTorch model
            config: Gradient clipping configuration
        """
        self.model = model
        self.config = config
        self.statistics = GradientStatistics(config.adaptive_window_size)
        
        # Gradient history for analysis
        self.gradient_history = []
        self.clipping_history = []
        
        logger.info(f"Initialized GradientHandler with {config.clip_method} clipping")
    
    def clip_gradients(self, optimizer: Optimizer) -> Dict[str, float]:
        """
        Apply gradient clipping and return clipping statistics.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            Dictionary with clipping statistics
        """
        
        # Calculate gradient norm before clipping
        grad_norm_before = self._calculate_gradient_norm()
        
        # Update statistics
        if self.config.monitor_gradients:
            self.statistics.update(self.model, grad_norm_before)
        
        # Detect anomalies
        anomalies = {}
        if self.config.detect_anomalies:
            anomalies = self.statistics.detect_gradient_anomalies(
                self.config.explosion_threshold,
                self.config.vanishing_threshold
            )
        
        # Apply gradient clipping
        if self.config.enabled:
            if self.config.clip_method == "norm":
                grad_norm_after = self._clip_by_norm()
            elif self.config.clip_method == "value":
                grad_norm_after = self._clip_by_value()
            elif self.config.clip_method == "adaptive":
                grad_norm_after = self._clip_adaptive()
            else:
                grad_norm_after = grad_norm_before
        else:
            grad_norm_after = grad_norm_before
        
        # Calculate clipping ratio
        clipping_ratio = grad_norm_after / max(grad_norm_before, 1e-8)
        
        # Store clipping history
        self.clipping_history.append({
            'step': self.statistics.step_count,
            'grad_norm_before': grad_norm_before,
            'grad_norm_after': grad_norm_after,
            'clipping_ratio': clipping_ratio
        })
        
        # Keep history size manageable
        if len(self.clipping_history) > 1000:
            self.clipping_history = self.clipping_history[-500:]
        
        # Prepare return statistics
        clip_stats = {
            'grad_norm_before': grad_norm_before,
            'grad_norm_after': grad_norm_after,
            'clipping_ratio': clipping_ratio,
            'clipping_applied': clipping_ratio < 0.99
        }
        
        # Add anomaly information
        clip_stats.update(anomalies)
        
        # Log if needed
        if (self.config.monitor_gradients and 
            self.statistics.step_count % self.config.log_frequency == 0):
            self._log_gradient_statistics(clip_stats)
        
        return clip_stats
    
    def _calculate_gradient_norm(self, norm_type: float = 2.0) -> float:
        """Calculate total gradient norm across all parameters."""
        
        total_norm = 0.0
        param_count = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
                param_count += 1
        
        if param_count == 0:
            return 0.0
        
        return total_norm ** (1.0 / norm_type)
    
    def _clip_by_norm(self) -> float:
        """Clip gradients by global norm."""
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_norm,
            norm_type=self.config.norm_type
        )
        
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    
    def _clip_by_value(self) -> float:
        """Clip gradients by value."""
        
        torch.nn.utils.clip_grad_value_(
            self.model.parameters(),
            clip_value=self.config.max_norm
        )
        
        return self._calculate_gradient_norm()
    
    def _clip_adaptive(self) -> float:
        """Clip gradients using adaptive threshold."""
        
        # Calculate adaptive clipping value
        adaptive_clip = self.statistics.get_adaptive_clip_value(
            self.config.adaptive_percentile,
            self.config.adaptive_min_norm,
            self.config.adaptive_max_norm
        )
        
        # Apply clipping with adaptive value
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=adaptive_clip,
            norm_type=self.config.norm_type
        )
        
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    
    def _log_gradient_statistics(self, clip_stats: Dict[str, float]):
        """Log gradient statistics for monitoring."""
        
        stats = self.statistics.get_statistics()
        
        logger.info(f"Step {stats['step_count']} - Gradient Statistics:")
        logger.info(f"  Norm (before/after): {clip_stats['grad_norm_before']:.4f} / {clip_stats['grad_norm_after']:.4f}")
        logger.info(f"  Clipping ratio: {clip_stats['clipping_ratio']:.4f}")
        logger.info(f"  Mean norm (window): {stats.get('grad_norm_mean', 0):.4f}")
        logger.info(f"  Std norm (window): {stats.get('grad_norm_std', 0):.4f}")
        
        # Log anomalies
        if clip_stats.get('gradient_explosion', False):
            logger.warning(f"  🚨 Gradient explosion detected! Norm: {clip_stats['grad_norm_before']:.4f}")
        if clip_stats.get('gradient_vanishing', False):
            logger.warning(f"  🔻 Gradient vanishing detected! Norm: {clip_stats['grad_norm_before']:.4f}")
        if clip_stats.get('gradient_nan', False):
            logger.error(f"  ❌ NaN gradients detected!")
        if clip_stats.get('gradient_inf', False):
            logger.error(f"  ♾️ Infinite gradients detected!")
    
    def get_gradient_statistics(self) -> Dict[str, Any]:
        """Get comprehensive gradient statistics."""
        
        stats = self.statistics.get_statistics()
        
        # Add clipping statistics
        if self.clipping_history:
            recent_clips = self.clipping_history[-100:]  # Last 100 steps
            clipping_ratios = [clip['clipping_ratio'] for clip in recent_clips]
            
            stats.update({
                'clipping_frequency': sum(1 for r in clipping_ratios if r < 0.99) / len(clipping_ratios),
                'avg_clipping_ratio': np.mean(clipping_ratios),
                'min_clipping_ratio': np.min(clipping_ratios),
                'clipping_applied_steps': sum(1 for r in clipping_ratios if r < 0.99)
            })
        
        return stats
    
    def get_parameter_gradient_norms(self) -> Dict[str, float]:
        """Get current gradient norms for each parameter group."""
        
        param_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norms[name] = param.grad.norm().item()
        
        return param_norms
    
    def reset_statistics(self):
        """Reset gradient statistics (useful for new training phases)."""
        
        self.statistics = GradientStatistics(self.config.adaptive_window_size)
        self.clipping_history = []
        logger.info("Reset gradient statistics")
    
    def update_config(self, **kwargs):
        """Update gradient handler configuration."""
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated gradient config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def check_gradient_health(self) -> Dict[str, Union[bool, str]]:
        """Check overall gradient health and provide recommendations."""
        
        stats = self.statistics.get_statistics()
        health_report = {
            'healthy': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check for gradient explosion
        if stats.get('grad_norm_current', 0) > self.config.explosion_threshold:
            health_report['healthy'] = False
            health_report['issues'].append('Gradient explosion')
            health_report['recommendations'].append('Reduce learning rate or increase gradient clipping')
        
        # Check for gradient vanishing
        if stats.get('grad_norm_current', 1) < self.config.vanishing_threshold:
            health_report['healthy'] = False
            health_report['issues'].append('Gradient vanishing')
            health_report['recommendations'].append('Increase learning rate or check model architecture')
        
        # Check gradient variance
        if stats.get('grad_norm_std', 0) > stats.get('grad_norm_mean', 1):
            health_report['issues'].append('High gradient variance')
            health_report['recommendations'].append('Consider adaptive gradient clipping')
        
        # Check clipping frequency
        if hasattr(self, 'clipping_history') and len(self.clipping_history) > 10:
            recent_clips = self.clipping_history[-50:]
            clip_freq = sum(1 for clip in recent_clips if clip['clipping_ratio'] < 0.99) / len(recent_clips)
            
            if clip_freq > 0.8:  # More than 80% of gradients clipped
                health_report['issues'].append('High clipping frequency')
                health_report['recommendations'].append('Increase gradient clipping threshold or reduce learning rate')
        
        return health_report
    
    def save_gradient_analysis(self, filepath: str):
        """Save gradient analysis to file for post-training analysis."""
        
        import json
        
        analysis_data = {
            'config': self.config.__dict__,
            'statistics': self.get_gradient_statistics(),
            'clipping_history': self.clipping_history[-500:],  # Last 500 steps
            'health_report': self.check_gradient_health()
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        logger.info(f"Saved gradient analysis to {filepath}")