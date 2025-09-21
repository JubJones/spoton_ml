"""
Advanced Loss Tracking for RF-DETR Training
Comprehensive loss monitoring, analysis, and anomaly detection
"""
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import math
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """Configuration for loss tracking and monitoring."""
    
    # Tracking configuration
    window_size: int = 1000
    smooth_window: int = 100
    log_frequency: int = 50
    
    # Loss component tracking
    track_components: bool = True
    component_names: List[str] = field(default_factory=lambda: [
        "loss_ce", "loss_bbox", "loss_giou", "class_error", "cardinality_error"
    ])
    
    # Anomaly detection
    anomaly_detection: bool = True
    spike_threshold: float = 3.0      # Standard deviations for spike detection
    plateau_threshold: int = 100      # Steps without improvement for plateau detection
    divergence_threshold: float = 10.0 # Multiplier for divergence detection
    
    # Loss analysis
    convergence_analysis: bool = True
    trend_analysis_window: int = 500
    correlation_analysis: bool = True
    
    # Persistence
    save_history: bool = True
    save_frequency: int = 1000
    save_path: Optional[str] = "logs/loss_history.json"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if not 1 <= self.smooth_window <= self.window_size:
            raise ValueError(f"smooth_window must be in [1, {self.window_size}]")


class LossStatistics:
    """Statistical analysis for loss values."""
    
    def __init__(self, window_size: int = 1000):
        self.values = deque(maxlen=window_size)
        self.window_size = window_size
    
    def add(self, value: float):
        """Add new loss value."""
        if not math.isnan(value) and not math.isinf(value):
            self.values.append(value)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary."""
        if not self.values:
            return {}
        
        values_array = np.array(list(self.values))
        
        return {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'median': float(np.median(values_array)),
            'q25': float(np.percentile(values_array, 25)),
            'q75': float(np.percentile(values_array, 75)),
            'current': float(values_array[-1]) if len(values_array) > 0 else 0.0,
            'count': len(values_array)
        }
    
    def get_trend(self, window: int = 100) -> Dict[str, float]:
        """Calculate trend statistics."""
        if len(self.values) < max(2, window):
            return {'trend': 0.0, 'slope': 0.0, 'r_squared': 0.0}
        
        recent_values = list(self.values)[-window:]
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        # Linear regression
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            
            # R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return {
                'trend': 1 if slope < 0 else -1,  # 1 for improving (decreasing), -1 for worsening
                'slope': float(slope),
                'r_squared': float(r_squared)
            }
        
        return {'trend': 0.0, 'slope': 0.0, 'r_squared': 0.0}
    
    def detect_anomalies(self, spike_threshold: float = 3.0, divergence_threshold: float = 10.0) -> Dict[str, bool]:
        """Detect loss anomalies."""
        anomalies = {
            'spike': False,
            'divergence': False,
            'nan_loss': False,
            'inf_loss': False
        }
        
        if len(self.values) < 10:
            return anomalies
        
        current_value = self.values[-1]
        
        # Check for NaN/Inf
        if math.isnan(current_value):
            anomalies['nan_loss'] = True
        if math.isinf(current_value):
            anomalies['inf_loss'] = True
        
        # Spike detection (z-score based)
        if len(self.values) >= 10:
            recent_values = list(self.values)[:-1]  # Exclude current value
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values)
            
            if std_val > 0:
                z_score = abs(current_value - mean_val) / std_val
                if z_score > spike_threshold:
                    anomalies['spike'] = True
        
        # Divergence detection (compared to initial values)
        if len(self.values) >= 50:
            initial_mean = np.mean(list(self.values)[:10])
            if initial_mean > 0 and current_value > initial_mean * divergence_threshold:
                anomalies['divergence'] = True
        
        return anomalies


class LossTracker:
    """
    Advanced loss tracking system for RF-DETR training.
    Provides comprehensive monitoring, analysis, and anomaly detection.
    """
    
    def __init__(self, config: LossConfig):
        """
        Initialize loss tracker.
        
        Args:
            config: Loss tracking configuration
        """
        self.config = config
        
        # Loss tracking
        self.total_loss_stats = LossStatistics(config.window_size)
        self.component_stats = {}
        
        # Step tracking
        self.step_count = 0
        self.epoch_count = 0
        
        # History for analysis
        self.loss_history = []
        self.anomaly_history = []
        
        # Convergence tracking
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.convergence_metrics = {}
        
        # Smoothing
        self.smoothed_losses = {}
        
        logger.info(f"Initialized LossTracker with window_size={config.window_size}")
    
    def update(
        self, 
        loss_dict: Dict[str, Union[torch.Tensor, float]], 
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ):
        """
        Update loss tracking with new loss values.
        
        Args:
            loss_dict: Dictionary of loss components
            step: Current training step
            epoch: Current epoch
        """
        
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
        
        if epoch is not None:
            self.epoch_count = epoch
        
        # Convert tensors to floats
        processed_losses = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                processed_losses[key] = value.item()
            else:
                processed_losses[key] = float(value)
        
        # Track total loss
        total_loss = processed_losses.get('loss', processed_losses.get('total_loss', 0.0))
        self.total_loss_stats.add(total_loss)
        
        # Track loss components
        if self.config.track_components:
            for component_name in self.config.component_names:
                if component_name in processed_losses:
                    if component_name not in self.component_stats:
                        self.component_stats[component_name] = LossStatistics(self.config.window_size)
                    self.component_stats[component_name].add(processed_losses[component_name])
        
        # Update smoothed losses
        self._update_smoothed_losses(processed_losses)
        
        # Detect anomalies
        anomalies = {}
        if self.config.anomaly_detection:
            anomalies = self._detect_anomalies(processed_losses)
        
        # Update convergence metrics
        if self.config.convergence_analysis:
            self._update_convergence_metrics(total_loss)
        
        # Store history entry
        history_entry = {
            'step': self.step_count,
            'epoch': self.epoch_count,
            'losses': processed_losses.copy(),
            'total_loss': total_loss,
            'anomalies': anomalies,
            'best_loss': self.best_loss,
            'steps_without_improvement': self.steps_without_improvement
        }
        
        self.loss_history.append(history_entry)
        
        # Keep history size manageable
        if len(self.loss_history) > self.config.window_size * 2:
            self.loss_history = self.loss_history[-self.config.window_size:]
        
        # Log if needed
        if self.step_count % self.config.log_frequency == 0:
            self._log_loss_summary(processed_losses, anomalies)
        
        # Save history periodically
        if (self.config.save_history and 
            self.step_count % self.config.save_frequency == 0 and 
            self.config.save_path):
            self.save_history(self.config.save_path)
    
    def _update_smoothed_losses(self, losses: Dict[str, float]):
        """Update exponentially smoothed loss values."""
        
        alpha = 2.0 / (self.config.smooth_window + 1)
        
        for key, value in losses.items():
            if key not in self.smoothed_losses:
                self.smoothed_losses[key] = value
            else:
                self.smoothed_losses[key] = alpha * value + (1 - alpha) * self.smoothed_losses[key]
    
    def _detect_anomalies(self, losses: Dict[str, float]) -> Dict[str, Any]:
        """Detect loss anomalies."""
        
        total_loss = losses.get('loss', losses.get('total_loss', 0.0))
        
        # Detect total loss anomalies
        total_anomalies = self.total_loss_stats.detect_anomalies(
            self.config.spike_threshold,
            self.config.divergence_threshold
        )
        
        # Detect component anomalies
        component_anomalies = {}
        for component_name, stats in self.component_stats.items():
            if component_name in losses:
                component_anomalies[component_name] = stats.detect_anomalies(
                    self.config.spike_threshold,
                    self.config.divergence_threshold
                )
        
        # Aggregate anomalies
        anomalies = {
            'total_loss': total_anomalies,
            'components': component_anomalies,
            'any_anomaly': any(total_anomalies.values()) or any(
                any(comp_anom.values()) for comp_anom in component_anomalies.values()
            )
        }
        
        # Store anomaly if detected
        if anomalies['any_anomaly']:
            self.anomaly_history.append({
                'step': self.step_count,
                'epoch': self.epoch_count,
                'anomalies': anomalies,
                'losses': losses.copy()
            })
        
        return anomalies
    
    def _update_convergence_metrics(self, total_loss: float):
        """Update convergence tracking metrics."""
        
        # Update best loss
        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        
        # Calculate convergence metrics
        stats = self.total_loss_stats.get_statistics()
        trend = self.total_loss_stats.get_trend(self.config.trend_analysis_window)
        
        self.convergence_metrics = {
            'best_loss': self.best_loss,
            'steps_without_improvement': self.steps_without_improvement,
            'is_converging': trend['trend'] > 0 and trend['r_squared'] > 0.1,
            'convergence_rate': abs(trend['slope']) if trend['slope'] != 0 else 0.0,
            'plateau_detected': self.steps_without_improvement >= self.config.plateau_threshold,
            'loss_variance': stats.get('std', 0.0),
            'loss_stability': 1.0 / (1.0 + stats.get('std', 1.0))  # Higher = more stable
        }
    
    def _log_loss_summary(self, losses: Dict[str, float], anomalies: Dict[str, Any]):
        """Log loss summary for monitoring."""
        
        total_loss = losses.get('loss', losses.get('total_loss', 0.0))
        stats = self.total_loss_stats.get_statistics()
        
        logger.info(f"Step {self.step_count} - Loss Summary:")
        logger.info(f"  Total Loss: {total_loss:.6f} (mean: {stats.get('mean', 0):.6f}, std: {stats.get('std', 0):.6f})")
        logger.info(f"  Best Loss: {self.best_loss:.6f} ({self.steps_without_improvement} steps without improvement)")
        
        # Log smoothed losses
        if self.smoothed_losses:
            smoothed_total = self.smoothed_losses.get('loss', self.smoothed_losses.get('total_loss', 0.0))
            logger.info(f"  Smoothed Loss: {smoothed_total:.6f}")
        
        # Log component losses
        if self.config.track_components and len(self.component_stats) > 1:
            component_summary = []
            for name, stats_obj in self.component_stats.items():
                if name in losses:
                    comp_stats = stats_obj.get_statistics()
                    component_summary.append(f"{name}={losses[name]:.4f}({comp_stats.get('mean', 0):.4f})")
            
            if component_summary:
                logger.info(f"  Components: {', '.join(component_summary)}")
        
        # Log anomalies
        if anomalies.get('any_anomaly', False):
            anomaly_types = []
            total_anom = anomalies.get('total_loss', {})
            
            if total_anom.get('spike', False):
                anomaly_types.append("SPIKE")
            if total_anom.get('divergence', False):
                anomaly_types.append("DIVERGENCE") 
            if total_anom.get('nan_loss', False):
                anomaly_types.append("NaN")
            if total_anom.get('inf_loss', False):
                anomaly_types.append("INF")
            
            if anomaly_types:
                logger.warning(f"  🚨 Anomalies detected: {', '.join(anomaly_types)}")
        
        # Log convergence status
        if self.config.convergence_analysis and self.convergence_metrics:
            if self.convergence_metrics.get('plateau_detected', False):
                logger.warning(f"  📉 Plateau detected ({self.steps_without_improvement} steps)")
            elif self.convergence_metrics.get('is_converging', False):
                logger.info(f"  📈 Converging (rate: {self.convergence_metrics.get('convergence_rate', 0):.2e})")
    
    def get_loss_summary(self) -> Dict[str, Any]:
        """Get comprehensive loss summary."""
        
        summary = {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'total_loss_stats': self.total_loss_stats.get_statistics(),
            'smoothed_losses': self.smoothed_losses.copy(),
            'convergence_metrics': self.convergence_metrics.copy(),
            'anomaly_count': len(self.anomaly_history),
            'config': self.config.__dict__
        }
        
        # Add component statistics
        if self.config.track_components:
            component_summary = {}
            for name, stats in self.component_stats.items():
                component_summary[name] = stats.get_statistics()
            summary['component_stats'] = component_summary
        
        # Add trend analysis
        if len(self.loss_history) >= 10:
            trend = self.total_loss_stats.get_trend(self.config.trend_analysis_window)
            summary['trend_analysis'] = trend
        
        return summary
    
    def get_recent_losses(self, window: int = 100) -> List[Dict[str, Any]]:
        """Get recent loss history."""
        return self.loss_history[-window:] if self.loss_history else []
    
    def get_anomaly_report(self) -> Dict[str, Any]:
        """Get anomaly detection report."""
        
        if not self.anomaly_history:
            return {'total_anomalies': 0, 'anomaly_types': {}, 'recent_anomalies': []}
        
        # Count anomaly types
        anomaly_types = defaultdict(int)
        for anomaly_event in self.anomaly_history:
            total_anom = anomaly_event['anomalies'].get('total_loss', {})
            for anom_type, detected in total_anom.items():
                if detected:
                    anomaly_types[anom_type] += 1
        
        return {
            'total_anomalies': len(self.anomaly_history),
            'anomaly_types': dict(anomaly_types),
            'recent_anomalies': self.anomaly_history[-10:],  # Last 10 anomalies
            'anomaly_rate': len(self.anomaly_history) / max(1, self.step_count)
        }
    
    def analyze_loss_correlation(self) -> Dict[str, float]:
        """Analyze correlation between loss components."""
        
        if not self.config.correlation_analysis or len(self.loss_history) < 50:
            return {}
        
        # Extract loss values for correlation analysis
        recent_history = self.loss_history[-min(500, len(self.loss_history)):]
        
        loss_arrays = defaultdict(list)
        for entry in recent_history:
            for loss_name, loss_value in entry['losses'].items():
                loss_arrays[loss_name].append(loss_value)
        
        # Calculate correlations
        correlations = {}
        loss_names = list(loss_arrays.keys())
        
        for i, name1 in enumerate(loss_names):
            for name2 in loss_names[i+1:]:
                if len(loss_arrays[name1]) == len(loss_arrays[name2]) and len(loss_arrays[name1]) > 10:
                    corr = np.corrcoef(loss_arrays[name1], loss_arrays[name2])[0, 1]
                    if not np.isnan(corr):
                        correlations[f"{name1}_vs_{name2}"] = float(corr)
        
        return correlations
    
    def save_history(self, filepath: str):
        """Save loss history to file."""
        
        try:
            save_data = {
                'config': self.config.__dict__,
                'summary': self.get_loss_summary(),
                'recent_history': self.get_recent_losses(500),  # Last 500 steps
                'anomaly_report': self.get_anomaly_report(),
                'correlation_analysis': self.analyze_loss_correlation()
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            logger.debug(f"Saved loss history to {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save loss history: {e}")
    
    def load_history(self, filepath: str):
        """Load loss history from file."""
        
        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            # Restore recent history
            if 'recent_history' in save_data:
                self.loss_history = save_data['recent_history']
            
            # Restore summary data
            if 'summary' in save_data:
                summary = save_data['summary']
                self.step_count = summary.get('step_count', 0)
                self.epoch_count = summary.get('epoch_count', 0)
                self.smoothed_losses = summary.get('smoothed_losses', {})
                self.convergence_metrics = summary.get('convergence_metrics', {})
                
                # Restore best loss tracking
                if 'convergence_metrics' in summary:
                    conv_metrics = summary['convergence_metrics']
                    self.best_loss = conv_metrics.get('best_loss', float('inf'))
                    self.steps_without_improvement = conv_metrics.get('steps_without_improvement', 0)
            
            logger.info(f"Loaded loss history from {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to load loss history: {e}")
    
    def reset(self):
        """Reset loss tracking state."""
        
        self.total_loss_stats = LossStatistics(self.config.window_size)
        self.component_stats = {}
        self.step_count = 0
        self.epoch_count = 0
        self.loss_history = []
        self.anomaly_history = []
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.convergence_metrics = {}
        self.smoothed_losses = {}
        
        logger.info("Reset loss tracker state")


def create_loss_tracker(
    window_size: int = 1000,
    log_frequency: int = 50,
    anomaly_detection: bool = True,
    **kwargs
) -> LossTracker:
    """
    Convenience function to create loss tracker.
    
    Args:
        window_size: Size of tracking window
        log_frequency: Frequency of loss logging
        anomaly_detection: Enable anomaly detection
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured loss tracker
    """
    
    config = LossConfig(
        window_size=window_size,
        log_frequency=log_frequency,
        anomaly_detection=anomaly_detection,
        **kwargs
    )
    
    return LossTracker(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    tracker = create_loss_tracker(window_size=100, log_frequency=10)
    
    # Simulate training with loss updates
    import random
    
    for step in range(50):
        # Simulate losses
        base_loss = 2.0 * math.exp(-step / 20) + 0.1  # Decreasing loss
        noise = random.gauss(0, 0.05)
        
        loss_dict = {
            'loss': base_loss + noise,
            'loss_ce': base_loss * 0.6 + random.gauss(0, 0.02),
            'loss_bbox': base_loss * 0.3 + random.gauss(0, 0.01),
            'loss_giou': base_loss * 0.1 + random.gauss(0, 0.005)
        }
        
        tracker.update(loss_dict, step=step)
    
    # Get summary
    summary = tracker.get_loss_summary()
    print(f"Final summary: convergence={summary['convergence_metrics']}")
    
    # Get anomaly report
    anomaly_report = tracker.get_anomaly_report()
    print(f"Anomaly report: {anomaly_report['total_anomalies']} anomalies detected")