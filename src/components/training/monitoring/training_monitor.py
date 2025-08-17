"""
Training Monitor for RF-DETR Training
Orchestrates EMA, loss tracking, and comprehensive monitoring
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Callable, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

from .ema_handler import EMAHandler, EMAConfig
from .loss_tracker import LossTracker, LossConfig
from .metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for comprehensive training monitoring."""
    
    # EMA configuration
    ema_enabled: bool = True
    ema_config: EMAConfig = field(default_factory=lambda: EMAConfig())
    
    # Loss tracking configuration
    loss_tracking_enabled: bool = True
    loss_config: LossConfig = field(default_factory=lambda: LossConfig())
    
    # Metrics logging configuration
    metrics_logging_enabled: bool = True
    metrics_log_frequency: int = 100
    
    # Checkpoint configuration
    checkpoint_enabled: bool = True
    checkpoint_frequency: int = 1000
    checkpoint_path: Optional[str] = "checkpoints/training_monitor"
    
    # Validation configuration
    validation_enabled: bool = True
    validation_frequency: int = 500
    validation_metric: str = "loss"  # Primary metric for tracking
    
    # Performance monitoring
    performance_monitoring: bool = True
    memory_monitoring: bool = True
    timing_enabled: bool = True
    
    # Early stopping configuration
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 1e-4
    
    def __post_init__(self):
        """Validate configuration."""
        if self.checkpoint_frequency <= 0:
            raise ValueError(f"checkpoint_frequency must be positive, got {self.checkpoint_frequency}")
        if self.validation_frequency <= 0:
            raise ValueError(f"validation_frequency must be positive, got {self.validation_frequency}")


class TrainingMonitor:
    """
    Comprehensive training monitor for RF-DETR training.
    Orchestrates EMA, loss tracking, metrics logging, and performance monitoring.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MonitorConfig,
        save_dir: Optional[str] = None
    ):
        """
        Initialize training monitor.
        
        Args:
            model: RF-DETR model to monitor
            config: Monitoring configuration
            save_dir: Directory for saving monitoring outputs
        """
        self.config = config
        self.model = model
        self.save_dir = Path(save_dir) if save_dir else Path("monitoring_outputs")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.ema_handler = None
        self.loss_tracker = None
        self.metrics_logger = None
        
        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.training_start_time = None
        self.epoch_start_time = None
        
        # Performance tracking
        self.step_times = []
        self.memory_usage = []
        self.validation_history = []
        
        # Early stopping state
        self.best_metric = None
        self.patience_counter = 0
        self.early_stopped = False
        
        # Initialize enabled components
        self._initialize_components()
        
        logger.info(f"Initialized TrainingMonitor with save_dir={self.save_dir}")
        logger.info(f"  EMA: {'enabled' if self.ema_handler else 'disabled'}")
        logger.info(f"  Loss tracking: {'enabled' if self.loss_tracker else 'disabled'}")
        logger.info(f"  Metrics logging: {'enabled' if self.metrics_logger else 'disabled'}")
    
    def _initialize_components(self):
        """Initialize monitoring components based on configuration."""
        
        # Initialize EMA handler
        if self.config.ema_enabled:
            self.ema_handler = EMAHandler(self.model, self.config.ema_config)
            logger.info("EMA handler initialized")
        
        # Initialize loss tracker
        if self.config.loss_tracking_enabled:
            self.loss_tracker = LossTracker(self.config.loss_config)
            logger.info("Loss tracker initialized")
        
        # Initialize metrics logger
        if self.config.metrics_logging_enabled:
            log_dir = self.save_dir / "metrics"
            from .metrics_logger import LoggerConfig
            logger_config = LoggerConfig(log_frequency=self.config.metrics_log_frequency)
            self.metrics_logger = MetricsLogger(
                log_dir=str(log_dir),
                config=logger_config
            )
            logger.info(f"Metrics logger initialized with log_dir={log_dir}")
    
    def start_training(self):
        """Mark the start of training."""
        self.training_start_time = time.time()
        self.step_count = 0
        self.epoch_count = 0
        
        logger.info("Training monitoring started")
        
        if self.metrics_logger:
            self.metrics_logger.log_event("training_start", {
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "device": str(next(self.model.parameters()).device)
            })
    
    def start_epoch(self, epoch: int):
        """Mark the start of an epoch."""
        self.epoch_count = epoch
        self.epoch_start_time = time.time()
        
        if self.metrics_logger and epoch % 10 == 0:
            logger.info(f"Starting epoch {epoch}")
    
    def end_epoch(self, epoch_metrics: Optional[Dict[str, Any]] = None):
        """Mark the end of an epoch."""
        epoch_duration = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        # Log epoch summary
        if self.metrics_logger:
            epoch_summary = {
                "epoch": self.epoch_count,
                "duration": epoch_duration,
                "steps": self.step_count
            }
            
            if epoch_metrics:
                epoch_summary.update(epoch_metrics)
            
            self.metrics_logger.log_metrics(epoch_summary, step=self.step_count)
        
        # Log epoch completion
        if self.epoch_count % 10 == 0 or self.epoch_count < 5:
            logger.info(f"Completed epoch {self.epoch_count} in {epoch_duration:.2f}s")
    
    def step(
        self,
        loss_dict: Dict[str, Union[torch.Tensor, float]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Update monitoring for a training step.
        
        Args:
            loss_dict: Dictionary of loss components
            step: Current training step
            epoch: Current epoch
            additional_metrics: Additional metrics to log
        """
        step_start_time = time.time()
        
        # Update step/epoch counters
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
        
        if epoch is not None:
            self.epoch_count = epoch
        
        # Update EMA
        if self.ema_handler:
            self.ema_handler.update(self.step_count)
        
        # Update loss tracking
        if self.loss_tracker:
            self.loss_tracker.update(loss_dict, self.step_count, self.epoch_count)
        
        # Performance monitoring
        if self.config.performance_monitoring:
            self._monitor_performance(step_start_time)
        
        # Memory monitoring
        if self.config.memory_monitoring:
            self._monitor_memory()
        
        # Log metrics
        if self.metrics_logger and self.step_count % self.config.metrics_log_frequency == 0:
            self._log_step_metrics(loss_dict, additional_metrics)
        
        # Checkpoint
        if (self.config.checkpoint_enabled and 
            self.step_count % self.config.checkpoint_frequency == 0):
            self.save_checkpoint()
    
    def _monitor_performance(self, step_start_time: float):
        """Monitor step performance timing."""
        step_duration = time.time() - step_start_time
        self.step_times.append(step_duration)
        
        # Keep only recent step times
        if len(self.step_times) > 1000:
            self.step_times = self.step_times[-500:]
    
    def _monitor_memory(self):
        """Monitor memory usage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            
            self.memory_usage.append({
                "step": self.step_count,
                "allocated": memory_allocated,
                "reserved": memory_reserved
            })
            
            # Keep only recent memory measurements
            if len(self.memory_usage) > 1000:
                self.memory_usage = self.memory_usage[-500:]
    
    def _log_step_metrics(
        self,
        loss_dict: Dict[str, Union[torch.Tensor, float]],
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log step-level metrics."""
        
        # Prepare metrics dictionary
        metrics = {
            "step": self.step_count,
            "epoch": self.epoch_count
        }
        
        # Add loss metrics
        total_loss = None
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                metrics[f"loss/{key}"] = value.item()
                if key in ["loss", "total_loss"]:
                    total_loss = value.item()
            else:
                metrics[f"loss/{key}"] = float(value)
                if key in ["loss", "total_loss"]:
                    total_loss = float(value)
        
        # Add loss tracker statistics
        if self.loss_tracker:
            loss_summary = self.loss_tracker.get_loss_summary()
            if loss_summary.get('total_loss_stats'):
                stats = loss_summary['total_loss_stats']
                metrics.update({
                    "loss_stats/mean": stats.get('mean', 0),
                    "loss_stats/std": stats.get('std', 0),
                    "loss_stats/min": stats.get('min', 0),
                    "loss_stats/max": stats.get('max', 0)
                })
            
            if loss_summary.get('convergence_metrics'):
                conv_metrics = loss_summary['convergence_metrics']
                metrics.update({
                    "convergence/is_converging": conv_metrics.get('is_converging', False),
                    "convergence/steps_without_improvement": conv_metrics.get('steps_without_improvement', 0),
                    "convergence/loss_stability": conv_metrics.get('loss_stability', 0)
                })
        
        # Add EMA statistics
        if self.ema_handler:
            ema_stats = self.ema_handler.get_ema_statistics()
            metrics.update({
                "ema/current_decay": ema_stats.get('current_decay', 0),
                "ema/update_count": ema_stats.get('update_count', 0),
                "ema/updates_per_step": ema_stats.get('updates_per_step', 0)
            })
        
        # Add performance metrics
        if self.step_times:
            recent_times = self.step_times[-100:]  # Last 100 steps
            metrics.update({
                "performance/step_time_mean": sum(recent_times) / len(recent_times),
                "performance/step_time_max": max(recent_times),
                "performance/steps_per_second": 1.0 / (sum(recent_times) / len(recent_times))
            })
        
        # Add memory metrics
        if self.memory_usage and torch.cuda.is_available():
            recent_memory = self.memory_usage[-1]
            metrics.update({
                "memory/allocated_mb": recent_memory["allocated"] / 1024 / 1024,
                "memory/reserved_mb": recent_memory["reserved"] / 1024 / 1024
            })
        
        # Add additional metrics
        if additional_metrics:
            for key, value in additional_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[f"custom/{key}"] = value.item()
                else:
                    metrics[f"custom/{key}"] = value
        
        # Log to metrics logger
        self.metrics_logger.log_metrics(metrics, step=self.step_count)
    
    def validate(
        self,
        validation_fn: Callable,
        *args,
        validation_metrics: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform validation and update monitoring.
        
        Args:
            validation_fn: Function that performs validation
            *args, **kwargs: Arguments for validation function
            validation_metrics: Additional validation metrics
            
        Returns:
            Validation results
        """
        
        if not self.config.validation_enabled:
            return {}
        
        logger.info(f"Running validation at step {self.step_count}")
        
        validation_start_time = time.time()
        
        # Run validation on source model
        self.model.eval()
        with torch.no_grad():
            source_results = validation_fn(self.model, *args, **kwargs)
        
        # Run validation on EMA model if available
        ema_results = {}
        if self.ema_handler:
            ema_model = self.ema_handler.get_ema_model()
            ema_model.eval()
            with torch.no_grad():
                ema_results = validation_fn(ema_model, *args, **kwargs)
        
        validation_duration = time.time() - validation_start_time
        
        # Compile validation summary
        validation_summary = {
            "step": self.step_count,
            "epoch": self.epoch_count,
            "duration": validation_duration,
            "source_results": source_results,
            "ema_results": ema_results
        }
        
        if validation_metrics:
            validation_summary["additional_metrics"] = validation_metrics
        
        # Store validation history
        self.validation_history.append(validation_summary)
        
        # Keep validation history manageable
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-50:]
        
        # Check for early stopping
        if self.config.early_stopping_enabled:
            self._check_early_stopping(source_results)
        
        # Log validation metrics
        if self.metrics_logger:
            val_metrics = {}
            
            # Log source model metrics
            for key, value in source_results.items():
                if isinstance(value, (int, float)):
                    val_metrics[f"validation/source_{key}"] = value
            
            # Log EMA model metrics
            for key, value in ema_results.items():
                if isinstance(value, (int, float)):
                    val_metrics[f"validation/ema_{key}"] = value
            
            val_metrics["validation/duration"] = validation_duration
            
            if validation_metrics:
                for key, value in validation_metrics.items():
                    if isinstance(value, (int, float)):
                        val_metrics[f"validation/{key}"] = value
            
            self.metrics_logger.log_metrics(val_metrics, step=self.step_count)
        
        logger.info(f"Validation completed in {validation_duration:.2f}s")
        
        return validation_summary
    
    def _check_early_stopping(self, validation_results: Dict[str, Any]):
        """Check if early stopping criteria are met."""
        
        # Get validation metric
        validation_metric = validation_results.get(self.config.validation_metric)
        
        if validation_metric is None:
            return
        
        # Initialize best metric
        if self.best_metric is None:
            self.best_metric = validation_metric
            self.patience_counter = 0
            return
        
        # Check for improvement
        if validation_metric < self.best_metric - self.config.early_stopping_threshold:
            self.best_metric = validation_metric
            self.patience_counter = 0
            logger.info(f"New best validation {self.config.validation_metric}: {validation_metric:.6f}")
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= self.config.early_stopping_patience:
                self.early_stopped = True
                logger.info(f"Early stopping triggered after {self.patience_counter} validations "
                           f"without improvement (best: {self.best_metric:.6f})")
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early."""
        return self.early_stopped
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        
        summary = {
            "step_count": self.step_count,
            "epoch_count": self.epoch_count,
            "training_duration": time.time() - self.training_start_time if self.training_start_time else 0,
            "early_stopped": self.early_stopped,
            "config": self.config.__dict__
        }
        
        # Add loss tracker summary
        if self.loss_tracker:
            summary["loss_tracking"] = self.loss_tracker.get_loss_summary()
        
        # Add EMA statistics
        if self.ema_handler:
            summary["ema_statistics"] = self.ema_handler.get_ema_statistics()
        
        # Add performance statistics
        if self.step_times:
            recent_times = self.step_times[-1000:]
            summary["performance"] = {
                "mean_step_time": sum(recent_times) / len(recent_times),
                "total_steps": len(self.step_times),
                "steps_per_second": len(recent_times) / sum(recent_times) if recent_times else 0
            }
        
        # Add validation history summary
        if self.validation_history:
            summary["validation_summary"] = {
                "total_validations": len(self.validation_history),
                "best_metric": self.best_metric,
                "patience_counter": self.patience_counter
            }
        
        return summary
    
    def save_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Save monitoring checkpoint."""
        
        if not self.config.checkpoint_enabled:
            return
        
        if checkpoint_path is None:
            checkpoint_dir = self.save_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoint_dir / f"monitor_checkpoint_step_{self.step_count}.json"
        
        checkpoint_data = {
            "step_count": self.step_count,
            "epoch_count": self.epoch_count,
            "config": self.config.__dict__,
            "monitoring_summary": self.get_monitoring_summary(),
            "validation_history": self.validation_history[-10:],  # Last 10 validations
        }
        
        # Add component states
        if self.ema_handler:
            checkpoint_data["ema_state"] = self.ema_handler.save_state()
        
        if self.loss_tracker:
            checkpoint_data["loss_summary"] = self.loss_tracker.get_loss_summary()
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            logger.debug(f"Saved monitoring checkpoint to {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save monitoring checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load monitoring checkpoint."""
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.step_count = checkpoint_data.get('step_count', 0)
            self.epoch_count = checkpoint_data.get('epoch_count', 0)
            
            if 'validation_history' in checkpoint_data:
                self.validation_history = checkpoint_data['validation_history']
            
            # Restore component states
            if self.ema_handler and 'ema_state' in checkpoint_data:
                self.ema_handler.load_state(checkpoint_data['ema_state'])
            
            logger.info(f"Loaded monitoring checkpoint from {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load monitoring checkpoint: {e}")
    
    def finalize_training(self):
        """Finalize training monitoring."""
        
        training_duration = time.time() - self.training_start_time if self.training_start_time else 0
        
        # Generate final summary
        final_summary = self.get_monitoring_summary()
        final_summary["training_completed"] = True
        final_summary["total_training_duration"] = training_duration
        
        # Save final summary
        summary_path = self.save_dir / "training_summary.json"
        try:
            with open(summary_path, 'w') as f:
                json.dump(final_summary, f, indent=2, default=str)
            
            logger.info(f"Saved final training summary to {summary_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save training summary: {e}")
        
        # Log final statistics
        logger.info(f"Training monitoring completed:")
        logger.info(f"  Total duration: {training_duration:.2f}s")
        logger.info(f"  Total steps: {self.step_count}")
        logger.info(f"  Total epochs: {self.epoch_count}")
        logger.info(f"  Early stopped: {self.early_stopped}")
        
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            logger.info(f"  Average step time: {avg_step_time:.4f}s")
        
        # Finalize components
        if self.metrics_logger:
            self.metrics_logger.close()
    
    def __enter__(self):
        """Context manager entry."""
        self.start_training()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize_training()


def create_training_monitor(
    model: nn.Module,
    save_dir: Optional[str] = None,
    ema_enabled: bool = True,
    loss_tracking_enabled: bool = True,
    **kwargs
) -> TrainingMonitor:
    """
    Convenience function to create training monitor.
    
    Args:
        model: RF-DETR model
        save_dir: Directory for saving outputs
        ema_enabled: Enable EMA monitoring
        loss_tracking_enabled: Enable loss tracking
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured training monitor
    """
    
    config = MonitorConfig(
        ema_enabled=ema_enabled,
        loss_tracking_enabled=loss_tracking_enabled,
        **kwargs
    )
    
    return TrainingMonitor(model, config, save_dir)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock model for testing
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = MockModel()
    
    # Create training monitor
    monitor = create_training_monitor(
        model=model,
        save_dir="test_monitoring",
        validation_frequency=10
    )
    
    # Simulate training
    with monitor:
        for epoch in range(3):
            monitor.start_epoch(epoch)
            
            for step in range(20):
                loss_dict = {
                    'loss': 1.0 - step * 0.01,
                    'loss_ce': 0.6 - step * 0.005,
                    'loss_bbox': 0.3 - step * 0.003
                }
                
                monitor.step(loss_dict)
                
                # Simulate validation
                if step % 10 == 0:
                    def mock_validation(model):
                        return {"accuracy": 0.8 + step * 0.01}
                    
                    monitor.validate(mock_validation)
            
            monitor.end_epoch({"epoch_loss": 0.5})
    
    print("Training monitoring completed successfully")