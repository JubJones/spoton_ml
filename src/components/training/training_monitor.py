"""
Advanced Training Monitor with Real-time Performance Tracking
Comprehensive monitoring system with MLflow integration for RF-DETR training
"""
import logging
import time
import json
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn
from threading import Thread, Event
import psutil

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.debug("GPUtil not available. GPU monitoring will use PyTorch fallback.")

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Logging will be local only.")

from ..models import ValidationEngine, ValidationResult, create_validation_engine
from ..models.surveillance_detector import create_surveillance_detector

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for advanced training monitoring."""
    
    # Monitoring intervals
    log_frequency: int = 50  # Steps between metric logging
    validation_frequency: int = 500  # Steps between validations
    checkpoint_frequency: int = 1000  # Steps between checkpoints
    
    # Performance monitoring
    system_monitoring: bool = True  # Monitor CPU, memory, GPU usage
    gradient_monitoring: bool = True  # Monitor gradient statistics
    activation_monitoring: bool = False  # Monitor layer activations
    
    # MLflow integration
    mlflow_enabled: bool = True
    experiment_name: str = "rfdetr_surveillance_training"
    run_name: Optional[str] = None
    mlflow_uri: Optional[str] = None
    
    # Real-time monitoring
    realtime_dashboard: bool = False  # Enable real-time web dashboard
    alert_system: bool = True  # Enable performance alerts
    
    # Storage configuration
    save_dir: Path = field(default_factory=lambda: Path("training_logs"))
    keep_n_checkpoints: int = 5
    save_optimizer_state: bool = True
    
    # Alert thresholds
    memory_alert_threshold: float = 0.9  # 90% memory usage
    gpu_memory_alert_threshold: float = 0.95  # 95% GPU memory
    loss_spike_threshold: float = 2.0  # Loss increase multiplier
    gradient_explosion_threshold: float = 10.0  # Gradient norm threshold
    
    # Validation configuration
    comprehensive_validation_frequency: int = 2000  # Steps for full validation
    early_stopping_patience: int = 10  # Validations without improvement
    early_stopping_min_delta: float = 0.001  # Minimum improvement threshold


@dataclass
class TrainingMetrics:
    """Training metrics for a single step."""
    
    step: int
    epoch: int
    timestamp: float
    
    # Loss metrics
    total_loss: float
    loss_components: Dict[str, float] = field(default_factory=dict)
    
    # Learning rate
    learning_rate: float = 0.0
    
    # Performance metrics
    batch_time: float = 0.0
    data_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    
    # System metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    
    # Gradient metrics
    grad_norm: float = 0.0
    grad_max: float = 0.0
    grad_min: float = 0.0
    
    # Model metrics
    param_norm: float = 0.0
    ema_decay: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class SystemAlert:
    """System alert for performance monitoring."""
    
    timestamp: float
    alert_type: str
    severity: str  # "warning", "error", "critical"
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self._stop_event = Event()
        self._monitor_thread = None
        self.current_metrics = {}
        
    def start(self):
        """Start system monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self._stop_event.clear()
            self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("System monitoring started")
    
    def stop(self):
        """Stop system monitoring."""
        if self.monitoring:
            self.monitoring = False
            self._stop_event.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=2.0)
            logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.update_interval):
            try:
                self.current_metrics = self._collect_system_metrics()
            except Exception as e:
                logger.warning(f"System monitoring error: {e}")
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        # CPU and memory
        metrics['cpu_usage'] = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        metrics['memory_usage'] = memory.percent / 100.0
        metrics['memory_available'] = memory.available / (1024**3)  # GB
        
        # GPU metrics if available
        try:
            if GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    metrics['gpu_usage'] = gpu.load
                    metrics['gpu_memory'] = gpu.memoryUtil
                    metrics['gpu_temperature'] = gpu.temperature
            elif torch.cuda.is_available():
                # Fallback to PyTorch CUDA info
                max_memory = torch.cuda.max_memory_allocated()
                current_memory = torch.cuda.memory_allocated()
                if max_memory > 0:
                    metrics['gpu_memory'] = current_memory / max_memory
                else:
                    metrics['gpu_memory'] = 0.0
                metrics['gpu_usage'] = 0.5  # Default assumption
        except Exception as e:
            logger.debug(f"GPU monitoring failed: {e}")
        
        return metrics
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        return self.current_metrics.copy()


class TrainingMonitor:
    """Advanced training monitor with real-time performance tracking."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.system_monitor = SystemMonitor() if config.system_monitoring else None
        self.validation_engine = None  # Will be set up when detector is available
        
        # Training state
        self.training_metrics = deque(maxlen=1000)  # Keep last 1000 metrics
        self.validation_results = []
        self.alerts = []
        
        # Performance tracking
        self.best_metrics = {}
        self.step_times = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        
        # MLflow setup
        self.mlflow_run = None
        if config.mlflow_enabled and MLFLOW_AVAILABLE:
            self._setup_mlflow()
        
        # Alert thresholds
        self.alert_cooldowns = defaultdict(float)  # Prevent alert spam
        
        logger.info(f"Training Monitor initialized:")
        logger.info(f"  Log frequency: {config.log_frequency}")
        logger.info(f"  Validation frequency: {config.validation_frequency}")
        logger.info(f"  MLflow enabled: {config.mlflow_enabled and MLFLOW_AVAILABLE}")
        logger.info(f"  System monitoring: {config.system_monitoring}")
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            if self.config.mlflow_uri:
                mlflow.set_tracking_uri(self.config.mlflow_uri)
            
            mlflow.set_experiment(self.config.experiment_name)
            
            # Start run
            run_name = self.config.run_name or f"rfdetr_run_{int(time.time())}"
            self.mlflow_run = mlflow.start_run(run_name=run_name)
            
            # Log configuration
            mlflow.log_params(asdict(self.config))
            
            logger.info(f"MLflow tracking initialized: {self.mlflow_run.info.run_id}")
            
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            self.config.mlflow_enabled = False
    
    def setup_validation(self, model: nn.Module, validation_loader, scene_loaders=None):
        """Setup validation engine."""
        if self.validation_engine is None:
            detector = create_surveillance_detector()
            self.validation_engine = create_validation_engine(
                detector=detector,
                validation_frequency=self.config.validation_frequency,
                scene_specific_validation=scene_loaders is not None,
                save_dir=self.config.save_dir / "validation"
            )
        
        self.validation_loader = validation_loader
        self.scene_loaders = scene_loaders or {}
    
    def start_training(self):
        """Start training monitoring."""
        if self.system_monitor:
            self.system_monitor.start()
        
        self.training_start_time = time.time()
        logger.info("Training monitoring started")
    
    def stop_training(self):
        """Stop training monitoring."""
        if self.system_monitor:
            self.system_monitor.stop()
        
        if self.mlflow_run:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"MLflow end run failed: {e}")
        
        training_duration = time.time() - getattr(self, 'training_start_time', time.time())
        logger.info(f"Training monitoring stopped. Duration: {training_duration:.2f}s")
    
    def log_step(
        self,
        step: int,
        epoch: int,
        total_loss: float,
        loss_components: Dict[str, float],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_time: float = 0.0,
        data_time: float = 0.0,
        forward_time: float = 0.0,
        backward_time: float = 0.0
    ) -> TrainingMetrics:
        """Log training step metrics."""
        
        # Create training metrics
        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            timestamp=time.time(),
            total_loss=total_loss,
            loss_components=loss_components.copy(),
            learning_rate=self._get_learning_rate(optimizer),
            batch_time=batch_time,
            data_time=data_time,
            forward_time=forward_time,
            backward_time=backward_time
        )
        
        # Add system metrics
        if self.system_monitor:
            system_metrics = self.system_monitor.get_current_metrics()
            metrics.cpu_usage = system_metrics.get('cpu_usage', 0.0)
            metrics.memory_usage = system_metrics.get('memory_usage', 0.0)
            metrics.gpu_usage = system_metrics.get('gpu_usage', 0.0)
            metrics.gpu_memory = system_metrics.get('gpu_memory', 0.0)
        
        # Add gradient metrics
        if self.config.gradient_monitoring:
            grad_stats = self._compute_gradient_stats(model)
            metrics.grad_norm = grad_stats.get('grad_norm', 0.0)
            metrics.grad_max = grad_stats.get('grad_max', 0.0)
            metrics.grad_min = grad_stats.get('grad_min', 0.0)
        
        # Add model metrics
        model_stats = self._compute_model_stats(model)
        metrics.param_norm = model_stats.get('param_norm', 0.0)
        
        # Store metrics
        self.training_metrics.append(metrics)
        self.loss_history.append(total_loss)
        self.step_times.append(batch_time)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Log to MLflow
        if self.config.mlflow_enabled and step % self.config.log_frequency == 0:
            self._log_to_mlflow(metrics)
        
        # Log to console
        if step % self.config.log_frequency == 0:
            self._log_to_console(metrics)
        
        return metrics
    
    def validate(
        self,
        model: nn.Module,
        step: int,
        epoch: int,
        validation_type: str = "full"
    ) -> Optional[ValidationResult]:
        """Run validation."""
        if self.validation_engine is None or self.validation_loader is None:
            logger.warning("Validation not configured")
            return None
        
        try:
            # Determine validation type based on frequency
            if step % self.config.comprehensive_validation_frequency == 0:
                validation_type = "comprehensive"
            
            result = self.validation_engine.validate(
                model=model,
                validation_loader=self.validation_loader,
                step=step,
                epoch=epoch,
                validation_type=validation_type,
                scene_loaders=self.scene_loaders
            )
            
            self.validation_results.append(result)
            
            # Update best metrics
            for metric, value in result.metrics.items():
                if metric not in self.best_metrics or value > self.best_metrics[metric]:
                    self.best_metrics[metric] = value
            
            # Log validation results
            self._log_validation_results(result)
            
            # Check early stopping
            if self._should_stop_early():
                logger.warning("Early stopping criteria met")
                return result
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return None
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        scheduler=None,
        **kwargs
    ) -> Path:
        """Save training checkpoint."""
        
        checkpoint_dir = self.config.save_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pth"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_metrics': self.best_metrics.copy(),
            'config': asdict(self.config),
            'timestamp': time.time()
        }
        
        # Add optimizer state if configured
        if self.config.save_optimizer_state:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add custom data
        checkpoint_data.update(kwargs)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints(checkpoint_dir)
        
        # Log to MLflow
        if self.config.mlflow_enabled:
            try:
                mlflow.log_artifact(str(checkpoint_path), "checkpoints")
            except Exception as e:
                logger.warning(f"MLflow checkpoint logging failed: {e}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def _get_learning_rate(self, optimizer: torch.optim.Optimizer) -> float:
        """Get current learning rate."""
        if optimizer.param_groups:
            return optimizer.param_groups[0]['lr']
        return 0.0
    
    def _compute_gradient_stats(self, model: nn.Module) -> Dict[str, float]:
        """Compute gradient statistics."""
        grad_norms = []
        grad_values = []
        
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm().item()
                grad_norms.append(grad_norm)
                grad_values.extend(param.grad.data.flatten().cpu().numpy())
        
        if not grad_values:
            return {'grad_norm': 0.0, 'grad_max': 0.0, 'grad_min': 0.0}
        
        grad_values = np.array(grad_values)
        
        return {
            'grad_norm': np.sqrt(sum(norm**2 for norm in grad_norms)),
            'grad_max': float(np.max(grad_values)),
            'grad_min': float(np.min(grad_values)),
            'grad_mean': float(np.mean(grad_values)),
            'grad_std': float(np.std(grad_values))
        }
    
    def _compute_model_stats(self, model: nn.Module) -> Dict[str, float]:
        """Compute model parameter statistics."""
        param_values = []
        
        for param in model.parameters():
            param_values.extend(param.data.flatten().cpu().numpy())
        
        if not param_values:
            return {'param_norm': 0.0}
        
        param_values = np.array(param_values)
        
        return {
            'param_norm': float(np.linalg.norm(param_values)),
            'param_mean': float(np.mean(param_values)),
            'param_std': float(np.std(param_values)),
            'param_max': float(np.max(param_values)),
            'param_min': float(np.min(param_values))
        }
    
    def _check_alerts(self, metrics: TrainingMetrics):
        """Check for performance alerts."""
        current_time = time.time()
        alerts = []
        
        # Memory alerts
        if metrics.memory_usage > self.config.memory_alert_threshold:
            if current_time - self.alert_cooldowns['memory'] > 300:  # 5 min cooldown
                alerts.append(SystemAlert(
                    timestamp=current_time,
                    alert_type="memory",
                    severity="warning",
                    message=f"High memory usage: {metrics.memory_usage:.1%}",
                    metrics={'memory_usage': metrics.memory_usage}
                ))
                self.alert_cooldowns['memory'] = current_time
        
        # GPU memory alerts
        if metrics.gpu_memory > self.config.gpu_memory_alert_threshold:
            if current_time - self.alert_cooldowns['gpu_memory'] > 300:
                alerts.append(SystemAlert(
                    timestamp=current_time,
                    alert_type="gpu_memory",
                    severity="warning",
                    message=f"High GPU memory usage: {metrics.gpu_memory:.1%}",
                    metrics={'gpu_memory': metrics.gpu_memory}
                ))
                self.alert_cooldowns['gpu_memory'] = current_time
        
        # Loss spike alerts
        if len(self.loss_history) > 10:
            recent_avg = np.mean(list(self.loss_history)[-10:])
            current_loss = metrics.total_loss
            
            if current_loss > recent_avg * self.config.loss_spike_threshold:
                if current_time - self.alert_cooldowns['loss_spike'] > 60:  # 1 min cooldown
                    alerts.append(SystemAlert(
                        timestamp=current_time,
                        alert_type="loss_spike",
                        severity="error",
                        message=f"Loss spike detected: {current_loss:.4f} vs avg {recent_avg:.4f}",
                        metrics={'current_loss': current_loss, 'recent_avg': recent_avg}
                    ))
                    self.alert_cooldowns['loss_spike'] = current_time
        
        # Gradient explosion alerts
        if metrics.grad_norm > self.config.gradient_explosion_threshold:
            if current_time - self.alert_cooldowns['grad_explosion'] > 60:
                alerts.append(SystemAlert(
                    timestamp=current_time,
                    alert_type="gradient_explosion",
                    severity="error",
                    message=f"Gradient explosion: norm {metrics.grad_norm:.2f}",
                    metrics={'grad_norm': metrics.grad_norm}
                ))
                self.alert_cooldowns['grad_explosion'] = current_time
        
        # Store and log alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"ALERT [{alert.severity}] {alert.alert_type}: {alert.message}")
    
    def _log_to_mlflow(self, metrics: TrainingMetrics):
        """Log metrics to MLflow."""
        try:
            # Core metrics
            mlflow.log_metrics({
                'loss/total': metrics.total_loss,
                'learning_rate': metrics.learning_rate,
                'performance/batch_time': metrics.batch_time,
                'performance/data_time': metrics.data_time,
                'performance/forward_time': metrics.forward_time,
                'performance/backward_time': metrics.backward_time,
                'system/cpu_usage': metrics.cpu_usage,
                'system/memory_usage': metrics.memory_usage,
                'system/gpu_usage': metrics.gpu_usage,
                'system/gpu_memory': metrics.gpu_memory,
                'gradients/norm': metrics.grad_norm,
                'gradients/max': metrics.grad_max,
                'gradients/min': metrics.grad_min,
                'parameters/norm': metrics.param_norm
            }, step=metrics.step)
            
            # Loss components
            for component, value in metrics.loss_components.items():
                mlflow.log_metric(f'loss/{component}', value, step=metrics.step)
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
    
    def _log_to_console(self, metrics: TrainingMetrics):
        """Log metrics to console."""
        # Calculate performance statistics
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        recent_loss = np.mean(list(self.loss_history)[-10:]) if len(self.loss_history) >= 10 else metrics.total_loss
        
        logger.info(
            f"Step {metrics.step:6d} | "
            f"Epoch {metrics.epoch:3d} | "
            f"Loss: {metrics.total_loss:.4f} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Time: {metrics.batch_time:.3f}s | "
            f"GPU: {metrics.gpu_memory:.1%} | "
            f"GradNorm: {metrics.grad_norm:.3f}"
        )
    
    def _log_validation_results(self, result: ValidationResult):
        """Log validation results."""
        logger.info(f"Validation Step {result.step}:")
        logger.info(f"  mAP: {result.metrics.get('mAP', 0):.4f}")
        logger.info(f"  Inference time: {result.inference_stats.get('avg_time', 0):.2f}ms")
        
        # Log to MLflow
        if self.config.mlflow_enabled:
            try:
                validation_metrics = {}
                for metric, value in result.metrics.items():
                    validation_metrics[f'validation/{metric}'] = value
                
                # Performance metrics
                validation_metrics['validation/inference_time'] = result.inference_stats.get('avg_time', 0)
                validation_metrics['validation/memory_usage'] = result.memory_stats.get('peak_memory', 0)
                
                mlflow.log_metrics(validation_metrics, step=result.step)
                
            except Exception as e:
                logger.warning(f"MLflow validation logging failed: {e}")
    
    def _should_stop_early(self) -> bool:
        """Check early stopping criteria."""
        if len(self.validation_results) < self.config.early_stopping_patience:
            return False
        
        # Check if validation performance has plateaued
        recent_results = self.validation_results[-self.config.early_stopping_patience:]
        
        # Use mAP as primary metric for early stopping
        recent_maps = [r.metrics.get('mAP', 0) for r in recent_results]
        
        if not recent_maps:
            return False
        
        # Check if there's been no significant improvement
        best_recent = max(recent_maps)
        current = recent_maps[-1]
        
        # No improvement if current performance is not within min_delta of best recent
        no_improvement = (best_recent - current) > self.config.early_stopping_min_delta
        
        if no_improvement:
            logger.info(f"Early stopping: no improvement in {self.config.early_stopping_patience} validations")
            return True
        
        return False
    
    def _cleanup_checkpoints(self, checkpoint_dir: Path):
        """Clean up old checkpoints."""
        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pth"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only the most recent N checkpoints
        for old_checkpoint in checkpoints[self.config.keep_n_checkpoints:]:
            try:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.training_metrics:
            return {}
        
        recent_metrics = list(self.training_metrics)[-100:]  # Last 100 steps
        
        summary = {
            'total_steps': len(self.training_metrics),
            'current_step': self.training_metrics[-1].step if self.training_metrics else 0,
            'total_validations': len(self.validation_results),
            'best_metrics': self.best_metrics.copy(),
            'recent_performance': {
                'avg_loss': np.mean([m.total_loss for m in recent_metrics]),
                'avg_step_time': np.mean([m.batch_time for m in recent_metrics]),
                'avg_gpu_usage': np.mean([m.gpu_usage for m in recent_metrics]),
                'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics])
            },
            'alerts': {
                'total_alerts': len(self.alerts),
                'unresolved_alerts': len([a for a in self.alerts if not a.resolved]),
                'recent_alerts': [asdict(a) for a in self.alerts[-10:]]  # Last 10 alerts
            }
        }
        
        return summary
    
    def __enter__(self):
        """Context manager entry."""
        self.start_training()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_training()


def create_training_monitor(
    log_frequency: int = 50,
    validation_frequency: int = 500,
    mlflow_enabled: bool = True,
    experiment_name: str = "rfdetr_surveillance_training",
    system_monitoring: bool = True,
    save_dir: Optional[Path] = None,
    **kwargs
) -> TrainingMonitor:
    """
    Create training monitor with configuration.
    
    Args:
        log_frequency: Steps between metric logging
        validation_frequency: Steps between validations
        mlflow_enabled: Enable MLflow tracking
        experiment_name: MLflow experiment name
        system_monitoring: Enable system resource monitoring
        save_dir: Directory for saving logs and checkpoints
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured training monitor
    """
    
    config = MonitoringConfig(
        log_frequency=log_frequency,
        validation_frequency=validation_frequency,
        mlflow_enabled=mlflow_enabled,
        experiment_name=experiment_name,
        system_monitoring=system_monitoring,
        save_dir=save_dir or Path("training_logs"),
        **kwargs
    )
    
    return TrainingMonitor(config)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Advanced Training Monitor")
    
    # Test monitor creation
    monitor = create_training_monitor(
        log_frequency=10,
        validation_frequency=50,
        mlflow_enabled=False,  # Disable for testing
        system_monitoring=True
    )
    
    print(f"✅ Training monitor created")
    
    # Mock model and optimizer
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Test monitoring
    try:
        with monitor:
            for step in range(5):
                # Simulate training step
                loss = 1.0 / (step + 1)  # Decreasing loss
                loss_components = {'mse': loss * 0.8, 'reg': loss * 0.2}
                
                metrics = monitor.log_step(
                    step=step,
                    epoch=0,
                    total_loss=loss,
                    loss_components=loss_components,
                    model=model,
                    optimizer=optimizer,
                    batch_time=0.1,
                    data_time=0.02
                )
                
                print(f"  Step {step}: Loss {metrics.total_loss:.4f}, "
                      f"GPU: {metrics.gpu_memory:.1%}, "
                      f"GradNorm: {metrics.grad_norm:.3f}")
                
                time.sleep(0.1)  # Simulate training time
        
        print("✅ Training monitoring test completed")
        
        # Test summary
        summary = monitor.get_training_summary()
        print(f"📊 Training Summary:")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Average loss: {summary['recent_performance']['avg_loss']:.4f}")
        print(f"  Total alerts: {summary['alerts']['total_alerts']}")
        
    except Exception as e:
        print(f"❌ Training monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Advanced Training Monitor testing completed")