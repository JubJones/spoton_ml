"""
Exponential Moving Average (EMA) Handler for RF-DETR Training
Advanced EMA implementation with decay scheduling and validation tracking
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from collections import OrderedDict
import copy
import math

logger = logging.getLogger(__name__)


@dataclass
class EMAConfig:
    """Configuration for Exponential Moving Average."""
    
    # EMA decay parameters
    decay: float = 0.9999
    tau: int = 2000  # EMA tau parameter for decay adjustment
    
    # Decay scheduling
    dynamic_decay: bool = True
    min_decay: float = 0.99
    max_decay: float = 0.9999
    
    # Update frequency
    update_frequency: int = 1  # Update EMA every N steps
    warmup_steps: int = 100    # Steps before EMA starts
    
    # Validation and checkpointing
    validate_on_ema: bool = True
    save_ema_state: bool = True
    
    # Advanced options
    decay_schedule: str = "exponential"  # "exponential", "linear", "cosine"
    momentum_based: bool = False         # Use momentum-based EMA
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")
        if not 0 < self.min_decay <= self.max_decay <= 1:
            raise ValueError(f"Invalid decay range: min_decay={self.min_decay}, max_decay={self.max_decay}")
        if self.decay_schedule not in ["exponential", "linear", "cosine"]:
            raise ValueError(f"decay_schedule must be 'exponential', 'linear', or 'cosine'")


class EMAHandler:
    """
    Advanced Exponential Moving Average handler for RF-DETR training.
    Provides model stabilization and improved validation performance.
    """
    
    def __init__(self, model: nn.Module, config: EMAConfig):
        """
        Initialize EMA handler.
        
        Args:
            model: Source model to create EMA for
            config: EMA configuration
        """
        self.config = config
        self.source_model = model
        
        # Create EMA model (deep copy)
        self.ema_model = self._create_ema_model(model)
        
        # Training state
        self.step_count = 0
        self.update_count = 0
        self.current_decay = config.decay
        
        # Performance tracking
        self.ema_metrics = {}
        self.source_metrics = {}
        self.performance_history = []
        
        # Move EMA model to same device as source
        device = next(model.parameters()).device
        self.ema_model = self.ema_model.to(device)
        
        logger.info(f"Initialized EMA handler with decay={config.decay:.4f}, tau={config.tau}")
        logger.info(f"EMA model parameters: {sum(p.numel() for p in self.ema_model.parameters()):,}")
    
    def _create_ema_model(self, model: nn.Module) -> nn.Module:
        """Create EMA model as deep copy of source model."""
        
        # Create deep copy
        ema_model = copy.deepcopy(model)
        
        # Set to eval mode
        ema_model.eval()
        
        # Disable gradients for all parameters
        for param in ema_model.parameters():
            param.requires_grad_(False)
        
        return ema_model
    
    def update(self, step: Optional[int] = None):
        """
        Update EMA model weights.
        
        Args:
            step: Current training step (optional, uses internal counter if None)
        """
        
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
        
        # Check if we should update (warmup and frequency)
        if self.step_count < self.config.warmup_steps:
            return
        
        if self.step_count % self.config.update_frequency != 0:
            return
        
        # Update decay if dynamic
        if self.config.dynamic_decay:
            self.current_decay = self._calculate_dynamic_decay()
        
        # Perform EMA update
        self._update_ema_weights()
        self.update_count += 1
        
        # Log occasionally
        if self.update_count % 1000 == 0:
            logger.debug(f"EMA update #{self.update_count}, step {self.step_count}, decay={self.current_decay:.6f}")
    
    def _calculate_dynamic_decay(self) -> float:
        """Calculate dynamic decay value based on training progress."""
        
        if self.config.decay_schedule == "exponential":
            # Exponential increase to target decay
            progress = min(1.0, self.step_count / self.config.tau)
            decay = self.config.min_decay + (self.config.max_decay - self.config.min_decay) * (1 - math.exp(-progress * 5))
            
        elif self.config.decay_schedule == "linear":
            # Linear increase to target decay
            progress = min(1.0, self.step_count / self.config.tau)
            decay = self.config.min_decay + (self.config.max_decay - self.config.min_decay) * progress
            
        elif self.config.decay_schedule == "cosine":
            # Cosine schedule
            progress = min(1.0, self.step_count / self.config.tau)
            decay = self.config.min_decay + (self.config.max_decay - self.config.min_decay) * (1 + math.cos(math.pi * (1 - progress))) / 2
            
        else:
            decay = self.config.decay
        
        return decay
    
    def _update_ema_weights(self):
        """Update EMA model weights using current decay."""
        
        if self.config.momentum_based:
            self._momentum_based_update()
        else:
            self._standard_ema_update()
    
    def _standard_ema_update(self):
        """Standard EMA weight update."""
        
        with torch.no_grad():
            for ema_param, source_param in zip(self.ema_model.parameters(), self.source_model.parameters()):
                if source_param.requires_grad:
                    ema_param.data.mul_(self.current_decay).add_(source_param.data, alpha=1 - self.current_decay)
    
    def _momentum_based_update(self):
        """Momentum-based EMA update (alternative formulation)."""
        
        momentum = 1 - self.current_decay
        
        with torch.no_grad():
            for ema_param, source_param in zip(self.ema_model.parameters(), self.source_model.parameters()):
                if source_param.requires_grad:
                    diff = source_param.data - ema_param.data
                    ema_param.data.add_(diff, alpha=momentum)
    
    def get_ema_model(self) -> nn.Module:
        """Get EMA model for inference or validation."""
        return self.ema_model
    
    def get_source_model(self) -> nn.Module:
        """Get source model."""
        return self.source_model
    
    def validate_ema_performance(self, validation_fn: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Compare EMA and source model performance on validation.
        
        Args:
            validation_fn: Function that takes model and returns metrics dict
            *args, **kwargs: Arguments for validation function
            
        Returns:
            Dictionary with comparison results
        """
        
        # Validate source model
        self.source_model.eval()
        with torch.no_grad():
            source_metrics = validation_fn(self.source_model, *args, **kwargs)
        
        # Validate EMA model
        self.ema_model.eval()
        with torch.no_grad():
            ema_metrics = validation_fn(self.ema_model, *args, **kwargs)
        
        # Store metrics
        self.source_metrics = source_metrics
        self.ema_metrics = ema_metrics
        
        # Calculate improvements
        comparison = {
            'source_metrics': source_metrics,
            'ema_metrics': ema_metrics,
            'improvements': {},
            'ema_better_count': 0,
            'total_metrics': 0
        }
        
        for metric_name in source_metrics:
            if isinstance(source_metrics[metric_name], (int, float)):
                source_val = source_metrics[metric_name]
                ema_val = ema_metrics[metric_name]
                
                # Calculate improvement (assuming higher is better for most metrics)
                improvement = ema_val - source_val
                comparison['improvements'][metric_name] = improvement
                
                # Count improvements
                comparison['total_metrics'] += 1
                if improvement > 0:
                    comparison['ema_better_count'] += 1
        
        # Calculate overall improvement ratio
        if comparison['total_metrics'] > 0:
            comparison['ema_improvement_ratio'] = comparison['ema_better_count'] / comparison['total_metrics']
        else:
            comparison['ema_improvement_ratio'] = 0.0
        
        # Store in history
        self.performance_history.append({
            'step': self.step_count,
            'update_count': self.update_count,
            'decay': self.current_decay,
            'comparison': comparison
        })
        
        # Keep history size manageable
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
        
        logger.info(f"EMA validation at step {self.step_count}: "
                   f"{comparison['ema_better_count']}/{comparison['total_metrics']} metrics improved")
        
        return comparison
    
    def get_ema_statistics(self) -> Dict[str, Any]:
        """Get EMA handler statistics."""
        
        stats = {
            'step_count': self.step_count,
            'update_count': self.update_count,
            'current_decay': self.current_decay,
            'config': self.config.__dict__,
            'updates_per_step': self.update_count / max(1, self.step_count)
        }
        
        # Add performance statistics if available
        if self.performance_history:
            recent_comparisons = self.performance_history[-10:]  # Last 10 validations
            improvement_ratios = [comp['comparison']['ema_improvement_ratio'] for comp in recent_comparisons]
            
            stats.update({
                'recent_improvement_ratio_mean': sum(improvement_ratios) / len(improvement_ratios),
                'recent_improvement_ratio_max': max(improvement_ratios),
                'recent_improvement_ratio_min': min(improvement_ratios),
                'validation_count': len(self.performance_history)
            })
        
        return stats
    
    def copy_ema_to_source(self):
        """Copy EMA model weights back to source model (for final training)."""
        
        logger.info("Copying EMA weights to source model")
        
        with torch.no_grad():
            for source_param, ema_param in zip(self.source_model.parameters(), self.ema_model.parameters()):
                if source_param.requires_grad:
                    source_param.data.copy_(ema_param.data)
    
    def reset_ema(self):
        """Reset EMA model to current source model state."""
        
        logger.info("Resetting EMA model to source model state")
        
        with torch.no_grad():
            for ema_param, source_param in zip(self.ema_model.parameters(), self.source_model.parameters()):
                if source_param.requires_grad:
                    ema_param.data.copy_(source_param.data)
        
        self.update_count = 0
    
    def save_state(self) -> Dict[str, Any]:
        """Save EMA handler state for checkpointing."""
        
        state = {
            'ema_model_state': self.ema_model.state_dict(),
            'step_count': self.step_count,
            'update_count': self.update_count,
            'current_decay': self.current_decay,
            'config': self.config.__dict__,
            'performance_history': self.performance_history[-10:]  # Last 10 for resuming
        }
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load EMA handler state from checkpoint."""
        
        self.ema_model.load_state_dict(state['ema_model_state'])
        self.step_count = state['step_count']
        self.update_count = state['update_count']
        self.current_decay = state['current_decay']
        
        if 'performance_history' in state:
            self.performance_history = state['performance_history']
        
        logger.info(f"Loaded EMA state: step={self.step_count}, updates={self.update_count}, decay={self.current_decay:.6f}")
    
    def get_decay_schedule_preview(self, steps: int = 10000) -> list[float]:
        """Generate preview of decay schedule."""
        
        if not self.config.dynamic_decay:
            return [self.config.decay] * steps
        
        original_step = self.step_count
        schedule = []
        
        for step in range(steps):
            self.step_count = step
            decay = self._calculate_dynamic_decay()
            schedule.append(decay)
        
        self.step_count = original_step
        return schedule
    
    def plot_decay_schedule(self, steps: int = 10000, save_path: Optional[str] = None):
        """Plot decay schedule for visualization."""
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot decay schedule")
            return
        
        schedule = self.get_decay_schedule_preview(steps)
        step_range = list(range(steps))
        
        plt.figure(figsize=(10, 6))
        plt.plot(step_range, schedule, linewidth=2, label=f'EMA Decay ({self.config.decay_schedule})')
        
        plt.xlabel('Training Step')
        plt.ylabel('EMA Decay')
        plt.title('EMA Decay Schedule')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Mark tau point
        if self.config.tau < steps:
            plt.axvline(x=self.config.tau, color='red', linestyle='--', alpha=0.7, label=f'Tau ({self.config.tau})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved EMA decay schedule plot to {save_path}")
        
        plt.show()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if needed."""
        pass


def create_ema_handler(
    model: nn.Module,
    decay: float = 0.9999,
    tau: int = 2000,
    warmup_steps: int = 100,
    **kwargs
) -> EMAHandler:
    """
    Convenience function to create EMA handler.
    
    Args:
        model: Source model
        decay: EMA decay rate
        tau: Tau parameter for decay scheduling
        warmup_steps: Steps before EMA starts
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured EMA handler
    """
    
    config = EMAConfig(
        decay=decay,
        tau=tau,
        warmup_steps=warmup_steps,
        **kwargs
    )
    
    return EMAHandler(model, config)


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
    ema_handler = create_ema_handler(model, decay=0.999, tau=1000)
    
    # Simulate training updates
    for step in range(100):
        ema_handler.update()
        
        if step % 20 == 0:
            stats = ema_handler.get_ema_statistics()
            print(f"Step {step}: decay={stats['current_decay']:.6f}")
    
    print(f"Final statistics: {ema_handler.get_ema_statistics()}")