"""
Cosine Annealing Learning Rate Scheduler with Warmup for RF-DETR Training
Advanced scheduling strategies optimized for transformer-based detection models
"""
import logging
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CosineSchedulerConfig:
    """Configuration for cosine annealing scheduler with warmup."""
    
    # Total training configuration
    total_epochs: int = 100
    warmup_epochs: int = 5
    
    # Learning rate bounds
    max_lr: Optional[float] = None  # Will use optimizer's initial LR if None
    min_lr_ratio: float = 0.01  # Minimum LR as ratio of max LR
    
    # Cosine annealing parameters
    restart_enabled: bool = False
    restart_period: int = 50
    restart_mult: float = 2.0
    
    # Warmup configuration
    warmup_method: str = "linear"  # "linear", "exponential", "cosine"
    warmup_start_factor: float = 0.1  # Starting factor for warmup
    
    # Layer-wise scheduling
    layer_wise_decay_enabled: bool = False
    layer_wise_decay_factor: float = 0.95
    
    # Advanced features
    plateau_detection: bool = False
    plateau_patience: int = 10
    plateau_threshold: float = 1e-4
    
    def __post_init__(self):
        """Validate configuration."""
        if self.total_epochs <= 0:
            raise ValueError(f"total_epochs must be positive, got {self.total_epochs}")
        if not 0 <= self.warmup_epochs < self.total_epochs:
            raise ValueError(f"warmup_epochs must be in [0, {self.total_epochs}), got {self.warmup_epochs}")
        if not 0 < self.min_lr_ratio <= 1:
            raise ValueError(f"min_lr_ratio must be in (0, 1], got {self.min_lr_ratio}")
        if self.warmup_method not in ["linear", "exponential", "cosine"]:
            raise ValueError(f"warmup_method must be 'linear', 'exponential', or 'cosine'")


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with configurable warmup.
    Designed for stable RF-DETR training with smooth learning rate transitions.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        config: CosineSchedulerConfig,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize cosine annealing scheduler with warmup.
        
        Args:
            optimizer: PyTorch optimizer
            config: Scheduler configuration
            last_epoch: Last epoch number for resuming training
            verbose: Whether to print learning rate updates
        """
        self.config = config
        
        # Store initial learning rates for each parameter group
        self.base_lrs = []
        self.min_lrs = []
        
        for group in optimizer.param_groups:
            base_lr = config.max_lr if config.max_lr is not None else group['lr']
            min_lr = base_lr * config.min_lr_ratio
            
            self.base_lrs.append(base_lr)
            self.min_lrs.append(min_lr)
        
        # Plateau detection variables
        self.plateau_count = 0
        self.best_metric = None
        self.plateau_detected = False
        
        # Restart variables
        self.restart_count = 0
        self.current_restart_period = config.restart_period
        self.epochs_since_restart = 0
        
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(f"Initialized CosineAnnealingWarmupScheduler:")
        logger.info(f"  Total epochs: {config.total_epochs}")
        logger.info(f"  Warmup epochs: {config.warmup_epochs}")
        logger.info(f"  Base LRs: {self.base_lrs}")
        logger.info(f"  Min LRs: {self.min_lrs}")
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current epoch."""
        
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        # Current epoch (0-indexed)
        epoch = self.last_epoch
        
        # Handle restarts if enabled
        if self.config.restart_enabled:
            epoch = self._handle_restarts(epoch)
        
        lrs = []
        
        for i, (base_lr, min_lr) in enumerate(zip(self.base_lrs, self.min_lrs)):
            
            if epoch < self.config.warmup_epochs:
                # Warmup phase
                lr = self._calculate_warmup_lr(epoch, base_lr, min_lr)
            else:
                # Cosine annealing phase
                lr = self._calculate_cosine_lr(epoch, base_lr, min_lr)
            
            # Apply layer-wise decay if enabled
            if self.config.layer_wise_decay_enabled:
                lr = self._apply_layer_wise_decay(lr, i)
            
            lrs.append(lr)
        
        return lrs
    
    def _handle_restarts(self, epoch: int) -> int:
        """Handle cosine restarts."""
        
        self.epochs_since_restart = epoch % self.current_restart_period
        
        # Check if we need to restart
        if epoch > 0 and epoch % self.current_restart_period == 0:
            self.restart_count += 1
            self.current_restart_period = int(self.config.restart_period * (self.config.restart_mult ** self.restart_count))
            logger.info(f"Cosine restart #{self.restart_count} at epoch {epoch}, next period: {self.current_restart_period}")
        
        return self.epochs_since_restart
    
    def _calculate_warmup_lr(self, epoch: int, base_lr: float, min_lr: float) -> float:
        """Calculate learning rate during warmup phase."""
        
        if self.config.warmup_epochs == 0:
            return base_lr
        
        # Progress through warmup (0 to 1)
        progress = epoch / self.config.warmup_epochs
        
        if self.config.warmup_method == "linear":
            # Linear warmup
            start_lr = base_lr * self.config.warmup_start_factor
            lr = start_lr + (base_lr - start_lr) * progress
            
        elif self.config.warmup_method == "exponential":
            # Exponential warmup
            start_lr = base_lr * self.config.warmup_start_factor
            lr = start_lr * ((base_lr / start_lr) ** progress)
            
        elif self.config.warmup_method == "cosine":
            # Cosine warmup
            start_lr = base_lr * self.config.warmup_start_factor
            lr = start_lr + (base_lr - start_lr) * (1 - math.cos(math.pi * progress)) / 2
            
        else:
            lr = base_lr
        
        return max(lr, min_lr)
    
    def _calculate_cosine_lr(self, epoch: int, base_lr: float, min_lr: float) -> float:
        """Calculate learning rate during cosine annealing phase."""
        
        # Adjust epoch for warmup
        cosine_epoch = epoch - self.config.warmup_epochs
        total_cosine_epochs = self.config.total_epochs - self.config.warmup_epochs
        
        if total_cosine_epochs <= 0:
            return base_lr
        
        # Handle restarts
        if self.config.restart_enabled:
            total_cosine_epochs = self.current_restart_period - self.config.warmup_epochs
            total_cosine_epochs = max(1, total_cosine_epochs)
        
        # Cosine annealing formula
        progress = cosine_epoch / total_cosine_epochs
        progress = max(0, min(1, progress))  # Clamp to [0, 1]
        
        lr = min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
        
        return max(lr, min_lr)
    
    def _apply_layer_wise_decay(self, lr: float, group_index: int) -> float:
        """Apply layer-wise learning rate decay."""
        
        # Decay factor increases with group index (deeper layers get higher LR)
        decay_factor = self.config.layer_wise_decay_factor ** (len(self.base_lrs) - 1 - group_index)
        return lr * decay_factor
    
    def step(self, epoch: Optional[int] = None, metrics: Optional[float] = None):
        """
        Step the scheduler.
        
        Args:
            epoch: Current epoch number
            metrics: Optional metric for plateau detection
        """
        
        # Handle plateau detection
        if self.config.plateau_detection and metrics is not None:
            self._handle_plateau_detection(metrics)
        
        super().step(epoch)
        
        # Log learning rates periodically
        if self.last_epoch % 10 == 0 or self.last_epoch < 10:
            current_lrs = self.get_last_lr()
            logger.debug(f"Epoch {self.last_epoch}: LRs = {[f'{lr:.2e}' for lr in current_lrs]}")
    
    def _handle_plateau_detection(self, metric: float):
        """Handle plateau detection and adjustment."""
        
        if self.best_metric is None:
            self.best_metric = metric
            return
        
        # Check if metric improved
        if metric < self.best_metric - self.config.plateau_threshold:
            self.best_metric = metric
            self.plateau_count = 0
            self.plateau_detected = False
        else:
            self.plateau_count += 1
            
            if self.plateau_count >= self.config.plateau_patience:
                if not self.plateau_detected:
                    logger.info(f"Plateau detected at epoch {self.last_epoch}, "
                               f"no improvement for {self.plateau_count} epochs")
                    self.plateau_detected = True
    
    def get_scheduler_state(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        
        state = {
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs,
            'min_lrs': self.min_lrs,
            'restart_count': self.restart_count,
            'current_restart_period': self.current_restart_period,
            'epochs_since_restart': self.epochs_since_restart,
            'plateau_count': self.plateau_count,
            'best_metric': self.best_metric,
            'plateau_detected': self.plateau_detected,
            'config': self.config.__dict__
        }
        
        return state
    
    def load_scheduler_state(self, state: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        
        self.last_epoch = state['last_epoch']
        self.base_lrs = state['base_lrs']
        self.min_lrs = state['min_lrs']
        self.restart_count = state['restart_count']
        self.current_restart_period = state['current_restart_period']
        self.epochs_since_restart = state['epochs_since_restart']
        self.plateau_count = state['plateau_count']
        self.best_metric = state['best_metric']
        self.plateau_detected = state['plateau_detected']
        
        logger.info(f"Loaded scheduler state for epoch {self.last_epoch}")
    
    def get_lr_schedule_preview(self, preview_epochs: Optional[int] = None) -> List[List[float]]:
        """
        Generate preview of learning rate schedule.
        
        Args:
            preview_epochs: Number of epochs to preview (default: total_epochs)
            
        Returns:
            List of learning rates for each epoch and parameter group
        """
        
        if preview_epochs is None:
            preview_epochs = self.config.total_epochs
        
        # Save current state
        current_epoch = self.last_epoch
        
        # Generate schedule
        schedule = []
        
        for epoch in range(preview_epochs):
            self.last_epoch = epoch
            lrs = self.get_lr()
            schedule.append(lrs.copy())
        
        # Restore state
        self.last_epoch = current_epoch
        
        return schedule
    
    def plot_lr_schedule(self, save_path: Optional[str] = None, preview_epochs: Optional[int] = None):
        """
        Plot learning rate schedule for visualization.
        
        Args:
            save_path: Path to save plot (optional)
            preview_epochs: Number of epochs to preview
        """
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, cannot plot LR schedule")
            return
        
        schedule = self.get_lr_schedule_preview(preview_epochs)
        epochs = list(range(len(schedule)))
        
        plt.figure(figsize=(12, 6))
        
        # Plot each parameter group
        for group_idx in range(len(schedule[0])):
            lrs = [epoch_lrs[group_idx] for epoch_lrs in schedule]
            plt.plot(epochs, lrs, label=f'Group {group_idx}', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('RF-DETR Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Mark warmup period
        if self.config.warmup_epochs > 0:
            plt.axvline(x=self.config.warmup_epochs, color='red', linestyle='--', alpha=0.7, label='Warmup End')
        
        # Mark restart periods if enabled
        if self.config.restart_enabled:
            restart_epochs = []
            period = self.config.restart_period
            while period < len(schedule):
                restart_epochs.append(period)
                period += int(period * self.config.restart_mult)
            
            for restart_epoch in restart_epochs:
                plt.axvline(x=restart_epoch, color='orange', linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved LR schedule plot to {save_path}")
        
        plt.show()
    
    def adjust_for_early_stopping(self, stopped_epoch: int):
        """Adjust scheduler parameters when training stops early."""
        
        if stopped_epoch < self.config.total_epochs:
            logger.info(f"Adjusting scheduler for early stopping at epoch {stopped_epoch}")
            self.config.total_epochs = stopped_epoch
    
    def get_current_phase(self) -> str:
        """Get current training phase description."""
        
        epoch = self.last_epoch
        
        if epoch < self.config.warmup_epochs:
            progress = (epoch + 1) / self.config.warmup_epochs * 100
            return f"Warmup ({progress:.1f}%)"
        
        elif self.plateau_detected:
            return "Plateau"
        
        elif self.config.restart_enabled:
            restart_progress = self.epochs_since_restart / self.current_restart_period * 100
            return f"Cosine Restart #{self.restart_count} ({restart_progress:.1f}%)"
        
        else:
            cosine_progress = (epoch - self.config.warmup_epochs) / (self.config.total_epochs - self.config.warmup_epochs) * 100
            return f"Cosine Annealing ({cosine_progress:.1f}%)"


def create_cosine_scheduler_with_warmup(
    optimizer: Optimizer,
    total_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr_ratio: float = 0.01,
    warmup_method: str = "linear",
    restart_enabled: bool = False,
    **kwargs
) -> CosineAnnealingWarmupScheduler:
    """
    Convenience function to create cosine annealing scheduler with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        total_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        min_lr_ratio: Minimum LR as ratio of max LR
        warmup_method: Warmup method ("linear", "exponential", "cosine")
        restart_enabled: Enable cosine restarts
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured cosine annealing scheduler
    """
    
    config = CosineSchedulerConfig(
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        min_lr_ratio=min_lr_ratio,
        warmup_method=warmup_method,
        restart_enabled=restart_enabled,
        **kwargs
    )
    
    return CosineAnnealingWarmupScheduler(optimizer, config)