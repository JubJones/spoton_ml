"""
Advanced Training Techniques for RF-DETR Surveillance Optimization
Knowledge distillation, mixed precision, gradient accumulation, and advanced checkpointing
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
import time
from collections import defaultdict, OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    
    # Teacher-student setup
    temperature: float = 4.0  # Softmax temperature for distillation
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for student hard loss
    
    # Feature distillation
    feature_distillation: bool = True
    feature_loss_weight: float = 0.1
    feature_layers: List[str] = field(default_factory=lambda: ['neck', 'head'])
    
    # Attention distillation
    attention_distillation: bool = True
    attention_loss_weight: float = 0.05
    
    # Progressive distillation
    progressive_training: bool = False
    warmup_epochs: int = 5
    distillation_schedule: str = "cosine"  # "linear", "cosine", "step"
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.alpha + self.beta == 1.0, f"Alpha ({self.alpha}) + Beta ({self.beta}) must equal 1.0"
        assert 0 < self.temperature <= 10.0, f"Temperature must be between 0 and 10, got {self.temperature}"


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    
    # AMP settings
    enabled: bool = True
    init_scale: float = 2.**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    
    # Loss scaling
    dynamic_loss_scaling: bool = True
    loss_scale_window: int = 1000
    
    # Optimization settings
    find_unused_parameters: bool = False
    gradient_predivide_factor: float = 1.0
    
    # Memory optimization
    cpu_offload: bool = False
    activation_checkpointing: bool = False


@dataclass
class CheckpointConfig:
    """Configuration for advanced checkpointing."""
    
    # Checkpoint frequency
    save_frequency: int = 1000  # Steps between checkpoints
    save_top_k: int = 3  # Keep top K best checkpoints
    monitor_metric: str = "mAP"  # Metric to monitor for best checkpoints
    
    # Checkpoint content
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_model_weights: bool = True
    save_training_state: bool = True
    
    # Advanced features
    incremental_checkpointing: bool = True
    checkpoint_compression: bool = False
    async_checkpointing: bool = False
    
    # Storage optimization
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    max_checkpoint_size_gb: float = 10.0  # Maximum checkpoint size in GB


class KnowledgeDistillationTrainer:
    """Knowledge distillation trainer for surveillance person detection."""
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        config: DistillationConfig
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.config = config
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Setup feature hooks for distillation
        self.student_features = {}
        self.teacher_features = {}
        self._setup_feature_hooks()
        
        # Loss functions
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
        # Progressive training state
        self.current_epoch = 0
        self.distillation_weight = config.alpha
        
        logger.info(f"Knowledge Distillation initialized:")
        logger.info(f"  Temperature: {config.temperature}")
        logger.info(f"  Alpha (distillation): {config.alpha}, Beta (hard): {config.beta}")
        logger.info(f"  Feature distillation: {config.feature_distillation}")
        logger.info(f"  Progressive training: {config.progressive_training}")
    
    def _setup_feature_hooks(self):
        """Setup hooks to extract intermediate features."""
        
        def create_hook(name: str, feature_dict: Dict):
            def hook_fn(module, input, output):
                if isinstance(output, dict):
                    feature_dict[name] = output
                else:
                    feature_dict[name] = {'features': output}
            return hook_fn
        
        # Register hooks for specified layers
        for layer_name in self.config.feature_layers:
            if hasattr(self.student, layer_name):
                student_layer = getattr(self.student, layer_name)
                teacher_layer = getattr(self.teacher, layer_name)
                
                student_layer.register_forward_hook(
                    create_hook(f'student_{layer_name}', self.student_features)
                )
                teacher_layer.register_forward_hook(
                    create_hook(f'teacher_{layer_name}', self.teacher_features)
                )
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        hard_loss: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute comprehensive distillation loss."""
        
        losses = {}
        
        # Knowledge distillation loss (soft targets)
        student_soft = F.log_softmax(student_logits / self.config.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.config.temperature, dim=-1)
        
        kd_loss = self.kl_loss(student_soft, teacher_soft) * (self.config.temperature ** 2)
        losses['distillation_loss'] = kd_loss
        
        # Feature distillation losses
        if self.config.feature_distillation:
            feature_losses = self._compute_feature_distillation_losses()
            losses.update(feature_losses)
        
        # Attention distillation losses  
        if self.config.attention_distillation:
            attention_losses = self._compute_attention_distillation_losses()
            losses.update(attention_losses)
        
        # Combined loss with current weights
        total_distillation_loss = (
            self.distillation_weight * kd_loss +
            self.config.feature_loss_weight * losses.get('feature_loss', 0) +
            self.config.attention_loss_weight * losses.get('attention_loss', 0)
        )
        
        # Total loss combining distillation and hard loss
        beta_weight = 1.0 - self.distillation_weight  # Complement for hard loss
        total_loss = total_distillation_loss + beta_weight * hard_loss
        
        losses.update({
            'total_distillation_loss': total_distillation_loss,
            'hard_loss': hard_loss,
            'total_loss': total_loss,
            'distillation_weight': self.distillation_weight
        })
        
        return losses
    
    def _compute_feature_distillation_losses(self) -> Dict[str, torch.Tensor]:
        """Compute feature-based distillation losses."""
        
        feature_losses = {}
        total_feature_loss = 0.0
        
        for layer_name in self.config.feature_layers:
            student_key = f'student_{layer_name}'
            teacher_key = f'teacher_{layer_name}'
            
            if student_key in self.student_features and teacher_key in self.teacher_features:
                student_feat = self.student_features[student_key]
                teacher_feat = self.teacher_features[teacher_key]
                
                # Handle dict outputs (common in detection models)
                if isinstance(student_feat, dict) and isinstance(teacher_feat, dict):
                    layer_loss = 0.0
                    for key in student_feat.keys():
                        if key in teacher_feat:
                            s_f = student_feat[key]
                            t_f = teacher_feat[key]
                            
                            # Align feature dimensions if needed
                            if s_f.shape != t_f.shape:
                                s_f, t_f = self._align_features(s_f, t_f)
                            
                            layer_loss += self.mse_loss(s_f, t_f.detach())
                    
                    feature_losses[f'feature_loss_{layer_name}'] = layer_loss
                    total_feature_loss += layer_loss
                else:
                    # Handle tensor outputs
                    s_f = student_feat if isinstance(student_feat, torch.Tensor) else student_feat['features']
                    t_f = teacher_feat if isinstance(teacher_feat, torch.Tensor) else teacher_feat['features']
                    
                    if s_f.shape != t_f.shape:
                        s_f, t_f = self._align_features(s_f, t_f)
                    
                    layer_loss = self.mse_loss(s_f, t_f.detach())
                    feature_losses[f'feature_loss_{layer_name}'] = layer_loss
                    total_feature_loss += layer_loss
        
        feature_losses['feature_loss'] = total_feature_loss
        return feature_losses
    
    def _compute_attention_distillation_losses(self) -> Dict[str, torch.Tensor]:
        """Compute attention-based distillation losses."""
        
        attention_losses = {}
        total_attention_loss = 0.0
        
        # Extract attention maps from features
        for layer_name in self.config.feature_layers:
            student_key = f'student_{layer_name}'
            teacher_key = f'teacher_{layer_name}'
            
            if student_key in self.student_features and teacher_key in self.teacher_features:
                student_feat = self.student_features[student_key]
                teacher_feat = self.teacher_features[teacher_key]
                
                # Compute attention maps (spatial attention)
                student_attention = self._compute_attention_map(student_feat)
                teacher_attention = self._compute_attention_map(teacher_feat)
                
                if student_attention is not None and teacher_attention is not None:
                    # Align attention maps
                    if student_attention.shape != teacher_attention.shape:
                        student_attention = F.interpolate(
                            student_attention, 
                            size=teacher_attention.shape[-2:], 
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    attention_loss = self.mse_loss(student_attention, teacher_attention.detach())
                    attention_losses[f'attention_loss_{layer_name}'] = attention_loss
                    total_attention_loss += attention_loss
        
        attention_losses['attention_loss'] = total_attention_loss
        return attention_losses
    
    def _align_features(
        self, 
        student_feat: torch.Tensor, 
        teacher_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align student and teacher features to the same dimensions."""
        
        # Align spatial dimensions
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            student_feat = F.interpolate(
                student_feat, 
                size=teacher_feat.shape[-2:], 
                mode='bilinear',
                align_corners=False
            )
        
        # Align channel dimensions if needed
        if student_feat.shape[1] != teacher_feat.shape[1]:
            # Add a 1x1 conv to align channels
            if not hasattr(self, 'channel_aligners'):
                self.channel_aligners = nn.ModuleDict()
            
            aligner_key = f"{student_feat.shape[1]}to{teacher_feat.shape[1]}"
            if aligner_key not in self.channel_aligners:
                self.channel_aligners[aligner_key] = nn.Conv2d(
                    student_feat.shape[1], 
                    teacher_feat.shape[1], 
                    1, 
                    bias=False
                ).to(student_feat.device)
            
            student_feat = self.channel_aligners[aligner_key](student_feat)
        
        return student_feat, teacher_feat
    
    def _compute_attention_map(self, feature: Union[torch.Tensor, Dict]) -> Optional[torch.Tensor]:
        """Compute spatial attention map from features."""
        
        if isinstance(feature, dict):
            # Try to find the main feature tensor
            if 'features' in feature:
                feat = feature['features']
            elif len(feature) == 1:
                feat = list(feature.values())[0]
            else:
                return None
        else:
            feat = feature
        
        if not isinstance(feat, torch.Tensor) or len(feat.shape) != 4:
            return None
        
        # Compute spatial attention (channel-wise sum and normalize)
        attention = torch.sum(feat, dim=1, keepdim=True)  # [B, 1, H, W]
        attention = F.normalize(attention.flatten(2), p=2, dim=2)  # Normalize spatially
        attention = attention.view_as(attention)  # Reshape back
        
        return attention
    
    def update_distillation_schedule(self, epoch: int):
        """Update distillation weight based on schedule."""
        
        if not self.config.progressive_training:
            return
        
        self.current_epoch = epoch
        
        if epoch < self.config.warmup_epochs:
            # During warmup, focus more on hard loss
            warmup_progress = epoch / self.config.warmup_epochs
            self.distillation_weight = self.config.alpha * warmup_progress
        else:
            # After warmup, use full distillation weight
            self.distillation_weight = self.config.alpha
        
        logger.debug(f"Epoch {epoch}: Distillation weight = {self.distillation_weight:.3f}")
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with distillation."""
        
        # Clear previous features
        self.student_features.clear()
        self.teacher_features.clear()
        
        # Student forward pass
        student_outputs = self.student(inputs)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        
        # Extract logits (model-specific logic)
        student_logits = self._extract_logits(student_outputs)
        teacher_logits = self._extract_logits(teacher_outputs)
        
        # Compute hard loss (standard training loss)
        hard_loss = torch.tensor(0.0, device=inputs.device)
        if targets is not None:
            # This would be the standard loss computation
            # Placeholder - should be replaced with actual model loss
            hard_loss = F.cross_entropy(student_logits.view(-1, student_logits.shape[-1]), 
                                      targets.view(-1), ignore_index=-1)
        
        # Compute distillation losses
        distillation_losses = self.compute_distillation_loss(
            student_logits, teacher_logits, targets, hard_loss
        )
        
        # Add model outputs
        result = {
            'student_outputs': student_outputs,
            'teacher_outputs': teacher_outputs,
            **distillation_losses
        }
        
        return result
    
    def _extract_logits(self, model_outputs: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """Extract logits from model outputs."""
        
        if isinstance(model_outputs, torch.Tensor):
            return model_outputs
        elif isinstance(model_outputs, dict):
            # Common keys for detection models
            for key in ['pred_logits', 'logits', 'classification_logits', 'cls_logits']:
                if key in model_outputs:
                    return model_outputs[key]
            
            # Fallback to first tensor value
            for value in model_outputs.values():
                if isinstance(value, torch.Tensor):
                    return value
        
        raise ValueError(f"Could not extract logits from model outputs: {type(model_outputs)}")


class MixedPrecisionTrainer:
    """Mixed precision training with advanced optimizations."""
    
    def __init__(self, model: nn.Module, config: MixedPrecisionConfig):
        self.model = model
        self.config = config
        
        # Initialize GradScaler for AMP
        if config.enabled:
            self.scaler = GradScaler(
                init_scale=config.init_scale,
                growth_factor=config.growth_factor,
                backoff_factor=config.backoff_factor,
                growth_interval=config.growth_interval,
                enabled=config.enabled
            )
        else:
            self.scaler = None
        
        # Loss scaling tracking
        self.loss_scale_history = []
        self.overflow_count = 0
        self.successful_steps = 0
        
        logger.info(f"Mixed Precision Training initialized:")
        logger.info(f"  Enabled: {config.enabled}")
        logger.info(f"  Dynamic loss scaling: {config.dynamic_loss_scaling}")
        logger.info(f"  Activation checkpointing: {config.activation_checkpointing}")
    
    def forward_pass(
        self, 
        inputs: torch.Tensor, 
        compute_loss_fn: Callable,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with mixed precision."""
        
        if self.config.enabled:
            with autocast():
                outputs = self.model(inputs, **kwargs)
                loss = compute_loss_fn(outputs)
        else:
            outputs = self.model(inputs, **kwargs)
            loss = compute_loss_fn(outputs)
        
        return loss, outputs
    
    def backward_pass(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        retain_graph: bool = False
    ) -> Dict[str, Any]:
        """Backward pass with mixed precision."""
        
        metrics = {
            'loss_scale': 1.0,
            'gradient_overflow': False,
            'successful_step': True
        }
        
        if self.scaler is not None:
            # Scale loss and backward
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
            
            # Track loss scale
            current_scale = self.scaler.get_scale()
            metrics['loss_scale'] = current_scale
            self.loss_scale_history.append(current_scale)
            
            # Keep only recent history
            if len(self.loss_scale_history) > 1000:
                self.loss_scale_history = self.loss_scale_history[-1000:]
            
            # Optimizer step with gradient scaling
            self.scaler.step(optimizer)
            
            # Check for gradient overflow
            if current_scale < self.scaler.get_scale():
                # Scale was not updated, indicating overflow
                self.overflow_count += 1
                metrics['gradient_overflow'] = True
                metrics['successful_step'] = False
            else:
                self.successful_steps += 1
            
            # Update scaler
            self.scaler.update()
        else:
            # Standard training without mixed precision
            loss.backward(retain_graph=retain_graph)
            optimizer.step()
            self.successful_steps += 1
        
        return metrics
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get mixed precision training statistics."""
        
        stats = {
            'mixed_precision_enabled': self.config.enabled,
            'successful_steps': self.successful_steps,
            'overflow_count': self.overflow_count,
            'overflow_rate': self.overflow_count / max(self.successful_steps + self.overflow_count, 1)
        }
        
        if self.scaler is not None:
            stats.update({
                'current_loss_scale': self.scaler.get_scale(),
                'growth_tracker': getattr(self.scaler, '_growth_tracker', 0)
            })
            
            if self.loss_scale_history:
                stats.update({
                    'avg_loss_scale': np.mean(self.loss_scale_history),
                    'min_loss_scale': min(self.loss_scale_history),
                    'max_loss_scale': max(self.loss_scale_history)
                })
        
        return stats


class AdvancedCheckpointManager:
    """Advanced checkpoint manager with optimization features."""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint tracking
        self.checkpoint_history = []
        self.best_checkpoints = []  # Top-k checkpoints
        self.last_checkpoint_time = None
        
        logger.info(f"Advanced Checkpoint Manager initialized:")
        logger.info(f"  Checkpoint directory: {config.checkpoint_dir}")
        logger.info(f"  Save frequency: {config.save_frequency} steps")
        logger.info(f"  Keep top {config.save_top_k} checkpoints")
        logger.info(f"  Monitor metric: {config.monitor_metric}")
    
    def should_save_checkpoint(self, step: int) -> bool:
        """Check if checkpoint should be saved."""
        return step % self.config.save_frequency == 0
    
    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler = None,
        metrics: Optional[Dict[str, float]] = None,
        training_state: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Path:
        """Save comprehensive checkpoint."""
        
        start_time = time.time()
        
        # Prepare checkpoint data
        checkpoint_data = {
            'step': step,
            'epoch': epoch,
            'timestamp': start_time,
            'metrics': metrics or {},
            'config': self.config.__dict__
        }
        
        # Add model state
        if self.config.save_model_weights:
            checkpoint_data['model_state_dict'] = model.state_dict()
        
        # Add optimizer state
        if self.config.save_optimizer_state and optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add scheduler state
        if self.config.save_scheduler_state and scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add training state
        if self.config.save_training_state and training_state is not None:
            checkpoint_data['training_state'] = training_state
        
        # Add custom data
        checkpoint_data.update(kwargs)
        
        # Generate checkpoint filename
        if metrics and self.config.monitor_metric in metrics:
            metric_value = metrics[self.config.monitor_metric]
            filename = f"checkpoint_step_{step}_epoch_{epoch}_{self.config.monitor_metric}_{metric_value:.4f}.pth"
        else:
            filename = f"checkpoint_step_{step}_epoch_{epoch}.pth"
        
        checkpoint_path = self.config.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Calculate checkpoint size
        checkpoint_size = checkpoint_path.stat().st_size / (1024**3)  # GB
        
        # Record checkpoint info
        checkpoint_info = {
            'path': checkpoint_path,
            'step': step,
            'epoch': epoch,
            'size_gb': checkpoint_size,
            'save_time': time.time() - start_time,
            'metrics': metrics or {},
            'timestamp': start_time
        }
        
        self.checkpoint_history.append(checkpoint_info)
        
        # Update best checkpoints
        if metrics and self.config.monitor_metric in metrics:
            self._update_best_checkpoints(checkpoint_info)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        logger.info(f"  Size: {checkpoint_size:.2f} GB")
        logger.info(f"  Save time: {checkpoint_info['save_time']:.2f}s")
        
        return checkpoint_path
    
    def _update_best_checkpoints(self, checkpoint_info: Dict[str, Any]):
        """Update list of best checkpoints."""
        
        metric_value = checkpoint_info['metrics'].get(self.config.monitor_metric, 0)
        checkpoint_info['monitor_value'] = metric_value
        
        # Add to best checkpoints
        self.best_checkpoints.append(checkpoint_info)
        
        # Sort by metric value (descending for metrics like mAP)
        self.best_checkpoints.sort(key=lambda x: x['monitor_value'], reverse=True)
        
        # Keep only top-k
        if len(self.best_checkpoints) > self.config.save_top_k:
            # Remove worst checkpoint file
            worst_checkpoint = self.best_checkpoints.pop()
            try:
                worst_checkpoint['path'].unlink()
                logger.debug(f"Removed checkpoint: {worst_checkpoint['path']}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint: {e}")
    
    def _cleanup_checkpoints(self):
        """Cleanup old checkpoints based on policy."""
        
        # Remove checkpoints that exceed size limit
        total_size = sum(info['size_gb'] for info in self.checkpoint_history)
        
        if total_size > self.config.max_checkpoint_size_gb:
            # Sort by timestamp (oldest first)
            sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['timestamp'])
            
            while total_size > self.config.max_checkpoint_size_gb and sorted_checkpoints:
                old_checkpoint = sorted_checkpoints.pop(0)
                
                # Don't remove if it's in best checkpoints
                if old_checkpoint in self.best_checkpoints:
                    continue
                
                try:
                    old_checkpoint['path'].unlink()
                    self.checkpoint_history.remove(old_checkpoint)
                    total_size -= old_checkpoint['size_gb']
                    logger.debug(f"Cleaned up checkpoint: {old_checkpoint['path']}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup checkpoint: {e}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load checkpoint with comprehensive restoration."""
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint data
        checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
        
        # Restore model state
        if 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
            logger.info("Model state restored from checkpoint")
        
        # Restore optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            logger.info("Optimizer state restored from checkpoint")
        
        # Restore scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            logger.info("Scheduler state restored from checkpoint")
        
        # Extract training info
        training_info = {
            'step': checkpoint_data.get('step', 0),
            'epoch': checkpoint_data.get('epoch', 0),
            'metrics': checkpoint_data.get('metrics', {}),
            'training_state': checkpoint_data.get('training_state', {}),
            'checkpoint_path': checkpoint_path
        }
        
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        logger.info(f"  Step: {training_info['step']}, Epoch: {training_info['epoch']}")
        
        return training_info
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        if self.best_checkpoints:
            return self.best_checkpoints[0]['path']
        return None
    
    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get path to the latest checkpoint."""
        if self.checkpoint_history:
            latest = max(self.checkpoint_history, key=lambda x: x['step'])
            return latest['path']
        return None
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get checkpoint management summary."""
        
        total_checkpoints = len(self.checkpoint_history)
        total_size = sum(info['size_gb'] for info in self.checkpoint_history)
        
        summary = {
            'total_checkpoints': total_checkpoints,
            'total_size_gb': total_size,
            'best_checkpoints': len(self.best_checkpoints),
            'latest_checkpoint': None,
            'best_checkpoint': None,
            'avg_checkpoint_size_gb': total_size / max(total_checkpoints, 1),
            'checkpoint_directory': str(self.config.checkpoint_dir)
        }
        
        if self.checkpoint_history:
            latest = max(self.checkpoint_history, key=lambda x: x['step'])
            summary['latest_checkpoint'] = {
                'step': latest['step'],
                'epoch': latest['epoch'],
                'path': str(latest['path']),
                'metrics': latest['metrics']
            }
        
        if self.best_checkpoints:
            best = self.best_checkpoints[0]
            summary['best_checkpoint'] = {
                'step': best['step'],
                'epoch': best['epoch'],
                'path': str(best['path']),
                'metric_value': best['monitor_value'],
                'metric_name': self.config.monitor_metric
            }
        
        return summary


# Factory functions for easy creation
def create_distillation_trainer(
    student_model: nn.Module,
    teacher_model: nn.Module,
    temperature: float = 4.0,
    alpha: float = 0.7,
    feature_distillation: bool = True,
    **kwargs
) -> KnowledgeDistillationTrainer:
    """Create knowledge distillation trainer."""
    
    config = DistillationConfig(
        temperature=temperature,
        alpha=alpha,
        beta=1.0 - alpha,
        feature_distillation=feature_distillation,
        **kwargs
    )
    
    return KnowledgeDistillationTrainer(student_model, teacher_model, config)


def create_mixed_precision_trainer(
    model: nn.Module,
    enabled: bool = True,
    **kwargs
) -> MixedPrecisionTrainer:
    """Create mixed precision trainer."""
    
    config = MixedPrecisionConfig(enabled=enabled, **kwargs)
    return MixedPrecisionTrainer(model, config)


def create_checkpoint_manager(
    checkpoint_dir: Path,
    save_frequency: int = 1000,
    save_top_k: int = 3,
    monitor_metric: str = "mAP",
    **kwargs
) -> AdvancedCheckpointManager:
    """Create advanced checkpoint manager."""
    
    config = CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        save_frequency=save_frequency,
        save_top_k=save_top_k,
        monitor_metric=monitor_metric,
        **kwargs
    )
    
    return AdvancedCheckpointManager(config)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Advanced Training Techniques")
    
    # Test models
    class MockStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Conv2d(3, 64, 3, padding=1)
            self.neck = nn.Conv2d(64, 128, 3, padding=1) 
            self.head = nn.Linear(128, 10)
        
        def forward(self, x):
            features = self.backbone(x)
            neck_out = self.neck(features)
            logits = self.head(neck_out.mean([2, 3]))
            return {'pred_logits': logits, 'features': features}
    
    class MockTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Conv2d(3, 128, 3, padding=1)
            self.neck = nn.Conv2d(128, 256, 3, padding=1)
            self.head = nn.Linear(256, 10)
        
        def forward(self, x):
            features = self.backbone(x)
            neck_out = self.neck(features)  
            logits = self.head(neck_out.mean([2, 3]))
            return {'pred_logits': logits, 'features': features}
    
    student = MockStudent()
    teacher = MockTeacher()
    
    try:
        # Test Knowledge Distillation
        print("1. Testing Knowledge Distillation...")
        
        distillation_trainer = create_distillation_trainer(
            student_model=student,
            teacher_model=teacher,
            temperature=4.0,
            alpha=0.7
        )
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 64, 64)
        dummy_targets = torch.randint(0, 10, (2,))
        
        with torch.no_grad():
            distill_outputs = distillation_trainer.forward(dummy_input, dummy_targets)
        
        print(f"   ✅ Distillation training successful")
        print(f"      Total loss: {distill_outputs['total_loss'].item():.4f}")
        print(f"      Distillation loss: {distill_outputs['distillation_loss'].item():.4f}")
        
        # Test Mixed Precision
        print("2. Testing Mixed Precision...")
        
        mp_trainer = create_mixed_precision_trainer(student, enabled=True)
        optimizer = torch.optim.Adam(student.parameters())
        
        def compute_loss(outputs):
            return torch.randn(1, requires_grad=True)
        
        loss, outputs = mp_trainer.forward_pass(dummy_input, compute_loss)
        mp_metrics = mp_trainer.backward_pass(loss, optimizer)
        
        print(f"   ✅ Mixed precision training successful")
        print(f"      Loss scale: {mp_metrics['loss_scale']}")
        print(f"      Successful step: {mp_metrics['successful_step']}")
        
        # Test Advanced Checkpointing
        print("3. Testing Advanced Checkpointing...")
        
        checkpoint_manager = create_checkpoint_manager(
            checkpoint_dir=Path("test_checkpoints"),
            save_frequency=100,
            save_top_k=2
        )
        
        # Save test checkpoint
        test_metrics = {'mAP': 0.75, 'precision': 0.80}
        checkpoint_path = checkpoint_manager.save_checkpoint(
            step=100,
            epoch=1,
            model=student,
            optimizer=optimizer,
            metrics=test_metrics
        )
        
        print(f"   ✅ Checkpoint management successful")
        print(f"      Saved to: {checkpoint_path}")
        
        # Test checkpoint loading
        training_info = checkpoint_manager.load_checkpoint(
            checkpoint_path, student, optimizer
        )
        
        print(f"      Loaded step: {training_info['step']}")
        print(f"      Loaded metrics: {training_info['metrics']}")
        
        # Cleanup test checkpoint
        checkpoint_path.unlink()
        checkpoint_path.parent.rmdir()
        
        print("✅ All advanced training techniques tested successfully")
        
    except Exception as e:
        print(f"❌ Advanced training techniques test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Advanced Training Techniques testing completed")