"""
Advanced AdamW Optimizer Configuration for RF-DETR Training
Optimized for transformer-based detection models with layer-wise learning rates
"""
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, List, Any, Optional, Union, Iterator
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class LayerWiseConfig:
    """Configuration for layer-wise learning rate decay."""
    
    # Base learning rates
    base_lr: float = 1e-4
    encoder_lr: float = 1.5e-4
    
    # Decay factors
    vit_layer_decay: float = 0.8
    component_decay: float = 0.7
    
    # Special component learning rates
    backbone_lr_multiplier: float = 0.8
    neck_lr_multiplier: float = 1.0
    head_lr_multiplier: float = 1.2
    
    # Layer-specific configurations
    freeze_backbone_epochs: int = 0
    freeze_batch_norm: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.vit_layer_decay <= 1.0:
            raise ValueError(f"vit_layer_decay must be in (0, 1], got {self.vit_layer_decay}")
        if not 0 < self.component_decay <= 1.0:
            raise ValueError(f"component_decay must be in (0, 1], got {self.component_decay}")


class RFDETRAdamWOptimizer:
    """
    Advanced AdamW optimizer with layer-wise learning rate decay for RF-DETR.
    Implements sophisticated parameter grouping and learning rate strategies.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: LayerWiseConfig,
        weight_decay: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False
    ):
        """
        Initialize RF-DETR AdamW optimizer with layer-wise configuration.
        
        Args:
            model: RF-DETR model
            config: Layer-wise learning rate configuration
            weight_decay: L2 regularization strength
            betas: Adam beta parameters
            eps: Numerical stability epsilon
            amsgrad: Whether to use AMSGrad variant
        """
        self.model = model
        self.config = config
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad
        
        # Parameter groups with layer-wise learning rates
        self.param_groups = self._create_parameter_groups()
        
        # Create AdamW optimizer
        self.optimizer = AdamW(
            self.param_groups,
            lr=config.base_lr,  # Will be overridden by param groups
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        
        logger.info(f"Created RFDETRAdamWOptimizer with {len(self.param_groups)} parameter groups")
        self._log_parameter_statistics()
    
    def _create_parameter_groups(self) -> List[Dict[str, Any]]:
        """Create parameter groups with layer-wise learning rates."""
        
        param_groups = []
        processed_params = set()
        
        # Group 1: Vision Transformer Backbone (layer-wise decay)
        vit_groups = self._create_vit_parameter_groups()
        param_groups.extend(vit_groups)
        for group in vit_groups:
            processed_params.update(id(p) for p in group['params'])
        
        # Group 2: Neck/FPN components
        neck_params = self._get_neck_parameters()
        if neck_params:
            neck_group = {
                'params': neck_params,
                'lr': self.config.base_lr * self.config.neck_lr_multiplier,
                'weight_decay': self.weight_decay,
                'name': 'neck'
            }
            param_groups.append(neck_group)
            processed_params.update(id(p) for p in neck_params)
        
        # Group 3: Detection Head
        head_params = self._get_head_parameters()
        if head_params:
            head_group = {
                'params': head_params,
                'lr': self.config.base_lr * self.config.head_lr_multiplier,
                'weight_decay': self.weight_decay,
                'name': 'detection_head'
            }
            param_groups.append(head_group)
            processed_params.update(id(p) for p in head_params)
        
        # Group 4: Bias parameters (no weight decay)
        bias_params = self._get_bias_parameters(processed_params)
        if bias_params:
            bias_group = {
                'params': bias_params,
                'lr': self.config.base_lr,
                'weight_decay': 0.0,  # No weight decay for biases
                'name': 'biases'
            }
            param_groups.append(bias_group)
            processed_params.update(id(p) for p in bias_params)
        
        # Group 5: Normalization layers (no weight decay)
        norm_params = self._get_normalization_parameters(processed_params)
        if norm_params:
            norm_group = {
                'params': norm_params,
                'lr': self.config.base_lr,
                'weight_decay': 0.0,  # No weight decay for normalization
                'name': 'normalization'
            }
            param_groups.append(norm_group)
            processed_params.update(id(p) for p in norm_params)
        
        # Group 6: Remaining parameters
        remaining_params = self._get_remaining_parameters(processed_params)
        if remaining_params:
            remaining_group = {
                'params': remaining_params,
                'lr': self.config.base_lr,
                'weight_decay': self.weight_decay,
                'name': 'remaining'
            }
            param_groups.append(remaining_group)
        
        return param_groups
    
    def _create_vit_parameter_groups(self) -> List[Dict[str, Any]]:
        """Create layer-wise parameter groups for Vision Transformer backbone."""
        
        vit_groups = []
        
        # Find ViT encoder layers
        encoder_layers = self._find_encoder_layers()
        
        if not encoder_layers:
            logger.warning("No ViT encoder layers found, using default backbone grouping")
            return self._create_default_backbone_groups()
        
        num_layers = len(encoder_layers)
        logger.info(f"Found {num_layers} ViT encoder layers for layer-wise decay")
        
        # Create groups with exponential decay
        for layer_idx, layer_params in enumerate(encoder_layers):
            if layer_params:
                # Calculate learning rate with layer-wise decay
                # Later layers (closer to output) get higher learning rates
                decay_factor = self.config.vit_layer_decay ** (num_layers - 1 - layer_idx)
                layer_lr = self.config.encoder_lr * decay_factor * self.config.backbone_lr_multiplier
                
                group = {
                    'params': layer_params,
                    'lr': layer_lr,
                    'weight_decay': self.weight_decay,
                    'name': f'encoder_layer_{layer_idx}'
                }
                vit_groups.append(group)
        
        # Add embedding and positional encoding parameters
        embedding_params = self._get_embedding_parameters()
        if embedding_params:
            # Embeddings get lowest learning rate
            embedding_lr = self.config.encoder_lr * (self.config.vit_layer_decay ** num_layers) * self.config.backbone_lr_multiplier
            embedding_group = {
                'params': embedding_params,
                'lr': embedding_lr,
                'weight_decay': self.weight_decay,
                'name': 'embeddings'
            }
            vit_groups.append(embedding_group)
        
        return vit_groups
    
    def _find_encoder_layers(self) -> List[List[nn.Parameter]]:
        """Find and group ViT encoder layer parameters."""
        
        encoder_layers = []
        
        # Common patterns for ViT encoder layers
        layer_patterns = [
            r'encoder\.layer\.(\d+)',
            r'backbone\.encoder\.layer\.(\d+)',
            r'transformer\.encoder\.layers\.(\d+)',
            r'model\.encoder\.layer\.(\d+)'
        ]
        
        # Group parameters by layer index
        layer_params = {}
        
        for name, param in self.model.named_parameters():
            for pattern in layer_patterns:
                match = re.search(pattern, name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in layer_params:
                        layer_params[layer_idx] = []
                    layer_params[layer_idx].append(param)
                    break
        
        # Convert to ordered list
        if layer_params:
            max_layer = max(layer_params.keys())
            for i in range(max_layer + 1):
                encoder_layers.append(layer_params.get(i, []))
        
        return encoder_layers
    
    def _create_default_backbone_groups(self) -> List[Dict[str, Any]]:
        """Create default backbone parameter groups when ViT structure is not found."""
        
        backbone_params = []
        
        # Common backbone parameter patterns
        backbone_patterns = [
            r'backbone\.',
            r'encoder\.',
            r'model\.encoder\.'
        ]
        
        for name, param in self.model.named_parameters():
            if any(re.search(pattern, name) for pattern in backbone_patterns):
                backbone_params.append(param)
        
        if backbone_params:
            group = {
                'params': backbone_params,
                'lr': self.config.encoder_lr * self.config.backbone_lr_multiplier,
                'weight_decay': self.weight_decay,
                'name': 'backbone'
            }
            return [group]
        
        return []
    
    def _get_embedding_parameters(self) -> List[nn.Parameter]:
        """Get embedding and positional encoding parameters."""
        
        embedding_params = []
        
        # Common embedding parameter patterns
        embedding_patterns = [
            r'embeddings\.',
            r'pos_embed',
            r'position_embedding',
            r'patch_embed',
            r'cls_token'
        ]
        
        for name, param in self.model.named_parameters():
            if any(re.search(pattern, name) for pattern in embedding_patterns):
                embedding_params.append(param)
        
        return embedding_params
    
    def _get_neck_parameters(self) -> List[nn.Parameter]:
        """Get neck/FPN parameters."""
        
        neck_params = []
        
        # Common neck parameter patterns
        neck_patterns = [
            r'neck\.',
            r'fpn\.',
            r'projector\.',
            r'adapter\.'
        ]
        
        for name, param in self.model.named_parameters():
            if any(re.search(pattern, name) for pattern in neck_patterns):
                neck_params.append(param)
        
        return neck_params
    
    def _get_head_parameters(self) -> List[nn.Parameter]:
        """Get detection head parameters."""
        
        head_params = []
        
        # Common head parameter patterns
        head_patterns = [
            r'head\.',
            r'class_embed',
            r'bbox_embed',
            r'query_embed',
            r'decoder\.'
        ]
        
        for name, param in self.model.named_parameters():
            if any(re.search(pattern, name) for pattern in head_patterns):
                head_params.append(param)
        
        return head_params
    
    def _get_bias_parameters(self, processed_params: set) -> List[nn.Parameter]:
        """Get bias parameters that haven't been processed yet."""
        
        bias_params = []
        
        for name, param in self.model.named_parameters():
            if (id(param) not in processed_params and 
                'bias' in name.lower() and 
                param.requires_grad):
                bias_params.append(param)
        
        return bias_params
    
    def _get_normalization_parameters(self, processed_params: set) -> List[nn.Parameter]:
        """Get normalization layer parameters that haven't been processed yet."""
        
        norm_params = []
        
        # Common normalization parameter patterns
        norm_patterns = [
            r'norm\.',
            r'bn\.',
            r'batch_norm',
            r'layer_norm',
            r'group_norm',
            r'\.norm$',
            r'\.bn$'
        ]
        
        for name, param in self.model.named_parameters():
            if (id(param) not in processed_params and 
                any(re.search(pattern, name) for pattern in norm_patterns) and
                param.requires_grad):
                norm_params.append(param)
        
        return norm_params
    
    def _get_remaining_parameters(self, processed_params: set) -> List[nn.Parameter]:
        """Get any remaining parameters that haven't been processed."""
        
        remaining_params = []
        
        for param in self.model.parameters():
            if id(param) not in processed_params and param.requires_grad:
                remaining_params.append(param)
        
        return remaining_params
    
    def _log_parameter_statistics(self):
        """Log parameter group statistics."""
        
        total_params = 0
        
        logger.info("Parameter group statistics:")
        for i, group in enumerate(self.param_groups):
            group_params = sum(p.numel() for p in group['params'])
            total_params += group_params
            
            logger.info(f"  Group {i} ({group['name']}): "
                       f"{group_params:,} parameters, "
                       f"lr={group['lr']:.2e}, "
                       f"weight_decay={group.get('weight_decay', 0.0)}")
        
        logger.info(f"Total optimized parameters: {total_params:,}")
    
    def get_optimizer(self) -> AdamW:
        """Get the underlying AdamW optimizer."""
        return self.optimizer
    
    def get_param_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups for inspection."""
        return self.param_groups
    
    def get_learning_rates(self) -> Dict[str, float]:
        """Get current learning rates for all parameter groups."""
        
        learning_rates = {}
        for group in self.optimizer.param_groups:
            learning_rates[group.get('name', 'unnamed')] = group['lr']
        
        return learning_rates
    
    def set_learning_rate(self, lr: float, group_name: Optional[str] = None):
        """
        Set learning rate for specific group or all groups.
        
        Args:
            lr: New learning rate
            group_name: Name of specific group, or None for all groups
        """
        
        for group in self.optimizer.param_groups:
            if group_name is None or group.get('name') == group_name:
                group['lr'] = lr
        
        if group_name:
            logger.debug(f"Set learning rate for {group_name}: {lr:.2e}")
        else:
            logger.debug(f"Set learning rate for all groups: {lr:.2e}")
    
    def scale_learning_rates(self, factor: float, group_name: Optional[str] = None):
        """
        Scale learning rates by a factor.
        
        Args:
            factor: Scaling factor
            group_name: Name of specific group, or None for all groups
        """
        
        for group in self.optimizer.param_groups:
            if group_name is None or group.get('name') == group_name:
                group['lr'] *= factor
        
        if group_name:
            logger.debug(f"Scaled learning rate for {group_name} by {factor}")
        else:
            logger.debug(f"Scaled learning rates for all groups by {factor}")
    
    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze or unfreeze backbone parameters.
        
        Args:
            freeze: Whether to freeze backbone parameters
        """
        
        backbone_patterns = [r'backbone\.', r'encoder\.', r'embeddings\.']
        
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if any(re.search(pattern, name) for pattern in backbone_patterns):
                param.requires_grad = not freeze
                frozen_count += 1
        
        status = "Frozen" if freeze else "Unfrozen"
        logger.info(f"{status} {frozen_count} backbone parameters")
    
    def get_optimizer_state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict with additional metadata."""
        
        state_dict = self.optimizer.state_dict()
        state_dict['config'] = {
            'layer_wise_config': self.config.__dict__,
            'weight_decay': self.weight_decay,
            'betas': self.betas,
            'eps': self.eps,
            'amsgrad': self.amsgrad
        }
        
        return state_dict
    
    def load_optimizer_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict."""
        
        # Extract config if present
        if 'config' in state_dict:
            config = state_dict.pop('config')
            logger.info(f"Loaded optimizer config: {config}")
        
        self.optimizer.load_state_dict(state_dict)
        logger.info("Loaded optimizer state dict")