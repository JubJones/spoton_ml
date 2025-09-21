"""
Optimizer Factory for RF-DETR Training
Centralized creation and configuration of optimizers with best practices
"""
import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, Any, Optional, Type, Union
from .adamw_optimizer import RFDETRAdamWOptimizer, LayerWiseConfig
from .gradient_handler import GradientHandler, GradientClipConfig

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """Factory for creating optimized RF-DETR training configurations."""
    
    # Predefined optimizer configurations
    OPTIMIZER_CONFIGS = {
        "basic": {
            "description": "Basic AdamW configuration for development",
            "optimizer_type": "adamw",
            "layer_wise": {
                "base_lr": 1e-4,
                "encoder_lr": 1.2e-4,
                "vit_layer_decay": 0.85,
                "component_decay": 0.8,
                "backbone_lr_multiplier": 0.9,
                "freeze_backbone_epochs": 0
            },
            "optimizer_params": {
                "weight_decay": 1e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8
            },
            "gradient_clip": {
                "enabled": True,
                "clip_method": "norm",
                "max_norm": 1.0,
                "monitor_gradients": True
            }
        },
        
        "standard": {
            "description": "Standard AdamW configuration for general training",
            "optimizer_type": "adamw",
            "layer_wise": {
                "base_lr": 1e-4,
                "encoder_lr": 1.5e-4,
                "vit_layer_decay": 0.8,
                "component_decay": 0.7,
                "backbone_lr_multiplier": 0.8,
                "neck_lr_multiplier": 1.0,
                "head_lr_multiplier": 1.2,
                "freeze_backbone_epochs": 0,
                "freeze_batch_norm": True
            },
            "optimizer_params": {
                "weight_decay": 1e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "amsgrad": False
            },
            "gradient_clip": {
                "enabled": True,
                "clip_method": "norm",
                "max_norm": 1.0,
                "norm_type": 2.0,
                "monitor_gradients": True,
                "log_frequency": 100,
                "detect_anomalies": True
            }
        },
        
        "advanced": {
            "description": "Advanced AdamW with adaptive gradient clipping",
            "optimizer_type": "adamw",
            "layer_wise": {
                "base_lr": 1e-4,
                "encoder_lr": 1.5e-4,
                "vit_layer_decay": 0.8,
                "component_decay": 0.7,
                "backbone_lr_multiplier": 0.75,
                "neck_lr_multiplier": 1.0,
                "head_lr_multiplier": 1.5,
                "freeze_backbone_epochs": 2,
                "freeze_batch_norm": True
            },
            "optimizer_params": {
                "weight_decay": 1e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "amsgrad": False
            },
            "gradient_clip": {
                "enabled": True,
                "clip_method": "adaptive",
                "max_norm": 1.0,
                "norm_type": 2.0,
                "adaptive_percentile": 95.0,
                "adaptive_window_size": 100,
                "adaptive_min_norm": 0.1,
                "adaptive_max_norm": 5.0,
                "monitor_gradients": True,
                "log_frequency": 50,
                "detect_anomalies": True,
                "explosion_threshold": 50.0,
                "vanishing_threshold": 1e-6
            }
        },
        
        "surveillance_optimized": {
            "description": "Optimized for surveillance person detection on MTMMC",
            "optimizer_type": "adamw",
            "layer_wise": {
                "base_lr": 8e-5,  # Slightly lower for stability
                "encoder_lr": 1.2e-4,
                "vit_layer_decay": 0.82,  # Conservative decay
                "component_decay": 0.75,
                "backbone_lr_multiplier": 0.7,  # Lower backbone LR for fine-tuning
                "neck_lr_multiplier": 1.0,
                "head_lr_multiplier": 1.3,  # Higher head LR for person detection
                "freeze_backbone_epochs": 3,  # Freeze backbone initially
                "freeze_batch_norm": True
            },
            "optimizer_params": {
                "weight_decay": 1.2e-4,  # Slightly higher regularization
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "amsgrad": False
            },
            "gradient_clip": {
                "enabled": True,
                "clip_method": "adaptive",
                "max_norm": 0.8,  # More conservative clipping
                "norm_type": 2.0,
                "adaptive_percentile": 90.0,
                "adaptive_window_size": 150,
                "adaptive_min_norm": 0.05,
                "adaptive_max_norm": 3.0,
                "monitor_gradients": True,
                "log_frequency": 100,
                "detect_anomalies": True,
                "explosion_threshold": 20.0,
                "vanishing_threshold": 1e-7
            }
        },
        
        "production": {
            "description": "Production-ready configuration with conservative settings",
            "optimizer_type": "adamw",
            "layer_wise": {
                "base_lr": 5e-5,  # Conservative learning rate
                "encoder_lr": 8e-5,
                "vit_layer_decay": 0.9,  # Minimal decay
                "component_decay": 0.85,
                "backbone_lr_multiplier": 0.8,
                "neck_lr_multiplier": 1.0,
                "head_lr_multiplier": 1.1,
                "freeze_backbone_epochs": 5,  # Extended freeze period
                "freeze_batch_norm": True
            },
            "optimizer_params": {
                "weight_decay": 8e-5,  # Lower regularization for stability
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "amsgrad": False
            },
            "gradient_clip": {
                "enabled": True,
                "clip_method": "norm",
                "max_norm": 0.5,  # Conservative clipping
                "norm_type": 2.0,
                "monitor_gradients": True,
                "log_frequency": 200,
                "detect_anomalies": True,
                "explosion_threshold": 10.0,
                "vanishing_threshold": 1e-8
            }
        }
    }
    
    @classmethod
    def create_optimizer(
        self,
        model: nn.Module,
        config_name: str = "standard",
        custom_config: Optional[Dict[str, Any]] = None,
        device_optimization: bool = True
    ) -> tuple[RFDETRAdamWOptimizer, GradientHandler]:
        """
        Create optimizer and gradient handler for RF-DETR training.
        
        Args:
            model: RF-DETR model
            config_name: Predefined configuration name
            custom_config: Custom configuration overrides
            device_optimization: Apply device-specific optimizations
            
        Returns:
            Tuple of (optimizer, gradient_handler)
        """
        
        # Get base configuration
        if config_name not in self.OPTIMIZER_CONFIGS:
            logger.warning(f"Unknown config '{config_name}', using 'standard'")
            config_name = "standard"
        
        config = self.OPTIMIZER_CONFIGS[config_name].copy()
        
        # Apply custom overrides
        if custom_config:
            config = self._merge_configs(config, custom_config)
        
        # Apply device-specific optimizations
        if device_optimization:
            config = self._apply_device_optimizations(config, model)
        
        logger.info(f"Creating optimizer with config: {config['description']}")
        
        # Create layer-wise configuration
        layer_wise_config = LayerWiseConfig(**config["layer_wise"])
        
        # Create optimizer
        optimizer = RFDETRAdamWOptimizer(
            model=model,
            config=layer_wise_config,
            **config["optimizer_params"]
        )
        
        # Create gradient handler
        gradient_config = GradientClipConfig(**config["gradient_clip"])
        gradient_handler = GradientHandler(model, gradient_config)
        
        return optimizer, gradient_handler
    
    @classmethod
    def _merge_configs(self, base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge custom configuration with base configuration."""
        
        merged = base_config.copy()
        
        for key, value in custom_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @classmethod
    def _apply_device_optimizations(self, config: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
        """Apply device-specific optimizations."""
        
        device = next(model.parameters()).device
        
        if device.type == "mps":  # Apple Silicon optimization
            logger.info("Applying MPS (Apple Silicon) optimizations")
            
            # More conservative settings for MPS
            config["layer_wise"]["base_lr"] *= 0.8
            config["layer_wise"]["encoder_lr"] *= 0.8
            config["gradient_clip"]["max_norm"] *= 0.8
            config["optimizer_params"]["eps"] = 1e-7  # Better numerical stability
            
        elif device.type == "cpu":  # CPU optimization
            logger.info("Applying CPU optimizations")
            
            # Even more conservative for CPU training
            config["layer_wise"]["base_lr"] *= 0.5
            config["layer_wise"]["encoder_lr"] *= 0.5
            config["gradient_clip"]["max_norm"] = 0.3
            config["optimizer_params"]["eps"] = 1e-6
            
        elif device.type == "cuda":  # CUDA optimization
            logger.info("Applying CUDA optimizations")
            
            # Check for mixed precision training capability
            if torch.cuda.get_device_capability(device.index)[0] >= 7:  # Tensor Cores available
                logger.info("Tensor Cores detected, enabling advanced optimizations")
                config["optimizer_params"]["eps"] = 1e-8
            
        return config
    
    @classmethod
    def get_available_configs(self) -> Dict[str, str]:
        """Get all available optimizer configurations."""
        return {name: config["description"] for name, config in self.OPTIMIZER_CONFIGS.items()}
    
    @classmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate optimizer configuration."""
        
        try:
            # Validate required sections
            required_sections = ["layer_wise", "optimizer_params", "gradient_clip"]
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required section: {section}")
                    return False
            
            # Validate layer_wise config
            LayerWiseConfig(**config["layer_wise"])
            
            # Validate gradient clip config
            GradientClipConfig(**config["gradient_clip"])
            
            logger.info("Optimizer configuration is valid")
            return True
            
        except Exception as e:
            logger.error(f"Invalid optimizer configuration: {e}")
            return False
    
    @classmethod
    def create_custom_optimizer(
        self,
        model: nn.Module,
        base_lr: float = 1e-4,
        encoder_lr: float = 1.5e-4,
        weight_decay: float = 1e-4,
        gradient_clip_norm: float = 1.0,
        **kwargs
    ) -> tuple[RFDETRAdamWOptimizer, GradientHandler]:
        """
        Create optimizer with simple custom parameters.
        
        Args:
            model: RF-DETR model
            base_lr: Base learning rate
            encoder_lr: Encoder learning rate
            weight_decay: Weight decay factor
            gradient_clip_norm: Gradient clipping norm
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (optimizer, gradient_handler)
        """
        
        custom_config = {
            "layer_wise": {
                "base_lr": base_lr,
                "encoder_lr": encoder_lr,
                **kwargs
            },
            "optimizer_params": {
                "weight_decay": weight_decay
            },
            "gradient_clip": {
                "max_norm": gradient_clip_norm
            }
        }
        
        return self.create_optimizer(model, "standard", custom_config)


def create_rfdetr_optimizer(
    model: nn.Module,
    config_name: str = "surveillance_optimized",
    **kwargs
) -> tuple[RFDETRAdamWOptimizer, GradientHandler]:
    """
    Convenience function to create RF-DETR optimizer.
    
    Args:
        model: RF-DETR model
        config_name: Configuration name
        **kwargs: Additional configuration parameters
        
    Returns:
        Tuple of (optimizer, gradient_handler)
    """
    return OptimizerFactory.create_optimizer(model, config_name, kwargs)


def get_optimizer_recommendations(model: nn.Module, dataset_size: int) -> Dict[str, str]:
    """
    Get optimizer recommendations based on model and dataset characteristics.
    
    Args:
        model: RF-DETR model
        dataset_size: Size of training dataset
        
    Returns:
        Dictionary with recommendations
    """
    
    recommendations = {}
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get device info
    device = next(model.parameters()).device
    
    # Base recommendations
    if dataset_size < 1000:
        recommendations["config"] = "production"
        recommendations["reason"] = "Small dataset - conservative settings to prevent overfitting"
    elif dataset_size < 10000:
        recommendations["config"] = "surveillance_optimized"
        recommendations["reason"] = "Medium dataset - balanced settings for surveillance detection"
    else:
        recommendations["config"] = "advanced"
        recommendations["reason"] = "Large dataset - aggressive settings for maximum performance"
    
    # Device-specific recommendations
    if device.type == "cpu":
        recommendations["config"] = "basic"
        recommendations["device_note"] = "CPU training detected - using conservative settings"
    elif device.type == "mps":
        recommendations["device_note"] = "Apple Silicon detected - optimized for MPS"
    
    # Model size recommendations
    if trainable_params > 50_000_000:  # Large model
        recommendations["memory_note"] = "Large model - consider gradient checkpointing"
    
    recommendations.update({
        "total_params": f"{total_params:,}",
        "trainable_params": f"{trainable_params:,}",
        "dataset_size": dataset_size,
        "device": str(device)
    })
    
    return recommendations


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock model for testing
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(100, 50)
            self.neck = nn.Linear(50, 25)
            self.head = nn.Linear(25, 10)
        
        def forward(self, x):
            return self.head(self.neck(self.backbone(x)))
    
    model = MockModel()
    
    # Test optimizer creation
    optimizer, gradient_handler = create_rfdetr_optimizer(model, "surveillance_optimized")
    
    print(f"Created optimizer with {len(optimizer.get_param_groups())} parameter groups")
    print(f"Learning rates: {optimizer.get_learning_rates()}")
    
    # Test recommendations
    recommendations = get_optimizer_recommendations(model, 5000)
    print(f"Recommendations: {recommendations}")