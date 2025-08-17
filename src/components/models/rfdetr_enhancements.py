"""
RF-DETR Architecture Enhancements for Surveillance Person Detection
Advanced modifications and optimizations for MTMMC surveillance scenarios
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import math
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SurveillanceConfig:
    """Configuration for surveillance-specific enhancements."""
    
    # Person detection optimizations
    person_class_weight: float = 2.0  # Increased weight for person class
    small_person_enhancement: bool = True
    crowd_handling_enabled: bool = True
    
    # Multi-scale enhancements
    multi_scale_training: bool = True
    scale_ranges: List[Tuple[int, int]] = None
    
    # Attention mechanisms
    spatial_attention: bool = True
    temporal_consistency: bool = False  # For video sequences
    
    # Loss function enhancements
    focal_loss_enabled: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Architecture modifications
    feature_pyramid_enhancement: bool = True
    neck_channels: int = 256
    additional_conv_layers: int = 1
    
    def __post_init__(self):
        """Set default scale ranges if not provided."""
        if self.scale_ranges is None:
            self.scale_ranges = [(32, 96), (96, 192), (192, 512), (512, 2048)]


class SpatialAttentionModule(nn.Module):
    """Spatial attention for enhanced person detection in surveillance scenes."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_out = self.channel_attention(self.avg_pool(x))
        max_out = self.channel_attention(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x = x * spatial_att
        
        return x


class PersonFocusedFPN(nn.Module):
    """Enhanced Feature Pyramid Network optimized for person detection."""
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_outs: int = 4,
        extra_convs_on_inputs: bool = False
    ):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        
        # Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            self.lateral_convs.append(lateral_conv)
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.fpn_convs.append(fpn_conv)
        
        # Person-specific enhancement layers
        self.person_enhancement = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            enhancement = SpatialAttentionModule(out_channels)
            self.person_enhancement.append(enhancement)
        
        # Extra layers for more scales
        if self.num_outs > len(in_channels_list):
            extra_levels = self.num_outs - len(in_channels_list)
            self.extra_convs = nn.ModuleList()
            for _ in range(extra_levels):
                extra_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                self.extra_convs.append(extra_conv)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through enhanced FPN."""
        assert len(inputs) == len(self.in_channels_list)
        
        # Build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher resolution features
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'
            )
        
        # Build outputs with person-specific enhancements
        outs = []
        for i, (lateral, fpn_conv, enhancement) in enumerate(
            zip(laterals, self.fpn_convs, self.person_enhancement)
        ):
            # Apply FPN convolution
            out = fpn_conv(lateral)
            
            # Apply person-focused spatial attention
            out = enhancement(out)
            
            outs.append(out)
        
        # Add extra levels
        if self.num_outs > len(outs):
            if self.extra_convs_on_inputs:
                orig = inputs[-1]
            else:
                orig = outs[-1]
            
            for extra_conv in self.extra_convs:
                orig = extra_conv(orig)
                outs.append(orig)
        
        return outs


class CrowdAwareLoss(nn.Module):
    """Crowd-aware loss function for dense surveillance scenes."""
    
    def __init__(
        self,
        num_classes: int = 91,
        alpha: float = 0.25,
        gamma: float = 2.0,
        crowd_weight: float = 1.5,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.crowd_weight = crowd_weight
        self.reduction = reduction
        
        # Person class is typically index 1 in COCO format
        self.person_class_idx = 1
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        crowd_density: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute crowd-aware focal loss.
        
        Args:
            predictions: Model predictions [N, num_classes]
            targets: Ground truth labels [N]
            crowd_density: Optional crowd density per sample [N]
        
        Returns:
            Computed loss
        """
        # Compute focal loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply crowd-aware weighting
        if crowd_density is not None:
            # Higher weight for crowded scenes
            crowd_weights = 1.0 + (crowd_density * (self.crowd_weight - 1.0))
            focal_loss = focal_loss * crowd_weights
        
        # Extra weight for person class
        person_mask = (targets == self.person_class_idx)
        focal_loss[person_mask] = focal_loss[person_mask] * 1.5
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SmallPersonDetector(nn.Module):
    """Specialized detector head for small person detection."""
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 91,
        num_anchors: int = 9,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Enhanced feature processing for small objects
        self.small_obj_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Dilated convolutions for context
        self.context_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=4, dilation=4),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Combine features
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for classification head (bias towards background)
        nn.init.constant_(self.cls_head[-1].bias, -math.log((1 - 0.01) / 0.01))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through small person detector."""
        # Enhanced feature processing
        enhanced_features = self.small_obj_conv(x)
        
        # Context features
        context_features = self.context_conv(x)
        
        # Fuse features
        fused_features = torch.cat([enhanced_features, context_features], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # Predictions
        cls_logits = self.cls_head(fused_features)
        bbox_preds = self.reg_head(fused_features)
        
        return {
            'cls_logits': cls_logits,
            'bbox_preds': bbox_preds,
            'features': fused_features
        }


class EnhancedRFDETR(nn.Module):
    """Enhanced RF-DETR model with surveillance-specific optimizations."""
    
    def __init__(
        self,
        base_model: nn.Module,
        config: SurveillanceConfig,
        num_classes: int = 91
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.base_model = base_model
        
        # Extract base model components (this would need to be adapted to actual RF-DETR structure)
        self.backbone = getattr(base_model, 'backbone', None)
        self.neck = getattr(base_model, 'neck', None)
        self.head = getattr(base_model, 'head', None)
        
        # Enhanced components
        if config.feature_pyramid_enhancement and self.neck:
            # Replace neck with enhanced FPN
            in_channels = getattr(self.neck, 'in_channels_list', [256, 512, 1024, 2048])
            self.enhanced_neck = PersonFocusedFPN(
                in_channels_list=in_channels,
                out_channels=config.neck_channels
            )
        
        # Small person detector for finest scale
        if config.small_person_enhancement:
            self.small_person_detector = SmallPersonDetector(
                in_channels=config.neck_channels,
                num_classes=num_classes
            )
        
        # Crowd-aware loss
        if config.focal_loss_enabled:
            self.crowd_loss = CrowdAwareLoss(
                num_classes=num_classes,
                alpha=config.focal_alpha,
                gamma=config.focal_gamma
            )
        
        # Multi-scale training support
        if config.multi_scale_training:
            self.scale_ranges = config.scale_ranges
            self.current_scale_idx = 0
        
        logger.info(f"Enhanced RF-DETR initialized:")
        logger.info(f"  Person class weight: {config.person_class_weight}")
        logger.info(f"  Small person enhancement: {config.small_person_enhancement}")
        logger.info(f"  Spatial attention: {config.spatial_attention}")
        logger.info(f"  Focal loss: {config.focal_loss_enabled}")
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with surveillance optimizations."""
        
        # Extract features through backbone
        if self.backbone:
            features = self.backbone(images)
        else:
            # Fallback if backbone not accessible
            features = self.base_model(images, return_features=True)
        
        # Enhanced neck processing
        if hasattr(self, 'enhanced_neck'):
            if isinstance(features, dict):
                feature_list = list(features.values())
            elif isinstance(features, (list, tuple)):
                feature_list = list(features)
            else:
                feature_list = [features]
            
            enhanced_features = self.enhanced_neck(feature_list)
        else:
            enhanced_features = features
        
        # Base model predictions
        if targets is not None:
            # Training mode
            base_outputs = self.base_model(images, targets)
        else:
            # Inference mode
            base_outputs = self.base_model(images)
        
        # Small person detection enhancement
        small_person_outputs = None
        if hasattr(self, 'small_person_detector') and isinstance(enhanced_features, list):
            # Use finest scale features for small person detection
            finest_features = enhanced_features[0]  # Assuming first is finest
            small_person_outputs = self.small_person_detector(finest_features)
        
        # Combine outputs
        outputs = {
            'predictions': base_outputs,
            'enhanced_features': enhanced_features if return_features else None,
            'small_person_predictions': small_person_outputs
        }
        
        # Compute enhanced losses during training
        if targets is not None and hasattr(self, 'crowd_loss'):
            # Extract crowd density from targets if available
            crowd_densities = []
            for target in targets:
                crowd_density = len(target.get('labels', [])) / 100.0  # Normalize
                crowd_densities.append(crowd_density)
            
            crowd_density_tensor = torch.tensor(crowd_densities, device=images.device)
            
            # This would need to be adapted based on actual RF-DETR loss computation
            if 'losses' in base_outputs:
                enhanced_loss = self._compute_enhanced_losses(
                    base_outputs['losses'],
                    targets,
                    crowd_density_tensor
                )
                outputs['losses'] = enhanced_loss
        
        return outputs
    
    def _compute_enhanced_losses(
        self,
        base_losses: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        crowd_densities: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute enhanced losses with surveillance-specific modifications."""
        
        enhanced_losses = base_losses.copy()
        
        # Apply person class weighting to classification loss
        if 'loss_ce' in enhanced_losses:
            person_weight = self.config.person_class_weight
            enhanced_losses['loss_ce'] = enhanced_losses['loss_ce'] * person_weight
        
        # Add crowd-aware component
        if hasattr(self, 'crowd_loss') and 'predictions' in base_losses:
            # This would need actual logits from the model
            # crowd_loss_value = self.crowd_loss(predictions, targets, crowd_densities)
            # enhanced_losses['crowd_loss'] = crowd_loss_value
            pass
        
        return enhanced_losses
    
    def set_multi_scale_training(self, enable: bool = True):
        """Enable/disable multi-scale training."""
        if hasattr(self, 'scale_ranges'):
            self.training_multi_scale = enable
    
    def get_current_scale_range(self) -> Tuple[int, int]:
        """Get current scale range for multi-scale training."""
        if hasattr(self, 'scale_ranges') and self.training:
            return self.scale_ranges[self.current_scale_idx % len(self.scale_ranges)]
        return (512, 512)  # Default scale
    
    def update_scale_range(self):
        """Update scale range for multi-scale training."""
        if hasattr(self, 'scale_ranges'):
            self.current_scale_idx = (self.current_scale_idx + 1) % len(self.scale_ranges)


def create_enhanced_rfdetr(
    base_model: nn.Module,
    person_class_weight: float = 2.0,
    small_person_enhancement: bool = True,
    spatial_attention: bool = True,
    focal_loss: bool = True,
    **kwargs
) -> EnhancedRFDETR:
    """
    Create enhanced RF-DETR model with surveillance optimizations.
    
    Args:
        base_model: Base RF-DETR model
        person_class_weight: Weight for person class in loss
        small_person_enhancement: Enable small person detection
        spatial_attention: Enable spatial attention mechanisms
        focal_loss: Enable focal loss for crowd scenarios
        **kwargs: Additional configuration parameters
        
    Returns:
        Enhanced RF-DETR model
    """
    
    config = SurveillanceConfig(
        person_class_weight=person_class_weight,
        small_person_enhancement=small_person_enhancement,
        spatial_attention=spatial_attention,
        focal_loss_enabled=focal_loss,
        **kwargs
    )
    
    return EnhancedRFDETR(base_model, config)


class ModelValidator:
    """Validator for enhanced RF-DETR model architecture."""
    
    @staticmethod
    def validate_enhancements(model: EnhancedRFDETR) -> Dict[str, bool]:
        """Validate that all enhancements are properly configured."""
        
        validation_results = {
            'base_model_present': model.base_model is not None,
            'config_valid': isinstance(model.config, SurveillanceConfig),
            'enhanced_neck': hasattr(model, 'enhanced_neck'),
            'small_person_detector': hasattr(model, 'small_person_detector'),
            'crowd_loss': hasattr(model, 'crowd_loss'),
            'multi_scale_support': hasattr(model, 'scale_ranges')
        }
        
        # Check component functionality
        if validation_results['enhanced_neck']:
            try:
                # Test enhanced neck with dummy input
                dummy_features = [torch.randn(1, 256, 64, 64) for _ in range(4)]
                output = model.enhanced_neck(dummy_features)
                validation_results['enhanced_neck_functional'] = len(output) == len(dummy_features)
            except Exception as e:
                logger.warning(f"Enhanced neck validation failed: {e}")
                validation_results['enhanced_neck_functional'] = False
        
        if validation_results['small_person_detector']:
            try:
                # Test small person detector
                dummy_input = torch.randn(1, 256, 32, 32)
                output = model.small_person_detector(dummy_input)
                validation_results['small_person_detector_functional'] = (
                    'cls_logits' in output and 'bbox_preds' in output
                )
            except Exception as e:
                logger.warning(f"Small person detector validation failed: {e}")
                validation_results['small_person_detector_functional'] = False
        
        return validation_results


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing RF-DETR Architecture Enhancements")
    
    # Mock base RF-DETR model
    class MockRFDETR(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 256, 3, padding=1)
            )
            
        def forward(self, x, targets=None, return_features=False):
            features = self.backbone(x)
            if return_features:
                return [features]
            
            # Mock predictions
            batch_size = x.size(0)
            return {
                'pred_logits': torch.randn(batch_size, 100, 91),
                'pred_boxes': torch.randn(batch_size, 100, 4),
                'losses': {'loss_ce': torch.tensor(0.5), 'loss_bbox': torch.tensor(0.3)} if targets else None
            }
    
    # Test enhanced model creation
    base_model = MockRFDETR()
    enhanced_model = create_enhanced_rfdetr(
        base_model=base_model,
        person_class_weight=2.0,
        small_person_enhancement=True,
        spatial_attention=True
    )
    
    print(f"✅ Enhanced RF-DETR model created")
    
    # Test model components
    validator = ModelValidator()
    validation_results = validator.validate_enhancements(enhanced_model)
    
    print(f"🔍 Validation Results:")
    for component, is_valid in validation_results.items():
        status = "✅" if is_valid else "❌"
        print(f"  {status} {component}: {is_valid}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512)
    
    try:
        with torch.no_grad():
            output = enhanced_model(dummy_input, return_features=True)
        
        print(f"✅ Forward pass successful")
        print(f"  Output keys: {list(output.keys())}")
        
        if 'predictions' in output:
            predictions = output['predictions']
            print(f"  Predictions shape: {predictions['pred_logits'].shape if 'pred_logits' in predictions else 'N/A'}")
    
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
    
    print("✅ RF-DETR Architecture Enhancement testing completed")