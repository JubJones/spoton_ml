"""
Advanced Data Augmentation Pipeline for RF-DETR Training
Surveillance-optimized augmentations for person detection in MTMMC dataset
"""

from .base import BaseAugmentation, AugmentationPipeline
from .geometric import GeometricAugmentations
from .photometric import PhotometricAugmentations
from .occlusion import OcclusionAugmentations
from .surveillance import SurveillanceAugmentations
from .pipeline import RFDETRAugmentationPipeline

__all__ = [
    'BaseAugmentation',
    'AugmentationPipeline', 
    'GeometricAugmentations',
    'PhotometricAugmentations',
    'OcclusionAugmentations',
    'SurveillanceAugmentations',
    'RFDETRAugmentationPipeline'
]