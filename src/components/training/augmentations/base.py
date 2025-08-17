"""
Base classes for RF-DETR augmentation pipeline
"""
import random
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters."""
    probability: float = 0.5
    severity: float = 0.5  # 0.0 (weak) to 1.0 (strong)
    enabled: bool = True
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        
        # Validate probability
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Probability must be between 0.0 and 1.0, got {self.probability}")
        
        # Validate severity
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"Severity must be between 0.0 and 1.0, got {self.severity}")


@dataclass 
class BoundingBox:
    """Bounding box representation for augmentations."""
    x: float  # x coordinate (top-left)
    y: float  # y coordinate (top-left) 
    width: float
    height: float
    class_id: int = 0
    confidence: float = 1.0
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def to_cxcywh(self) -> Tuple[float, float, float, float]:
        """Convert to (center_x, center_y, width, height) format."""
        return (
            self.x + self.width / 2,
            self.y + self.height / 2,
            self.width,
            self.height
        )
    
    def area(self) -> float:
        """Calculate bounding box area."""
        return self.width * self.height
    
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (height/width)."""
        return self.height / self.width if self.width > 0 else 0
    
    def is_valid(self) -> bool:
        """Check if bounding box is valid."""
        return self.width > 0 and self.height > 0
    
    def clip_to_image(self, img_width: int, img_height: int) -> 'BoundingBox':
        """Clip bounding box to image boundaries."""
        x1, y1, x2, y2 = self.to_xyxy()
        
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        return BoundingBox(
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            class_id=self.class_id,
            confidence=self.confidence
        )


@dataclass
class AugmentationResult:
    """Result of applying augmentation to an image and bounding boxes."""
    image: Union[np.ndarray, Image.Image, torch.Tensor]
    bboxes: List[BoundingBox]
    applied_augmentations: List[str]
    transform_matrix: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAugmentation(ABC):
    """Base class for all augmentation techniques."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.name = self.__class__.__name__
        self._validate_config()
    
    def _validate_config(self):
        """Validate augmentation configuration."""
        if not isinstance(self.config, AugmentationConfig):
            raise TypeError("Config must be an AugmentationConfig instance")
    
    @abstractmethod
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """
        Apply augmentation to image and bounding boxes.
        
        Args:
            image: Input image as numpy array (H, W, C)
            bboxes: List of bounding boxes
            
        Returns:
            Tuple of (augmented_image, augmented_bboxes, metadata)
        """
        pass
    
    def __call__(
        self, 
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        bboxes: List[BoundingBox]
    ) -> AugmentationResult:
        """
        Apply augmentation with probability check.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes
            
        Returns:
            AugmentationResult with augmented data
        """
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:  # CHW -> HWC
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.cpu().numpy()
        
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:  # Normalized image
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        applied_augmentations = []
        metadata = {}
        
        # Apply augmentation based on probability
        if self.config.enabled and random.random() < self.config.probability:
            try:
                image, bboxes, aug_metadata = self._apply_augmentation(image, bboxes)
                applied_augmentations.append(self.name)
                metadata.update(aug_metadata)
                
                # Filter out invalid bounding boxes
                valid_bboxes = [bbox for bbox in bboxes if bbox.is_valid()]
                if len(valid_bboxes) != len(bboxes):
                    logger.debug(f"{self.name}: Filtered {len(bboxes) - len(valid_bboxes)} invalid bboxes")
                    bboxes = valid_bboxes
                
            except Exception as e:
                logger.warning(f"Error applying {self.name}: {e}")
                # Return original data on error
        
        return AugmentationResult(
            image=image,
            bboxes=bboxes,
            applied_augmentations=applied_augmentations,
            metadata=metadata
        )
    
    def set_severity(self, severity: float):
        """Adjust augmentation severity."""
        self.config.severity = max(0.0, min(1.0, severity))
    
    def enable(self):
        """Enable augmentation."""
        self.config.enabled = True
    
    def disable(self):
        """Disable augmentation."""
        self.config.enabled = False


class AugmentationPipeline:
    """Pipeline for applying multiple augmentations in sequence."""
    
    def __init__(
        self, 
        augmentations: List[BaseAugmentation],
        shuffle_order: bool = False,
        max_augmentations: Optional[int] = None
    ):
        self.augmentations = augmentations
        self.shuffle_order = shuffle_order
        self.max_augmentations = max_augmentations
        
        logger.info(f"Initialized augmentation pipeline with {len(augmentations)} augmentations")
        for aug in augmentations:
            logger.debug(f"  - {aug.name}: p={aug.config.probability}, severity={aug.config.severity}")
    
    def __call__(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        bboxes: List[BoundingBox]
    ) -> AugmentationResult:
        """
        Apply all augmentations in the pipeline.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes
            
        Returns:
            Final augmentation result
        """
        current_image = image
        current_bboxes = bboxes
        all_applied_augmentations = []
        all_metadata = {}
        
        # Select augmentations to apply
        augmentations_to_apply = self.augmentations.copy()
        
        if self.shuffle_order:
            random.shuffle(augmentations_to_apply)
        
        if self.max_augmentations is not None:
            augmentations_to_apply = augmentations_to_apply[:self.max_augmentations]
        
        # Apply augmentations sequentially
        for augmentation in augmentations_to_apply:
            if not current_bboxes:  # Skip if no bboxes left
                logger.debug("No bounding boxes left, skipping remaining augmentations")
                break
            
            result = augmentation(current_image, current_bboxes)
            
            current_image = result.image
            current_bboxes = result.bboxes
            all_applied_augmentations.extend(result.applied_augmentations)
            all_metadata.update(result.metadata)
        
        logger.debug(f"Applied augmentations: {all_applied_augmentations}")
        
        return AugmentationResult(
            image=current_image,
            bboxes=current_bboxes,
            applied_augmentations=all_applied_augmentations,
            metadata=all_metadata
        )
    
    def add_augmentation(self, augmentation: BaseAugmentation):
        """Add augmentation to pipeline."""
        self.augmentations.append(augmentation)
        logger.info(f"Added {augmentation.name} to pipeline")
    
    def remove_augmentation(self, augmentation_name: str):
        """Remove augmentation from pipeline by name."""
        self.augmentations = [aug for aug in self.augmentations if aug.name != augmentation_name]
        logger.info(f"Removed {augmentation_name} from pipeline")
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get statistics about augmentations in the pipeline."""
        stats = {
            "total_augmentations": len(self.augmentations),
            "enabled_augmentations": sum(1 for aug in self.augmentations if aug.config.enabled),
            "average_probability": np.mean([aug.config.probability for aug in self.augmentations]),
            "augmentation_details": []
        }
        
        for aug in self.augmentations:
            details = {
                "name": aug.name,
                "enabled": aug.config.enabled,
                "probability": aug.config.probability,
                "severity": aug.config.severity
            }
            stats["augmentation_details"].append(details)
        
        return stats
    
    def set_global_severity(self, severity: float):
        """Set severity for all augmentations in the pipeline."""
        for aug in self.augmentations:
            aug.set_severity(severity)
        logger.info(f"Set global severity to {severity}")
    
    def enable_all(self):
        """Enable all augmentations in the pipeline."""
        for aug in self.augmentations:
            aug.enable()
        logger.info("Enabled all augmentations")
    
    def disable_all(self):
        """Disable all augmentations in the pipeline."""
        for aug in self.augmentations:
            aug.disable()
        logger.info("Disabled all augmentations")


def create_bboxes_from_coco(annotations: List[Dict[str, Any]]) -> List[BoundingBox]:
    """
    Create BoundingBox objects from COCO format annotations.
    
    Args:
        annotations: List of COCO annotation dictionaries
        
    Returns:
        List of BoundingBox objects
    """
    bboxes = []
    for ann in annotations:
        bbox_data = ann.get('bbox', [])  # [x, y, width, height]
        if len(bbox_data) == 4:
            bbox = BoundingBox(
                x=bbox_data[0],
                y=bbox_data[1],
                width=bbox_data[2],
                height=bbox_data[3],
                class_id=ann.get('category_id', 0),
                confidence=1.0
            )
            if bbox.is_valid():
                bboxes.append(bbox)
    
    return bboxes


def convert_bboxes_to_coco(bboxes: List[BoundingBox]) -> List[Dict[str, Any]]:
    """
    Convert BoundingBox objects to COCO format.
    
    Args:
        bboxes: List of BoundingBox objects
        
    Returns:
        List of COCO format annotation dictionaries
    """
    annotations = []
    for bbox in bboxes:
        if bbox.is_valid():
            ann = {
                'bbox': [bbox.x, bbox.y, bbox.width, bbox.height],
                'category_id': bbox.class_id,
                'area': bbox.area(),
                'iscrowd': 0
            }
            annotations.append(ann)
    
    return annotations