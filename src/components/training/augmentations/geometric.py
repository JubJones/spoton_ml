"""
Geometric Augmentations for Surveillance Person Detection
Optimized for MTMMC dataset and RF-DETR training
"""
import random
import logging
import math
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
from .base import BaseAugmentation, AugmentationConfig, BoundingBox

logger = logging.getLogger(__name__)


class HorizontalFlipAugmentation(BaseAugmentation):
    """
    Horizontal flip augmentation optimized for surveillance person detection.
    Essential for simulating people moving in opposite directions.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.5,
                severity=1.0,  # Binary operation, severity not applicable
                enabled=True
            )
        super().__init__(config)
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply horizontal flip to image and bounding boxes."""
        
        h, w = image.shape[:2]
        
        # Flip image horizontally
        flipped_image = cv2.flip(image, 1)
        
        # Transform bounding boxes
        flipped_bboxes = []
        for bbox in bboxes:
            # Flip x coordinate: new_x = width - (old_x + old_width)
            new_x = w - (bbox.x + bbox.width)
            
            flipped_bbox = BoundingBox(
                x=new_x,
                y=bbox.y,
                width=bbox.width,
                height=bbox.height,
                class_id=bbox.class_id,
                confidence=bbox.confidence
            )
            
            # Ensure bbox is within image bounds
            flipped_bbox = flipped_bbox.clip_to_image(w, h)
            if flipped_bbox.is_valid():
                flipped_bboxes.append(flipped_bbox)
        
        metadata = {
            "flip_direction": "horizontal",
            "original_bboxes": len(bboxes),
            "filtered_bboxes": len(flipped_bboxes)
        }
        
        return flipped_image, flipped_bboxes, metadata


class RandomScaleAugmentation(BaseAugmentation):
    """
    Random scaling augmentation to simulate varying distances from cameras.
    Maintains aspect ratio for person detection.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.7,
                severity=0.5,
                enabled=True,
                parameters={
                    "scale_min": 0.8,
                    "scale_max": 1.2,
                    "maintain_aspect_ratio": True
                }
            )
        super().__init__(config)
    
    def _get_scale_range(self) -> Tuple[float, float]:
        """Calculate scale range based on severity."""
        params = self.config.parameters
        base_min = params.get("scale_min", 0.8)
        base_max = params.get("scale_max", 1.2)
        
        # Adjust range based on severity (higher severity = more variation)
        scale_variation = self.config.severity * 0.3  # Max 30% additional variation
        
        scale_min = max(0.5, base_min - scale_variation)
        scale_max = min(2.0, base_max + scale_variation)
        
        return scale_min, scale_max
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply random scaling to image and bounding boxes."""
        
        h, w = image.shape[:2]
        scale_min, scale_max = self._get_scale_range()
        
        # Generate random scale factor
        scale_factor = random.uniform(scale_min, scale_max)
        
        # Calculate new dimensions
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Resize image
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # If image is larger than original, crop from center
        if scale_factor > 1.0:
            # Calculate crop coordinates
            crop_x = (new_w - w) // 2
            crop_y = (new_h - h) // 2
            
            # Crop to original size
            scaled_image = scaled_image[crop_y:crop_y+h, crop_x:crop_x+w]
            
            # Adjust bounding boxes for cropping
            scaled_bboxes = []
            for bbox in bboxes:
                # Scale bbox coordinates
                scaled_x = bbox.x * scale_factor - crop_x
                scaled_y = bbox.y * scale_factor - crop_y
                scaled_width = bbox.width * scale_factor
                scaled_height = bbox.height * scale_factor
                
                scaled_bbox = BoundingBox(
                    x=scaled_x,
                    y=scaled_y,
                    width=scaled_width,
                    height=scaled_height,
                    class_id=bbox.class_id,
                    confidence=bbox.confidence
                )
                
                # Clip to image bounds and validate
                scaled_bbox = scaled_bbox.clip_to_image(w, h)
                if scaled_bbox.is_valid():
                    scaled_bboxes.append(scaled_bbox)
        
        else:  # scale_factor <= 1.0
            # Pad image to original size
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            
            # Create padded image
            padded_image = np.zeros((h, w, image.shape[2]), dtype=image.dtype)
            padded_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = scaled_image
            scaled_image = padded_image
            
            # Adjust bounding boxes for padding
            scaled_bboxes = []
            for bbox in bboxes:
                # Scale and translate bbox coordinates
                scaled_x = bbox.x * scale_factor + pad_x
                scaled_y = bbox.y * scale_factor + pad_y
                scaled_width = bbox.width * scale_factor
                scaled_height = bbox.height * scale_factor
                
                scaled_bbox = BoundingBox(
                    x=scaled_x,
                    y=scaled_y,
                    width=scaled_width,
                    height=scaled_height,
                    class_id=bbox.class_id,
                    confidence=bbox.confidence
                )
                
                # Clip to image bounds and validate
                scaled_bbox = scaled_bbox.clip_to_image(w, h)
                if scaled_bbox.is_valid():
                    scaled_bboxes.append(scaled_bbox)
        
        metadata = {
            "scale_factor": scale_factor,
            "original_size": (w, h),
            "scaled_size": (new_w, new_h),
            "original_bboxes": len(bboxes),
            "filtered_bboxes": len(scaled_bboxes)
        }
        
        return scaled_image, scaled_bboxes, metadata


class RandomTranslationAugmentation(BaseAugmentation):
    """
    Random translation augmentation to simulate camera movement and positioning variations.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.6,
                severity=0.4,
                enabled=True,
                parameters={
                    "max_translation_ratio": 0.1  # Max 10% of image dimension
                }
            )
        super().__init__(config)
    
    def _get_translation_range(self, img_width: int, img_height: int) -> Tuple[int, int]:
        """Calculate translation range based on severity."""
        max_ratio = self.config.parameters.get("max_translation_ratio", 0.1)
        
        # Scale translation by severity
        actual_ratio = max_ratio * self.config.severity
        
        max_tx = int(img_width * actual_ratio)
        max_ty = int(img_height * actual_ratio)
        
        return max_tx, max_ty
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply random translation to image and bounding boxes."""
        
        h, w = image.shape[:2]
        max_tx, max_ty = self._get_translation_range(w, h)
        
        # Generate random translation
        tx = random.randint(-max_tx, max_tx)
        ty = random.randint(-max_ty, max_ty)
        
        # Create translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply translation to image
        translated_image = cv2.warpAffine(image, translation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Transform bounding boxes
        translated_bboxes = []
        for bbox in bboxes:
            # Translate bbox coordinates
            new_x = bbox.x + tx
            new_y = bbox.y + ty
            
            translated_bbox = BoundingBox(
                x=new_x,
                y=new_y,
                width=bbox.width,
                height=bbox.height,
                class_id=bbox.class_id,
                confidence=bbox.confidence
            )
            
            # Clip to image bounds and validate
            translated_bbox = translated_bbox.clip_to_image(w, h)
            if translated_bbox.is_valid():
                translated_bboxes.append(translated_bbox)
        
        metadata = {
            "translation_x": tx,
            "translation_y": ty,
            "max_translation": (max_tx, max_ty),
            "original_bboxes": len(bboxes),
            "filtered_bboxes": len(translated_bboxes)
        }
        
        return translated_image, translated_bboxes, metadata


class PerspectiveAugmentation(BaseAugmentation):
    """
    Perspective transformation augmentation to simulate slight camera angle variations.
    Conservative angles to maintain realism for surveillance scenarios.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.3,
                severity=0.3,
                enabled=True,
                parameters={
                    "max_rotation_degrees": 5,
                    "max_perspective_ratio": 0.05  # Max 5% perspective distortion
                }
            )
        super().__init__(config)
    
    def _generate_perspective_transform(self, width: int, height: int) -> np.ndarray:
        """Generate random perspective transformation matrix."""
        max_angle = self.config.parameters.get("max_rotation_degrees", 5) * self.config.severity
        max_perspective = self.config.parameters.get("max_perspective_ratio", 0.05) * self.config.severity
        
        # Generate small random angle
        angle = random.uniform(-max_angle, max_angle)
        angle_rad = math.radians(angle)
        
        # Generate perspective distortion
        perspective_x = random.uniform(-max_perspective, max_perspective) * width
        perspective_y = random.uniform(-max_perspective, max_perspective) * height
        
        # Original corners
        src_points = np.float32([
            [0, 0],
            [width, 0], 
            [width, height],
            [0, height]
        ])
        
        # Apply rotation and perspective to corners
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        center_x, center_y = width / 2, height / 2
        
        dst_points = []
        for x, y in src_points:
            # Translate to origin
            x_centered = x - center_x
            y_centered = y - center_y
            
            # Apply rotation
            x_rot = x_centered * cos_a - y_centered * sin_a
            y_rot = x_centered * sin_a + y_centered * cos_a
            
            # Translate back
            x_final = x_rot + center_x
            y_final = y_rot + center_y
            
            # Add perspective distortion
            if x < width / 2:  # Left side
                x_final += perspective_x
            else:  # Right side
                x_final -= perspective_x
            
            if y < height / 2:  # Top side
                y_final += perspective_y
            else:  # Bottom side
                y_final -= perspective_y
            
            dst_points.append([x_final, y_final])
        
        dst_points = np.float32(dst_points)
        
        # Get perspective transform matrix
        return cv2.getPerspectiveTransform(src_points, dst_points)
    
    def _transform_bbox(
        self, 
        bbox: BoundingBox, 
        transform_matrix: np.ndarray, 
        img_width: int, 
        img_height: int
    ) -> BoundingBox:
        """Transform bounding box using perspective transformation."""
        
        # Get bbox corners
        x1, y1, x2, y2 = bbox.to_xyxy()
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ], dtype=np.float32)
        
        # Apply perspective transformation to corners
        transformed_corners = []
        for corner in corners:
            transformed = transform_matrix @ corner.T
            # Convert from homogeneous coordinates
            x = transformed[0] / transformed[2]
            y = transformed[1] / transformed[2]
            transformed_corners.append([x, y])
        
        transformed_corners = np.array(transformed_corners)
        
        # Find bounding box of transformed corners
        min_x = np.min(transformed_corners[:, 0])
        max_x = np.max(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_y = np.max(transformed_corners[:, 1])
        
        # Create new bounding box
        transformed_bbox = BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            class_id=bbox.class_id,
            confidence=bbox.confidence
        )
        
        return transformed_bbox.clip_to_image(img_width, img_height)
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply perspective transformation to image and bounding boxes."""
        
        h, w = image.shape[:2]
        
        # Generate perspective transformation matrix
        transform_matrix = self._generate_perspective_transform(w, h)
        
        # Apply transformation to image
        transformed_image = cv2.warpPerspective(
            image, 
            transform_matrix, 
            (w, h), 
            borderMode=cv2.BORDER_REFLECT
        )
        
        # Transform bounding boxes
        transformed_bboxes = []
        for bbox in bboxes:
            transformed_bbox = self._transform_bbox(bbox, transform_matrix, w, h)
            
            if transformed_bbox.is_valid():
                transformed_bboxes.append(transformed_bbox)
        
        metadata = {
            "transform_matrix": transform_matrix.tolist(),
            "original_bboxes": len(bboxes),
            "filtered_bboxes": len(transformed_bboxes)
        }
        
        return transformed_image, transformed_bboxes, metadata


class GeometricAugmentations:
    """Factory class for creating geometric augmentation pipelines."""
    
    @staticmethod
    def create_basic_geometric_pipeline() -> List[BaseAugmentation]:
        """Create basic geometric augmentation pipeline for surveillance."""
        return [
            HorizontalFlipAugmentation(AugmentationConfig(probability=0.5)),
            RandomScaleAugmentation(AugmentationConfig(probability=0.6, severity=0.4)),
            RandomTranslationAugmentation(AugmentationConfig(probability=0.4, severity=0.3))
        ]
    
    @staticmethod
    def create_advanced_geometric_pipeline() -> List[BaseAugmentation]:
        """Create advanced geometric augmentation pipeline with perspective transforms."""
        return [
            HorizontalFlipAugmentation(AugmentationConfig(probability=0.5)),
            RandomScaleAugmentation(AugmentationConfig(probability=0.7, severity=0.5)),
            RandomTranslationAugmentation(AugmentationConfig(probability=0.5, severity=0.4)),
            PerspectiveAugmentation(AugmentationConfig(probability=0.3, severity=0.3))
        ]
    
    @staticmethod
    def create_conservative_geometric_pipeline() -> List[BaseAugmentation]:
        """Create conservative geometric augmentation pipeline for production."""
        return [
            HorizontalFlipAugmentation(AugmentationConfig(probability=0.5)),
            RandomScaleAugmentation(AugmentationConfig(probability=0.5, severity=0.3)),
            RandomTranslationAugmentation(AugmentationConfig(probability=0.3, severity=0.2))
        ]