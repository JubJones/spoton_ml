"""
Advanced Occlusion Augmentations for Surveillance Person Detection
Robust augmentations for handling occlusions and crowded scenes in MTMMC dataset
"""
import random
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
import math
from .base import BaseAugmentation, AugmentationConfig, BoundingBox

logger = logging.getLogger(__name__)


class RandomErasingAugmentation(BaseAugmentation):
    """
    Random erasing augmentation for simulating partial occlusions.
    Removes rectangular regions to improve robustness to occlusion.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.5,
                severity=0.4,
                enabled=True,
                parameters={
                    "area_range": (0.02, 0.15),       # 2%-15% of image area
                    "aspect_ratio_range": (0.3, 3.3), # Height/width ratio range
                    "max_attempts": 100,               # Maximum attempts to place erasing region
                    "fill_mode": "random"              # "random", "mean", "black", "white"
                }
            )
        super().__init__(config)
    
    def _get_erasing_parameters(self, img_height: int, img_width: int) -> Dict[str, Any]:
        """Calculate erasing parameters based on severity and image size."""
        params = self.config.parameters
        area_range = params.get("area_range", (0.02, 0.15))
        aspect_ratio_range = params.get("aspect_ratio_range", (0.3, 3.3))
        
        # Scale area by severity
        min_area_ratio = area_range[0] * self.config.severity
        max_area_ratio = area_range[1] * self.config.severity
        
        # Calculate target area
        img_area = img_height * img_width
        target_area = random.uniform(min_area_ratio, max_area_ratio) * img_area
        
        # Random aspect ratio
        aspect_ratio = random.uniform(*aspect_ratio_range)
        
        # Calculate dimensions
        erase_height = int(math.sqrt(target_area * aspect_ratio))
        erase_width = int(math.sqrt(target_area / aspect_ratio))
        
        # Ensure dimensions are within image bounds
        erase_height = min(erase_height, img_height)
        erase_width = min(erase_width, img_width)
        
        return {
            "erase_height": erase_height,
            "erase_width": erase_width,
            "target_area": target_area,
            "aspect_ratio": aspect_ratio
        }
    
    def _generate_fill_value(self, image: np.ndarray, fill_mode: str) -> np.ndarray:
        """Generate fill value for erased region."""
        if fill_mode == "random":
            # Random pixel values
            return np.random.randint(0, 256, size=image.shape[2], dtype=np.uint8)
        elif fill_mode == "mean":
            # Mean color of the image
            return image.mean(axis=(0, 1)).astype(np.uint8)
        elif fill_mode == "black":
            return np.zeros(image.shape[2], dtype=np.uint8)
        elif fill_mode == "white":
            return np.full(image.shape[2], 255, dtype=np.uint8)
        else:
            # Default to random
            return np.random.randint(0, 256, size=image.shape[2], dtype=np.uint8)
    
    def _check_bbox_overlap(self, bbox: BoundingBox, erase_x: int, erase_y: int, 
                           erase_width: int, erase_height: int) -> float:
        """Calculate overlap ratio between bounding box and erasing region."""
        # Calculate intersection
        x1 = max(bbox.x, erase_x)
        y1 = max(bbox.y, erase_y)
        x2 = min(bbox.x + bbox.width, erase_x + erase_width)
        y2 = min(bbox.y + bbox.height, erase_y + erase_height)
        
        if x1 >= x2 or y1 >= y2:
            return 0.0  # No overlap
        
        intersection_area = (x2 - x1) * (y2 - y1)
        bbox_area = bbox.area()
        
        return intersection_area / bbox_area if bbox_area > 0 else 0.0
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply random erasing to image."""
        
        h, w = image.shape[:2]
        params = self.config.parameters
        max_attempts = params.get("max_attempts", 100)
        fill_mode = params.get("fill_mode", "random")
        
        erase_params = self._get_erasing_parameters(h, w)
        erase_height = erase_params["erase_height"]
        erase_width = erase_params["erase_width"]
        
        erased_image = image.copy()
        applied_erasings = []
        
        # Try multiple erasing attempts
        num_erasings = random.randint(1, 3)  # 1-3 erasing regions
        
        for attempt in range(num_erasings):
            best_position = None
            min_critical_overlap = 1.0
            
            # Find position with minimal critical overlap
            for _ in range(max_attempts // num_erasings):
                erase_x = random.randint(0, max(1, w - erase_width))
                erase_y = random.randint(0, max(1, h - erase_height))
                
                # Check overlap with all bounding boxes
                max_overlap = 0.0
                for bbox in bboxes:
                    overlap = self._check_bbox_overlap(bbox, erase_x, erase_y, erase_width, erase_height)
                    max_overlap = max(max_overlap, overlap)
                
                # Prefer positions with less overlap with person bboxes
                if max_overlap < min_critical_overlap:
                    min_critical_overlap = max_overlap
                    best_position = (erase_x, erase_y)
                
                # If we found a position with minimal overlap, use it
                if max_overlap < 0.3:  # Less than 30% overlap
                    break
            
            if best_position is not None:
                erase_x, erase_y = best_position
                
                # Generate fill value
                fill_value = self._generate_fill_value(erased_image, fill_mode)
                
                # Apply erasing
                erased_image[erase_y:erase_y+erase_height, erase_x:erase_x+erase_width] = fill_value
                
                applied_erasings.append({
                    "position": (erase_x, erase_y),
                    "size": (erase_width, erase_height),
                    "overlap": min_critical_overlap
                })
        
        metadata = {
            "num_erasings": len(applied_erasings),
            "erasings": applied_erasings,
            "fill_mode": fill_mode,
            "target_dimensions": (erase_width, erase_height)
        }
        
        # Bounding boxes remain unchanged
        return erased_image, bboxes, metadata


class MosaicAugmentation(BaseAugmentation):
    """
    Mosaic augmentation that combines 4 images into one training sample.
    Improves multi-scale detection and context understanding.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.3,
                severity=0.5,
                enabled=True,
                parameters={
                    "mosaic_ratio": 0.5,    # Ratio for dividing image into quadrants
                    "border_value": 114     # Fill value for borders (ImageNet mean)
                }
            )
        super().__init__(config)
        
        # Store additional images and bboxes for mosaic creation
        self._image_cache = []
        self._bbox_cache = []
    
    def add_cache_sample(self, image: np.ndarray, bboxes: List[BoundingBox]):
        """Add an image and its bboxes to the cache for mosaic creation."""
        self._image_cache.append(image.copy())
        self._bbox_cache.append(bboxes.copy())
        
        # Keep only the last 10 samples to avoid memory issues
        if len(self._image_cache) > 10:
            self._image_cache.pop(0)
            self._bbox_cache.pop(0)
    
    def _create_mosaic(
        self, 
        images: List[np.ndarray], 
        bbox_lists: List[List[BoundingBox]], 
        output_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[BoundingBox]]:
        """Create mosaic from 4 images."""
        
        h, w = output_size
        mosaic_ratio = self.config.parameters.get("mosaic_ratio", 0.5)
        border_value = self.config.parameters.get("border_value", 114)
        
        # Calculate center point with some randomness
        center_x = int(w * (mosaic_ratio + random.uniform(-0.1, 0.1)))
        center_y = int(h * (mosaic_ratio + random.uniform(-0.1, 0.1)))
        center_x = max(w//4, min(3*w//4, center_x))
        center_y = max(h//4, min(3*h//4, center_y))
        
        # Create output mosaic image
        mosaic_image = np.full((h, w, 3), border_value, dtype=np.uint8)
        mosaic_bboxes = []
        
        # Define quadrant positions and scales
        quadrants = [
            (0, 0, center_x, center_y),           # Top-left
            (center_x, 0, w - center_x, center_y), # Top-right
            (0, center_y, center_x, h - center_y), # Bottom-left
            (center_x, center_y, w - center_x, h - center_y) # Bottom-right
        ]
        
        for i, (img, bboxes) in enumerate(zip(images, bbox_lists)):
            if i >= 4:  # Only use first 4 images
                break
                
            quad_x, quad_y, quad_w, quad_h = quadrants[i]
            
            # Resize image to fit quadrant
            img_resized = cv2.resize(img, (quad_w, quad_h), interpolation=cv2.INTER_LINEAR)
            
            # Place image in mosaic
            mosaic_image[quad_y:quad_y+quad_h, quad_x:quad_x+quad_w] = img_resized
            
            # Transform bounding boxes
            orig_h, orig_w = img.shape[:2]
            scale_x = quad_w / orig_w
            scale_y = quad_h / orig_h
            
            for bbox in bboxes:
                # Scale and translate bbox
                new_x = bbox.x * scale_x + quad_x
                new_y = bbox.y * scale_y + quad_y
                new_width = bbox.width * scale_x
                new_height = bbox.height * scale_y
                
                mosaic_bbox = BoundingBox(
                    x=new_x,
                    y=new_y,
                    width=new_width,
                    height=new_height,
                    class_id=bbox.class_id,
                    confidence=bbox.confidence
                )
                
                # Clip to mosaic bounds and validate
                mosaic_bbox = mosaic_bbox.clip_to_image(w, h)
                if mosaic_bbox.is_valid() and mosaic_bbox.area() > 100:  # Minimum area threshold
                    mosaic_bboxes.append(mosaic_bbox)
        
        return mosaic_image, mosaic_bboxes
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply mosaic augmentation."""
        
        # Add current sample to cache
        self.add_cache_sample(image, bboxes)
        
        # Need at least 3 cached samples to create mosaic (current + 3 cached)
        if len(self._image_cache) < 3:
            return image, bboxes, {"mosaic_applied": False, "reason": "insufficient_cache"}
        
        # Select 3 random samples from cache + current image
        selected_indices = random.sample(range(len(self._image_cache)), 3)
        selected_images = [self._image_cache[i] for i in selected_indices]
        selected_bboxes = [self._bbox_cache[i] for i in selected_indices]
        
        # Add current sample
        selected_images.append(image)
        selected_bboxes.append(bboxes)
        
        # Create mosaic
        output_size = image.shape[:2]
        mosaic_image, mosaic_bboxes = self._create_mosaic(
            selected_images, 
            selected_bboxes, 
            output_size
        )
        
        metadata = {
            "mosaic_applied": True,
            "num_source_images": len(selected_images),
            "original_bboxes": sum(len(bbox_list) for bbox_list in selected_bboxes),
            "mosaic_bboxes": len(mosaic_bboxes),
            "cache_size": len(self._image_cache)
        }
        
        return mosaic_image, mosaic_bboxes, metadata


class CopyPasteAugmentation(BaseAugmentation):
    """
    Copy-paste augmentation for increasing person instance diversity.
    Copies person instances and pastes them in plausible locations.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.2,
                severity=0.3,
                enabled=True,
                parameters={
                    "max_paste_instances": 3,    # Maximum instances to paste per image
                    "min_instance_area": 500,    # Minimum area for source instances
                    "overlap_threshold": 0.3,    # Maximum IoU overlap for placement
                    "scale_range": (0.8, 1.2),  # Scale variation for pasted instances
                    "blend_mode": "alpha"        # "alpha", "replace", "gaussian"
                }
            )
        super().__init__(config)
        
        # Cache for storing person instances
        self._person_cache = []
    
    def _extract_person_instances(self, image: np.ndarray, bboxes: List[BoundingBox]) -> List[Dict[str, Any]]:
        """Extract person instances from image for potential copy-paste."""
        instances = []
        min_area = self.config.parameters.get("min_instance_area", 500)
        
        for bbox in bboxes:
            if bbox.area() >= min_area and bbox.is_valid():
                # Extract person region with some padding
                pad = 5  # Small padding around bbox
                x1 = max(0, int(bbox.x - pad))
                y1 = max(0, int(bbox.y - pad))
                x2 = min(image.shape[1], int(bbox.x + bbox.width + pad))
                y2 = min(image.shape[0], int(bbox.y + bbox.height + pad))
                
                person_patch = image[y1:y2, x1:x2].copy()
                
                if person_patch.size > 0:
                    instance = {
                        "patch": person_patch,
                        "bbox": bbox,
                        "original_position": (x1, y1),
                        "size": (x2 - x1, y2 - y1)
                    }
                    instances.append(instance)
        
        return instances
    
    def _add_to_person_cache(self, instances: List[Dict[str, Any]]):
        """Add person instances to cache."""
        self._person_cache.extend(instances)
        
        # Keep cache size reasonable
        if len(self._person_cache) > 50:
            # Remove oldest instances
            self._person_cache = self._person_cache[-50:]
    
    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1.to_xyxy()
        x1_2, y1_2, x2_2, y2_2 = bbox2.to_xyxy()
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = bbox1.area()
        area2 = bbox2.area()
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _find_valid_paste_location(
        self, 
        patch_size: Tuple[int, int], 
        existing_bboxes: List[BoundingBox],
        img_size: Tuple[int, int],
        max_attempts: int = 50
    ) -> Optional[Tuple[int, int]]:
        """Find valid location to paste instance without excessive overlap."""
        
        img_h, img_w = img_size
        patch_w, patch_h = patch_size
        overlap_threshold = self.config.parameters.get("overlap_threshold", 0.3)
        
        for _ in range(max_attempts):
            # Random position
            x = random.randint(0, max(1, img_w - patch_w))
            y = random.randint(0, max(1, img_h - patch_h))
            
            # Create candidate bbox
            candidate_bbox = BoundingBox(x=x, y=y, width=patch_w, height=patch_h)
            
            # Check overlap with existing bboxes
            max_overlap = 0.0
            for existing_bbox in existing_bboxes:
                overlap = self._calculate_iou(candidate_bbox, existing_bbox)
                max_overlap = max(max_overlap, overlap)
            
            if max_overlap < overlap_threshold:
                return (x, y)
        
        return None
    
    def _blend_patch(
        self, 
        image: np.ndarray, 
        patch: np.ndarray, 
        position: Tuple[int, int], 
        blend_mode: str = "alpha"
    ) -> np.ndarray:
        """Blend patch into image at specified position."""
        
        x, y = position
        patch_h, patch_w = patch.shape[:2]
        
        # Ensure patch fits in image
        end_x = min(x + patch_w, image.shape[1])
        end_y = min(y + patch_h, image.shape[0])
        actual_patch_w = end_x - x
        actual_patch_h = end_y - y
        
        if actual_patch_w <= 0 or actual_patch_h <= 0:
            return image
        
        # Crop patch if necessary
        cropped_patch = patch[:actual_patch_h, :actual_patch_w]
        
        result_image = image.copy()
        
        if blend_mode == "replace":
            # Simple replacement
            result_image[y:end_y, x:end_x] = cropped_patch
        
        elif blend_mode == "alpha":
            # Alpha blending with edge feathering
            alpha = 0.8  # Blend factor
            
            # Create alpha mask with edge feathering
            mask = np.ones((actual_patch_h, actual_patch_w), dtype=np.float32) * alpha
            
            # Feather edges
            feather_size = min(5, actual_patch_w//4, actual_patch_h//4)
            if feather_size > 0:
                # Top and bottom edges
                for i in range(feather_size):
                    factor = (i + 1) / feather_size * alpha
                    mask[i, :] = factor
                    mask[-(i+1), :] = factor
                
                # Left and right edges  
                for j in range(feather_size):
                    factor = (j + 1) / feather_size * alpha
                    mask[:, j] = np.minimum(mask[:, j], factor)
                    mask[:, -(j+1)] = np.minimum(mask[:, -(j+1)], factor)
            
            # Apply blending
            for c in range(3):
                result_image[y:end_y, x:end_x, c] = (
                    mask * cropped_patch[:, :, c] + 
                    (1 - mask) * result_image[y:end_y, x:end_x, c]
                ).astype(np.uint8)
        
        else:  # Default to replace
            result_image[y:end_y, x:end_x] = cropped_patch
        
        return result_image
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply copy-paste augmentation."""
        
        # Extract and cache person instances from current image
        current_instances = self._extract_person_instances(image, bboxes)
        self._add_to_person_cache(current_instances)
        
        # If cache is empty, return original
        if not self._person_cache:
            return image, bboxes, {"copy_paste_applied": False, "reason": "no_cached_instances"}
        
        # Parameters
        max_paste = self.config.parameters.get("max_paste_instances", 3)
        scale_range = self.config.parameters.get("scale_range", (0.8, 1.2))
        blend_mode = self.config.parameters.get("blend_mode", "alpha")
        
        # Determine number of instances to paste based on severity
        num_to_paste = random.randint(1, max(1, int(max_paste * self.config.severity)))
        
        # Select random instances from cache
        available_instances = [inst for inst in self._person_cache if inst["patch"].size > 0]
        if len(available_instances) == 0:
            return image, bboxes, {"copy_paste_applied": False, "reason": "no_valid_instances"}
        
        num_to_paste = min(num_to_paste, len(available_instances))
        selected_instances = random.sample(available_instances, num_to_paste)
        
        # Apply copy-paste
        result_image = image.copy()
        result_bboxes = bboxes.copy()
        pasted_instances = []
        
        for instance in selected_instances:
            patch = instance["patch"]
            original_bbox = instance["bbox"]
            
            # Random scale
            scale_factor = random.uniform(*scale_range)
            
            # Resize patch
            new_width = int(patch.shape[1] * scale_factor)
            new_height = int(patch.shape[0] * scale_factor)
            
            if new_width > 0 and new_height > 0:
                scaled_patch = cv2.resize(patch, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
                # Find valid paste location
                paste_location = self._find_valid_paste_location(
                    (new_width, new_height),
                    result_bboxes,
                    image.shape[:2]
                )
                
                if paste_location is not None:
                    x, y = paste_location
                    
                    # Blend patch into image
                    result_image = self._blend_patch(result_image, scaled_patch, paste_location, blend_mode)
                    
                    # Create new bounding box for pasted instance
                    new_bbox = BoundingBox(
                        x=x,
                        y=y,
                        width=new_width,
                        height=new_height,
                        class_id=original_bbox.class_id,
                        confidence=original_bbox.confidence * 0.9  # Slightly lower confidence
                    )
                    
                    result_bboxes.append(new_bbox)
                    pasted_instances.append({
                        "position": paste_location,
                        "size": (new_width, new_height),
                        "scale_factor": scale_factor
                    })
        
        metadata = {
            "copy_paste_applied": True,
            "instances_pasted": len(pasted_instances),
            "pasted_details": pasted_instances,
            "cache_size": len(self._person_cache),
            "blend_mode": blend_mode
        }
        
        return result_image, result_bboxes, metadata


class OcclusionAugmentations:
    """Factory class for creating occlusion augmentation pipelines."""
    
    @staticmethod
    def create_basic_occlusion_pipeline() -> List[BaseAugmentation]:
        """Create basic occlusion augmentation pipeline."""
        return [
            RandomErasingAugmentation(AugmentationConfig(probability=0.5, severity=0.4))
        ]
    
    @staticmethod
    def create_advanced_occlusion_pipeline() -> List[BaseAugmentation]:
        """Create advanced occlusion augmentation pipeline with mosaic and copy-paste."""
        return [
            RandomErasingAugmentation(AugmentationConfig(probability=0.5, severity=0.4)),
            MosaicAugmentation(AugmentationConfig(probability=0.3, severity=0.5)),
            CopyPasteAugmentation(AugmentationConfig(probability=0.2, severity=0.3))
        ]
    
    @staticmethod
    def create_surveillance_occlusion_pipeline() -> List[BaseAugmentation]:
        """Create surveillance-optimized occlusion pipeline."""
        return [
            RandomErasingAugmentation(AugmentationConfig(
                probability=0.6,
                severity=0.5,
                parameters={
                    "area_range": (0.01, 0.12),  # Smaller areas for person-scale occlusions
                    "aspect_ratio_range": (0.5, 2.0),  # More conservative aspect ratios
                    "fill_mode": "mean"  # Use image mean for more natural occlusions
                }
            )),
            CopyPasteAugmentation(AugmentationConfig(
                probability=0.25,
                severity=0.4,
                parameters={
                    "max_paste_instances": 2,  # Conservative for surveillance
                    "overlap_threshold": 0.2,  # Allow some overlap for crowded scenes
                    "blend_mode": "alpha"  # Smooth blending
                }
            ))
        ]