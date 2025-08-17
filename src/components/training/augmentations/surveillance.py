"""
Surveillance-Specific Augmentations for MTMMC Person Detection
Specialized augmentations for surveillance scenarios: crowd simulation, lighting variations, weather effects
"""
import random
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
import math
from .base import BaseAugmentation, AugmentationConfig, BoundingBox

logger = logging.getLogger(__name__)


class CrowdSimulationAugmentation(BaseAugmentation):
    """
    Crowd simulation augmentation to artificially increase person density in sparse scenes.
    Uses existing person instances to create more crowded surveillance scenarios.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.3,
                severity=0.4,
                enabled=True,
                parameters={
                    "max_density_multiplier": 2.5,    # Maximum crowd density increase
                    "placement_strategy": "realistic", # "realistic", "random", "grid"
                    "size_variation": 0.3,            # Size variation for crowd members
                    "occlusion_simulation": True,      # Simulate partial occlusions
                    "depth_simulation": True           # Simulate depth-based sizing
                }
            )
        super().__init__(config)
        
        # Cache for person templates
        self._person_templates = []
    
    def _extract_person_templates(self, image: np.ndarray, bboxes: List[BoundingBox]) -> List[Dict[str, Any]]:
        """Extract person templates with various attributes for crowd simulation."""
        templates = []
        
        for bbox in bboxes:
            if bbox.is_valid() and bbox.area() > 200:  # Minimum size threshold
                # Extract person region with padding
                padding = 10
                x1 = max(0, int(bbox.x - padding))
                y1 = max(0, int(bbox.y - padding))
                x2 = min(image.shape[1], int(bbox.x + bbox.width + padding))
                y2 = min(image.shape[0], int(bbox.y + bbox.height + padding))
                
                person_patch = image[y1:y2, x1:x2].copy()
                
                if person_patch.size > 0:
                    # Analyze person attributes
                    template = {
                        "patch": person_patch,
                        "bbox": bbox,
                        "aspect_ratio": bbox.aspect_ratio(),
                        "size_category": self._categorize_person_size(bbox),
                        "position": (bbox.x + bbox.width/2, bbox.y + bbox.height/2),
                        "original_dimensions": (x2-x1, y2-y1)
                    }
                    templates.append(template)
        
        return templates
    
    def _categorize_person_size(self, bbox: BoundingBox) -> str:
        """Categorize person size for realistic crowd generation."""
        area = bbox.area()
        if area > 5000:
            return "large"
        elif area > 2000:
            return "medium"
        else:
            return "small"
    
    def _generate_realistic_positions(
        self, 
        image_shape: Tuple[int, int], 
        existing_bboxes: List[BoundingBox],
        num_new_persons: int
    ) -> List[Tuple[float, float, str]]:
        """Generate realistic positions for new crowd members."""
        
        h, w = image_shape
        positions = []
        
        # Analyze existing person distribution
        existing_positions = [(bbox.x + bbox.width/2, bbox.y + bbox.height/2) for bbox in existing_bboxes]
        
        if not existing_positions:
            # If no existing persons, generate random positions
            for _ in range(num_new_persons):
                x = random.uniform(0.1 * w, 0.9 * w)
                y = random.uniform(0.3 * h, 0.9 * h)  # Avoid sky area
                size_category = random.choice(["small", "medium", "large"])
                positions.append((x, y, size_category))
        else:
            # Generate positions near existing persons (crowd clustering)
            for _ in range(num_new_persons):
                # Choose a reference person
                ref_x, ref_y = random.choice(existing_positions)
                
                # Generate position near reference with some randomness
                cluster_radius = min(w, h) * 0.15  # 15% of image dimension
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(50, cluster_radius)
                
                new_x = ref_x + distance * math.cos(angle)
                new_y = ref_y + distance * math.sin(angle)
                
                # Ensure position is within image bounds
                new_x = max(0.05 * w, min(0.95 * w, new_x))
                new_y = max(0.2 * h, min(0.95 * h, new_y))
                
                # Determine size based on vertical position (depth simulation)
                if self.config.parameters.get("depth_simulation", True):
                    # People closer to bottom are typically larger (closer to camera)
                    depth_factor = (new_y - 0.2 * h) / (0.75 * h)  # Normalized depth
                    if depth_factor > 0.7:
                        size_category = random.choice(["medium", "large"])
                    elif depth_factor > 0.3:
                        size_category = "medium"
                    else:
                        size_category = random.choice(["small", "medium"])
                else:
                    size_category = random.choice(["small", "medium", "large"])
                
                positions.append((new_x, new_y, size_category))
        
        return positions
    
    def _check_position_validity(
        self, 
        position: Tuple[float, float], 
        template_size: Tuple[int, int],
        existing_bboxes: List[BoundingBox],
        max_overlap: float = 0.4
    ) -> bool:
        """Check if a position is valid for placing a new person."""
        
        x, y = position
        w, h = template_size
        
        # Create candidate bbox
        candidate = BoundingBox(x=x-w/2, y=y-h/2, width=w, height=h)
        
        # Check overlap with existing bboxes
        for bbox in existing_bboxes:
            overlap = self._calculate_overlap(candidate, bbox)
            if overlap > max_overlap:
                return False
        
        return True
    
    def _calculate_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate overlap ratio between two bounding boxes."""
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
        
        return intersection / area1 if area1 > 0 else 0.0
    
    def _place_person_in_crowd(
        self,
        image: np.ndarray,
        template: Dict[str, Any],
        position: Tuple[float, float],
        size_variation: float,
        occlusion_simulation: bool
    ) -> Tuple[np.ndarray, BoundingBox]:
        """Place a person template at specified position with realistic effects."""
        
        patch = template["patch"]
        x, y = position
        
        # Apply size variation
        scale_factor = 1.0 + random.uniform(-size_variation, size_variation)
        new_w = max(10, int(patch.shape[1] * scale_factor))
        new_h = max(10, int(patch.shape[0] * scale_factor))
        
        # Resize patch
        resized_patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate placement coordinates
        place_x = int(x - new_w/2)
        place_y = int(y - new_h/2)
        
        # Ensure patch fits in image
        place_x = max(0, min(place_x, image.shape[1] - new_w))
        place_y = max(0, min(place_y, image.shape[0] - new_h))
        
        # Apply occlusion simulation if enabled
        if occlusion_simulation and random.random() < 0.3:
            # Simulate partial occlusion by masking part of the person
            mask = np.ones((new_h, new_w), dtype=np.float32)
            
            # Random occlusion type
            occlusion_type = random.choice(["bottom", "side", "corner"])
            
            if occlusion_type == "bottom":
                occlusion_height = random.randint(new_h//4, new_h//2)
                mask[-occlusion_height:, :] = 0
            elif occlusion_type == "side":
                occlusion_width = random.randint(new_w//4, new_w//2)
                if random.random() < 0.5:
                    mask[:, :occlusion_width] = 0  # Left side
                else:
                    mask[:, -occlusion_width:] = 0  # Right side
            elif occlusion_type == "corner":
                corner_size = min(new_w//3, new_h//3)
                mask[:corner_size, :corner_size] = 0  # Top-left corner
            
            # Apply mask to patch
            for c in range(3):
                resized_patch[:, :, c] = (resized_patch[:, :, c] * mask).astype(np.uint8)
        
        # Blend patch into image with alpha blending
        result_image = image.copy()
        alpha = 0.9  # Slight blending for realism
        
        for c in range(3):
            result_image[place_y:place_y+new_h, place_x:place_x+new_w, c] = (
                alpha * resized_patch[:, :, c] + 
                (1 - alpha) * result_image[place_y:place_y+new_h, place_x:place_x+new_w, c]
            ).astype(np.uint8)
        
        # Create bounding box for new person
        new_bbox = BoundingBox(
            x=place_x,
            y=place_y,
            width=new_w,
            height=new_h,
            class_id=template["bbox"].class_id,
            confidence=0.85  # Slightly lower confidence for synthetic persons
        )
        
        return result_image, new_bbox
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply crowd simulation augmentation."""
        
        # Extract person templates from current image
        templates = self._extract_person_templates(image, bboxes)
        
        # Add to person template cache
        self._person_templates.extend(templates)
        
        # Keep cache size reasonable
        if len(self._person_templates) > 30:
            self._person_templates = self._person_templates[-30:]
        
        # If no templates available, return original
        if not self._person_templates:
            return image, bboxes, {"crowd_simulation_applied": False, "reason": "no_templates"}
        
        # Calculate number of persons to add based on severity
        current_density = len(bboxes)
        max_multiplier = self.config.parameters.get("max_density_multiplier", 2.5)
        
        # Scale addition by severity
        max_additions = int((max_multiplier - 1) * current_density * self.config.severity)
        num_to_add = random.randint(1, max(1, max_additions))
        
        # Generate realistic positions
        positions = self._generate_realistic_positions(image.shape[:2], bboxes, num_to_add)
        
        # Parameters
        size_variation = self.config.parameters.get("size_variation", 0.3)
        occlusion_simulation = self.config.parameters.get("occlusion_simulation", True)
        
        # Place new crowd members
        result_image = image.copy()
        result_bboxes = bboxes.copy()
        added_persons = []
        
        for pos_x, pos_y, size_category in positions:
            # Select appropriate template based on size category
            suitable_templates = [t for t in self._person_templates if t["size_category"] == size_category]
            if not suitable_templates:
                suitable_templates = self._person_templates  # Fallback to all templates
            
            template = random.choice(suitable_templates)
            
            # Check if position is valid
            template_size = template["original_dimensions"]
            if self._check_position_validity((pos_x, pos_y), template_size, result_bboxes):
                # Place person
                result_image, new_bbox = self._place_person_in_crowd(
                    result_image, template, (pos_x, pos_y), size_variation, occlusion_simulation
                )
                
                result_bboxes.append(new_bbox)
                added_persons.append({
                    "position": (pos_x, pos_y),
                    "size_category": size_category,
                    "bbox": new_bbox
                })
        
        metadata = {
            "crowd_simulation_applied": True,
            "original_person_count": len(bboxes),
            "added_person_count": len(added_persons),
            "final_person_count": len(result_bboxes),
            "density_increase": len(added_persons) / max(1, len(bboxes)),
            "template_cache_size": len(self._person_templates)
        }
        
        return result_image, result_bboxes, metadata


class LightingVariationAugmentation(BaseAugmentation):
    """
    Advanced lighting variation augmentation simulating different times of day and lighting conditions.
    Includes shadow simulation, directional lighting, and time-of-day effects.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.6,
                severity=0.5,
                enabled=True,
                parameters={
                    "time_of_day_effects": True,
                    "shadow_simulation": True,
                    "directional_lighting": True,
                    "artificial_lighting": True,
                    "lighting_scenarios": ["dawn", "noon", "dusk", "night", "artificial"]
                }
            )
        super().__init__(config)
    
    def _apply_time_of_day_effect(self, image: np.ndarray, scenario: str) -> np.ndarray:
        """Apply time-of-day lighting effects."""
        
        result = image.astype(np.float32)
        
        if scenario == "dawn":
            # Warm, soft lighting with slight orange tint
            color_matrix = np.array([
                [1.1, 0.05, 0.0],    # Red channel
                [0.05, 1.0, 0.0],    # Green channel  
                [0.0, 0.0, 0.9]      # Blue channel
            ])
            brightness_boost = 10
            
        elif scenario == "noon":
            # Bright, neutral lighting with high contrast
            color_matrix = np.array([
                [1.2, 0.0, 0.0],
                [0.0, 1.2, 0.0], 
                [0.0, 0.0, 1.2]
            ])
            brightness_boost = 20
            
        elif scenario == "dusk":
            # Warm, golden lighting
            color_matrix = np.array([
                [1.3, 0.1, 0.0],
                [0.1, 1.1, 0.0],
                [0.0, 0.0, 0.7]
            ])
            brightness_boost = -10
            
        elif scenario == "night":
            # Cool, dim lighting with blue tint
            color_matrix = np.array([
                [0.6, 0.0, 0.1],
                [0.0, 0.7, 0.1],
                [0.1, 0.1, 1.0]
            ])
            brightness_boost = -40
            
        elif scenario == "artificial":
            # Artificial lighting with slight yellow/green tint
            color_matrix = np.array([
                [1.0, 0.1, 0.0],
                [0.1, 1.1, 0.0],
                [0.0, 0.0, 0.8]
            ])
            brightness_boost = 5
            
        else:  # Default/neutral
            return image
        
        # Apply color transformation
        for c in range(3):
            for target_c in range(3):
                if color_matrix[c, target_c] != 0:
                    result[:, :, c] += color_matrix[c, target_c] * image[:, :, target_c]
        
        # Apply brightness adjustment
        result += brightness_boost
        
        # Clip and convert back
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _add_directional_shadows(self, image: np.ndarray, bboxes: List[BoundingBox]) -> np.ndarray:
        """Add realistic directional shadows for persons."""
        
        if not bboxes:
            return image
        
        result = image.copy()
        
        # Random shadow parameters
        shadow_direction = random.uniform(0, 2 * math.pi)  # Shadow angle
        shadow_length = random.uniform(0.3, 1.2)           # Shadow length multiplier
        shadow_intensity = random.uniform(0.3, 0.7)        # Shadow darkness
        
        for bbox in bboxes:
            if bbox.is_valid() and bbox.area() > 100:
                # Calculate shadow offset
                person_width = bbox.width
                person_height = bbox.height
                
                offset_x = int(shadow_length * person_width * 0.3 * math.cos(shadow_direction))
                offset_y = int(shadow_length * person_height * 0.2 * math.sin(shadow_direction))
                
                # Create shadow region
                shadow_x = int(bbox.x + offset_x)
                shadow_y = int(bbox.y + bbox.height + offset_y)
                shadow_width = int(person_width * 0.8)
                shadow_height = int(person_height * 0.3)
                
                # Ensure shadow is within image bounds
                shadow_x = max(0, min(shadow_x, image.shape[1] - shadow_width))
                shadow_y = max(0, min(shadow_y, image.shape[0] - shadow_height))
                shadow_width = min(shadow_width, image.shape[1] - shadow_x)
                shadow_height = min(shadow_height, image.shape[0] - shadow_y)
                
                if shadow_width > 0 and shadow_height > 0:
                    # Create shadow mask with gradient
                    shadow_mask = np.ones((shadow_height, shadow_width), dtype=np.float32)
                    
                    # Add gradient for realistic shadow falloff
                    for y in range(shadow_height):
                        fade_factor = 1.0 - (y / shadow_height) * 0.5
                        shadow_mask[y, :] *= fade_factor
                    
                    # Apply shadow (darken the region)
                    shadow_factor = 1.0 - (shadow_intensity * shadow_mask)
                    
                    for c in range(3):
                        result[shadow_y:shadow_y+shadow_height, shadow_x:shadow_x+shadow_width, c] = (
                            result[shadow_y:shadow_y+shadow_height, shadow_x:shadow_x+shadow_width, c] * 
                            shadow_factor
                        ).astype(np.uint8)
        
        return result
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply lighting variation augmentation."""
        
        params = self.config.parameters
        result_image = image.copy()
        applied_effects = []
        
        # Select lighting scenario
        scenarios = params.get("lighting_scenarios", ["dawn", "noon", "dusk", "night", "artificial"])
        scenario = random.choice(scenarios)
        
        # Apply time-of-day effects
        if params.get("time_of_day_effects", True):
            result_image = self._apply_time_of_day_effect(result_image, scenario)
            applied_effects.append(f"time_of_day_{scenario}")
        
        # Add directional shadows
        if params.get("shadow_simulation", True) and random.random() < 0.6:
            result_image = self._add_directional_shadows(result_image, bboxes)
            applied_effects.append("directional_shadows")
        
        metadata = {
            "lighting_scenario": scenario,
            "applied_effects": applied_effects,
            "severity": self.config.severity
        }
        
        # Bounding boxes remain unchanged
        return result_image, bboxes, metadata


class WeatherEffectAugmentation(BaseAugmentation):
    """
    Weather effect augmentation for outdoor surveillance scenarios.
    Simulates rain, fog, and atmospheric conditions.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.2,
                severity=0.3,
                enabled=True,
                parameters={
                    "weather_types": ["rain", "fog", "haze"],
                    "rain_intensity_range": (0.1, 0.6),
                    "fog_density_range": (0.1, 0.5),
                    "preserve_person_visibility": True
                }
            )
        super().__init__(config)
    
    def _add_rain_effect(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Add realistic rain effect to image."""
        
        result = image.copy()
        h, w = image.shape[:2]
        
        # Generate rain drops
        num_drops = int(intensity * w * h * 0.001)  # Scale by image size
        
        for _ in range(num_drops):
            # Random drop position and properties
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            
            # Rain drop characteristics
            drop_length = random.randint(5, 15)
            drop_width = 1
            drop_angle = random.uniform(-15, 15)  # Slight angle variation
            
            # Calculate end position
            end_x = int(x + drop_length * math.sin(math.radians(drop_angle)))
            end_y = int(y + drop_length * math.cos(math.radians(drop_angle)))
            
            # Ensure end position is within bounds
            end_x = max(0, min(end_x, w - 1))
            end_y = max(0, min(end_y, h - 1))
            
            # Draw rain drop (bright line)
            cv2.line(result, (x, y), (end_x, end_y), (200, 200, 200), drop_width)
        
        # Add overall atmospheric effect (slight brightness reduction and blur)
        atmospheric_factor = 1.0 - (intensity * 0.2)
        result = (result * atmospheric_factor).astype(np.uint8)
        
        # Slight blur for atmospheric scattering
        if intensity > 0.3:
            kernel_size = int(intensity * 3) * 2 + 1  # Odd kernel size
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0.5)
        
        return result
    
    def _add_fog_effect(self, image: np.ndarray, density: float) -> np.ndarray:
        """Add fog/haze effect to image."""
        
        # Create fog overlay
        fog_color = np.array([200, 200, 200], dtype=np.float32)  # Light gray
        
        # Convert to float for blending
        result = image.astype(np.float32)
        
        # Create distance-based fog intensity
        h, w = image.shape[:2]
        
        # Fog typically increases with distance (higher y values)
        y_coords = np.arange(h).reshape(-1, 1)
        distance_factor = (y_coords - h * 0.3) / (h * 0.7)  # Normalize distance
        distance_factor = np.clip(distance_factor, 0, 1)
        
        # Apply fog gradient
        fog_intensity = density * distance_factor
        fog_intensity = np.repeat(fog_intensity, w, axis=1)
        fog_intensity = np.expand_dims(fog_intensity, axis=2)
        
        # Blend with fog color
        result = result * (1 - fog_intensity) + fog_color * fog_intensity
        
        # Add slight blur for atmospheric scattering
        if density > 0.2:
            kernel_size = int(density * 5) * 2 + 1
            result = cv2.GaussianBlur(result.astype(np.uint8), (kernel_size, kernel_size), 1.0)
            result = result.astype(np.float32)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply weather effect augmentation."""
        
        params = self.config.parameters
        weather_types = params.get("weather_types", ["rain", "fog", "haze"])
        
        # Select random weather effect
        weather_type = random.choice(weather_types)
        
        result_image = image.copy()
        effect_intensity = 0.0
        
        if weather_type == "rain":
            rain_range = params.get("rain_intensity_range", (0.1, 0.6))
            intensity = random.uniform(*rain_range) * self.config.severity
            result_image = self._add_rain_effect(result_image, intensity)
            effect_intensity = intensity
            
        elif weather_type in ["fog", "haze"]:
            fog_range = params.get("fog_density_range", (0.1, 0.5))
            density = random.uniform(*fog_range) * self.config.severity
            result_image = self._add_fog_effect(result_image, density)
            effect_intensity = density
        
        metadata = {
            "weather_type": weather_type,
            "effect_intensity": effect_intensity,
            "severity_applied": self.config.severity
        }
        
        # Bounding boxes remain unchanged
        return result_image, bboxes, metadata


class SurveillanceAugmentations:
    """Factory class for creating surveillance-specific augmentation pipelines."""
    
    @staticmethod
    def create_basic_surveillance_pipeline() -> List[BaseAugmentation]:
        """Create basic surveillance-specific augmentation pipeline."""
        return [
            LightingVariationAugmentation(AugmentationConfig(probability=0.6, severity=0.4))
        ]
    
    @staticmethod
    def create_advanced_surveillance_pipeline() -> List[BaseAugmentation]:
        """Create advanced surveillance augmentation pipeline with all effects."""
        return [
            CrowdSimulationAugmentation(AugmentationConfig(probability=0.3, severity=0.4)),
            LightingVariationAugmentation(AugmentationConfig(probability=0.7, severity=0.5)),
            WeatherEffectAugmentation(AugmentationConfig(probability=0.2, severity=0.3))
        ]
    
    @staticmethod
    def create_mtmmc_optimized_pipeline() -> List[BaseAugmentation]:
        """Create MTMMC dataset optimized surveillance pipeline."""
        return [
            LightingVariationAugmentation(AugmentationConfig(
                probability=0.8,
                severity=0.6,
                parameters={
                    "lighting_scenarios": ["dawn", "noon", "dusk", "artificial"],  # Skip night for campus/factory
                    "shadow_simulation": True,
                    "directional_lighting": True
                }
            )),
            CrowdSimulationAugmentation(AugmentationConfig(
                probability=0.25,
                severity=0.3,
                parameters={
                    "max_density_multiplier": 2.0,  # Conservative for MTMMC scenes
                    "placement_strategy": "realistic",
                    "depth_simulation": True
                }
            )),
            WeatherEffectAugmentation(AugmentationConfig(
                probability=0.15,
                severity=0.2,
                parameters={
                    "weather_types": ["haze"],  # Subtle effects for indoor/outdoor mix
                    "preserve_person_visibility": True
                }
            ))
        ]