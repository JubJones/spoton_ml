"""
Photometric Augmentations for Surveillance Person Detection
Environmental adaptation for varying lighting conditions in MTMMC dataset
"""
import random
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
from .base import BaseAugmentation, AugmentationConfig, BoundingBox

logger = logging.getLogger(__name__)


class BrightnessContrastAugmentation(BaseAugmentation):
    """
    Brightness and contrast augmentation for handling day/night surveillance variations.
    Critical for MTMMC dataset which includes various lighting conditions.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.8,
                severity=0.5,
                enabled=True,
                parameters={
                    "brightness_range": (-0.2, 0.2),  # ±20% brightness variation
                    "contrast_range": (0.8, 1.2),    # 80%-120% contrast variation
                    "independent_channels": False      # Apply same adjustment to all channels
                }
            )
        super().__init__(config)
    
    def _get_adjustment_range(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate brightness and contrast ranges based on severity."""
        params = self.config.parameters
        base_brightness = params.get("brightness_range", (-0.2, 0.2))
        base_contrast = params.get("contrast_range", (0.8, 1.2))
        
        # Scale ranges by severity
        brightness_scale = self.config.severity
        contrast_scale = self.config.severity
        
        brightness_range = (
            base_brightness[0] * brightness_scale,
            base_brightness[1] * brightness_scale
        )
        
        # Adjust contrast range symmetrically around 1.0
        contrast_deviation = max(abs(base_contrast[0] - 1.0), abs(base_contrast[1] - 1.0))
        scaled_deviation = contrast_deviation * contrast_scale
        contrast_range = (
            max(0.1, 1.0 - scaled_deviation),
            min(3.0, 1.0 + scaled_deviation)
        )
        
        return brightness_range, contrast_range
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply brightness and contrast adjustment."""
        
        brightness_range, contrast_range = self._get_adjustment_range()
        
        # Generate random adjustments
        brightness_delta = random.uniform(*brightness_range)
        contrast_factor = random.uniform(*contrast_range)
        
        # Convert to float for processing
        image_float = image.astype(np.float32)
        
        # Apply contrast (multiply)
        adjusted_image = image_float * contrast_factor
        
        # Apply brightness (add)
        adjusted_image = adjusted_image + (brightness_delta * 255)
        
        # Clip to valid range and convert back to uint8
        adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
        
        metadata = {
            "brightness_delta": brightness_delta,
            "contrast_factor": contrast_factor,
            "brightness_range": brightness_range,
            "contrast_range": contrast_range
        }
        
        # Bounding boxes remain unchanged for photometric augmentations
        return adjusted_image, bboxes, metadata


class ColorJitterAugmentation(BaseAugmentation):
    """
    Color jitter augmentation to handle different camera color profiles.
    Adjusts hue, saturation, and color balance.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.6,
                severity=0.4,
                enabled=True,
                parameters={
                    "hue_range": (-10, 10),           # ±10 degrees hue shift
                    "saturation_range": (0.85, 1.15), # 85%-115% saturation
                    "color_shift_range": (-15, 15)    # ±15 color channel shift
                }
            )
        super().__init__(config)
    
    def _get_color_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Calculate color adjustment ranges based on severity."""
        params = self.config.parameters
        base_hue = params.get("hue_range", (-10, 10))
        base_saturation = params.get("saturation_range", (0.85, 1.15))
        base_color_shift = params.get("color_shift_range", (-15, 15))
        
        severity = self.config.severity
        
        return {
            "hue": (base_hue[0] * severity, base_hue[1] * severity),
            "saturation": (
                1.0 - (1.0 - base_saturation[0]) * severity,
                1.0 + (base_saturation[1] - 1.0) * severity
            ),
            "color_shift": (base_color_shift[0] * severity, base_color_shift[1] * severity)
        }
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply color jittering."""
        
        ranges = self._get_color_ranges()
        
        # Convert BGR to HSV for hue/saturation adjustment
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Apply hue shift
        hue_shift = random.uniform(*ranges["hue"])
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
        
        # Apply saturation adjustment
        saturation_factor = random.uniform(*ranges["saturation"])
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
        
        # Convert back to BGR
        adjusted_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Apply random color channel shifts
        color_shifts = [random.uniform(*ranges["color_shift"]) for _ in range(3)]
        adjusted_image = adjusted_image.astype(np.float32)
        
        for channel in range(3):
            adjusted_image[:, :, channel] = np.clip(
                adjusted_image[:, :, channel] + color_shifts[channel], 
                0, 255
            )
        
        adjusted_image = adjusted_image.astype(np.uint8)
        
        metadata = {
            "hue_shift": hue_shift,
            "saturation_factor": saturation_factor,
            "color_shifts": color_shifts,
            "ranges": ranges
        }
        
        return adjusted_image, bboxes, metadata


class GaussianNoiseAugmentation(BaseAugmentation):
    """
    Gaussian noise augmentation to simulate low-quality camera feeds.
    Helps model robustness to sensor noise.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.3,
                severity=0.3,
                enabled=True,
                parameters={
                    "noise_std_range": (5, 25),  # Standard deviation range for noise
                    "noise_mean": 0               # Mean of gaussian noise
                }
            )
        super().__init__(config)
    
    def _get_noise_parameters(self) -> Tuple[float, float]:
        """Calculate noise parameters based on severity."""
        params = self.config.parameters
        std_range = params.get("noise_std_range", (5, 25))
        mean = params.get("noise_mean", 0)
        
        # Scale noise standard deviation by severity
        min_std = std_range[0] * self.config.severity
        max_std = std_range[1] * self.config.severity
        
        return mean, random.uniform(min_std, max_std)
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply Gaussian noise to image."""
        
        mean, std = self._get_noise_parameters()
        
        # Generate noise with same shape as image
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        
        # Add noise to image
        noisy_image = image.astype(np.float32) + noise
        
        # Clip to valid range and convert back to uint8
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        metadata = {
            "noise_mean": mean,
            "noise_std": std,
            "noise_intensity": std / 255.0  # Normalized intensity
        }
        
        return noisy_image, bboxes, metadata


class MotionBlurAugmentation(BaseAugmentation):
    """
    Motion blur augmentation to simulate fast-moving persons and camera shake.
    Uses directional kernels to simulate realistic motion blur.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.25,
                severity=0.4,
                enabled=True,
                parameters={
                    "kernel_size_range": (3, 15),     # Blur kernel size range
                    "angle_range": (0, 180),          # Blur direction in degrees
                    "blur_types": ["linear", "radial"] # Types of motion blur
                }
            )
        super().__init__(config)
    
    def _generate_motion_blur_kernel(self, kernel_size: int, angle: float, blur_type: str) -> np.ndarray:
        """Generate motion blur kernel based on type and parameters."""
        
        if blur_type == "linear":
            # Create linear motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            
            # Calculate line coordinates
            center = kernel_size // 2
            angle_rad = np.radians(angle)
            
            for i in range(kernel_size):
                x = int(center + (i - center) * np.cos(angle_rad))
                y = int(center + (i - center) * np.sin(angle_rad))
                
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1.0
            
            # Normalize kernel
            kernel_sum = kernel.sum()
            if kernel_sum > 0:
                kernel /= kernel_sum
            else:
                kernel[center, center] = 1.0
        
        elif blur_type == "radial":
            # Create radial motion blur kernel (zoom blur effect)
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            center = kernel_size // 2
            
            # Create radial pattern
            y, x = np.ogrid[:kernel_size, :kernel_size]
            distance = np.sqrt((x - center)**2 + (y - center)**2)
            
            # Create radial mask
            max_distance = kernel_size // 2
            mask = distance <= max_distance
            kernel[mask] = 1.0 / (distance[mask] + 1e-6)
            
            # Normalize
            kernel /= kernel.sum()
        
        else:
            # Fallback to simple averaging kernel
            kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
            kernel /= (kernel_size * kernel_size)
        
        return kernel
    
    def _get_blur_parameters(self) -> Tuple[int, float, str]:
        """Get blur parameters based on severity."""
        params = self.config.parameters
        kernel_range = params.get("kernel_size_range", (3, 15))
        angle_range = params.get("angle_range", (0, 180))
        blur_types = params.get("blur_types", ["linear", "radial"])
        
        # Scale kernel size by severity
        min_size = kernel_range[0]
        max_size = min_size + int((kernel_range[1] - min_size) * self.config.severity)
        kernel_size = random.randrange(min_size, max_size + 1, 2)  # Ensure odd size
        
        # Random angle
        angle = random.uniform(*angle_range)
        
        # Random blur type
        blur_type = random.choice(blur_types)
        
        return kernel_size, angle, blur_type
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply motion blur to image."""
        
        kernel_size, angle, blur_type = self._get_blur_parameters()
        
        # Generate blur kernel
        blur_kernel = self._generate_motion_blur_kernel(kernel_size, angle, blur_type)
        
        # Apply convolution
        blurred_image = cv2.filter2D(image, -1, blur_kernel)
        
        metadata = {
            "kernel_size": kernel_size,
            "blur_angle": angle,
            "blur_type": blur_type,
            "kernel_shape": blur_kernel.shape
        }
        
        return blurred_image, bboxes, metadata


class RandomGammaAugmentation(BaseAugmentation):
    """
    Gamma correction augmentation for simulating different exposure conditions.
    Useful for handling underexposed or overexposed surveillance footage.
    """
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig(
                probability=0.4,
                severity=0.4,
                enabled=True,
                parameters={
                    "gamma_range": (0.7, 1.3)  # Gamma correction range
                }
            )
        super().__init__(config)
    
    def _get_gamma_range(self) -> Tuple[float, float]:
        """Calculate gamma range based on severity."""
        base_range = self.config.parameters.get("gamma_range", (0.7, 1.3))
        
        # Scale deviation from 1.0 by severity
        deviation = max(abs(base_range[0] - 1.0), abs(base_range[1] - 1.0))
        scaled_deviation = deviation * self.config.severity
        
        gamma_min = max(0.3, 1.0 - scaled_deviation)
        gamma_max = min(2.5, 1.0 + scaled_deviation)
        
        return gamma_min, gamma_max
    
    def _apply_augmentation(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> Tuple[np.ndarray, List[BoundingBox], Dict[str, Any]]:
        """Apply gamma correction."""
        
        gamma_min, gamma_max = self._get_gamma_range()
        gamma = random.uniform(gamma_min, gamma_max)
        
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 
                               for i in np.arange(0, 256)]).astype(np.uint8)
        
        # Apply gamma correction using lookup table
        gamma_corrected = cv2.LUT(image, lookup_table)
        
        metadata = {
            "gamma": gamma,
            "gamma_range": (gamma_min, gamma_max),
            "inverse_gamma": inv_gamma
        }
        
        return gamma_corrected, bboxes, metadata


class PhotometricAugmentations:
    """Factory class for creating photometric augmentation pipelines."""
    
    @staticmethod
    def create_basic_photometric_pipeline() -> List[BaseAugmentation]:
        """Create basic photometric augmentation pipeline for surveillance."""
        return [
            BrightnessContrastAugmentation(AugmentationConfig(probability=0.7, severity=0.4)),
            ColorJitterAugmentation(AugmentationConfig(probability=0.5, severity=0.3)),
            GaussianNoiseAugmentation(AugmentationConfig(probability=0.3, severity=0.3))
        ]
    
    @staticmethod
    def create_advanced_photometric_pipeline() -> List[BaseAugmentation]:
        """Create advanced photometric augmentation pipeline with motion blur."""
        return [
            BrightnessContrastAugmentation(AugmentationConfig(probability=0.8, severity=0.5)),
            ColorJitterAugmentation(AugmentationConfig(probability=0.6, severity=0.4)),
            GaussianNoiseAugmentation(AugmentationConfig(probability=0.3, severity=0.3)),
            MotionBlurAugmentation(AugmentationConfig(probability=0.25, severity=0.4)),
            RandomGammaAugmentation(AugmentationConfig(probability=0.4, severity=0.4))
        ]
    
    @staticmethod
    def create_surveillance_optimized_pipeline() -> List[BaseAugmentation]:
        """Create surveillance-optimized photometric pipeline emphasizing lighting variations."""
        return [
            BrightnessContrastAugmentation(AugmentationConfig(
                probability=0.9, 
                severity=0.6,
                parameters={
                    "brightness_range": (-0.3, 0.3),  # Wider brightness range for day/night
                    "contrast_range": (0.7, 1.4)      # Stronger contrast variations
                }
            )),
            RandomGammaAugmentation(AugmentationConfig(probability=0.6, severity=0.5)),
            ColorJitterAugmentation(AugmentationConfig(probability=0.4, severity=0.3)),
            GaussianNoiseAugmentation(AugmentationConfig(
                probability=0.4, 
                severity=0.4,
                parameters={"noise_std_range": (3, 20)}  # More conservative noise
            ))
        ]