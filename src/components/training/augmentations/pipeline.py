"""
RF-DETR Augmentation Pipeline Integration
Comprehensive pipeline orchestrator for surveillance person detection training
"""
import logging
from typing import Dict, Any, List, Union, Optional
from .base import BaseAugmentation, AugmentationPipeline, AugmentationConfig, BoundingBox
from .geometric import GeometricAugmentations
from .photometric import PhotometricAugmentations
from .occlusion import OcclusionAugmentations
from .surveillance import SurveillanceAugmentations

logger = logging.getLogger(__name__)


class RFDETRAugmentationPipeline:
    """
    Specialized augmentation pipeline for RF-DETR training on MTMMC dataset.
    Provides pre-configured pipelines optimized for surveillance person detection.
    """
    
    # Predefined pipeline configurations
    PIPELINE_CONFIGS = {
        "basic": {
            "description": "Basic augmentation pipeline for development and testing",
            "geometric_level": "basic",
            "photometric_level": "basic", 
            "occlusion_level": "basic",
            "surveillance_level": "basic",
            "max_augmentations_per_category": 2
        },
        "standard": {
            "description": "Standard augmentation pipeline for general training",
            "geometric_level": "advanced",
            "photometric_level": "advanced",
            "occlusion_level": "basic",
            "surveillance_level": "basic",
            "max_augmentations_per_category": 3
        },
        "advanced": {
            "description": "Advanced augmentation pipeline with all techniques",
            "geometric_level": "advanced",
            "photometric_level": "advanced",
            "occlusion_level": "advanced",
            "surveillance_level": "advanced",
            "max_augmentations_per_category": None
        },
        "surveillance_optimized": {
            "description": "MTMMC surveillance-optimized pipeline",
            "geometric_level": "conservative",
            "photometric_level": "surveillance_optimized", 
            "occlusion_level": "surveillance",
            "surveillance_level": "mtmmc_optimized",
            "max_augmentations_per_category": None
        },
        "production": {
            "description": "Conservative pipeline for production deployment",
            "geometric_level": "conservative",
            "photometric_level": "surveillance_optimized",
            "occlusion_level": "basic",
            "surveillance_level": "basic",
            "max_augmentations_per_category": 2
        }
    }
    
    def __init__(self, config_name: str = "standard", custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialize RF-DETR augmentation pipeline.
        
        Args:
            config_name: Name of predefined configuration or "custom"
            custom_config: Custom configuration dict if config_name is "custom"
        """
        self.config_name = config_name
        
        if config_name == "custom" and custom_config:
            self.config = custom_config
        elif config_name in self.PIPELINE_CONFIGS:
            self.config = self.PIPELINE_CONFIGS[config_name].copy()
        else:
            logger.warning(f"Unknown config '{config_name}', using 'standard'")
            self.config = self.PIPELINE_CONFIGS["standard"].copy()
        
        self.pipeline = self._build_pipeline()
        
        logger.info(f"Initialized RF-DETR augmentation pipeline: {self.config['description']}")
        
    def _build_pipeline(self) -> AugmentationPipeline:
        """Build the augmentation pipeline based on configuration."""
        
        all_augmentations = []
        
        # Add geometric augmentations
        geometric_level = self.config.get("geometric_level", "basic")
        if geometric_level == "basic":
            all_augmentations.extend(GeometricAugmentations.create_basic_geometric_pipeline())
        elif geometric_level == "advanced":
            all_augmentations.extend(GeometricAugmentations.create_advanced_geometric_pipeline())
        elif geometric_level == "conservative":
            all_augmentations.extend(GeometricAugmentations.create_conservative_geometric_pipeline())
        
        # Add photometric augmentations  
        photometric_level = self.config.get("photometric_level", "basic")
        if photometric_level == "basic":
            all_augmentations.extend(PhotometricAugmentations.create_basic_photometric_pipeline())
        elif photometric_level == "advanced":
            all_augmentations.extend(PhotometricAugmentations.create_advanced_photometric_pipeline())
        elif photometric_level == "surveillance_optimized":
            all_augmentations.extend(PhotometricAugmentations.create_surveillance_optimized_pipeline())
        
        # Add occlusion augmentations
        occlusion_level = self.config.get("occlusion_level", "basic")
        if occlusion_level == "basic":
            all_augmentations.extend(OcclusionAugmentations.create_basic_occlusion_pipeline())
        elif occlusion_level == "advanced":
            all_augmentations.extend(OcclusionAugmentations.create_advanced_occlusion_pipeline())
        elif occlusion_level == "surveillance":
            all_augmentations.extend(OcclusionAugmentations.create_surveillance_occlusion_pipeline())
        
        # Add surveillance-specific augmentations
        surveillance_level = self.config.get("surveillance_level", "basic")
        if surveillance_level == "basic":
            all_augmentations.extend(SurveillanceAugmentations.create_basic_surveillance_pipeline())
        elif surveillance_level == "advanced":
            all_augmentations.extend(SurveillanceAugmentations.create_advanced_surveillance_pipeline())
        elif surveillance_level == "mtmmc_optimized":
            all_augmentations.extend(SurveillanceAugmentations.create_mtmmc_optimized_pipeline())
        
        # Create pipeline with configuration
        max_augs = self.config.get("max_augmentations_per_category", None)
        
        pipeline = AugmentationPipeline(
            augmentations=all_augmentations,
            shuffle_order=True,  # Randomize augmentation order
            max_augmentations=max_augs
        )
        
        return pipeline
    
    def __call__(self, image, bboxes):
        """Apply augmentation pipeline to image and bounding boxes."""
        return self.pipeline(image, bboxes)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get detailed information about the current pipeline."""
        stats = self.pipeline.get_augmentation_stats()
        
        info = {
            "config_name": self.config_name,
            "config": self.config,
            "pipeline_stats": stats,
            "augmentation_summary": self._get_augmentation_summary()
        }
        
        return info
    
    def _get_augmentation_summary(self) -> Dict[str, List[str]]:
        """Get summary of augmentations by category."""
        summary = {
            "geometric": [],
            "photometric": [], 
            "occlusion": [],
            "surveillance": []
        }
        
        for aug in self.pipeline.augmentations:
            aug_name = aug.__class__.__name__
            
            if "Flip" in aug_name or "Scale" in aug_name or "Translation" in aug_name or "Perspective" in aug_name:
                summary["geometric"].append(aug_name)
            elif "Brightness" in aug_name or "Color" in aug_name or "Noise" in aug_name or "Blur" in aug_name or "Gamma" in aug_name:
                summary["photometric"].append(aug_name)
            elif "Erasing" in aug_name or "Mosaic" in aug_name or "CopyPaste" in aug_name:
                summary["occlusion"].append(aug_name)
            elif "Crowd" in aug_name or "Lighting" in aug_name or "Weather" in aug_name:
                summary["surveillance"].append(aug_name)
        
        return summary
    
    def set_global_severity(self, severity: float):
        """Set severity for all augmentations in the pipeline."""
        self.pipeline.set_global_severity(severity)
        logger.info(f"Set global augmentation severity to {severity}")
    
    def enable_category(self, category: str):
        """Enable all augmentations in a specific category."""
        category_map = {
            "geometric": ["Flip", "Scale", "Translation", "Perspective"],
            "photometric": ["Brightness", "Color", "Noise", "Blur", "Gamma"],
            "occlusion": ["Erasing", "Mosaic", "CopyPaste"],
            "surveillance": ["Crowd", "Lighting", "Weather"]
        }
        
        if category in category_map:
            keywords = category_map[category]
            for aug in self.pipeline.augmentations:
                if any(keyword in aug.__class__.__name__ for keyword in keywords):
                    aug.enable()
            logger.info(f"Enabled {category} augmentations")
    
    def disable_category(self, category: str):
        """Disable all augmentations in a specific category."""
        category_map = {
            "geometric": ["Flip", "Scale", "Translation", "Perspective"],
            "photometric": ["Brightness", "Color", "Noise", "Blur", "Gamma"],
            "occlusion": ["Erasing", "Mosaic", "CopyPaste"],
            "surveillance": ["Crowd", "Lighting", "Weather"]
        }
        
        if category in category_map:
            keywords = category_map[category]
            for aug in self.pipeline.augmentations:
                if any(keyword in aug.__class__.__name__ for keyword in keywords):
                    aug.disable()
            logger.info(f"Disabled {category} augmentations")
    
    def get_effective_augmentations(self) -> List[str]:
        """Get list of currently enabled augmentations."""
        enabled_augs = []
        for aug in self.pipeline.augmentations:
            if aug.config.enabled:
                enabled_augs.append(f"{aug.__class__.__name__} (p={aug.config.probability:.2f})")
        return enabled_augs
    
    @staticmethod
    def get_available_configs() -> Dict[str, str]:
        """Get all available pipeline configurations."""
        return {name: config["description"] for name, config in RFDETRAugmentationPipeline.PIPELINE_CONFIGS.items()}
    
    @staticmethod 
    def create_custom_pipeline(
        geometric_augmentations: Optional[List[BaseAugmentation]] = None,
        photometric_augmentations: Optional[List[BaseAugmentation]] = None,
        occlusion_augmentations: Optional[List[BaseAugmentation]] = None,
        surveillance_augmentations: Optional[List[BaseAugmentation]] = None,
        shuffle_order: bool = True,
        max_augmentations: Optional[int] = None
    ) -> AugmentationPipeline:
        """
        Create a custom augmentation pipeline with specific augmentations.
        
        Args:
            geometric_augmentations: List of geometric augmentations
            photometric_augmentations: List of photometric augmentations
            occlusion_augmentations: List of occlusion augmentations
            surveillance_augmentations: List of surveillance-specific augmentations
            shuffle_order: Whether to shuffle augmentation order
            max_augmentations: Maximum number of augmentations to apply
            
        Returns:
            Custom augmentation pipeline
        """
        all_augmentations = []
        
        if geometric_augmentations:
            all_augmentations.extend(geometric_augmentations)
        if photometric_augmentations:
            all_augmentations.extend(photometric_augmentations)
        if occlusion_augmentations:
            all_augmentations.extend(occlusion_augmentations)
        if surveillance_augmentations:
            all_augmentations.extend(surveillance_augmentations)
        
        return AugmentationPipeline(
            augmentations=all_augmentations,
            shuffle_order=shuffle_order,
            max_augmentations=max_augmentations
        )


def create_rfdetr_augmentation_pipeline(
    config_name: str = "surveillance_optimized",
    severity: float = 0.5,
    custom_params: Optional[Dict[str, Any]] = None
) -> RFDETRAugmentationPipeline:
    """
    Convenience function to create RF-DETR augmentation pipeline.
    
    Args:
        config_name: Pipeline configuration name
        severity: Global augmentation severity (0.0-1.0)
        custom_params: Custom parameters to override defaults
        
    Returns:
        Configured RF-DETR augmentation pipeline
    """
    if custom_params and config_name != "custom":
        # Merge custom params with predefined config
        base_config = RFDETRAugmentationPipeline.PIPELINE_CONFIGS.get(config_name, {}).copy()
        base_config.update(custom_params)
        pipeline = RFDETRAugmentationPipeline("custom", base_config)
    else:
        pipeline = RFDETRAugmentationPipeline(config_name, custom_params)
    
    # Set global severity
    pipeline.set_global_severity(severity)
    
    return pipeline


def validate_augmentation_pipeline(pipeline: RFDETRAugmentationPipeline) -> Dict[str, Any]:
    """
    Validate augmentation pipeline configuration and performance.
    
    Args:
        pipeline: RF-DETR augmentation pipeline to validate
        
    Returns:
        Validation results dictionary
    """
    import numpy as np
    from PIL import Image
    
    validation_results = {
        "pipeline_valid": True,
        "issues": [],
        "warnings": [],
        "performance_metrics": {},
        "augmentation_coverage": {}
    }
    
    try:
        # Test with dummy data
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_bboxes = [
            BoundingBox(x=100, y=100, width=50, height=120, class_id=0),
            BoundingBox(x=300, y=200, width=60, height=140, class_id=0)
        ]
        
        # Test pipeline execution
        result = pipeline(dummy_image, dummy_bboxes)
        
        if result.image is None or len(result.bboxes) == 0:
            validation_results["issues"].append("Pipeline produced invalid output")
            validation_results["pipeline_valid"] = False
        
        # Check augmentation coverage
        info = pipeline.get_pipeline_info()
        augmentation_summary = info["augmentation_summary"]
        
        for category, augs in augmentation_summary.items():
            validation_results["augmentation_coverage"][category] = len(augs)
        
        # Performance test (simplified)
        import time
        start_time = time.time()
        
        for _ in range(10):
            _ = pipeline(dummy_image, dummy_bboxes)
        
        avg_time = (time.time() - start_time) / 10
        validation_results["performance_metrics"]["avg_processing_time_ms"] = avg_time * 1000
        
        if avg_time > 0.1:  # 100ms threshold
            validation_results["warnings"].append(f"Slow processing time: {avg_time*1000:.1f}ms")
        
        # Check for enabled augmentations
        enabled_augs = pipeline.get_effective_augmentations()
        validation_results["performance_metrics"]["enabled_augmentations"] = len(enabled_augs)
        
        if len(enabled_augs) == 0:
            validation_results["warnings"].append("No augmentations are enabled")
        elif len(enabled_augs) > 15:
            validation_results["warnings"].append(f"Many augmentations enabled ({len(enabled_augs)}), may be slow")
        
    except Exception as e:
        validation_results["issues"].append(f"Pipeline validation failed: {str(e)}")
        validation_results["pipeline_valid"] = False
    
    return validation_results


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Test different pipeline configurations
    configs_to_test = ["basic", "standard", "advanced", "surveillance_optimized", "production"]
    
    for config_name in configs_to_test:
        print(f"\nTesting {config_name} pipeline:")
        pipeline = create_rfdetr_augmentation_pipeline(config_name, severity=0.5)
        
        info = pipeline.get_pipeline_info()
        print(f"Description: {info['config']['description']}")
        print(f"Total augmentations: {info['pipeline_stats']['total_augmentations']}")
        print(f"Enabled augmentations: {info['pipeline_stats']['enabled_augmentations']}")
        
        # Validate pipeline
        validation = validate_augmentation_pipeline(pipeline)
        status = "✅ VALID" if validation["pipeline_valid"] else "❌ INVALID"
        print(f"Validation: {status}")
        
        if validation["warnings"]:
            for warning in validation["warnings"]:
                print(f"  ⚠️ {warning}")
        
        if validation["issues"]:
            for issue in validation["issues"]:
                print(f"  ❌ {issue}")
        
        print(f"Processing time: {validation['performance_metrics'].get('avg_processing_time_ms', 0):.1f}ms")
        print("-" * 50)