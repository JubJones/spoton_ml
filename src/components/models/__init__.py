"""
Enhanced RF-DETR Models for Surveillance Person Detection
Model architecture enhancements and advanced training validation
"""

from .rfdetr_enhancements import (
    EnhancedRFDETR,
    SurveillanceConfig,
    SpatialAttentionModule,
    PersonFocusedFPN,
    CrowdAwareLoss,
    SmallPersonDetector,
    ModelValidator,
    create_enhanced_rfdetr
)

from .surveillance_detector import (
    CrowdAwareDetector,
    DetectorConfig,
    ScaleAwareNMS,
    SurveillanceMetrics,
    create_surveillance_detector
)

from .training_validator import (
    ValidationEngine,
    ValidationConfig,
    ValidationResult,
    PerformanceTracker,
    ModelAnalyzer,
    create_validation_engine
)

__all__ = [
    # Enhanced RF-DETR Architecture
    'EnhancedRFDETR',
    'SurveillanceConfig',
    'SpatialAttentionModule',
    'PersonFocusedFPN',
    'CrowdAwareLoss',
    'SmallPersonDetector',
    'ModelValidator',
    'create_enhanced_rfdetr',
    
    # Surveillance Detection
    'CrowdAwareDetector',
    'DetectorConfig',
    'ScaleAwareNMS', 
    'SurveillanceMetrics',
    'create_surveillance_detector',
    
    # Advanced Validation
    'ValidationEngine',
    'ValidationConfig', 
    'ValidationResult',
    'PerformanceTracker',
    'ModelAnalyzer',
    'create_validation_engine'
]