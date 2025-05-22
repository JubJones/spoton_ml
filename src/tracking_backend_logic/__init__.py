"""
This package adapts core tracking and Re-ID components from the SpotOn backend
for use within the SpotOn ML MLOps framework.
"""
from .common_types_adapter import (
    CameraID, TrackID, GlobalID, FeatureVector, BoundingBoxXYXY,
    TrackKey, QuadrantName, ExitRuleModelAdapter, CameraHandoffDetailConfigAdapter,
    HandoffTriggerInfo, QUADRANT_REGIONS_TEMPLATE
)
from .botsort_tracker_adapter import BotSortTrackerAdapter
from .camera_tracker_factory_adapter import CameraTrackerFactoryAdapter
from .reid_manager_adapter import ReIDManagerAdapter

__all__ = [
    "CameraID", "TrackID", "GlobalID", "FeatureVector", "BoundingBoxXYXY",
    "TrackKey", "QuadrantName", "ExitRuleModelAdapter", "CameraHandoffDetailConfigAdapter",
    "HandoffTriggerInfo", "QUADRANT_REGIONS_TEMPLATE",
    "BotSortTrackerAdapter",
    "CameraTrackerFactoryAdapter",
    "ReIDManagerAdapter",
] 