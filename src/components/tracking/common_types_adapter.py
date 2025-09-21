"""
Module for shared type aliases and Pydantic data structures adapted from the
SpotOn backend, for use in the backend-style tracking/Re-ID pipeline.
"""
from typing import Dict, List, NewType, NamedTuple, Callable, Tuple, Optional, Set # Added Set for Python 3.8 compatibility

import numpy as np
from pydantic import BaseModel, Field

# --- Basic Types from Backend ---
CameraID = NewType("CameraID", str)
TrackID = NewType("TrackID", int)  # Intra-camera track ID
GlobalID = NewType("GlobalID", str) # System-wide unique person ID (UUID or similar string)
FeatureVector = NewType("FeatureVector", np.ndarray) # Assuming np.ndarray for features
BoundingBoxXYXY = NewType("BoundingBoxXYXY", List[float]) # [x1, y1, x2, y2]
TrackKey = Tuple[CameraID, TrackID] # Uniquely identifies a track within a camera

# --- Handoff Logic Types Adapted from Backend ---
QuadrantName = NewType("QuadrantName", str)  # e.g., 'upper_left', 'lower_right'

class ExitRuleModelAdapter(BaseModel):
    """
    Pydantic model for an exit rule, adapted from backend's ExitRuleModel.
    Used in YAML configuration.
    """
    source_exit_quadrant: QuadrantName = Field(
        ...,
        description="The source quadrant in the current camera that triggers this rule (e.g., 'upper_right')."
    )
    target_cam_id: CameraID = Field(
        ...,
        description="The camera ID this rule targets for handoff."
    )
    target_entry_area: str = Field(
        ...,
        description="Descriptive name of the entry area in the target camera (e.g., 'lower_left')."
    )
    notes: Optional[str] = Field(
        None,
        description="Optional notes about this rule."
    )

class CameraHandoffDetailConfigAdapter(BaseModel):
    """
    Pydantic model for detailed camera configuration including handoff rules,
    adapted from backend's CameraHandoffDetailConfig. Used in YAML.
    """
    exit_rules: List[ExitRuleModelAdapter] = Field(default_factory=list)
    homography_matrix_path: Optional[str] = Field(
        None,
        description="Path to the .npz file containing homography points for this camera and scene, "
                    "relative to handoff_config.homography_data_dir. (For config parity, not used by current handoff logic)."
    )

class HandoffTriggerInfo(NamedTuple):
    """
    Holds information about a triggered handoff event for a specific track.
    Adapted from backend.
    """
    source_track_key: TrackKey
    rule: ExitRuleModelAdapter # Uses the adapter model
    source_bbox: BoundingBoxXYXY # BBox that triggered the rule

# --- Quadrant Calculation Map from Backend ---
QUADRANT_REGIONS_TEMPLATE: Dict[QuadrantName, Callable[[int, int], Tuple[int, int, int, int]]] = {
    QuadrantName('upper_left'): lambda W, H: (0, 0, W // 2, H // 2),
    QuadrantName('upper_right'): lambda W, H: (W // 2, 0, W, H // 2),
    QuadrantName('lower_left'): lambda W, H: (0, H // 2, W // 2, H),
    QuadrantName('lower_right'): lambda W, H: (W // 2, H // 2, W, H),
}

# Type for storing parsed handoff config from YAML
ParsedCameraHandoffConfigs = Dict[Tuple[str, str], CameraHandoffDetailConfigAdapter] 