"""
Pipeline for backend-style tracking and Re-ID evaluation using ground truth bounding boxes.
This pipeline adapts the core tracking and Re-ID logic from the SpotOn backend.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.tracking_backend_logic.camera_tracker_factory_adapter import CameraTrackerFactoryAdapter
from src.tracking_backend_logic.handoff_logic_adapter import HandoffLogicAdapter
from src.tracking_backend_logic.reid_manager_adapter import ReIDManagerAdapter
from src.tracking_backend_logic.common_types_adapter import (
    CameraID,
    FeatureVector,
    GlobalID,
    TrackID
)

logger = logging.getLogger(__name__)

class BackendStyleTrackingReIDPipeline:
    """
    Pipeline for evaluating tracking and Re-ID performance using ground truth bounding boxes.
    This pipeline adapts the core tracking and Re-ID logic from the SpotOn backend.
    """
    def __init__(
        self,
        reid_weights_path: Path,
        handoff_config_path: Path,
        device: torch.device,
        half_precision: bool = False,
        per_class: bool = False,
        similarity_threshold: float = 0.7
    ):
        """
        Initializes the BackendStyleTrackingReIDPipeline.

        Args:
            reid_weights_path: Path to the Re-ID model weights file.
            handoff_config_path: Path to the handoff configuration YAML file.
            device: The torch.device to use for models.
            half_precision: Whether to use half precision for model operations.
            per_class: Whether trackers should operate on a per-class basis.
            similarity_threshold: Threshold for considering two features as a match.
        """
        self.device = device
        self.half_precision = half_precision
        self.per_class = per_class
        self.similarity_threshold = similarity_threshold

        # Initialize components
        self.tracker_factory = CameraTrackerFactoryAdapter(
            reid_weights_path=reid_weights_path,
            device=device,
            half_precision=half_precision,
            per_class=per_class
        )

        self.reid_manager = ReIDManagerAdapter(
            model_path=reid_weights_path,
            device=device,
            half_precision=half_precision,
            similarity_threshold=similarity_threshold
        )

        self.handoff_logic = HandoffLogicAdapter(config_path=handoff_config_path)

        # State tracking
        self._global_id_counter = 0
        self._global_id_to_features: Dict[GlobalID, FeatureVector] = {}
        self._camera_track_to_global_id: Dict[Tuple[CameraID, TrackID], GlobalID] = {}

        logger.info(
            f"BackendStyleTrackingReIDPipeline initialized with ReID weights: {reid_weights_path}, "
            f"Handoff config: {handoff_config_path}, Device: {device}, "
            f"Half: {half_precision}, PerClass: {per_class}, "
            f"Similarity threshold: {similarity_threshold}"
        )

    def load_models(self):
        """Loads all required models and configurations."""
        logger.info("Loading models and configurations...")
        
        # Load Re-ID model
        self.reid_manager.load_model()
        
        # Load handoff configuration
        self.handoff_logic.load_config()
        
        logger.info("Models and configurations loaded successfully")

    def warmup(self, dummy_image_shape: Tuple[int, int, int] = (640, 480, 3)):
        """Warms up all models with dummy data."""
        logger.info("Warming up models...")
        
        # Warm up Re-ID model
        self.reid_manager.warmup(dummy_image_shape)
        
        # Warm up trackers (will be created on demand)
        logger.info("Models warmed up successfully")

    def process_frame(
        self,
        camera_id: CameraID,
        frame: np.ndarray,
        gt_bboxes: np.ndarray,
        gt_classes: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[TrackID, GlobalID]]:
        """
        Processes a single frame with ground truth bounding boxes.

        Args:
            camera_id: ID of the camera providing the frame.
            frame: Frame as a NumPy array (BGR).
            gt_bboxes: Ground truth bounding boxes in [x1, y1, x2, y2] format.
            gt_classes: Optional ground truth class IDs.

        Returns:
            Tuple of (tracked_objects, track_to_global_id_mapping)
            tracked_objects: NumPy array of tracked objects
            track_to_global_id_mapping: Dictionary mapping track IDs to global IDs
        """
        # Get or create tracker for this camera
        tracker = self.tracker_factory.get_tracker(camera_id)

        # Prepare detections with ground truth boxes
        if gt_classes is not None:
            detections = np.column_stack([gt_bboxes, np.ones(len(gt_bboxes)), gt_classes])
        else:
            detections = np.column_stack([gt_bboxes, np.ones(len(gt_bboxes)), np.zeros(len(gt_bboxes))])

        # Update tracker with ground truth detections
        tracked_objects = tracker.update(detections, frame)

        # Process tracked objects for Re-ID and handoff
        track_to_global_id = {}
        if len(tracked_objects) > 0:
            # Extract features for each tracked object
            for track in tracked_objects:
                track_id = int(track[4])  # Assuming track ID is in column 5
                bbox = track[:4]  # [x1, y1, x2, y2]

                # Extract image patch
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue

                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                # Convert to PIL Image for feature extraction
                patch_pil = Image.fromarray(patch)
                
                # Extract features
                features = self.reid_manager.extract_features(patch_pil)

                # Check for handoff
                handoff_info = self.handoff_logic.check_exit_rules(
                    camera_id, bbox, frame.shape[:2]
                )

                if handoff_info is not None:
                    # Handle handoff
                    target_cameras = self.handoff_logic.get_target_cameras(camera_id)
                    if target_cameras:
                        # Store features for handoff
                        self._global_id_to_features[self._global_id_counter] = features
                        track_to_global_id[track_id] = self._global_id_counter
                        self._camera_track_to_global_id[(camera_id, track_id)] = self._global_id_counter
                        self._global_id_counter += 1
                else:
                    # Try to match with existing global IDs
                    global_id = self.reid_manager.assign_global_id(
                        features, self._global_id_to_features
                    )

                    if global_id is not None:
                        # Update existing global ID
                        track_to_global_id[track_id] = global_id
                        self._camera_track_to_global_id[(camera_id, track_id)] = global_id
                    else:
                        # Assign new global ID
                        self._global_id_to_features[self._global_id_counter] = features
                        track_to_global_id[track_id] = self._global_id_counter
                        self._camera_track_to_global_id[(camera_id, track_id)] = self._global_id_counter
                        self._global_id_counter += 1

        return tracked_objects, track_to_global_id

    def reset(self):
        """Resets the pipeline state."""
        logger.info("Resetting pipeline state...")
        
        # Reset trackers
        self.tracker_factory.reset_all_trackers()
        
        # Reset state tracking
        self._global_id_counter = 0
        self._global_id_to_features.clear()
        self._camera_track_to_global_id.clear()
        
        logger.info("Pipeline state reset successfully")

    def get_global_id_mapping(self) -> Dict[Tuple[CameraID, TrackID], GlobalID]:
        """
        Gets the current mapping of camera-track pairs to global IDs.

        Returns:
            Dictionary mapping (camera_id, track_id) tuples to global IDs.
        """
        return self._camera_track_to_global_id.copy() 