"""
Adapter for the handoff logic from the SpotOn backend, for use in the MLflow framework.
This class mirrors the structure and expected behavior of the backend's handoff logic.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.tracking_backend_logic.common_types_adapter import (
    CameraID,
    CameraHandoffDetailConfigAdapter,
    ExitRuleModelAdapter,
    HandoffTriggerInfo,
    QUADRANT_REGIONS_TEMPLATE
)

logger = logging.getLogger(__name__)

class HandoffLogicAdapter:
    """
    Manages handoff logic between cameras, adapted from the SpotOn backend.
    This class handles exit rule checking and handoff triggering.
    """
    def __init__(self, config_path: Path):
        """
        Initializes the HandoffLogicAdapter.

        Args:
            config_path: Path to the handoff configuration YAML file.
        """
        self.config_path = config_path
        self._config: Dict[CameraID, CameraHandoffDetailConfigAdapter] = {}
        self._homography_matrices: Dict[CameraID, np.ndarray] = {}
        self._quadrant_regions: Dict[CameraID, Dict[str, np.ndarray]] = {}

        logger.info(f"HandoffLogicAdapter initialized with config: {self.config_path}")

    def load_config(self):
        """
        Loads handoff configuration from YAML file.
        This method is synchronous to match typical config loading patterns.
        """
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Clear existing config
            self._config.clear()
            self._homography_matrices.clear()
            self._quadrant_regions.clear()

            # Load camera configurations
            for camera_id, camera_config in config_data.items():
                # Convert to Pydantic model
                camera_handoff_config = CameraHandoffDetailConfigAdapter(
                    camera_id=camera_id,
                    exit_rules=[
                        ExitRuleModelAdapter(**rule)
                        for rule in camera_config.get('exit_rules', [])
                    ],
                    homography_matrix_path=camera_config.get('homography_matrix_path')
                )
                self._config[camera_id] = camera_handoff_config

                # Load homography matrix if path is provided
                if camera_handoff_config.homography_matrix_path:
                    try:
                        matrix_path = Path(camera_handoff_config.homography_matrix_path)
                        if matrix_path.exists():
                            self._homography_matrices[camera_id] = np.load(str(matrix_path))
                            logger.info(f"Loaded homography matrix for camera {camera_id}")
                        else:
                            logger.warning(
                                f"Homography matrix file not found for camera {camera_id}: "
                                f"{matrix_path}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error loading homography matrix for camera {camera_id}: {e}",
                            exc_info=True
                        )

                # Initialize quadrant regions
                self._quadrant_regions[camera_id] = QUADRANT_REGIONS_TEMPLATE.copy()

            logger.info(f"Loaded handoff configuration for {len(self._config)} cameras")
        except Exception as e:
            logger.error(f"Error loading handoff configuration: {e}", exc_info=True)
            raise

    def check_exit_rules(
        self,
        camera_id: CameraID,
        bbox: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> Optional[HandoffTriggerInfo]:
        """
        Checks if a bounding box triggers any exit rules for the given camera.

        Args:
            camera_id: ID of the camera to check exit rules for.
            bbox: Bounding box in [x1, y1, x2, y2] format.
            frame_shape: Shape of the frame (height, width).

        Returns:
            HandoffTriggerInfo if an exit rule is triggered, None otherwise.
        """
        if camera_id not in self._config:
            logger.warning(f"No handoff configuration found for camera {camera_id}")
            return None

        camera_config = self._config[camera_id]
        if not camera_config.exit_rules:
            return None

        # Get frame dimensions
        frame_height, frame_width = frame_shape

        # Check each exit rule
        for rule in camera_config.exit_rules:
            # Get quadrant regions for this camera
            quadrant_regions = self._quadrant_regions[camera_id]

            # Check if bbox is in the specified quadrant
            if rule.quadrant in quadrant_regions:
                region = quadrant_regions[rule.quadrant]
                bbox_center = np.array([
                    (bbox[0] + bbox[2]) / 2,  # x center
                    (bbox[1] + bbox[3]) / 2   # y center
                ])

                # Check if bbox center is in the quadrant region
                if self._is_point_in_region(bbox_center, region):
                    # Check if bbox is close enough to the edge
                    if self._is_bbox_near_edge(bbox, frame_width, frame_height, rule.edge_threshold):
                        return HandoffTriggerInfo(
                            camera_id=camera_id,
                            exit_rule=rule,
                            bbox=bbox
                        )

        return None

    def _is_point_in_region(self, point: np.ndarray, region: np.ndarray) -> bool:
        """
        Checks if a point is inside a region defined by a polygon.

        Args:
            point: Point coordinates [x, y].
            region: Region polygon vertices.

        Returns:
            True if point is inside region, False otherwise.
        """
        # Simple bounding box check first
        min_x, min_y = np.min(region, axis=0)
        max_x, max_y = np.max(region, axis=0)
        if not (min_x <= point[0] <= max_x and min_y <= point[1] <= max_y):
            return False

        # Ray casting algorithm for point-in-polygon
        n = len(region)
        inside = False
        j = n - 1
        for i in range(n):
            if ((region[i][1] > point[1]) != (region[j][1] > point[1]) and
                point[0] < (region[j][0] - region[i][0]) * (point[1] - region[i][1]) /
                (region[j][1] - region[i][1]) + region[i][0]):
                inside = not inside
            j = i
        return inside

    def _is_bbox_near_edge(
        self,
        bbox: np.ndarray,
        frame_width: int,
        frame_height: int,
        threshold: float
    ) -> bool:
        """
        Checks if a bounding box is near the edge of the frame.

        Args:
            bbox: Bounding box in [x1, y1, x2, y2] format.
            frame_width: Width of the frame.
            frame_height: Height of the frame.
            threshold: Distance threshold from edge.

        Returns:
            True if bbox is near edge, False otherwise.
        """
        # Check distance to each edge
        dist_to_left = bbox[0]
        dist_to_right = frame_width - bbox[2]
        dist_to_top = bbox[1]
        dist_to_bottom = frame_height - bbox[3]

        return any(d < threshold for d in [dist_to_left, dist_to_right, dist_to_top, dist_to_bottom])

    def get_target_cameras(self, camera_id: CameraID) -> List[CameraID]:
        """
        Gets the list of target cameras for handoff from a source camera.

        Args:
            camera_id: ID of the source camera.

        Returns:
            List of target camera IDs.
        """
        if camera_id not in self._config:
            return []

        camera_config = self._config[camera_id]
        target_cameras = set()
        for rule in camera_config.exit_rules:
            target_cameras.update(rule.target_cameras)
        return list(target_cameras)

    def get_homography_matrix(self, camera_id: CameraID) -> Optional[np.ndarray]:
        """
        Gets the homography matrix for a camera.

        Args:
            camera_id: ID of the camera.

        Returns:
            Homography matrix if available, None otherwise.
        """
        return self._homography_matrices.get(camera_id) 