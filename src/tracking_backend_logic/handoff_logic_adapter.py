"""
Adapter for the handoff logic from the SpotOn backend, for use in the MLflow framework.
This class mirrors the structure and expected behavior of the backend's handoff logic.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
import yaml # Keep yaml for potential direct loading if adapter design changes

from src.tracking_backend_logic.common_types_adapter import (
    CameraID,
    CameraHandoffDetailConfigAdapter,
    ExitRuleModelAdapter,
    HandoffTriggerInfo,
    QUADRANT_REGIONS_TEMPLATE,
    QuadrantName # Ensure QuadrantName is imported
)

logger = logging.getLogger(__name__)

class HandoffLogicAdapter:
    """
    Manages handoff logic between cameras, adapted from the SpotOn backend.
    This class handles exit rule checking and handoff triggering.
    """
    def __init__(self, handoff_config_dict: Dict[str, Any], project_root: Path):
        """
        Initializes the HandoffLogicAdapter.

        Args:
            handoff_config_dict: The 'handoff_config' section from the main configuration.
            project_root: The root directory of the project, for resolving relative paths.
        """
        self.handoff_config_dict = handoff_config_dict
        self.project_root = project_root
        self._camera_configs: Dict[Tuple[str, str], CameraHandoffDetailConfigAdapter] = {} # Key: (env_id_str, cam_id_str)
        self._homography_matrices: Dict[Tuple[str, str], np.ndarray] = {} # Key: (env_id_str, cam_id_str)

        logger.info(f"HandoffLogicAdapter initialized.")
        self._parse_config_dict()

    def _parse_config_dict(self):
        """
        Parses the handoff_config dictionary provided during initialization.
        """
        try:
            camera_details_config = self.handoff_config_dict.get("camera_details", {})
            homography_base_dir_rel = self.handoff_config_dict.get("homography_data_dir", "homography_data_ml")
            homography_base_dir_abs = (self.project_root / homography_base_dir_rel).resolve()

            self._camera_configs.clear()
            self._homography_matrices.clear()

            for cam_env_tuple_str, camera_config_entry_dict in camera_details_config.items():
                try:
                    # Safely parse the string tuple "('env', 'cam_id')"
                    parsed_tuple = eval(cam_env_tuple_str) # Using eval as per original structure, assumes trusted config
                    if not (isinstance(parsed_tuple, tuple) and len(parsed_tuple) == 2 and
                            all(isinstance(s, str) for s in parsed_tuple)):
                        raise ValueError("Key format error")
                    env_id_str, cam_id_str = parsed_tuple
                    current_key = (env_id_str, cam_id_str)
                except Exception as e:
                    logger.error(f"Could not parse camera_details key '{cam_env_tuple_str}': {e}. Skipping.")
                    continue

                # Pydantic model validation for the entry
                camera_handoff_config = CameraHandoffDetailConfigAdapter(**camera_config_entry_dict)
                self._camera_configs[current_key] = camera_handoff_config

                if camera_handoff_config.homography_matrix_path:
                    try:
                        matrix_path = homography_base_dir_abs / camera_handoff_config.homography_matrix_path
                        if matrix_path.exists():
                            loaded_data = np.load(str(matrix_path))
                            if isinstance(loaded_data, np.lib.npyio.NpzFile) and list(loaded_data.files):
                                matrix_key_in_npz = list(loaded_data.files)[0] # Assume first array is the matrix
                                self._homography_matrices[current_key] = loaded_data[matrix_key_in_npz]
                                logger.info(f"Loaded homography matrix (key: {matrix_key_in_npz}) for {current_key} from NPZ {matrix_path}")
                                loaded_data.close()
                            elif isinstance(loaded_data, np.ndarray):
                                self._homography_matrices[current_key] = loaded_data
                                logger.info(f"Loaded homography matrix for key {current_key} from {matrix_path}")
                            else:
                                logger.warning(f"Homography data for {current_key} at {matrix_path} is not a direct NumPy array or recognizable NPZ. Type: {type(loaded_data)}")
                        else:
                            logger.warning(f"Homography matrix file not found for key {current_key}: {matrix_path}")
                    except Exception as e:
                        logger.error(
                            f"Error loading homography matrix for key {current_key} from path '{camera_handoff_config.homography_matrix_path}': {e}",
                            exc_info=False # Keep log cleaner for this common warning
                        )
            num_configs = len(self._camera_configs)
            num_matrices = len(self._homography_matrices)
            logger.info(f"Parsed handoff config. Loaded details for {num_configs} camera-env pairs and {num_matrices} homography matrices.")

        except Exception as e:
            logger.error(f"Error parsing handoff configuration dictionary: {e}", exc_info=True)
            raise

    def check_exit_rules(
        self,
        environment_id: str,
        camera_id: CameraID,
        bbox_xyxy: np.ndarray, # Expect [x1, y1, x2, y2]
        frame_shape: Tuple[int, int] # (height, width)
    ) -> Optional[HandoffTriggerInfo]:
        """
        Checks if a bounding box triggers any exit rules for the given camera and environment.
        """
        current_key = (str(environment_id), str(camera_id))
        camera_config = self._camera_configs.get(current_key)

        if not camera_config or not camera_config.exit_rules:
            # logger.debug(f"No handoff configuration or exit rules for camera {camera_id} in env {environment_id}.")
            return None

        frame_height, frame_width = frame_shape
        bbox_center_x = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
        bbox_center_y = (bbox_xyxy[1] + bbox_xyxy[3]) / 2

        for rule in camera_config.exit_rules:
            quadrant_func = QUADRANT_REGIONS_TEMPLATE.get(QuadrantName(rule.source_exit_quadrant))
            if not quadrant_func:
                logger.warning(f"Unknown quadrant name '{rule.source_exit_quadrant}' for {current_key}. Skipping rule.")
                continue

            qx1, qy1, qx2, qy2 = quadrant_func(frame_width, frame_height)

            if (qx1 <= bbox_center_x <= qx2 and qy1 <= bbox_center_y <= qy2):
                # Simplified check: if center of bbox is in the quadrant, consider it triggered.
                # Backend's logic might involve overlap ratios or edge proximity.
                # This adapter currently relies on the center being within the defined quadrant.
                logger.debug(f"Handoff rule triggered for {current_key}: Box center in {rule.source_exit_quadrant}, "
                             f"targets {rule.target_cam_id}")
                return HandoffTriggerInfo(
                    source_track_key=(camera_id, TrackID(-1)), # Actual TrackID filled by pipeline
                    rule=rule,
                    source_bbox=list(bbox_xyxy) # Ensure it's a list of floats
                )
        return None

    def get_target_cameras(self, environment_id: str, camera_id: CameraID) -> List[CameraID]:
        """
        Gets the list of unique target camera IDs for handoff from a source camera and environment.
        """
        current_key = (str(environment_id), str(camera_id))
        camera_config = self._camera_configs.get(current_key)
        if not camera_config:
            return []

        target_cameras: Set[CameraID] = set()
        for rule in camera_config.exit_rules:
            target_cameras.add(rule.target_cam_id)
        return list(target_cameras)

    def get_homography_matrix(self, environment_id: str, camera_id: CameraID) -> Optional[np.ndarray]:
        """
        Gets the pre-loaded homography matrix for a camera in a specific environment.
        """
        current_key = (str(environment_id), str(camera_id))
        return self._homography_matrices.get(current_key)
    
    