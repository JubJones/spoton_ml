import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple, Iterator, Optional, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def sorted_alphanumeric(data: List[str]) -> List[str]:
    """Sorts a list of strings alphanumerically."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


class FrameDataLoader:
    """Loads and iterates through frames for specified cameras in a selected scene."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the data loader based on the full configuration.
        """
        self.config = config
        data_config = config.get('data', {})

        self.base_path = Path(data_config.get("base_path", ""))
        if not self.base_path.is_dir():
            raise FileNotFoundError(f"Data base path not found: {self.base_path}")

        self.selected_env = data_config.get("selected_environment")
        if not self.selected_env:
            raise ValueError("Missing 'data.selected_environment' in configuration.")

        env_config = data_config.get(self.selected_env)
        if not env_config or not isinstance(env_config, dict):
            raise ValueError(
                f"Configuration for selected environment '{self.selected_env}' not found or invalid under 'data'.")

        self.scene_id = env_config.get("scene_id", "")
        self.camera_ids = env_config.get("camera_ids", [])
        self.max_frames_per_camera = env_config.get("max_frames_per_camera", -1)

        if not self.scene_id:
            raise ValueError(f"Missing 'scene_id' for selected environment '{self.selected_env}'.")
        if not self.camera_ids:
            raise ValueError(f"Missing 'camera_ids' for selected environment '{self.selected_env}'.")

        self.scene_path = self.base_path / "train" / "train" / self.scene_id
        if not self.scene_path.is_dir():
            raise FileNotFoundError(f"Scene directory not found at the expected path: {self.scene_path}")
        logger.info(f"Using scene path: {self.scene_path}")

        self.camera_image_dirs: Dict[str, Path] = {}
        self.image_filenames: List[str] = []
        self.frame_count: int = 0

        self._discover_data()

    def _find_image_dir_mtmmc(self, cam_id: str) -> Optional[Path]:
        """Finds the 'rgb' image directory specifically for MTMMC structure."""
        potential_img_dir = self.scene_path / cam_id / "rgb"
        if potential_img_dir.is_dir():
            image_files = list(potential_img_dir.glob('*.jpg')) + list(potential_img_dir.glob('*.png')) + list(
                potential_img_dir.glob('*.bmp'))
            if image_files:
                logger.debug(f"Found valid image directory for {cam_id}: {potential_img_dir}")
                return potential_img_dir
        logger.warning(
            f"Could not find a valid 'rgb' directory with images for camera {cam_id} under {self.scene_path / cam_id}")
        return None

    def _discover_data(self):
        """Finds 'rgb' image directories and the list of frame filenames."""
        first_valid_cam_path = None
        valid_cameras = []
        for cam_id in self.camera_ids:
            img_dir = self._find_image_dir_mtmmc(cam_id)
            if img_dir:
                self.camera_image_dirs[cam_id] = img_dir
                valid_cameras.append(cam_id)
                if first_valid_cam_path is None:
                    first_valid_cam_path = img_dir
            else:
                logger.warning(f"Skipping camera {cam_id}: No valid 'rgb' image directory found.")

        self.active_camera_ids = valid_cameras  # Store only the cameras found

        if not self.camera_image_dirs:
            raise FileNotFoundError(
                f"No valid 'rgb' image directories found for any specified camera in scene {self.scene_id}.")

        if first_valid_cam_path:
            try:
                filenames = [
                    f.name for f in first_valid_cam_path.iterdir()
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
                ]
                self.image_filenames = sorted_alphanumeric(filenames)
                total_frames_in_sequence = len(self.image_filenames)

                if self.max_frames_per_camera > 0 and self.max_frames_per_camera < total_frames_in_sequence:
                    self.image_filenames = self.image_filenames[:self.max_frames_per_camera]
                    logger.info(f"Limited processing to the first {len(self.image_filenames)} frames per camera.")
                else:
                    logger.info(f"Processing all {total_frames_in_sequence} frames found in sequence.")

                self.frame_count = len(self.image_filenames)
                ref_cam = first_valid_cam_path.parent.name
                logger.info(f"Found {self.frame_count} frames using '{ref_cam}' as reference.")
                logger.info(f"Processing valid cameras found: {self.active_camera_ids}")
            except Exception as e:
                logger.error(f"Error listing files in {first_valid_cam_path}: {e}", exc_info=True)
                raise RuntimeError("Failed to list image files.") from e
        else:
            raise RuntimeError("Cannot determine frame sequence: No valid camera directories found.")

        if not self.image_filenames:
            raise RuntimeError(f"No image files found in the reference camera directory: {first_valid_cam_path}")

    def __len__(self) -> int:
        """Returns the number of frame *indices* to iterate through."""
        return self.frame_count

    def __iter__(self) -> Iterator[Tuple[int, str, str, Optional[np.ndarray]]]:
        """
        Iterates through frames, yielding data for each *active* camera for a given frame index.

        Yields:
            tuple: (frame_index, camera_id, image_filename, frame_bgr)
                   frame_bgr is None if the image fails to load for that camera.
        """
        num_files_to_process = len(self.image_filenames)
        logger.info(f"Starting data iteration for {num_files_to_process} frame indices...")

        for frame_idx, filename in enumerate(self.image_filenames):
            for cam_id in self.active_camera_ids:  # Iterate only over cameras found
                cam_dir_path = self.camera_image_dirs[cam_id]
                image_path = cam_dir_path / filename
                frame_bgr: Optional[np.ndarray] = None
                if image_path.is_file():
                    try:
                        # Read with imdecode to handle potential path issues on some OS
                        img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
                        frame_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                        if frame_bgr is None or frame_bgr.size == 0:
                            logger.warning(
                                f"[{cam_id}] Failed to decode image (empty): {image_path} - Frame {frame_idx}")
                            frame_bgr = None
                    except Exception as e:
                        logger.error(f"[{cam_id}] Error reading image file {image_path} - Frame {frame_idx}: {e}")
                        frame_bgr = None

                yield frame_idx, cam_id, filename, frame_bgr

            if frame_idx + 1 >= num_files_to_process:
                logger.info(f"Finished data iteration after {frame_idx + 1} frame indices.")
                break
