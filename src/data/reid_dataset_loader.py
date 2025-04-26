import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ReidCropInfo:
    """Holds information needed for a single Re-ID crop."""
    instance_id: int  # Global person ID
    camera_id: str  # Camera identifier (e.g., 'c01')
    frame_path: str  # Full path to the source frame image
    bbox_xyxy: List[float]  # Bounding box [xmin, ymin, xmax, ymax] in the original frame
    frame_index: Optional[int] = None  # Frame index within the sequence
    image_id: Optional[int] = None  # Original image ID from JSON


class ReidDatasetLoader:
    """Loads and prepares ground truth crop data for Re-ID evaluation from split MTMMC JSON files."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the loader based on the configuration.
        """
        self.config = config
        data_config = config.get('data', {})
        env_config = config.get('environment', {})
        self.seed = env_config.get('seed', 42)
        random.seed(self.seed)

        self.base_path = Path(data_config.get("base_path", ""))
        if not self.base_path.is_dir():
            raise FileNotFoundError(f"Data base path not found: {self.base_path}")

        self.split_type = data_config.get("split_type", "val")  # 'train' or 'val'

        self.selected_env_key = data_config.get("selected_environment")
        if not self.selected_env_key:
            raise ValueError("Missing 'data.selected_environment' in configuration.")
        env_specific_config = data_config.get(self.selected_env_key, {})

        self.scene_annotation_filename = env_specific_config.get("scene_annotation_file")
        if not self.scene_annotation_filename:
            raise ValueError(f"Missing 'scene_annotation_file' for '{self.selected_env_key}' in configuration.")

        # Construct path to the specific scene JSON file
        self.json_annotation_path = self.base_path / "split" / self.split_type / self.scene_annotation_filename
        if not self.json_annotation_path.is_file():
            raise FileNotFoundError(f"Scene JSON annotation file not found: {self.json_annotation_path}")

        self.filter_camera_ids = set(env_specific_config.get("camera_ids", []))
        self.frame_sample_rate = env_specific_config.get("frame_sample_rate", 1)
        self.max_crops_per_id_per_cam = env_specific_config.get("max_crops_per_id_per_cam", -1)

        logger.info(f"Initializing ReidDatasetLoader:")
        logger.info(f"  Split Type: {self.split_type}")
        logger.info(f"  Scene JSON Path: {self.json_annotation_path}")
        logger.info(f"  Filter Cameras: {self.filter_camera_ids if self.filter_camera_ids else 'All Cameras in JSON'}")
        logger.info(f"  Frame Sample Rate: {self.frame_sample_rate}")
        logger.info(f"  Max Crops/ID/Cam: {self.max_crops_per_id_per_cam}")

        self._all_crops_data: List[ReidCropInfo] = []
        self._loaded = False

    def _load_and_prepare_data(self):
        """Loads JSON, filters annotations, and prepares the crop list."""
        logger.info(f"Loading annotations from {self.json_annotation_path}...")
        try:
            with open(self.json_annotation_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load/parse JSON: {e}", exc_info=True); raise RuntimeError(
                "JSON loading failed") from e

        annotations = data.get('annotations', [])
        images_info = {img['id']: img for img in data.get('images', [])}

        logger.info(f"Loaded {len(annotations)} annotations and {len(images_info)} image entries from scene JSON.")
        if not annotations or not images_info: logger.error(
            "Scene JSON missing 'annotations'/'images' or empty."); return

        valid_crops: List[ReidCropInfo] = []
        logger.info("Filtering annotations and preparing crop list...")
        for ann in tqdm(annotations, desc="Processing Annotations"):
            image_id = ann.get('image_id')
            instance_id = ann.get('instance_id')
            frame_id_rel = ann.get('frame_id')
            bbox_xywh = ann.get('bbox')
            if image_id is None or instance_id is None or frame_id_rel is None or not bbox_xywh or len(
                bbox_xywh) != 4: continue

            image_data = images_info.get(image_id)
            if not image_data: continue

            # --- Determine Camera ID (using heuristics - might need adjustment based on exact JSON format) ---
            camera_id_str = None
            raw_cam_id = image_data.get('camera_id')
            video_path_parts = image_data.get('video', '').split('/')
            file_path_parts = image_data.get('file_name', '').split('/')

            # Prioritize parts containing 'cXX' format
            if len(video_path_parts) >= 3 and video_path_parts[-1].startswith('c'):
                camera_id_str = video_path_parts[-1]
            elif len(file_path_parts) >= 4 and file_path_parts[-3].startswith('c'):
                camera_id_str = file_path_parts[-3]
            elif isinstance(raw_cam_id, str) and raw_cam_id.startswith('c'):
                camera_id_str = raw_cam_id
            elif isinstance(raw_cam_id, int):  # Attempt mapping if int
                potential_cam_str = f"c{raw_cam_id + 1:02d}"  # Heuristic mapping
                # Check against *all* cameras in this scene if filter is empty, else check against filter
                scene_cameras = self.filter_camera_ids if self.filter_camera_ids else set(
                    f"c{i + 1:02d}" for i in range(20))  # Assume max 20 cams if no filter
                if potential_cam_str in scene_cameras:
                    camera_id_str = potential_cam_str
                else:
                    camera_id_str = f"cam_int_{raw_cam_id}"
            if not camera_id_str: continue

            frame_index = image_data.get('frame_id', frame_id_rel)

            # --- Filtering ---
            # 1. Filter by Camera ID specified in the config for the selected environment
            if self.filter_camera_ids and camera_id_str not in self.filter_camera_ids:
                continue
            # 2. Filter by Frame Sample Rate
            if self.frame_sample_rate > 1 and frame_index % self.frame_sample_rate != 0:
                continue

            # --- Prepare Crop Info ---
            relative_image_path = image_data.get('file_name')
            if not relative_image_path:
                video_prefix = image_data.get('video')
                if video_prefix:
                    relative_image_path = f"{video_prefix}/rgb/{frame_index:06d}.jpg"
                else:
                    continue

            # Construct full path relative to the dataset base_path
            full_image_path = self.base_path / relative_image_path
            x, y, w, h = bbox_xywh
            if w <= 0 or h <= 0: continue
            bbox_xyxy = [x, y, x + w, y + h]

            crop_info = ReidCropInfo(
                instance_id=instance_id,
                camera_id=camera_id_str,
                frame_path=str(full_image_path),
                bbox_xyxy=bbox_xyxy,
                frame_index=frame_index,
                image_id=image_id
            )
            valid_crops.append(crop_info)

        logger.info(f"Found {len(valid_crops)} potentially valid crops after initial filtering.")

        # Apply max_crops_per_id_per_cam sampling
        if self.max_crops_per_id_per_cam > 0:
            sampled_crops: List[ReidCropInfo] = [];
            crops_by_id_cam = defaultdict(list)
            for crop in valid_crops: crops_by_id_cam[(crop.instance_id, crop.camera_id)].append(crop)
            logger.info(f"Applying max_crops_per_id_per_cam sampling ({self.max_crops_per_id_per_cam})...")
            for key, crops_list in tqdm(crops_by_id_cam.items(), desc="Sampling Crops"):
                if len(crops_list) > self.max_crops_per_id_per_cam:
                    sampled_crops.extend(random.sample(crops_list, self.max_crops_per_id_per_cam))
                else:
                    sampled_crops.extend(crops_list)
            logger.info(f"Retained {len(sampled_crops)} crops after sampling.")
            self._all_crops_data = sampled_crops
        else:
            self._all_crops_data = valid_crops
        self._loaded = True

    def get_data(self) -> List[ReidCropInfo]:
        """Returns the list of prepared Re-ID crop information."""
        if not self._loaded: self._load_and_prepare_data()
        return self._all_crops_data

    def __len__(self) -> int:
        """Returns the number of prepared crops."""
        if not self._loaded: self.get_data()
        return len(self._all_crops_data)
