"""
This file contains the dataset class for the MTMMC dataset.
It is used to load the data for the training and validation sets.
"""

import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable, Set

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from src.data.loader import sorted_alphanumeric

logger = logging.getLogger(__name__)

# Type alias for annotations loaded from gt.txt for a single frame
# List[Tuple[obj_id, cx, cy, w, h]]
FrameAnnotations = List[Tuple[int, float, float, float, float]]
# Mapping from filename to its annotations
AnnotationMap = Dict[str, FrameAnnotations]


class MTMMCDetectionDataset(Dataset):
    """
    PyTorch Dataset for loading MTMMC image frames and ground truth bounding boxes
    from gt.txt files for object detection training.
    Handles multiple scenes/cameras and data subsetting.
    Uses tv_tensors for compatibility with torchvision.transforms.v2.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        mode: str = "train",
        transforms: Optional[Callable] = None,
    ):
        """
        Initializes the dataset.
        """
        super().__init__()
        assert mode in ["train", "val"], f"Mode must be 'train' or 'val', got {mode}"
        self.config = config
        self.mode = mode
        self.transforms = transforms

        data_config = config["data"]
        self.base_path = Path(data_config["base_path"])
        self.scenes_to_include = data_config.get("scenes_to_include", [])
        if not self.scenes_to_include:
            raise ValueError("Config 'data.scenes_to_include' cannot be empty.")

        self.val_split_ratio = data_config.get("val_split_ratio", 0.2)
        # Data subsetting
        self.use_data_subset = data_config.get("use_data_subset", False)
        self.data_subset_fraction = data_config.get("data_subset_fraction", 0.1)

        # Stores tuples of (image_path, annotation_list)
        self.data_samples: List[Tuple[Path, FrameAnnotations]] = []
        self._load_data_samples()

        # --- Split Data ---
        self.samples_split: List[Tuple[Path, FrameAnnotations]] = []
        self._prepare_split()

        # --- Class Mapping ---
        # FasterRCNN needs background=0, person=1, ...
        self.class_name = "person"
        self.class_id = 1  # Assign ID 1 to person (background is implicitly 0)

        logger.info(
            f"MTMMC Dataset '{self.mode}' split initialized. Scenes: {[s['scene_id'] for s in self.scenes_to_include]}. "
            f"Subset: {self.use_data_subset} ({self.data_subset_fraction * 100}%). "
            f"Number of samples: {len(self.samples_split)}"
        )
        if len(self.samples_split) == 0:
            logger.warning(
                f"'{self.mode}' split has 0 samples. Check config, data paths, and subset fraction."
            )

    def _load_annotations_for_camera(
        self, scene_path: Path, cam_id: str
    ) -> Tuple[AnnotationMap, List[str]]:
        """Loads annotations from gt.txt for a specific camera within a scene."""
        annotations: AnnotationMap = {}
        image_filenames: List[str] = []

        rgb_dir = scene_path / cam_id / "rgb"
        if not rgb_dir.is_dir():
            logger.warning(
                f"RGB directory not found: {rgb_dir}. Skipping camera {cam_id}."
            )
            return {}, []

        try:
            filenames_unsorted = [
                f.name
                for f in rgb_dir.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ]
            image_filenames = sorted_alphanumeric(filenames_unsorted)
        except Exception as e:
            logger.error(f"Error listing files in {rgb_dir}: {e}", exc_info=True)
            return {}, []

        if not image_filenames:
            logger.warning(
                f"No image files found in {rgb_dir}. Skipping camera {cam_id}."
            )
            return {}, []

        # Load GT
        gt_file_path = scene_path / cam_id / "gt" / "gt.txt"
        if not gt_file_path.is_file():
            logger.warning(
                f"Ground truth file not found: {gt_file_path}. Camera {cam_id} will have no GT boxes."
            )
            for fname in image_filenames:
                annotations[fname] = []
            return annotations, image_filenames

        frame_map = {i: fname for i, fname in enumerate(image_filenames)}

        try:
            with open(gt_file_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue
                    try:
                        frame_idx_txt = int(parts[0])
                        if frame_idx_txt not in frame_map:
                            continue

                        filename = frame_map[frame_idx_txt]
                        obj_id = int(parts[1])
                        bb_left = float(parts[2])
                        bb_top = float(parts[3])
                        bb_width = float(parts[4])
                        bb_height = float(parts[5])

                        if bb_width <= 0 or bb_height <= 0:
                            continue

                        center_x = bb_left + bb_width / 2
                        center_y = bb_top + bb_height / 2

                        if filename not in annotations:
                            annotations[filename] = []
                        annotations[filename].append(
                            (obj_id, center_x, center_y, bb_width, bb_height)
                        )
                    except ValueError:
                        continue

            # Ensure every image file has an entry in annotations, even if empty
            for fname in image_filenames:
                if fname not in annotations:
                    annotations[fname] = []

        except Exception as e:
            logger.error(
                f"Error reading ground truth file {gt_file_path}: {e}", exc_info=True
            )
            # Ensure annotations dict exists but is empty for all files on error
            annotations = {fname: [] for fname in image_filenames}

        return annotations, image_filenames

    def _load_data_samples(self):
        """Loads image paths and annotations from all scenes/cameras specified in config."""
        self.data_samples = []
        unique_image_paths: Set[Path] = set()  # Prevent duplicates if cameras overlap strangely

        for scene_info in self.scenes_to_include:
            scene_id = scene_info["scene_id"]
            camera_ids = scene_info["camera_ids"]
            scene_path = self.base_path / "train" / scene_id
            logger.info(f"Loading data from Scene: {scene_id} (Path: {scene_path})")
            if not scene_path.is_dir():
                logger.warning(f"Scene directory not found: {scene_path}. Skipping.")
                continue

            for cam_id in camera_ids:
                cam_annotations, cam_filenames = self._load_annotations_for_camera(
                    scene_path, cam_id
                )

                if not cam_filenames:
                    continue

                rgb_dir = scene_path / cam_id / "rgb"
                for filename in cam_filenames:
                    img_path = rgb_dir / filename
                    if img_path in unique_image_paths:
                        continue  # Skip if already added

                    annotations = cam_annotations.get(
                        filename, []
                    )  # Get potentially empty list
                    if img_path.is_file():
                        self.data_samples.append((img_path, annotations))
                        unique_image_paths.add(img_path)
                    else:
                        logger.warning(f"Image file expected but not found: {img_path}")

        if not self.data_samples:
            raise RuntimeError(
                f"No data samples could be loaded for the specified scenes/cameras."
            )

        logger.info(
            f"Loaded {len(self.data_samples)} total unique samples (image path, annotations) before subsetting/splitting."
        )

    def _prepare_split(self):
        """Applies subsetting (if enabled) and splits data into train/val."""
        working_samples = self.data_samples

        # Apply subsetting first
        if self.use_data_subset:
            num_total = len(working_samples)
            num_subset = int(num_total * self.data_subset_fraction)
            if num_subset == 0 and num_total > 0:
                num_subset = 1  # Ensure at least one sample if possible
            logger.info(
                f"Applying data subset: Using {num_subset}/{num_total} samples ({self.data_subset_fraction * 100:.1f}%)."
            )
            # Shuffle before taking subset for better representation
            random.shuffle(working_samples)
            working_samples = working_samples[:num_subset]

        # Shuffle before splitting into train/val
        random.shuffle(working_samples)
        num_samples = len(working_samples)
        num_val = int(num_samples * self.val_split_ratio)
        # Ensure num_val is at least 1 if possible and validation is needed
        if num_val == 0 and num_samples > 1 and self.val_split_ratio > 0:
            num_val = 1
        num_train = num_samples - num_val

        if self.mode == "train":
            self.samples_split = working_samples[:num_train]
            logger.info(f"Using {num_train} samples for training split.")
        else:  # mode == "val"
            self.samples_split = working_samples[num_train:]
            logger.info(f"Using {num_val} samples for validation split.")

        if not self.samples_split:
            logger.warning(
                f"No samples available for the '{self.mode}' split after subsetting/splitting."
            )

    def __len__(self) -> int:
        return len(self.samples_split)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        """
        Loads and returns a single sample (image tensor/container, target dict with tv_tensors).
        Handles cases where no ground truth annotations exist for a frame.
        """
        if idx >= len(self.samples_split):
            raise IndexError(
                f"Index {idx} out of bounds for dataset split with length {len(self.samples_split)}"
            )

        img_path, annotations = self.samples_split[idx]
        dummy_image_tensor = None

        try:
            img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
            image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[:2]

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}. Returning dummy data.")
            img_h, img_w = 256, 256
            dummy_image_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            dummy_image_container = T.ToImage()(dummy_image_np)

            dummy_boxes = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format="XYXY",
                canvas_size=(img_h, img_w),
            )
            dummy_target = {
                "boxes": dummy_boxes,
                "labels": torch.zeros(0, dtype=torch.int64),
            }
            # Apply transforms to dummy data to ensure consistent output types
            if self.transforms:
                try:
                    dummy_image_tensor, dummy_target = self.transforms(
                        dummy_image_container, dummy_target
                    )
                except Exception as dummy_transform_err:
                    logger.error(
                        f"Error applying transforms even to dummy data: {dummy_transform_err}"
                    )
                    # Fallback to basic tensor if transforms fail on dummy
                    dummy_image_tensor = T.ToTensor()(dummy_image_np)
            else:
                dummy_image_tensor = T.ToTensor()(dummy_image_np)

            return dummy_image_tensor, dummy_target

        # --- Process Annotations ---
        boxes = []
        labels = []
        for _, cx, cy, w, h in annotations:
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            # Basic validity check (clip happens in BoundingBoxes or transforms)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_id)

        # Use torch.float32 for boxes, torch.int64 for labels
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        # Ensure tensors are correctly shaped even if empty
        if boxes_tensor.nelement() == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        if labels_tensor.nelement() == 0:
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target_boxes = tv_tensors.BoundingBoxes(
            boxes_tensor, format="XYXY", canvas_size=(img_h, img_w)
        )

        target = {"boxes": target_boxes, "labels": labels_tensor}

        # Wrap image in T.Image container BEFORE applying transforms
        image_container = T.ToImage()(image)

        # Apply transforms
        if self.transforms:
            try:
                image_transformed, target_transformed = self.transforms(
                    image_container, target
                )
            except Exception as transform_err:
                # This is where the "No bounding boxes found" error was happening
                logger.error(
                    f"Error applying transforms to {img_path}: {transform_err}",
                    exc_info=True,
                )

                # Return consistent dummy data on transform error
                img_h, img_w = 256, 256
                dummy_image_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                dummy_image_container = T.ToImage()(dummy_image_np)
                dummy_boxes = tv_tensors.BoundingBoxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    format="XYXY",
                    canvas_size=(img_h, img_w),
                )
                dummy_target = {
                    "boxes": dummy_boxes,
                    "labels": torch.zeros(0, dtype=torch.int64),
                }

                # Try applying transforms again to the dummy data
                try:
                    dummy_image_tensor, dummy_target = self.transforms(
                        dummy_image_container, dummy_target
                    )
                except Exception as dummy_transform_err_inner:
                    logger.error(
                        f"Error applying transforms even to dummy data (inner): {dummy_transform_err_inner}"
                    )
                    # Fallback to basic tensor if transforms fail on dummy
                    dummy_image_tensor = T.ToTensor()(dummy_image_np)

                return dummy_image_tensor, dummy_target
        else:
            # Fallback: Basic conversion without transforms if none provided
            image_transformed = T.ToTensor()(image_container)
            target_transformed = target

        # The model expects a Tensor for the image, not T.Image, so convert if needed
        if isinstance(image_transformed, tv_tensors.Image):
            image_output = image_transformed.to(torch.float32)
        elif isinstance(image_transformed, torch.Tensor):
            image_output = image_transformed
        else:
            logger.warning(
                f"Unexpected image type after transforms: {type(image_transformed)}. Attempting conversion."
            )
            image_output = T.ToTensor()(image_transformed)

        return image_output, target_transformed
