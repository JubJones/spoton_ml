import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
import pandas as pd
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from .loader import sorted_alphanumeric  # Reuse sorting utility

logger = logging.getLogger(__name__)

# Type alias for raw annotation tuple
AnnotationRecord = Tuple[
    str, str, int, int, float, float, float, float
]  # scene, cam, frame, obj, x, y, w, h


def discover_data_assets(
    base_path: Path,
    selection_strategy: str,
    scenes_to_analyze: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Scans the dataset structure based on configuration to find scenes, cameras,
    RGB folders, GT files, and estimates frame counts.

    Returns:
        A dictionary structured as:
        { scene_id: { camera_id: { 'rgb_path': Path, 'gt_path': Path, 'frame_count_jpg': int, 'frame_count_gt': int } } }
        Includes only cameras with both rgb path and gt path found.
    """
    logger.info(f"Starting data discovery from base path: {base_path}")
    train_base = base_path / "train"
    if not train_base.is_dir():
        raise FileNotFoundError(
            f"Train directory not found at expected path: {train_base}"
        )

    discovered_assets: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    potential_scenes = sorted(
        [p for p in train_base.iterdir() if p.is_dir() and p.name.startswith("s")]
    )
    logger.info(f"Found {len(potential_scenes)} potential scene directories.")

    scenes_in_scope: List[str] = []
    allowed_cameras_per_scene: Dict[str, Optional[List[str]]] = {}  # None means 'all'

    if selection_strategy == "all":
        scenes_in_scope = [s.name for s in potential_scenes]
        for scene_id in scenes_in_scope:
            allowed_cameras_per_scene[scene_id] = None  # All cameras
        logger.info(
            "Selection strategy: 'all'. Analyzing all found scenes and cameras."
        )
    elif selection_strategy == "list":
        if not scenes_to_analyze:
            raise ValueError(
                "Selection strategy is 'list' but 'scenes_to_analyze' is empty in config."
            )
        scenes_in_scope = [s_info["scene_id"] for s_info in scenes_to_analyze]
        for s_info in scenes_to_analyze:
            scene_id = s_info["scene_id"]
            # Handle 'all' keyword for cameras within a listed scene
            cam_ids = s_info.get("camera_ids")
            if isinstance(cam_ids, str) and cam_ids.lower() == "all":
                allowed_cameras_per_scene[scene_id] = None
            elif isinstance(cam_ids, list):
                allowed_cameras_per_scene[scene_id] = cam_ids
            else:
                allowed_cameras_per_scene[scene_id] = (
                    None  # Default to all if invalid format
                )
                logger.warning(
                    f"Invalid 'camera_ids' format for scene {scene_id}. Defaulting to all cameras."
                )

        logger.info(f"Selection strategy: 'list'. Analyzing scenes: {scenes_in_scope}")
    else:
        raise ValueError(f"Unknown selection_strategy: {selection_strategy}")

    # --- Iterate through scenes and cameras ---
    pbar_scenes = tqdm(potential_scenes, desc="Discovering Scenes")
    for scene_path in pbar_scenes:
        scene_id = scene_path.name
        if scene_id not in scenes_in_scope:
            continue
        pbar_scenes.set_postfix_str(f"Scene {scene_id}")

        potential_cameras = sorted(
            [p for p in scene_path.iterdir() if p.is_dir() and p.name.startswith("c")]
        )
        allowed_cameras = allowed_cameras_per_scene.get(scene_id)

        for cam_path in potential_cameras:
            cam_id = cam_path.name
            # Check if this camera is allowed by the config
            if allowed_cameras is not None and cam_id not in allowed_cameras:
                continue

            rgb_path = cam_path / "rgb"
            gt_path = cam_path / "gt" / "gt.txt"

            has_rgb = rgb_path.is_dir()
            has_gt = gt_path.is_file()

            if not has_rgb or not has_gt:
                continue  # Skip camera if essential components missing

            # --- Estimate Frame Counts ---
            frame_count_jpg = 0
            frame_count_gt = -1  # -1 indicates GT not processed or empty

            # Count JPGs
            try:
                jpg_files = [f for f in rgb_path.glob("*.jpg") if f.is_file()]
                frame_count_jpg = len(jpg_files)
            except Exception as e:
                logger.error(
                    f"[{scene_id}/{cam_id}] Error counting JPG files in {rgb_path}: {e}"
                )

            # Estimate max frame index from GT
            try:
                max_frame_idx_gt = -1
                with open(gt_path, "r") as f_gt:
                    for line in f_gt:
                        parts = line.strip().split(",")
                        if len(parts) >= 1:
                            try:
                                max_frame_idx_gt = max(max_frame_idx_gt, int(parts[0]))
                            except ValueError:
                                pass  # Ignore non-integer frame indices
                # GT index is 0-based, so count is max_index + 1
                frame_count_gt = max_frame_idx_gt + 1 if max_frame_idx_gt >= 0 else 0
            except Exception as e:
                logger.error(
                    f"[{scene_id}/{cam_id}] Error reading GT file {gt_path} for max frame index: {e}"
                )

            # Store results for this valid camera
            discovered_assets[scene_id][cam_id] = {
                "rgb_path": rgb_path,
                "gt_path": gt_path,
                "frame_count_jpg": frame_count_jpg,
                "frame_count_gt": frame_count_gt,
            }
            logger.debug(
                f"[{scene_id}/{cam_id}] Assets discovered. JPGs: {frame_count_jpg}, GT Max Idx: {max_frame_idx_gt}"
            )

    logger.info(
        f"Data discovery complete. Found assets for {len(discovered_assets)} scenes "
        f"and {sum(len(cams) for cams in discovered_assets.values())} cameras."
    )
    return dict(discovered_assets)


def load_all_ground_truth(
    discovered_assets: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], int]]:
    """
    Loads ground truth data from all discovered gt.txt files into a Pandas DataFrame.

    Args:
        discovered_assets: The dictionary returned by discover_data_assets.

    Returns:
        A tuple containing:
        - pd.DataFrame: Columns: ['scene_id', 'camera_id', 'frame_idx', 'obj_id', 'x', 'y', 'w', 'h']
        - dict: Mapping from (scene_id, camera_id) to the max frame index found in that camera's GT.
    """
    logger.info("Loading all ground truth annotations...")
    all_annotations: List[AnnotationRecord] = []
    max_frame_indices: Dict[Tuple[str, str], int] = {}
    total_annotations_loaded = 0
    total_cameras = sum(len(cams) for cams in discovered_assets.values())

    pbar_load = tqdm(total=total_cameras, desc="Loading GT")
    for scene_id, cameras in discovered_assets.items():
        for cam_id, assets in cameras.items():
            pbar_load.set_postfix_str(f"{scene_id}/{cam_id}")
            gt_path = assets["gt_path"]
            cam_max_frame = -1
            try:
                with open(gt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) < 6:
                            continue  # frame, id, x, y, w, h

                        try:
                            frame_idx = int(parts[0])
                            obj_id = int(parts[1])
                            bb_left = float(parts[2])
                            bb_top = float(parts[3])
                            bb_width = float(parts[4])
                            bb_height = float(parts[5])

                            # Store x,y,w,h (original format for analysis)
                            all_annotations.append(
                                (
                                    scene_id,
                                    cam_id,
                                    frame_idx,
                                    obj_id,
                                    bb_left,
                                    bb_top,
                                    bb_width,
                                    bb_height,
                                )
                            )
                            cam_max_frame = max(cam_max_frame, frame_idx)
                            total_annotations_loaded += 1
                        except ValueError:
                            logger.debug(
                                f"Skipping GT line due to parsing error in {scene_id}/{cam_id}: {line.strip()}"
                            )
                            continue
                max_frame_indices[(scene_id, cam_id)] = cam_max_frame
            except Exception as e:
                logger.error(f"Error reading GT file {gt_path}: {e}", exc_info=True)
            pbar_load.update(1)

    pbar_load.close()
    logger.info(
        f"Finished loading ground truth. Total annotations loaded: {total_annotations_loaded}"
    )

    if not all_annotations:
        logger.warning("No annotations were loaded. Returning empty DataFrame.")
        return (
            pd.DataFrame(
                columns=[
                    "scene_id",
                    "camera_id",
                    "frame_idx",
                    "obj_id",
                    "x",
                    "y",
                    "w",
                    "h",
                ]
            ),
            max_frame_indices,
        )

    # Create DataFrame
    df_gt = pd.DataFrame(
        all_annotations,
        columns=["scene_id", "camera_id", "frame_idx", "obj_id", "x", "y", "w", "h"],
    )
    logger.info(f"Created GT DataFrame with shape: {df_gt.shape}")

    return df_gt, max_frame_indices


def load_sample_image_data(
    discovered_assets: Dict[str, Dict[str, Dict[str, Any]]], sample_size: int
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Loads basic info (path, dimensions) for a sample of images per camera."""
    logger.info(f"Loading sample image data (up to {sample_size} images per camera)...")
    sample_data = defaultdict(list)
    total_cameras = sum(len(cams) for cams in discovered_assets.values())

    pbar_sample = tqdm(total=total_cameras, desc="Sampling Images")
    for scene_id, cameras in discovered_assets.items():
        for cam_id, assets in cameras.items():
            pbar_sample.set_postfix_str(f"{scene_id}/{cam_id}")
            rgb_path = assets["rgb_path"]
            image_files = []
            try:
                image_files = sorted_alphanumeric(
                    [f.name for f in rgb_path.glob("*.jpg") if f.is_file()]
                )
            except Exception as e:
                logger.error(
                    f"[{scene_id}/{cam_id}] Failed to list images for sampling: {e}"
                )
                pbar_sample.update(1)
                continue

            num_to_sample = min(sample_size, len(image_files))
            sampled_files = image_files[:num_to_sample]

            for filename in sampled_files:
                img_path = rgb_path / filename
                try:
                    # Use imdecode for robust path handling + shape check
                    img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
                    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                    if img is not None:
                        h, w = img.shape[:2]
                        dtype = str(img.dtype)
                        sample_data[(scene_id, cam_id)].append(
                            {
                                "path": str(img_path),
                                "width": w,
                                "height": h,
                                "dtype": dtype,
                            }
                        )
                    else:
                        logger.warning(
                            f"[{scene_id}/{cam_id}] Failed to decode sample image: {filename}"
                        )
                except Exception as e:
                    logger.error(
                        f"[{scene_id}/{cam_id}] Error loading sample image {filename}: {e}"
                    )
            pbar_sample.update(1)
    pbar_sample.close()
    logger.info(f"Sample image data collected for {len(sample_data)} camera streams.")
    return dict(sample_data)
