"""
Core pipeline for detection model performance analysis.
"""
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import cv2
import numpy as np
import torchvision
from tqdm import tqdm

from src.utils.mlflow_utils import download_best_model_checkpoint, setup_mlflow_experiment
from src.components.training.runner import get_fasterrcnn_model, get_transform
from src.components.data.training_dataset import MTMMCDetectionDataset

logger = logging.getLogger(__name__)


def _save_failure_visualization_for_person(
    image: np.ndarray,
    missed_box: torch.Tensor,
    missed_id: int,
    output_dir: Path,
    base_filename: str,
):
    """
    Draws a single missed ground truth box on an image and saves it.

    Args:
        image: The original image in RGB format (numpy array).
        missed_box: The specific ground truth bounding box that was not detected (Tensor).
        missed_id: The ID of the missed person.
        output_dir: The directory to save the image.
        base_filename: The base name for the output file (e.g., from the original frame).
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        img_to_draw = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        purple_color_bgr = (128, 0, 128)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        x1, y1, x2, y2 = map(int, missed_box.squeeze())
        cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), purple_color_bgr, thickness)
        cv2.putText(img_to_draw, f"Missed ID: {missed_id}", (x1, y1 - 10), font, font_scale, purple_color_bgr, thickness)

        new_filename = f"{Path(base_filename).stem}_id_{missed_id}_failure.png"
        output_path = output_dir / new_filename
        cv2.imwrite(str(output_path), img_to_draw)

    except Exception as e:
        logger.error(f"Failed to save failure visualization for {base_filename}: {e}", exc_info=True)


def _get_detection_status_by_gt_id(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_ids: torch.Tensor,
    iou_threshold: float
) -> Tuple[set, set]:
    """
    Partitions ground truth IDs from a frame into 'missed' and 'detected' sets.

    Args:
        pred_boxes: Predicted bounding boxes.
        gt_boxes: Ground truth bounding boxes.
        gt_ids: Ground truth object IDs.
        iou_threshold: IoU threshold for a match.

    Returns:
        A tuple containing (missed_ids_set, detected_ids_set).
    """
    if gt_boxes.shape[0] == 0:
        return set(), set()

    all_gt_ids_in_frame = set(gt_ids.tolist())
    detected_ids = set()

    if pred_boxes.shape[0] > 0:
        iou_matrix = torchvision.ops.box_iou(gt_boxes, pred_boxes)
        if iou_matrix.shape[1] > 0:
            max_ious, _ = torch.max(iou_matrix, dim=1)
            matched_mask = max_ious >= iou_threshold
            detected_ids.update(gt_ids[matched_mask].tolist())

    missed_ids = all_gt_ids_in_frame - detected_ids
    return missed_ids, detected_ids


def run_analysis(config: Dict[str, Any], device: torch.device, project_root: Path):
    """
    Runs the full detection analysis pipeline.
    """
    logger.info("--- Starting Detection Analysis Pipeline ---")
    model_path = None

    # --- Determine model path: local file OR MLflow download ---
    local_path_str = config.get("local_model_path")
    if local_path_str:
        logger.info(f"Prioritizing local model path from config: '{local_path_str}'")
        candidate_path = project_root / local_path_str
        if candidate_path.is_file():
            model_path = candidate_path
        else:
            logger.critical(f"Local model file not found at: {candidate_path}")
            return
    else:
        logger.info("Local model path not provided. Attempting MLflow download...")
        # 1. Setup MLflow and download the model
        setup_mlflow_experiment(config, "Dummy")
        run_id = config.get("mlflow_run_id")
        if not run_id:
            logger.critical("Config error: 'local_model_path' or 'mlflow_run_id' must be provided.")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded_path = download_best_model_checkpoint(run_id, Path(tmpdir))
            if not downloaded_path:
                logger.critical(f"Could not download a model for run_id {run_id}. Aborting.")
                return
            # The model is loaded within the temp context, so we do it here
            model_path = downloaded_path

    if not model_path:
        logger.critical("Could not determine a valid model path. Aborting.")
        return

    # 2. Load Model
    logger.info("Loading model architecture...")
    model = get_fasterrcnn_model(config)
    logger.info(f"Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # Adjust for potential key mismatch if saved differently
    state_dict_key = 'model_state_dict'
    if state_dict_key not in checkpoint:
        logger.warning(f"'{state_dict_key}' not in checkpoint. Assuming checkpoint is the state_dict itself.")
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint[state_dict_key])
    model.to(device)
    model.eval()
    logger.info("Model loaded and in evaluation mode.")

    # 3. Load Dataset
    logger.info("Loading validation dataset...")
    # We load the 'val' split of the dataset as defined in the config
    val_transforms = get_transform(train=False, config=config)
    full_val_dataset = MTMMCDetectionDataset(
        config=config,
        mode='val',
        transforms=val_transforms
    )
    if len(full_val_dataset) == 0:
        logger.error("Validation dataset is empty. Cannot perform analysis.")
        return
    logger.info(f"Loaded {len(full_val_dataset)} samples in the validation set.")

    # 4. Run Analysis Per Camera
    logger.info("Discovering all available cameras from the dataset...")
    all_scene_camera_pairs = set()
    for i in range(len(full_val_dataset)):
        info = full_val_dataset.get_sample_info(i)
        if info:
            all_scene_camera_pairs.add((info['scene_id'], info['camera_id']))

    sorted_pairs = sorted(list(all_scene_camera_pairs))
    logger.info(f"Found {len(sorted_pairs)} unique scene/camera pairs. "
                f"Ignoring 'scenes_to_include' in config and running on all.")

    for scene_id, camera_id in sorted_pairs:
        _analyze_camera(
            scene_id=scene_id,
            camera_id=camera_id,
            model=model,
            dataset=full_val_dataset,
            device=device,
            config=config,
        )


def _analyze_camera(
    scene_id: str,
    camera_id: str,
    model: torch.nn.Module,
    dataset: MTMMCDetectionDataset,
    device: torch.device,
    config: Dict[str, Any],
):
    """
    Analyzes all frames for a single camera to find and save initial detection
    failures for each unique person ID.
    """
    logger.info(f"--- Analyzing Scene: {scene_id}, Camera: {camera_id} for unique detection failures ---")

    analysis_params = config["analysis"]
    iou_threshold = analysis_params["iou_threshold"]

    camera_indices = []
    for i in range(len(dataset)):
        info = dataset.get_sample_info(i)
        if info and info['scene_id'] == scene_id and info['camera_id'] == camera_id:
            camera_indices.append(i)

    if not camera_indices:
        logger.warning(f"No validation data found for {scene_id}/{camera_id}. Skipping.")
        return

    total_frames = len(camera_indices)
    logger.info(f"Found {total_frames} frames. Analyzing every frame to find unique failures.")

    failures_saved = 0
    output_dir = Path(config["analysis"]["output_dir"]) / scene_id / camera_id / "failures"
    missed_ids_being_tracked = set()

    for i in tqdm(range(total_frames), desc=f"Processing {camera_id}"):
        dataset_idx = camera_indices[i]
        image_tensor, target = dataset[dataset_idx]
        original_image_path = dataset.get_image_path(dataset_idx)

        if not original_image_path:
            logger.warning(f"Could not retrieve image path for dataset index {dataset_idx}. Skipping.")
            continue

        with torch.no_grad():
            prediction = model([image_tensor.to(device)])[0]

        pred_cpu = {k: v.cpu() for k, v in prediction.items()}
        target_cpu = {k: v.cpu() for k, v in target.items()}

        gt_ids = target_cpu.get("labels")
        if gt_ids is None:
            logger.warning(f"Frame {original_image_path.name} has no 'labels' (GT IDs). Skipping.")
            continue

        missed_this_frame, detected_this_frame = _get_detection_status_by_gt_id(
            pred_boxes=pred_cpu["boxes"],
            gt_boxes=target_cpu["boxes"],
            gt_ids=gt_ids,
            iou_threshold=iou_threshold,
        )

        newly_missed_ids = missed_this_frame - missed_ids_being_tracked
        if newly_missed_ids:
            original_image = cv2.imread(str(original_image_path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            for missed_id in newly_missed_ids:
                missed_idx_mask = (gt_ids == missed_id)
                if torch.any(missed_idx_mask):
                    box_to_save = target_cpu["boxes"][missed_idx_mask]
                    _save_failure_visualization_for_person(
                        image=original_image,
                        missed_box=box_to_save,
                        missed_id=missed_id,
                        output_dir=output_dir,
                        base_filename=original_image_path.name,
                    )
                    failures_saved += 1

        all_gt_ids_in_frame = set(gt_ids.tolist())
        missed_ids_being_tracked.update(newly_missed_ids)
        missed_ids_being_tracked -= detected_this_frame
        missed_ids_being_tracked &= all_gt_ids_in_frame

    if failures_saved > 0:
        logger.info(f"Finished processing. Saved {failures_saved} unique detection failure images.")
    else:
        logger.info("Finished processing. No new detection failures found for this camera.") 