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
from src.training.runner import get_fasterrcnn_model, get_transform
from src.data.training_dataset import MTMMCDetectionDataset

logger = logging.getLogger(__name__)


def _save_failure_visualization(
    image: np.ndarray,
    missed_boxes: torch.Tensor,
    output_dir: Path,
    base_filename: str,
):
    """
    Draws only the missed ground truth boxes on an image and saves it.

    Args:
        image: The original image in RGB format (numpy array).
        missed_boxes: Ground truth bounding boxes that were not detected (Tensor).
        output_dir: The directory to save the image.
        base_filename: The original name of the image file.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        img_to_draw = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Define color and style for missed detections
        purple_color_bgr = (128, 0, 128)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Draw missed Ground Truth boxes
        for box in missed_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), purple_color_bgr, thickness)
            cv2.putText(img_to_draw, "Missed GT", (x1, y1 - 10), font, font_scale, purple_color_bgr, thickness)

        # Save the image
        new_filename = f"{Path(base_filename).stem}_failure.png"
        output_path = output_dir / new_filename
        cv2.imwrite(str(output_path), img_to_draw)

    except Exception as e:
        logger.error(f"Failed to save failure visualization for {base_filename}: {e}", exc_info=True)


def _find_missed_gt_boxes(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    """
    Finds ground truth boxes that have no matching prediction box above the IoU threshold.

    Args:
        pred_boxes: Predicted bounding boxes (Tensor, shape [N, 4]).
        gt_boxes: Ground truth bounding boxes (Tensor, shape [M, 4]).
        iou_threshold: The IoU threshold for a match.

    Returns:
        A tensor of ground truth boxes that were missed.
    """
    if gt_boxes.shape[0] == 0:
        return torch.empty((0, 4), dtype=gt_boxes.dtype)

    if pred_boxes.shape[0] == 0:
        return gt_boxes  # All GT boxes are missed if there are no predictions

    # Calculate IoU matrix
    iou_matrix = torchvision.ops.box_iou(gt_boxes, pred_boxes)

    # If there are no predictions, all GT boxes are missed
    if iou_matrix.shape[1] == 0:
        return gt_boxes

    # Find the max IoU for each ground truth box
    max_ious, _ = torch.max(iou_matrix, dim=1)

    # A GT box is missed if its max IoU with any prediction is below the threshold
    missed_mask = max_ious < iou_threshold
    return gt_boxes[missed_mask]


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
    checkpoint = torch.load(model_path, map_location=device)
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
    scenes_to_analyze = config.get("data", {}).get("scenes_to_include", [])

    for scene_info in scenes_to_analyze:
        scene_id = scene_info["scene_id"]
        for camera_id in scene_info["camera_ids"]:
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
    """Analyzes all frames for a single camera to find and save detection failures."""
    logger.info(f"--- Analyzing Scene: {scene_id}, Camera: {camera_id} for detection failures ---")

    analysis_params = config["analysis"]
    iou_threshold = analysis_params["iou_threshold"]

    # Filter dataset for the current camera
    camera_indices = []
    for i in range(len(dataset)):
        info = dataset.get_sample_info(i)
        if info and info['scene_id'] == scene_id and info['camera_id'] == camera_id:
            camera_indices.append(i)

    if not camera_indices:
        logger.warning(f"No validation data found for {scene_id}/{camera_id}. Skipping.")
        return

    # Per user request, we analyze every frame, ignoring the sample percentage.
    total_frames = len(camera_indices)
    stride = 1
    logger.info(f"Found {total_frames} frames. Analyzing every frame (stride={stride}) to find all failures.")

    failures_found = 0
    output_dir = Path(config["analysis"]["output_dir"]) / scene_id / camera_id / "failures"

    for i in tqdm(range(0, total_frames, stride), desc=f"Processing {camera_id}"):
        dataset_idx = camera_indices[i]

        # Get data for model input
        image_tensor, target = dataset[dataset_idx]

        # Get data for visualization
        original_image_path = dataset.get_image_path(dataset_idx)
        if not original_image_path:
            logger.warning(f"Could not retrieve image path for dataset index {dataset_idx}. Skipping frame.")
            continue

        original_image = cv2.imread(str(original_image_path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Perform inference
        with torch.no_grad():
            prediction = model([image_tensor.to(device)])[0]

        # Move prediction and target to CPU
        pred_cpu = {k: v.cpu() for k, v in prediction.items()}
        target_cpu = {k: v.cpu() for k, v in target.items()}

        # Find missed ground truth boxes
        missed_gt_boxes = _find_missed_gt_boxes(
            pred_boxes=pred_cpu["boxes"],
            gt_boxes=target_cpu["boxes"],
            iou_threshold=iou_threshold
        )

        if len(missed_gt_boxes) > 0:
            failures_found += 1
            _save_failure_visualization(
                image=original_image,
                missed_boxes=missed_gt_boxes,
                output_dir=output_dir,
                base_filename=original_image_path.name,
            )

    if failures_found > 0:
        logger.info(f"Finished processing. Found and saved {failures_found} frames with detection failures.")
    else:
        logger.info("Finished processing. No detection failures found for this camera.") 