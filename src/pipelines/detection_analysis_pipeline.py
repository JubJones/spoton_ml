"""
Core pipeline for detection model performance analysis.
"""
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Dict
import heapq

import torch
import cv2
import numpy as np
from tqdm import tqdm

from src.utils.mlflow_utils import download_best_model_checkpoint, setup_mlflow_experiment
from src.training.runner import get_fasterrcnn_model, get_transform
from src.data.training_dataset import MTMMCDetectionDataset
from src.evaluation.metrics import calculate_frame_detection_score
from src.explainability.visualization import save_analysis_visualization

logger = logging.getLogger(__name__)

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
    analysis_params = config["analysis"]
    num_images_per_category = analysis_params["num_images_per_category"]

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
    """Analyzes all frames for a single camera."""
    logger.info(f"--- Analyzing Scene: {scene_id}, Camera: {camera_id} ---")

    analysis_params = config["analysis"]
    num_to_keep = analysis_params["num_images_per_category"]
    sample_percent = analysis_params["frame_sample_percent"]
    iou_threshold = analysis_params["iou_threshold"]

    # Filter dataset for the current camera
    camera_indices = [
        i for i, info in enumerate(dataset.image_info)
        if info['scene_id'] == scene_id and info['camera_id'] == camera_id
    ]

    if not camera_indices:
        logger.warning(f"No validation data found for {scene_id}/{camera_id}. Skipping.")
        return

    # Calculate stride for sampling
    total_frames = len(camera_indices)
    stride = max(1, int(total_frames * (sample_percent / 100.0)))
    logger.info(f"Found {total_frames} frames. Sampling every {stride} frames.")

    best_frames = [] # Min-heap, stores (score, data)
    worst_frames = [] # Max-heap, stores (-score, data)

    # Use a raw transform to get images for visualization without normalization
    vis_transform = get_transform(train=False, config={}) # Basic ToTensor transform

    for i in tqdm(range(0, total_frames, stride), desc=f"Processing {camera_id}"):
        dataset_idx = camera_indices[i]
        
        # Get data for model input
        image_tensor, target = dataset[dataset_idx]
        
        # Get data for visualization
        original_image_path = dataset.get_image_path(dataset_idx)
        original_image = cv2.imread(str(original_image_path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Perform inference
        with torch.no_grad():
            prediction = model([image_tensor.to(device)])[0]

        # Move prediction and target to CPU for scoring
        pred_cpu = {k: v.cpu() for k, v in prediction.items()}
        target_cpu = {k: v.cpu() for k, v in target.items()}
        
        score = calculate_frame_detection_score(pred_cpu, target_cpu, iou_threshold)

        frame_data = {
            "image": original_image,
            "pred": pred_cpu,
            "gt": target_cpu,
            "score": score,
            "path": original_image_path,
        }

        # Update best frames (min-heap)
        if len(best_frames) < num_to_keep:
            heapq.heappush(best_frames, (score, frame_data))
        elif score > best_frames[0][0]:
            heapq.heappushpop(best_frames, (score, frame_data))

        # Update worst frames (max-heap, storing negative score)
        if len(worst_frames) < num_to_keep:
            heapq.heappush(worst_frames, (-score, frame_data))
        elif -score > worst_frames[0][0]:
            heapq.heappushpop(worst_frames, (-score, frame_data))

    logger.info(f"Finished processing. Saving {len(best_frames)} best and {len(worst_frames)} worst examples.")
    
    # Save visualizations
    output_dir = Path(config["analysis"]["output_dir"])
    
    # Save best frames
    for score, data in sorted(best_frames, key=lambda x: x[0], reverse=True):
        save_analysis_visualization(
            image=data["image"],
            pred_boxes=data["pred"]["boxes"],
            pred_scores=data["pred"]["scores"],
            gt_boxes=data["gt"]["boxes"],
            score=score,
            output_dir=output_dir / scene_id / camera_id / "best",
            base_filename=data["path"].name,
            config=config,
        )

    # Save worst frames (invert score back)
    for neg_score, data in sorted(worst_frames, key=lambda x: x[0], reverse=True):
        save_analysis_visualization(
            image=data["image"],
            pred_boxes=data["pred"]["boxes"],
            pred_scores=data["pred"]["scores"],
            gt_boxes=data["gt"]["boxes"],
            score=-neg_score,
            output_dir=output_dir / scene_id / camera_id / "worst",
            base_filename=data["path"].name,
            config=config,
        ) 