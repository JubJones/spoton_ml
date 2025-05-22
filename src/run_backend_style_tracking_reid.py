"""
Main script for running backend-style tracking and Re-ID evaluation using ground truth bounding boxes.
This script adapts the core tracking and Re-ID logic from the SpotOn backend.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.pipelines.backend_style_tracking_reid_pipeline import BackendStyleTrackingReIDPipeline
from src.tracking_backend_logic.common_types_adapter import CameraID, GlobalID, TrackID
from src.utils.device_utils import get_selected_device
from src.utils.mlflow_utils import setup_mlflow_experiment

logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> dict:
    """
    Loads configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_ground_truth(data_path: Path, camera_id: CameraID) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Loads ground truth data for a camera.

    Args:
        data_path: Path to the data directory.
        camera_id: ID of the camera to load data for.

    Returns:
        Tuple of (frames, bboxes, classes)
        frames: List of frame arrays
        bboxes: List of bounding box arrays
        classes: List of class ID arrays
    """
    camera_dir = data_path / camera_id
    if not camera_dir.exists():
        raise FileNotFoundError(f"Data directory not found for camera {camera_id}: {camera_dir}")

    # Load frames
    frames_dir = camera_dir / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    frames = [np.load(str(f)) for f in sorted(frames_dir.glob("*.npy"))]

    # Load ground truth bounding boxes
    bboxes_dir = camera_dir / "gt_bboxes"
    if not bboxes_dir.exists():
        raise FileNotFoundError(f"Ground truth bboxes directory not found: {bboxes_dir}")
    bboxes = [np.load(str(f)) for f in sorted(bboxes_dir.glob("*.npy"))]

    # Load ground truth classes (if available)
    classes_dir = camera_dir / "gt_classes"
    if classes_dir.exists():
        classes = [np.load(str(f)) for f in sorted(classes_dir.glob("*.npy"))]
    else:
        classes = [None] * len(frames)

    return frames, bboxes, classes

def evaluate_tracking_reid(
    pipeline: BackendStyleTrackingReIDPipeline,
    frames: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    gt_classes: List[Optional[np.ndarray]],
    camera_id: CameraID
) -> Dict[str, float]:
    """
    Evaluates tracking and Re-ID performance on a sequence of frames.

    Args:
        pipeline: BackendStyleTrackingReIDPipeline instance.
        frames: List of frame arrays.
        gt_bboxes: List of ground truth bounding box arrays.
        gt_classes: List of ground truth class ID arrays.
        camera_id: ID of the camera being evaluated.

    Returns:
        Dictionary containing evaluation metrics.
    """
    # Initialize metrics
    total_frames = len(frames)
    total_objects = sum(len(bboxes) for bboxes in gt_bboxes)
    total_global_ids = 0
    total_handoffs = 0

    # Process each frame
    for frame, bboxes, classes in tqdm(
        zip(frames, gt_bboxes, gt_classes),
        total=total_frames,
        desc=f"Processing camera {camera_id}"
    ):
        # Process frame
        tracked_objects, track_to_global_id = pipeline.process_frame(
            camera_id, frame, bboxes, classes
        )

        # Update metrics
        total_global_ids = max(total_global_ids, len(track_to_global_id))
        if len(track_to_global_id) > 0:
            total_handoffs += 1

    # Calculate metrics
    metrics = {
        "total_frames": total_frames,
        "total_objects": total_objects,
        "total_global_ids": total_global_ids,
        "total_handoffs": total_handoffs,
        "handoff_rate": total_handoffs / total_frames if total_frames > 0 else 0.0
    }

    return metrics

def main():
    """Main function for running the backend-style tracking and Re-ID evaluation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run backend-style tracking and Re-ID evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))

    # Setup MLflow
    setup_mlflow_experiment(
        experiment_name=config["mlflow"]["experiment_name"],
        run_name=config["parent_run_name"]
    )

    # Get device
    device = get_selected_device(config["environment"]["device"])

    # Initialize pipeline
    pipeline = BackendStyleTrackingReIDPipeline(
        reid_weights_path=Path(config["reid_params"]["model_path"]),
        handoff_config_path=Path(config["handoff_config"]["config_path"]),
        device=device,
        half_precision=config["environment"]["half_precision"],
        per_class=config["tracker_params"]["per_class"],
        similarity_threshold=config["reid_params"]["similarity_threshold"]
    )

    # Load models and warmup
    pipeline.load_models()
    pipeline.warmup()

    # Get list of cameras to process
    data_path = Path(config["data"]["base_path"])
    camera_ids = [d.name for d in data_path.iterdir() if d.is_dir()]

    # Process each camera
    all_metrics = {}
    for camera_id in camera_ids:
        try:
            # Load ground truth data
            frames, gt_bboxes, gt_classes = load_ground_truth(data_path, camera_id)

            # Evaluate tracking and Re-ID
            metrics = evaluate_tracking_reid(
                pipeline, frames, gt_bboxes, gt_classes, camera_id
            )

            # Log metrics
            all_metrics[camera_id] = metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{camera_id}_{metric_name}", metric_value)

            # Reset pipeline for next camera
            pipeline.reset()

        except Exception as e:
            logger.error(f"Error processing camera {camera_id}: {e}", exc_info=True)
            continue

    # Log overall metrics
    overall_metrics = {
        "total_cameras": len(camera_ids),
        "processed_cameras": len(all_metrics),
        "average_handoff_rate": np.mean([
            metrics["handoff_rate"]
            for metrics in all_metrics.values()
        ]) if all_metrics else 0.0
    }

    for metric_name, metric_value in overall_metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main() 