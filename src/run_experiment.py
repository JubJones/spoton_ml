import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import dagshub
import mlflow
import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
print(f"PROJECT_ROOT added to sys.path: {PROJECT_ROOT}")

# Local imports
try:
    from src.utils.config_loader import load_config
    from src.tracking.device_utils import get_selected_device
    from src.data.loader import FrameDataLoader
    from src.tracking.strategies import get_strategy, DetectionTrackingStrategy
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Ensure you are running this script from the project root or that src is in PYTHONPATH.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = PROJECT_ROOT / "experiment.log"
if log_file.exists():
    open(log_file, 'w').close()  # Clear log file on start

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)

# Structure: {(frame_idx, cam_id): [(object_id, x, y, w, h), ...], ...}
GroundTruthData = Dict[Tuple[int, str], List[Tuple[int, float, float, float, float]]]


# --- Helper Functions ---
def setup_environment_and_mlflow(config_path_str: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Loads config, initializes MLflow, and returns config and experiment_id."""
    config = load_config(config_path_str)
    if not config:
        logger.critical("Failed to load configuration.")
        return None, None

    mlflow_config = config.get("mlflow", {})
    dotenv_path = PROJECT_ROOT / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logger.info("Loaded environment variables from .env file.")
    else:
        logger.info(".env file not found, relying on environment or defaults.")

    try:
        repo_owner = "Jwizzed"
        repo_name = "spoton_ml"
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        logger.info(f"Dagshub initialized successfully for {repo_owner}/{repo_name}.")
        tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"MLflow tracking URI automatically set by Dagshub: {tracking_uri}")
    except Exception as dag_err:
        logger.warning(f"Dagshub initialization failed: {dag_err}. Attempting manual URI setup.")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set from environment variable: {tracking_uri}")
        else:
            logger.warning("MLFLOW_TRACKING_URI not set and Dagshub init failed. Using local tracking.")
            (PROJECT_ROOT / "mlruns").mkdir(exist_ok=True)

    experiment_name = mlflow_config.get("experiment_name", "Default Experiment")
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: '{experiment_name}'")

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.info(f"Creating experiment '{experiment_name}'.")
            experiment_id = client.create_experiment(experiment_name)
        elif experiment.lifecycle_stage != 'active':
            logger.error(f"Experiment '{experiment_name}' is deleted or archived.")
            return config, None
        else:
            experiment_id = experiment.experiment_id
        logger.info(f"Using experiment ID: {experiment_id}")
        return config, experiment_id
    except Exception as client_err:
        logger.error(f"Failed to connect to MLflow or get/create experiment: {client_err}")
        return config, None


def log_params_recursive(params_dict, parent_key=""):
    """Recursively logs parameters to MLflow."""
    for key, value in params_dict.items():
        mlflow_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            log_params_recursive(value, mlflow_key)
        elif isinstance(value, list):
            try:
                mlflow.log_param(mlflow_key, json.dumps(value))
            except TypeError:
                mlflow.log_param(mlflow_key, str(value)[:250])
        else:
            mlflow.log_param(mlflow_key, str(value)[:250])


def load_ground_truth(
        scene_path: Path,
        active_camera_ids: List[str],
        frame_filenames: List[str]
) -> GroundTruthData:
    """Loads ground truth data from gt.txt files for active cameras."""
    gt_data: GroundTruthData = defaultdict(list)
    num_frames_to_load = len(frame_filenames)
    logger.info(
        f"Loading ground truth for {len(active_camera_ids)} cameras up to frame index {num_frames_to_load - 1}...")

    for cam_id in active_camera_ids:
        gt_file_path = scene_path / cam_id / "gt" / "gt.txt"
        if not gt_file_path.is_file():
            logger.warning(f"Ground truth file not found for camera {cam_id} at {gt_file_path}. Skipping.")
            continue

        try:
            with open(gt_file_path, 'r') as f:
                lines_read = 0
                for line in f:
                    lines_read += 1
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        logger.debug(f"Skipping malformed GT line in {cam_id}: {line.strip()}")
                        continue
                    try:
                        # MOT format: frame_id, object_id, bb_left, bb_top, bb_width, bb_height, ...
                        frame_id = int(parts[0]) - 1
                        if frame_id >= num_frames_to_load:
                            continue  # Don't load GT for frames we won't process

                        obj_id = int(parts[1])
                        x = float(parts[2])
                        y = float(parts[3])
                        w = float(parts[4])
                        h = float(parts[5])

                        # Basic validation
                        if w <= 0 or h <= 0:
                            logger.debug(f"Skipping GT box with non-positive dimensions in {cam_id}, frame {frame_id}: w={w}, h={h}")
                            continue

                        gt_data[(frame_id, cam_id)].append((obj_id, x, y, w, h))

                    except ValueError as ve:
                        logger.warning(f"Skipping GT line due to parsing error in {cam_id}: {line.strip()} ({ve})")
            logger.info(
                f"Loaded {sum(len(v) for v in gt_data.values())} GT entries from {cam_id} ({lines_read} lines read).")

        except Exception as e:
            logger.error(f"Error reading ground truth file {gt_file_path}: {e}", exc_info=True)

    logger.info(f"Finished loading ground truth. Found entries for {len(gt_data)} (frame, camera) pairs.")
    if not gt_data:
        logger.warning("No ground truth data was loaded successfully.")
    return dict(gt_data)  # Convert back to regular dict


def calculate_iou(boxA_xywh: List[float], boxB_xywh: List[float]) -> float:
    """Calculates Intersection over Union (IoU) for two boxes in xywh format."""
    # Convert xywh to xyxy
    boxA_xyxy = [boxA_xywh[0] - boxA_xywh[2] / 2, boxA_xywh[1] - boxA_xywh[3] / 2,
                 boxA_xywh[0] + boxA_xywh[2] / 2, boxA_xywh[1] + boxA_xywh[3] / 2]
    boxB_xyxy = [boxB_xywh[0] - boxB_xywh[2] / 2, boxB_xywh[1] - boxB_xywh[3] / 2,
                 boxB_xywh[0] + boxB_xywh[2] / 2, boxB_xywh[1] + boxB_xywh[3] / 2]

    # Determine the coordinates of the intersection rectangle
    x_left = max(boxA_xyxy[0], boxB_xyxy[0])
    y_top = max(boxA_xyxy[1], boxB_xyxy[1])
    x_right = min(boxA_xyxy[2], boxB_xyxy[2])
    y_bottom = min(boxA_xyxy[3], boxB_xyxy[3])

    # Compute the area of intersection rectangle
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    if intersection_area == 0:
        return 0.0

    # Compute the area of both bounding boxes
    boxA_area = boxA_xywh[2] * boxA_xywh[3]
    boxB_area = boxB_xywh[2] * boxB_xywh[3]

    # Compute the area of union
    union_area = boxA_area + boxB_area - intersection_area

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou


def evaluate_frame_detections(
        detected_boxes_xywh: List[List[float]],
        gt_boxes_xywh: List[List[float]],
        iou_threshold: float
) -> Tuple[int, int, int]:
    """Matches detections to ground truth using IoU and calculates TP, FP, FN for a frame."""
    tp = 0
    fp = 0
    fn = 0

    num_gt = len(gt_boxes_xywh)
    num_det = len(detected_boxes_xywh)

    if num_gt == 0 and num_det == 0:
        return 0, 0, 0
    if num_gt == 0:
        return 0, num_det, 0  # All detections are false positives
    if num_det == 0:
        return 0, 0, num_gt  # All ground truths are false negatives

    # Build IoU matrix
    iou_matrix = np.zeros((num_det, num_gt))
    for i, det_box in enumerate(detected_boxes_xywh):
        for j, gt_box in enumerate(gt_boxes_xywh):
            iou_matrix[i, j] = calculate_iou(det_box, gt_box)

    # Greedy matching based on highest IoU
    matched_gt = [False] * num_gt
    matched_det = [False] * num_det

    # Get row (det) and col (gt) indices for IoUs >= threshold
    match_indices = np.argwhere(iou_matrix >= iou_threshold)
    # Get the IoU values for these potential matches
    iou_values = iou_matrix[match_indices[:, 0], match_indices[:, 1]]
    # Sort potential matches by IoU value (descending)
    sorted_match_indices = match_indices[np.argsort(iou_values)[::-1]]

    for det_idx, gt_idx in sorted_match_indices:
        if not matched_det[det_idx] and not matched_gt[gt_idx]:
            tp += 1
            matched_det[det_idx] = True
            matched_gt[gt_idx] = True

    # Calculate FP (unmatched detections)
    fp = num_det - sum(matched_det)

    # Calculate FN (unmatched ground truths)
    fn = num_gt - sum(matched_gt)

    return tp, fp, fn


def initialize_components(
        config: Dict[str, Any],
        actual_device: torch.device
) -> Tuple[Optional[DetectionTrackingStrategy], Optional[FrameDataLoader], Optional[GroundTruthData]]:
    """Initializes detection strategy, data loader, and loads ground truth."""
    try:
        model_config = config.get("model", {})
        detection_strategy = get_strategy(model_config, actual_device)
        logger.info(f"Using detection strategy: {detection_strategy.__class__.__name__} on device: {actual_device}")
    except (ValueError, ImportError, Exception) as e:
        logger.critical(f"Failed to initialize detection strategy: {e}", exc_info=True)
        return None, None, None

    try:
        data_loader = FrameDataLoader(config)
        if len(data_loader) == 0:
            raise ValueError("Data loader found 0 frame indices to process.")
        logger.info(
            f"Data loader initialized. Processing {len(data_loader)} frame indices across "
            f"{len(data_loader.active_camera_ids)} cameras: {data_loader.active_camera_ids}."
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.critical(f"Failed to initialize data loader: {e}", exc_info=True)
        return detection_strategy, None, None

    try:
        ground_truth_data = load_ground_truth(
            data_loader.scene_path,
            data_loader.active_camera_ids,
            data_loader.image_filenames
        )
    except Exception as e:
        logger.error(f"Failed to load ground truth data: {e}", exc_info=True)
        # Continue without GT if loading fails, but evaluation won't happen
        ground_truth_data = None

    return detection_strategy, data_loader, ground_truth_data


def process_frames_and_evaluate(
        data_loader: FrameDataLoader,
        detection_strategy: DetectionTrackingStrategy,
        ground_truth_data: Optional[GroundTruthData],
        iou_threshold: float
) -> Dict[str, Any]:
    """Processes frames, performs detection, evaluates against GT, and collects raw results."""
    logger.info("Starting frame processing and evaluation loop...")
    results = defaultdict(float)
    results['frame_counter'] = 0
    results['total_tp'] = 0
    results['total_fp'] = 0
    results['total_fn'] = 0
    results['total_gt_boxes'] = 0
    processed_indices = set()
    detections_per_camera = defaultdict(int)
    inference_time_per_camera = defaultdict(float)
    frame_count_per_camera = defaultdict(int)
    gt_exists_for_run = ground_truth_data is not None

    start_time_total = time.perf_counter()
    total_frames_to_process = len(data_loader) * len(data_loader.active_camera_ids)

    for frame_idx, cam_id, filename, frame_bgr in data_loader:
        if frame_bgr is None:
            continue

        results['frame_counter'] += 1
        frame_count_per_camera[cam_id] += 1
        processed_indices.add(frame_idx)

        # --- Detection ---
        start_time_inference = time.perf_counter()
        boxes_xywh, _, confidences = detection_strategy.process_frame(frame_bgr)
        inference_time_ms = (time.perf_counter() - start_time_inference) * 1000

        num_detections_frame = len(boxes_xywh)
        results['total_inference_time_ms'] += inference_time_ms
        results['total_person_detections'] += num_detections_frame
        detections_per_camera[cam_id] += num_detections_frame
        inference_time_per_camera[cam_id] += inference_time_ms

        # --- Evaluation (if GT available) ---
        frame_tp, frame_fp, frame_fn = 0, 0, 0
        if gt_exists_for_run:
            gt_for_frame = ground_truth_data.get((frame_idx, cam_id), [])
            gt_boxes_xywh = [[x, y, w, h] for _, x, y, w, h in gt_for_frame]
            results['total_gt_boxes'] += len(gt_boxes_xywh)

            if num_detections_frame > 0 or len(gt_boxes_xywh) > 0:
                frame_tp, frame_fp, frame_fn = evaluate_frame_detections(
                    boxes_xywh, gt_boxes_xywh, iou_threshold
                )
                results['total_tp'] += frame_tp
                results['total_fp'] += frame_fp
                results['total_fn'] += frame_fn
        else:
            pass

        if int(results['frame_counter']) % 100 == 0:
            log_msg = (
                f"Processed {int(results['frame_counter'])}/{total_frames_to_process} frames... "
                f"(Idx {frame_idx}/{len(data_loader) - 1}) "
                f"Cam {cam_id}: {num_detections_frame} dets, {inference_time_ms:.2f}ms"
            )
            if gt_exists_for_run:
                log_msg += f" | Eval: TP={frame_tp}, FP={frame_fp}, FN={frame_fn} (GT={len(gt_boxes_xywh)})"
            logger.info(log_msg)

    results['total_processing_time_sec'] = time.perf_counter() - start_time_total
    results['unique_frame_indices_processed'] = len(processed_indices)
    results['detections_per_camera'] = dict(detections_per_camera)
    results['inference_time_per_camera'] = dict(inference_time_per_camera)
    results['frame_count_per_camera'] = dict(frame_count_per_camera)
    results['gt_available'] = gt_exists_for_run

    logger.info("--- Frame Processing Finished ---")
    return dict(results)


def calculate_and_log_metrics(
        results: Dict[str, Any],
        config: Dict[str, Any],
        active_camera_ids: List[str]
):
    """Calculates aggregate and evaluation metrics and logs them to MLflow."""
    logger.info("Calculating and logging metrics...")

    frame_counter = results.get('frame_counter', 0)
    if frame_counter == 0:
        logger.warning("No frames were processed successfully. No metrics to log.")
        return

    # --- Performance Metrics ---
    total_processing_time_sec = results.get('total_processing_time_sec', 0)
    avg_inference_time_ms = results.get('total_inference_time_ms', 0) / frame_counter
    avg_detections = results.get('total_person_detections', 0) / frame_counter
    processing_fps = frame_counter / total_processing_time_sec if total_processing_time_sec > 0 else 0

    logger.info(f"Total Frames Processed: {frame_counter}")
    logger.info(f"Unique Frame Indices Processed: {results.get('unique_frame_indices_processed', 0)}")
    logger.info(f"Total Processing Time: {total_processing_time_sec:.2f} seconds")
    logger.info(f"Overall Average Inference Time: {avg_inference_time_ms:.2f} ms/frame")
    logger.info(f"Overall Average Person Detections: {avg_detections:.2f} per frame")
    logger.info(f"Overall Processing FPS: {processing_fps:.2f} frames/sec")
    logger.info(f"Total Person Detections Found: {results.get('total_person_detections', 0)}")

    mlflow.log_metric("total_frames_processed", frame_counter)
    mlflow.log_metric("unique_frame_indices_processed", results.get('unique_frame_indices_processed', 0))
    mlflow.log_metric("total_processing_time_sec", round(total_processing_time_sec, 2))
    mlflow.log_metric("avg_inference_time_ms_per_frame", round(avg_inference_time_ms, 2))
    mlflow.log_metric("total_person_detections", results.get('total_person_detections', 0))
    mlflow.log_metric("avg_detections_per_frame", round(avg_detections, 2))
    mlflow.log_metric("processing_fps", round(processing_fps, 2))

    # --- Per-Camera Performance Metrics ---
    frame_count_per_camera = results.get('frame_count_per_camera', {})
    detections_per_camera = results.get('detections_per_camera', {})
    inference_time_per_camera = results.get('inference_time_per_camera', {})

    for cam_id in active_camera_ids:
        cam_frames = frame_count_per_camera.get(cam_id, 0)
        cam_dets = detections_per_camera.get(cam_id, 0)
        cam_inf_time = inference_time_per_camera.get(cam_id, 0)
        avg_inf_cam = (cam_inf_time / cam_frames) if cam_frames > 0 else 0
        avg_dets_cam = (cam_dets / cam_frames) if cam_frames > 0 else 0

        mlflow.log_metric(f"frames_cam_{cam_id}", cam_frames)
        mlflow.log_metric(f"detections_cam_{cam_id}", cam_dets)
        mlflow.log_metric(f"avg_inf_ms_cam_{cam_id}", round(avg_inf_cam, 2))
        mlflow.log_metric(f"avg_dets_cam_{cam_id}", round(avg_dets_cam, 2))

    # --- Evaluation Metrics (if GT was available) ---
    if results.get('gt_available', False):
        total_tp = results.get('total_tp', 0)
        total_fp = results.get('total_fp', 0)
        total_fn = results.get('total_fn', 0)
        total_gt = results.get('total_gt_boxes', 0)  # Should equal total_tp + total_fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        logger.info("--- Evaluation Results ---")
        logger.info(f"Total Ground Truth Boxes: {total_gt}")
        logger.info(f"Total True Positives (TP): {total_tp}")
        logger.info(f"Total False Positives (FP): {total_fp}")
        logger.info(f"Total False Negatives (FN): {total_fn}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1_score:.4f}")

        mlflow.log_metric("eval_total_gt_boxes", total_gt)
        mlflow.log_metric("eval_total_tp", total_tp)
        mlflow.log_metric("eval_total_fp", total_fp)
        mlflow.log_metric("eval_total_fn", total_fn)
        mlflow.log_metric("eval_precision", round(precision, 4))
        mlflow.log_metric("eval_recall", round(recall, 4))
        mlflow.log_metric("eval_f1_score", round(f1_score, 4))
    else:
        logger.warning("Ground truth data was not available or failed to load. Skipping evaluation metrics.")
        mlflow.log_metric("eval_total_gt_boxes", 0)


# --- Main Experiment Logic ---
def run_experiment():
    """Main function to set up and run the detection experiment."""
    logger.info("--- Starting Experiment ---")
    config_path_str = "configs/experiment_config.yaml"
    run_status = "FAILED"  # Default status

    config, experiment_id = setup_environment_and_mlflow(config_path_str)
    if not config or not experiment_id:
        logger.critical("Environment or MLflow setup failed. Exiting.")
        sys.exit(1)

    # Determine device
    requested_device_name = config.get("environment", {}).get("device", "auto")
    requested_device = get_selected_device(requested_device_name)

    # Handle potential FasterRCNN CPU override
    model_type = config.get("model", {}).get("type", "").lower()
    actual_device = requested_device
    device_override_reason = "None"
    if model_type == "fasterrcnn" and requested_device.type != 'cuda':
        logger.warning(f"FasterRCNN selected but device is '{requested_device.type}'. Forcing to CPU.")
        actual_device = torch.device('cpu')
        device_override_reason = "FasterRCNN requires CUDA, fell back to CPU"

    run_name = config.get("run_name", "unnamed_run")
    run_id = None

    try:
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            logger.info(f"Starting MLflow run: {run_name} (ID: {run_id})")

            # --- Log Parameters ---
            logger.info("Logging parameters...")
            log_params_recursive(config)
            mlflow.log_param("environment.requested_device", str(requested_device))
            mlflow.log_param("environment.actual_device_used", str(actual_device))
            mlflow.log_param("environment.device_override_reason", device_override_reason)
            mlflow.log_artifact(str(PROJECT_ROOT / config_path_str), artifact_path="config")
            logger.info("Parameters and config artifact logged.")

            # --- Initialize Components ---
            detection_strategy, data_loader, ground_truth_data = initialize_components(config, actual_device)
            if not detection_strategy or not data_loader:
                raise RuntimeError("Failed to initialize detection strategy or data loader.")

            # Log actual data used
            mlflow.log_param("data.actual_cameras_used", ",".join(data_loader.active_camera_ids))
            mlflow.log_param("data.actual_frame_indices_processed", len(data_loader))

            # --- Processing Loop ---
            iou_threshold = config.get("evaluation", {}).get("iou_threshold", 0.5)
            results = process_frames_and_evaluate(data_loader, detection_strategy, ground_truth_data, iou_threshold)

            # --- Calculate Aggregate Metrics & Log ---
            if results['frame_counter'] > 0:
                calculate_and_log_metrics(results, config, data_loader.active_camera_ids)
                logger.info("Metrics logged successfully.")
                run_status = "FINISHED"
            else:
                logger.warning("No frames processed, run will be marked as FAILED.")
                run_status = "FAILED"


    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user (KeyboardInterrupt).")
        run_status = "KILLED"
        sys.exit(1)  # Exit cleanly
    except Exception as e:
        logger.critical(f"An error occurred during the experiment run: {e}", exc_info=True)
        run_status = "FAILED"
    finally:
        if run_id and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
            mlflow.end_run(status=run_status)
            logger.info(f"MLflow run {run_id} ended with status: {run_status}")
        elif run_id:
            logger.warning(f"Attempting to end run {run_id} outside active context (final status: {run_status})")
            try:
                mlflow.tracking.MlflowClient().set_terminated(run_id, status=run_status)
            except Exception as client_err:
                logger.error(f"Could not explicitly set final status for run {run_id}: {client_err}")

    logger.info(f"--- Experiment Run {run_id if run_id else 'UNKNOWN'} Completed ({run_status}) ---")
    if run_status != "FINISHED":
        sys.exit(1)


if __name__ == "__main__":
    run_experiment()
