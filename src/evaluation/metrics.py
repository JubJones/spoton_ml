import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import torch

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    MeanAveragePrecision = None
    TORCHMETRICS_AVAILABLE = False
    logging.warning("torchmetrics library not found or failed to import. mAP calculation unavailable. "
                   "Install with: pip install torchmetrics[detection]")

logger = logging.getLogger(__name__)

# Structure for ground truth remains similar initially, but will be converted
GroundTruthData = Dict[Tuple[int, str], List[Tuple[int, float, float, float, float]]] # (frame_idx, cam_id) -> list of (obj_id, cx, cy, w, h)

# Structure for predictions used by torchmetrics mAP
MAPPred = Dict[str, torch.Tensor] # keys: 'boxes', 'scores', 'labels'
MAPTarget = Dict[str, torch.Tensor] # keys: 'boxes', 'labels'


def xywh_to_xyxy(boxes_xywh: List[List[float]]) -> List[List[float]]:
    """Converts [[center_x, center_y, width, height], ...] to [[xmin, ymin, xmax, ymax], ...]"""
    boxes_xyxy = []
    for box in boxes_xywh:
        cx, cy, w, h = box
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2
        boxes_xyxy.append([x_min, y_min, x_max, y_max])
    return boxes_xyxy


def gt_tuples_to_xyxy(gt_tuples: List[Tuple[int, float, float, float, float]]) -> List[List[float]]:
    """Converts GT tuples [(obj_id, cx, cy, w, h), ...] to [[xmin, ymin, xmax, ymax], ...]"""
    boxes_xyxy = []
    for _, cx, cy, w, h in gt_tuples:
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2
        boxes_xyxy.append([x_min, y_min, x_max, y_max])
    return boxes_xyxy


def load_ground_truth(
    scene_path: Path,
    active_camera_ids: List[str],
    frame_filenames: List[str], # List of filenames defining the sequence length
    person_class_index: int = 0 # Class ID to assign to GT boxes (relevant for mAP)
) -> Tuple[Optional[GroundTruthData], int]:
    """
    Loads ground truth data from gt.txt files for active cameras.
    Filters based on the provided frame_filenames list.
    """
    gt_data: GroundTruthData = defaultdict(list)
    num_frames_to_load = len(frame_filenames)
    frame_filename_to_idx = {fname: idx for idx, fname in enumerate(frame_filenames)}
    logger.info(
        f"Loading ground truth for {len(active_camera_ids)} cameras, matching {num_frames_to_load} frame filenames..."
    )
    found_gt = False

    for cam_id in active_camera_ids:
        gt_file_path = scene_path / cam_id / "gt" / "gt.txt"
        if not gt_file_path.is_file():
            logger.warning(f"Ground truth file not found for camera {cam_id} at {gt_file_path}. Skipping.")
            continue

        found_gt = True
        try:
            with open(gt_file_path, 'r') as f:
                lines_read = 0
                gt_count_cam = 0
                for line in f:
                    lines_read += 1
                    parts = line.strip().split(',')
                    if len(parts) < 6: continue # Need frame, id, x, y, w, h

                    try:
                        frame_idx_txt = int(parts[0]) # Use 0-based index from gt.txt format
                        # This requires knowing the exact frame filenames and their order
                        # We need to map this frame_idx_txt back to our internal index if needed,
                        # BUT it's simpler to filter GT based on filenames present in the dataset split.
                        # For now, we assume frame_idx_txt corresponds to the order in frame_filenames
                        # A robust way is to parse filename from a different column if available, or
                        # match frame_idx_txt to the index in the sorted list of all potential frames.

                        # *** Assuming frame_idx_txt directly maps to index in `frame_filenames` ***
                        if frame_idx_txt >= num_frames_to_load:
                            continue # Only load GT for frames we intend to process based on filenames list

                        obj_id = int(parts[1])
                        bb_left = float(parts[2])
                        bb_top = float(parts[3])
                        bb_width = float(parts[4])
                        bb_height = float(parts[5])

                        if bb_width <= 0 or bb_height <= 0: continue

                        center_x = bb_left + bb_width / 2
                        center_y = bb_top + bb_height / 2

                        # Use the index from the provided `frame_filenames` list
                        internal_frame_idx = frame_idx_txt
                        gt_data[(internal_frame_idx, cam_id)].append((obj_id, center_x, center_y, bb_width, bb_height))
                        gt_count_cam += 1

                    except ValueError as ve:
                        logger.debug(f"Skipping GT line due to parsing error in {cam_id}: {line.strip()} ({ve})")

                logger.info(f"Loaded {gt_count_cam} valid GT entries from {cam_id} ({lines_read} lines read).")

        except Exception as e:
            logger.error(f"Error reading ground truth file {gt_file_path}: {e}", exc_info=True)

    total_gt_boxes = sum(len(v) for v in gt_data.values())
    logger.info(
        f"Finished loading ground truth. Found {total_gt_boxes} total entries for {len(gt_data)} (frame, camera) pairs.")

    if not found_gt:
        logger.warning("No ground truth files (gt.txt) were found. mAP calculation will be skipped.")
        return None, person_class_index
    if not gt_data:
        logger.warning("GT files found, but no valid entries loaded (check format/frame range). mAP will be skipped.")
        return None, person_class_index

    logger.info(f"Using Person Class Index: {person_class_index} for GT labels during mAP calculation.")

    return dict(gt_data), person_class_index


# --- Combined Metrics Calculation (for Detection Pipeline) ---
# This function might not be directly used by the training pipeline,
# but the mAP logic within it is relevant for the training evaluation engine.

def calculate_detection_metrics_with_map(
    results: Dict[str, Any], # Performance metrics dict
    active_camera_ids: List[str],
    all_predictions: List[MAPPred], # List of predictions per image for mAP
    all_targets: List[MAPTarget],     # List of targets per image for mAP
    person_class_index: int, # The index corresponding to the 'person' class (usually 1)
    device: torch.device # Device for metric calculation (can be CPU)
) -> Dict[str, Any]:
    """
    Calculates aggregate performance metrics AND mAP using torchmetrics.
    Designed for the output of the detection pipeline.
    """
    metrics = {}
    logger.info("Calculating aggregate performance and mAP metrics...")

    # --- Basic Performance Metrics ---
    frame_counter = results.get('frame_counter', 0)
    if frame_counter == 0:
        logger.warning("No frames were processed successfully. Cannot calculate performance metrics.")
    else:
        total_processing_time_sec = results.get('total_processing_time_sec', 0)
        total_inference_time_ms = results.get('total_inference_time_ms', 0)
        avg_inference_time_ms = total_inference_time_ms / frame_counter if frame_counter > 0 else 0
        total_person_detections = results.get('total_person_detections', 0)
        avg_detections = total_person_detections / frame_counter if frame_counter > 0 else 0
        processing_fps = frame_counter / total_processing_time_sec if total_processing_time_sec > 0 else 0

        metrics["perf_total_frames_processed"] = frame_counter
        metrics["perf_unique_frame_indices_processed"] = results.get('unique_frame_indices_processed', 0)
        metrics["perf_total_processing_time_sec"] = round(total_processing_time_sec, 2)
        metrics["perf_avg_inference_time_ms_per_frame"] = round(avg_inference_time_ms, 2)
        metrics["perf_total_person_detections"] = total_person_detections
        metrics["perf_avg_detections_per_frame"] = round(avg_detections, 2)
        metrics["perf_processing_fps"] = round(processing_fps, 2)

        logger.info(f"Performance Metrics:")
        for k, v in metrics.items():
             if k.startswith("perf_"): logger.info(f"  {k}: {v}")

        # Per-Camera Performance
        frame_count_per_camera = results.get('frame_count_per_camera', {})
        detections_per_camera = results.get('detections_per_camera', {})
        inference_time_per_camera = results.get('inference_time_per_camera', {})

        for cam_id in active_camera_ids:
            cam_frames = frame_count_per_camera.get(cam_id, 0)
            cam_dets = detections_per_camera.get(cam_id, 0)
            cam_inf_time = inference_time_per_camera.get(cam_id, 0)
            avg_inf_cam = (cam_inf_time / cam_frames) if cam_frames > 0 else 0
            avg_dets_cam = (cam_dets / cam_frames) if cam_frames > 0 else 0
            metrics[f"perf_frames_cam_{cam_id}"] = cam_frames
            metrics[f"perf_detections_cam_{cam_id}"] = cam_dets
            metrics[f"perf_avg_inf_ms_cam_{cam_id}"] = round(avg_inf_cam, 2)
            metrics[f"perf_avg_dets_cam_{cam_id}"] = round(avg_dets_cam, 2)

    # --- mAP Calculation ---
    if not TORCHMETRICS_AVAILABLE:
        logger.error("torchmetrics not available. Skipping mAP calculation.")
        metrics["eval_map"] = -1.0 # Use -1 to indicate unavailable
        metrics["eval_map_50"] = -1.0
        metrics["eval_map_75"] = -1.0
        return metrics

    if not all_predictions or not all_targets:
        logger.warning("No predictions or targets collected (or GT not available). Skipping mAP calculation.")
        metrics["eval_map"] = -1.0
        metrics["eval_map_50"] = -1.0
        metrics["eval_map_75"] = -1.0
        return metrics

    if len(all_predictions) != len(all_targets):
         logger.error(f"Mismatch between number of predictions ({len(all_predictions)}) and targets ({len(all_targets)}). Cannot calculate mAP.")
         metrics["eval_map"] = -1.0
         metrics["eval_map_50"] = -1.0
         metrics["eval_map_75"] = -1.0
         return metrics

    logger.info(f"Calculating mAP using torchmetrics on device '{device}'...")
    map_results = compute_map_metrics(all_predictions, all_targets, device, person_class_index)
    metrics.update(map_results) # Add computed mAP metrics to the main dict

    logger.info("--- mAP Results (COCO Standard IoUs) ---")
    for k, v in metrics.items():
         if k.startswith("eval_"): logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


def compute_map_metrics(
    predictions: List[MAPPred],
    targets: List[MAPTarget],
    device: torch.device,
    person_class_id: int = 1,
) -> Dict[str, float]:
    """
    Computes mAP metrics using torchmetrics. Helper function.

    Args:
        predictions: List of prediction dicts [{'boxes': T(N,4), 'scores': T(N), 'labels': T(N)}, ...]
        targets: List of target dicts [{'boxes': T(M,4), 'labels': T(M)}, ...]
        device: Device to perform calculation on (CPU recommended for large datasets).
        person_class_id: The integer ID representing the 'person' class.

    Returns:
        Dictionary containing computed mAP metrics (e.g., 'eval_map', 'eval_map_50').
        Returns placeholder values if calculation fails.
    """
    map_metrics = { # Placeholder values
        "eval_map": -1.0, "eval_map_50": -1.0, "eval_map_75": -1.0,
        "eval_map_small": -1.0, "eval_map_medium": -1.0, "eval_map_large": -1.0,
        "eval_mar_1": -1.0, "eval_mar_10": -1.0, "eval_mar_100": -1.0,
        "eval_mar_small": -1.0, "eval_mar_medium": -1.0, "eval_mar_large": -1.0,
        "eval_ap_person": -1.0
    }
    if not TORCHMETRICS_AVAILABLE:
        logger.error("Torchmetrics not installed, cannot compute mAP.")
        return map_metrics

    if not predictions or not targets or len(predictions) != len(targets):
         logger.warning(f"Invalid input for mAP: #Preds={len(predictions)}, #Targets={len(targets)}. Check data loading.")
         return map_metrics

    try:
        # It's often safer and less memory intensive to run metrics on CPU
        metric_device = torch.device('cpu')
        logger.info(f"Using device '{metric_device}' for mAP calculation.")
        metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(metric_device)

        # Move data to metric device chunk by chunk if necessary, but often feasible all at once on CPU
        preds_on_device = [{k: v.to(metric_device) for k, v in p.items()} for p in predictions]
        targets_on_device = [{k: v.to(metric_device) for k, v in t.items()} for t in targets]

        metric.update(preds_on_device, targets_on_device)
        computed_results = metric.compute()

        # Extract results safely
        main_map_keys = ['map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large']
        main_mar_keys = ['mar_1', 'mar_10', 'mar_100', 'mar_small', 'mar_medium', 'mar_large']

        for key in main_map_keys + main_mar_keys:
            if key in computed_results:
                value = computed_results[key].item()
                map_metrics[f"eval_{key}"] = round(value, 4)

        # Per-Class AP for Person
        if 'map_per_class' in computed_results and computed_results['map_per_class'] is not None:
            map_per_class_tensor = computed_results['map_per_class']
            classes_tensor = computed_results.get('classes', None) # Tensor of class indices evaluated

            if classes_tensor is not None and isinstance(classes_tensor, torch.Tensor) and \
               isinstance(map_per_class_tensor, torch.Tensor) and \
               classes_tensor.ndim == 1 and map_per_class_tensor.ndim == 1 and \
               len(classes_tensor) == len(map_per_class_tensor):

                classes_np = classes_tensor.cpu().numpy()
                map_per_class_np = map_per_class_tensor.cpu().numpy()
                class_ap_map = dict(zip(classes_np, map_per_class_np))

                if person_class_id in class_ap_map:
                    person_ap = class_ap_map[person_class_id]
                    map_metrics["eval_ap_person"] = round(person_ap.item(), 4)
                else:
                    logger.warning(f"Person class index {person_class_id} not found in computed per-class AP indices: {classes_np}")
            else:
                 logger.warning("Could not parse per-class AP results (unexpected format or missing 'classes' tensor).")

        return map_metrics

    except Exception as e:
        logger.error(f"Failed to compute mAP metrics: {e}", exc_info=True)
        # Return placeholders on error
        map_metrics = {k: -1.0 for k in map_metrics} # Reset all to -1.0 on error
        return map_metrics