import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import torch

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    MeanAveragePrecision = None
    logging.error("torchmetrics library not found or failed to import. mAP calculation unavailable. "
                  "Install with: pip install torchmetrics[detection]")

logger = logging.getLogger(__name__)

# Structure for ground truth remains similar initially, but will be converted
GroundTruthData = Dict[Tuple[int, str], List[Tuple[int, float, float, float, float]]]


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
        frame_filenames: List[str],
        person_class_index: int = 0  # Typically 0 for COCO-style datasets if person is first class
) -> Tuple[Optional[GroundTruthData], int]:
    """
    Loads ground truth data from gt.txt files for active cameras.
    Returns the data and the assumed class index for 'person'.
    NOTE: Assumes GT provides the 'person' class relevant for this evaluation.
    """
    gt_data: GroundTruthData = defaultdict(list)
    num_frames_to_load = len(frame_filenames)
    logger.info(
        f"Loading ground truth for {len(active_camera_ids)} cameras up to frame index {num_frames_to_load - 1}...")
    found_gt = False

    for cam_id in active_camera_ids:
        gt_file_path = scene_path / cam_id / "gt" / "gt.txt"
        if not gt_file_path.is_file():
            logger.warning(f"Ground truth file not found for camera {cam_id} at {gt_file_path}. Skipping.")
            continue

        found_gt = True  # Mark that we found at least one GT file
        try:
            with open(gt_file_path, 'r') as f:
                lines_read = 0
                gt_count_cam = 0
                for line in f:
                    lines_read += 1
                    parts = line.strip().split(',')
                    # Standard MOT format often has more fields, we need at least 6 (frame, id, bb_left, bb_top, bb_width, bb_height)
                    if len(parts) < 6:
                        logger.debug(f"Skipping malformed GT line in {cam_id}: {line.strip()}")
                        continue
                    try:
                        frame_id = int(parts[0]) - 1  # MOT format is 1-based, adjust to 0-based
                        if frame_id >= num_frames_to_load:
                            continue  # Only load GT for frames we intend to process

                        obj_id = int(parts[1])
                        bb_left = float(parts[2])
                        bb_top = float(parts[3])
                        bb_width = float(parts[4])
                        bb_height = float(parts[5])

                        # Basic validation
                        if bb_width <= 0 or bb_height <= 0:
                            logger.debug(
                                f"Skipping GT box with non-positive dimensions in {cam_id}, frame {frame_id}: w={bb_width}, h={bb_height}")
                            continue

                        # Convert to center_x, center_y, width, height for internal consistency before potential conversion
                        center_x = bb_left + bb_width / 2
                        center_y = bb_top + bb_height / 2

                        # We don't strictly need the obj_id for mAP, but keep it for potential future use
                        gt_data[(frame_id, cam_id)].append((obj_id, center_x, center_y, bb_width, bb_height))
                        gt_count_cam += 1

                    except ValueError as ve:
                        logger.warning(f"Skipping GT line due to parsing error in {cam_id}: {line.strip()} ({ve})")

                logger.info(f"Loaded {gt_count_cam} GT entries from {cam_id} ({lines_read} lines read).")

        except Exception as e:
            logger.error(f"Error reading ground truth file {gt_file_path}: {e}", exc_info=True)

    total_gt_boxes = sum(len(v) for v in gt_data.values())
    logger.info(
        f"Finished loading ground truth. Found {total_gt_boxes} total entries for {len(gt_data)} (frame, camera) pairs.")

    if not found_gt:
        logger.warning(
            "No ground truth files (gt.txt) were found for any active camera. mAP calculation will be skipped.")
        return None, person_class_index
    if not gt_data:
        logger.warning(
            "Ground truth files were found, but no valid entries could be loaded (check format and frame range). mAP calculation will be skipped.")
        return None, person_class_index

    logger.info(f"Using Person Class Index: {person_class_index} for mAP calculation.")

    return dict(gt_data), person_class_index


def calculate_metrics_with_map(
    results: Dict[str, Any],
    active_camera_ids: List[str],
    all_predictions: List[Dict[str, torch.Tensor]], # List of predictions per image
    all_targets: List[Dict[str, torch.Tensor]],     # List of targets per image
    person_class_index: int, # The index corresponding to the 'person' class
    device: torch.device # Device for metric calculation (can be CPU)
) -> Dict[str, Any]:
    """
    Calculates aggregate performance metrics AND mAP using torchmetrics.
    """
    metrics = {}
    logger.info("Calculating aggregate performance and mAP metrics...")

    # --- Standard Performance Metrics (Copied from original) ---
    # ... (This part remains unchanged) ...
    frame_counter = results.get('frame_counter', 0)
    if frame_counter == 0:
        logger.warning("No frames were processed successfully. Cannot calculate metrics.")
        return metrics

    total_processing_time_sec = results.get('total_processing_time_sec', 0)
    total_inference_time_ms = results.get('total_inference_time_ms', 0)
    avg_inference_time_ms = total_inference_time_ms / frame_counter if frame_counter > 0 else 0
    total_person_detections = results.get('total_person_detections', 0)
    avg_detections = total_person_detections / frame_counter if frame_counter > 0 else 0
    processing_fps = frame_counter / total_processing_time_sec if total_processing_time_sec > 0 else 0

    metrics["total_frames_processed"] = frame_counter
    metrics["unique_frame_indices_processed"] = results.get('unique_frame_indices_processed', 0)
    metrics["total_processing_time_sec"] = round(total_processing_time_sec, 2)
    metrics["avg_inference_time_ms_per_frame"] = round(avg_inference_time_ms, 2)
    metrics["total_person_detections"] = total_person_detections
    metrics["avg_detections_per_frame"] = round(avg_detections, 2)
    metrics["processing_fps"] = round(processing_fps, 2)

    logger.info(f"Total Frames Processed: {metrics['total_frames_processed']}")
    logger.info(f"Unique Frame Indices Processed: {metrics['unique_frame_indices_processed']}")
    logger.info(f"Total Processing Time: {metrics['total_processing_time_sec']:.2f} seconds")
    logger.info(f"Overall Average Inference Time: {metrics['avg_inference_time_ms_per_frame']:.2f} ms/frame")
    logger.info(f"Overall Average Person Detections: {metrics['avg_detections_per_frame']:.2f} per frame")
    logger.info(f"Overall Processing FPS: {metrics['processing_fps']:.2f} frames/sec")
    logger.info(f"Total Person Detections Found: {metrics['total_person_detections']}")


    # --- Per-Camera Performance Metrics (Copied from original) ---
    # ... (This part remains unchanged) ...
    frame_count_per_camera = results.get('frame_count_per_camera', {})
    detections_per_camera = results.get('detections_per_camera', {})
    inference_time_per_camera = results.get('inference_time_per_camera', {})

    for cam_id in active_camera_ids:
        cam_frames = frame_count_per_camera.get(cam_id, 0)
        cam_dets = detections_per_camera.get(cam_id, 0)
        cam_inf_time = inference_time_per_camera.get(cam_id, 0)
        avg_inf_cam = (cam_inf_time / cam_frames) if cam_frames > 0 else 0
        avg_dets_cam = (cam_dets / cam_frames) if cam_frames > 0 else 0

        metrics[f"frames_cam_{cam_id}"] = cam_frames
        metrics[f"detections_cam_{cam_id}"] = cam_dets
        metrics[f"avg_inf_ms_cam_{cam_id}"] = round(avg_inf_cam, 2)
        metrics[f"avg_dets_cam_{cam_id}"] = round(avg_dets_cam, 2)


    # --- mAP Calculation ---
    # ...(Initialization and checks remain the same)...
    if MeanAveragePrecision is None:
        logger.error("torchmetrics not available. Skipping mAP calculation.")
        # Add placeholder metrics
        metrics["eval_map"] = 0.0
        metrics["eval_map_50"] = 0.0
        metrics["eval_map_75"] = 0.0
        return metrics

    if not all_predictions or not all_targets:
        logger.warning("No predictions or targets were collected (or GT not available). Skipping mAP calculation.")
        # Add placeholder metrics
        metrics["eval_map"] = 0.0
        metrics["eval_map_50"] = 0.0
        metrics["eval_map_75"] = 0.0
        return metrics

    if len(all_predictions) != len(all_targets):
         logger.error(f"Mismatch between number of predictions ({len(all_predictions)}) and targets ({len(all_targets)}). Cannot calculate mAP.")
         # Add placeholder metrics
         metrics["eval_map"] = 0.0
         metrics["eval_map_50"] = 0.0
         metrics["eval_map_75"] = 0.0
         return metrics


    logger.info("Calculating mAP using torchmetrics...")
    metric_device = torch.device('cpu')
    logger.info(f"Using device '{metric_device}' for mAP calculation.")

    try:
        metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(metric_device)

        preds_on_device = [{k: v.to(metric_device) for k, v in p.items()} for p in all_predictions]
        targets_on_device = [{k: v.to(metric_device) for k, v in t.items()} for t in all_targets]

        metric.update(preds_on_device, targets_on_device)
        map_results = metric.compute()

        logger.info("--- mAP Results (COCO Standard IoUs) ---")
        main_map_keys = ['map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large']
        main_mar_keys = ['mar_1', 'mar_10', 'mar_100', 'mar_small', 'mar_medium', 'mar_large']

        for key in main_map_keys + main_mar_keys:
            if key in map_results:
                value = map_results[key].item() # Get scalar value from tensor
                metrics[f"eval_{key}"] = round(value, 4)
                logger.info(f"{key:<12}: {metrics[f'eval_{key}']:.4f}")

        # --- Per-Class AP Logging (Robust check added) ---
        if 'map_per_class' in map_results and map_results['map_per_class'] is not None:
            map_per_class_tensor = map_results['map_per_class'] # Keep as tensor initially
            classes_tensor = map_results.get('classes', None)

            # *** MODIFIED CHECK ***
            # Check if both are 1D tensors and have matching lengths before zipping
            valid_per_class_data = (
                classes_tensor is not None and isinstance(classes_tensor, torch.Tensor) and classes_tensor.ndim == 1 and
                map_per_class_tensor is not None and isinstance(map_per_class_tensor, torch.Tensor) and map_per_class_tensor.ndim == 1 and
                len(classes_tensor) == len(map_per_class_tensor)
            )
            # *** END MODIFIED CHECK ***

            if valid_per_class_data:
                map_per_class_np = map_per_class_tensor.cpu().numpy()
                classes_np = classes_tensor.cpu().numpy() # Convert class indices to numpy

                logger.info("--- AP per Class ---")
                found_person_ap = False
                for cls_idx_np, cls_ap_np in zip(classes_np, map_per_class_np):
                    cls_idx = int(cls_idx_np) # Ensure it's a Python int
                    cls_ap = cls_ap_np.item() # Get scalar value
                    metric_key = f"eval_ap_class_{cls_idx}"
                    metrics[metric_key] = round(cls_ap, 4)
                    logger.info(f"Class {cls_idx:<5}: {metrics[metric_key]:.4f}")

                    if cls_idx == person_class_index:
                        metrics["eval_ap_person"] = metrics[metric_key]
                        logger.info(f"  -> Person AP: {metrics['eval_ap_person']:.4f}")
                        found_person_ap = True

                if not found_person_ap:
                     logger.warning(f"Person class index {person_class_index} not found in computed per-class AP results.")
                     metrics["eval_ap_person"] = 0.0 # Add placeholder if not found

            else:
                # Fallback if classes tensor isn't valid or doesn't match map_per_class
                logger.warning("Could not parse per-class AP results as expected (check tensor dimensions/types). Attempting fallback for person AP.")
                # Try to extract person AP based on index if map_per_class is usable
                if map_per_class_tensor is not None and isinstance(map_per_class_tensor, torch.Tensor):
                    try:
                        # Handle scalar case (0-d tensor) - likely only one class result
                        if map_per_class_tensor.ndim == 0:
                            person_ap_value = map_per_class_tensor.item()
                            metrics["eval_ap_person"] = round(person_ap_value, 4)
                            logger.info(f"AP for Person Class (Index {person_class_index} - Fallback, scalar result): {metrics['eval_ap_person']:.4f}")
                        # Handle 1-d tensor case, check index bounds
                        elif map_per_class_tensor.ndim == 1 and person_class_index < len(map_per_class_tensor):
                            person_ap_value = map_per_class_tensor[person_class_index].item()
                            metrics["eval_ap_person"] = round(person_ap_value, 4)
                            logger.info(f"AP for Person Class (Index {person_class_index} - Fallback): {metrics['eval_ap_person']:.4f}")
                        else:
                            logger.warning(f"Cannot determine person AP from map_per_class tensor with shape {map_per_class_tensor.shape}")
                            metrics["eval_ap_person"] = 0.0
                    except IndexError:
                        logger.warning(f"Person class index {person_class_index} out of bounds for map_per_class tensor during fallback.")
                        metrics["eval_ap_person"] = 0.0
                    except Exception as fallback_err:
                        logger.warning(f"Error during fallback person AP extraction: {fallback_err}")
                        metrics["eval_ap_person"] = 0.0
                else:
                     logger.warning("map_per_class tensor not available or not a tensor for fallback.")
                     metrics["eval_ap_person"] = 0.0


    except Exception as e:
        logger.error(f"Failed to compute mAP: {e}", exc_info=True)
        # Add placeholder metrics
        metrics["eval_map"] = 0.0
        metrics["eval_map_50"] = 0.0
        metrics["eval_map_75"] = 0.0
        metrics["eval_ap_person"] = 0.0 # Add placeholder here too

    # --- Deprecated TP/FP/FN based metrics (optional: keep for reference or remove) ---
    metrics["eval_total_gt_boxes"] = results.get('total_gt_boxes', 0)
    metrics["eval_precision_single_iou"] = 0.0 # Mark as deprecated/unused
    metrics["eval_recall_single_iou"] = 0.0
    metrics["eval_f1_score_single_iou"] = 0.0
    metrics["eval_total_tp_single_iou"] = 0
    metrics["eval_total_fp_single_iou"] = 0
    metrics["eval_total_fn_single_iou"] = 0
    logger.info("Note: Precision/Recall/F1 based on single IoU threshold are deprecated in favor of mAP.")

    return metrics
