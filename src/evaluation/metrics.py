import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Structure: {(frame_idx, cam_id): [(object_id, x, y, w, h), ...], ...}
GroundTruthData = Dict[Tuple[int, str], List[Tuple[int, float, float, float, float]]]


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
                        frame_id = int(parts[0]) - 1
                        if frame_id >= num_frames_to_load:
                            continue

                        obj_id = int(parts[1])
                        # Convert gt format (top-left x, top-left y, width, height) to center x, center y, width, height
                        bb_left = float(parts[2])
                        bb_top = float(parts[3])
                        bb_width = float(parts[4])
                        bb_height = float(parts[5])

                        # Basic validation
                        if bb_width <= 0 or bb_height <= 0:
                            logger.debug(
                                f"Skipping GT box with non-positive dimensions in {cam_id}, frame {frame_id}: "
                                f"w={bb_width}, h={bb_height}")
                            continue

                        center_x = bb_left + bb_width / 2
                        center_y = bb_top + bb_height / 2

                        gt_data[(frame_id, cam_id)].append((obj_id, center_x, center_y, bb_width, bb_height))

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
    # Convert xywh to xyxy (top-left, bottom-right)
    boxA_xyxy = [boxA_xywh[0] - boxA_xywh[2] / 2, boxA_xywh[1] - boxA_xywh[3] / 2,
                 boxA_xywh[0] + boxA_xywh[2] / 2, boxA_xywh[1] + boxA_xywh[3] / 2]
    boxB_xyxy = [boxB_xywh[0] - boxB_xywh[2] / 2, boxB_xywh[1] - boxB_xywh[3] / 2,
                 boxB_xywh[0] + boxB_xywh[2] / 2, boxB_xywh[1] + boxB_xywh[3] / 2]

    x_left = max(boxA_xyxy[0], boxB_xyxy[0])
    y_top = max(boxA_xyxy[1], boxB_xyxy[1])
    x_right = min(boxA_xyxy[2], boxB_xyxy[2])
    y_bottom = min(boxA_xyxy[3], boxB_xyxy[3])

    # Compute the area of intersection rectangle
    intersection_w = x_right - x_left
    intersection_h = y_bottom - y_top
    if intersection_w <= 0 or intersection_h <= 0:
        return 0.0
    intersection_area = intersection_w * intersection_h

    # Compute the area of both bounding boxes
    boxA_area = boxA_xywh[2] * boxA_xywh[3]
    boxB_area = boxB_xywh[2] * boxB_xywh[3]

    # Compute the area of union
    union_area = boxA_area + boxB_area - intersection_area

    if union_area <= 0:
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

    match_indices = np.argwhere(iou_matrix >= iou_threshold)
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


def calculate_aggregate_metrics(
        results: Dict[str, Any],
        active_camera_ids: List[str]
) -> Dict[str, Any]:
    """Calculates aggregate and evaluation metrics from raw processing results."""
    metrics = {}
    logger.info("Calculating aggregate metrics...")

    frame_counter = results.get('frame_counter', 0)
    if frame_counter == 0:
        logger.warning("No frames were processed successfully. No metrics to calculate.")
        return metrics

    # --- Performance Metrics ---
    total_processing_time_sec = results.get('total_processing_time_sec', 0)
    avg_inference_time_ms = results.get('total_inference_time_ms', 0) / frame_counter
    avg_detections = results.get('total_person_detections', 0) / frame_counter
    processing_fps = frame_counter / total_processing_time_sec if total_processing_time_sec > 0 else 0

    metrics["total_frames_processed"] = frame_counter
    metrics["unique_frame_indices_processed"] = results.get('unique_frame_indices_processed', 0)
    metrics["total_processing_time_sec"] = round(total_processing_time_sec, 2)
    metrics["avg_inference_time_ms_per_frame"] = round(avg_inference_time_ms, 2)
    metrics["total_person_detections"] = results.get('total_person_detections', 0)
    metrics["avg_detections_per_frame"] = round(avg_detections, 2)
    metrics["processing_fps"] = round(processing_fps, 2)

    logger.info(f"Total Frames Processed: {metrics['total_frames_processed']}")
    logger.info(f"Unique Frame Indices Processed: {metrics['unique_frame_indices_processed']}")
    logger.info(f"Total Processing Time: {metrics['total_processing_time_sec']:.2f} seconds")
    logger.info(f"Overall Average Inference Time: {metrics['avg_inference_time_ms_per_frame']:.2f} ms/frame")
    logger.info(f"Overall Average Person Detections: {metrics['avg_detections_per_frame']:.2f} per frame")
    logger.info(f"Overall Processing FPS: {metrics['processing_fps']:.2f} frames/sec")
    logger.info(f"Total Person Detections Found: {metrics['total_person_detections']}")

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

        metrics[f"frames_cam_{cam_id}"] = cam_frames
        metrics[f"detections_cam_{cam_id}"] = cam_dets
        metrics[f"avg_inf_ms_cam_{cam_id}"] = round(avg_inf_cam, 2)
        metrics[f"avg_dets_cam_{cam_id}"] = round(avg_dets_cam, 2)

    # --- Evaluation Metrics (if GT was available) ---
    if results.get('gt_available', False):
        total_tp = results.get('total_tp', 0)
        total_fp = results.get('total_fp', 0)
        total_fn = results.get('total_fn', 0)
        total_gt = results.get('total_gt_boxes', 0)  # Should equal total_tp + total_fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics["eval_total_gt_boxes"] = total_gt
        metrics["eval_total_tp"] = total_tp
        metrics["eval_total_fp"] = total_fp
        metrics["eval_total_fn"] = total_fn
        metrics["eval_precision"] = round(precision, 4)
        metrics["eval_recall"] = round(recall, 4)
        metrics["eval_f1_score"] = round(f1_score, 4)

        logger.info("--- Evaluation Results ---")
        logger.info(f"Total Ground Truth Boxes: {metrics['eval_total_gt_boxes']}")
        logger.info(f"Total True Positives (TP): {metrics['eval_total_tp']}")
        logger.info(f"Total False Positives (FP): {metrics['eval_total_fp']}")
        logger.info(f"Total False Negatives (FN): {metrics['eval_total_fn']}")
        logger.info(f"Precision: {metrics['eval_precision']:.4f}")
        logger.info(f"Recall: {metrics['eval_recall']:.4f}")
        logger.info(f"F1 Score: {metrics['eval_f1_score']:.4f}")
    else:
        logger.warning("Ground truth data was not available or failed to load. Skipping evaluation metrics.")
        metrics["eval_total_gt_boxes"] = 0
        metrics["eval_total_tp"] = 0
        metrics["eval_total_fp"] = 0
        metrics["eval_total_fn"] = 0
        metrics["eval_precision"] = 0.0
        metrics["eval_recall"] = 0.0
        metrics["eval_f1_score"] = 0.0

    return metrics
