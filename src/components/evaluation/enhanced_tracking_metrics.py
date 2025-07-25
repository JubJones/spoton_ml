import logging
from typing import Dict, Any, Tuple, List, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np

# Optional imports for advanced metrics
try:
    import pandas as pd
    import motmetrics as mm
    MOTMETRICS_AVAILABLE = True
except ImportError:
    pd = None
    mm = None
    MOTMETRICS_AVAILABLE = False

logger = logging.getLogger(__name__)

TrackingResultSummary = Dict[str, Any]
RawTrackerOutputs = Dict[Tuple[int, str], np.ndarray]  # {(frame_idx, cam_id): tracker_output_array}
GroundTruthData = Dict[Tuple[int, str], List[Tuple[int, float, float, float, float]]]  # {(frame_idx, cam_id): [(obj_id, cx, cy, w, h), ...]}

@dataclass
class TrackingQualityAnalysis:
    """Enhanced tracking quality analysis results"""
    # Standard MOT metrics
    mota: float
    motp: float
    idf1: float
    idp: float
    idr: float
    
    # Identity analysis
    id_switches: int
    id_switch_rate: float
    fragmentations: int
    fragmentation_rate: float
    
    # Detection quality
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    
    # Trajectory analysis
    avg_trajectory_length: float
    trajectory_consistency_score: float
    temporal_stability_score: float
    
    # Camera-specific metrics
    per_camera_metrics: Dict[str, Dict[str, float]]
    
    # ID mapping analysis
    unique_gt_ids: int
    unique_pred_ids: int
    id_mapping_quality: float


def calculate_enhanced_tracking_metrics(
    raw_outputs: RawTrackerOutputs, 
    ground_truth_data: GroundTruthData,
    active_camera_ids: List[str]
) -> TrackingQualityAnalysis:
    """
    Calculate comprehensive tracking quality metrics including failure analysis.
    
    Args:
        raw_outputs: Dictionary mapping (frame_idx, cam_id) to tracker output arrays
        ground_truth_data: Dictionary mapping (frame_idx, cam_id) to GT data
        active_camera_ids: List of active camera IDs
    
    Returns:
        TrackingQualityAnalysis object with comprehensive metrics
    """
    logger.info("Calculating enhanced tracking quality metrics...")
    
    # Initialize metrics
    standard_metrics = _calculate_standard_mot_metrics(raw_outputs, ground_truth_data, active_camera_ids)
    identity_metrics = _calculate_identity_metrics(raw_outputs, ground_truth_data, active_camera_ids)
    trajectory_metrics = _calculate_trajectory_metrics(raw_outputs, ground_truth_data, active_camera_ids)
    camera_metrics = _calculate_per_camera_metrics(raw_outputs, ground_truth_data, active_camera_ids)
    
    return TrackingQualityAnalysis(
        **standard_metrics,
        **identity_metrics,
        **trajectory_metrics,
        per_camera_metrics=camera_metrics
    )


def _calculate_standard_mot_metrics(
    raw_outputs: RawTrackerOutputs,
    ground_truth_data: GroundTruthData,
    active_camera_ids: List[str]
) -> Dict[str, float]:
    """Calculate standard MOT metrics using motmetrics library"""
    default_metrics = {
        'mota': 0.0, 'motp': 0.0, 'idf1': 0.0, 'idp': 0.0, 'idr': 0.0,
        'false_positives': 0, 'false_negatives': 0, 'precision': 0.0, 'recall': 0.0
    }
    
    if not MOTMETRICS_AVAILABLE:
        logger.warning("motmetrics not available, returning default MOT metrics")
        return default_metrics
    
    try:
        acc = mm.MOTAccumulator(auto_id=True)
        
        # Get all frame indices
        all_frames = set()
        all_frames.update(frame_idx for frame_idx, _ in raw_outputs.keys())
        all_frames.update(frame_idx for frame_idx, _ in ground_truth_data.keys())
        
        for frame_idx in sorted(all_frames):
            frame_gt_ids = []
            frame_gt_boxes = []
            frame_hyp_ids = []
            frame_hyp_boxes = []
            
            # Collect GT data across all cameras for this frame
            for cam_id in active_camera_ids:
                gt_data = ground_truth_data.get((frame_idx, cam_id), [])
                for obj_id, cx, cy, w, h in gt_data:
                    if w > 0 and h > 0:
                        x1, y1 = cx - w/2, cy - h/2
                        frame_gt_ids.append(obj_id)
                        frame_gt_boxes.append([x1, y1, w, h])
            
            # Collect tracker predictions across all cameras for this frame
            for cam_id in active_camera_ids:
                tracker_output = raw_outputs.get((frame_idx, cam_id))
                if tracker_output is not None and tracker_output.size > 0:
                    for row in tracker_output:
                        if len(row) >= 5:
                            x1, y1, x2, y2, track_id = row[:5]
                            w, h = x2 - x1, y2 - y1
                            if w > 0 and h > 0:
                                frame_hyp_ids.append(int(track_id))
                                frame_hyp_boxes.append([x1, y1, w, h])
            
            # Update accumulator
            if frame_gt_boxes and frame_hyp_boxes:
                distances = mm.distances.iou_matrix(frame_gt_boxes, frame_hyp_boxes, max_iou=0.5)
                acc.update(frame_gt_ids, frame_hyp_ids, distances)
            elif frame_gt_ids:
                acc.update(frame_gt_ids, [], [])
            elif frame_hyp_ids:
                acc.update([], frame_hyp_ids, [])
        
        # Compute metrics
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'idp', 'idr', 'num_false_positives', 'num_misses'],
                           name='enhanced_summary')
        
        if not summary.empty:
            result = {}
            metric_mapping = {
                'mota': 'mota', 'motp': 'motp', 'idf1': 'idf1', 'idp': 'idp', 'idr': 'idr',
                'num_false_positives': 'false_positives', 'num_misses': 'false_negatives'
            }
            
            for mot_name, result_name in metric_mapping.items():
                try:
                    value = summary.loc['enhanced_summary', mot_name]
                    result[result_name] = float(value) if not pd.isna(value) else 0.0
                except (KeyError, IndexError):
                    result[result_name] = 0.0
            
            # Calculate precision and recall
            tp = max(0, result.get('false_positives', 0) + result.get('false_negatives', 0))  # Approximation
            fp = result.get('false_positives', 0)
            fn = result.get('false_negatives', 0)
            
            result['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            result['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            return result
        else:
            return default_metrics
            
    except Exception as e:
        logger.warning(f"Error calculating standard MOT metrics: {e}")
        return default_metrics


def _calculate_identity_metrics(
    raw_outputs: RawTrackerOutputs,
    ground_truth_data: GroundTruthData,
    active_camera_ids: List[str]
) -> Dict[str, Any]:
    """Calculate identity-related metrics including ID switches and fragmentations"""
    
    # Track GT ID to predicted ID mapping over time
    gt_to_pred_history = defaultdict(dict)  # {gt_id: {frame: pred_id}}
    id_switches = 0
    fragmentations = 0
    
    # Get all frame indices
    all_frames = sorted(set(frame_idx for frame_idx, _ in raw_outputs.keys()))
    
    for frame_idx in all_frames:
        # Match GT to predictions for this frame
        matches = _match_gt_to_predictions_frame(frame_idx, raw_outputs, ground_truth_data, active_camera_ids)
        
        for gt_id, pred_id in matches.items():
            # Check for ID switches
            if gt_id in gt_to_pred_history:
                recent_frames = [f for f in gt_to_pred_history[gt_id].keys() if frame_idx - f <= 3]
                if recent_frames:
                    recent_pred_ids = [gt_to_pred_history[gt_id][f] for f in recent_frames]
                    most_common_pred_id = Counter(recent_pred_ids).most_common(1)[0][0]
                    
                    if pred_id != most_common_pred_id:
                        id_switches += 1
            
            gt_to_pred_history[gt_id][frame_idx] = pred_id
    
    # Calculate fragmentations (gaps in tracking)
    for gt_id, frame_dict in gt_to_pred_history.items():
        frames = sorted(frame_dict.keys())
        for i in range(1, len(frames)):
            if frames[i] - frames[i-1] > 1:  # Gap detected
                fragmentations += 1
    
    # Calculate unique IDs
    unique_gt_ids = len(set(
        obj_id for gt_data in ground_truth_data.values() 
        for obj_id, _, _, _, _ in gt_data
    ))
    
    unique_pred_ids = len(set(
        int(row[4]) for output in raw_outputs.values()
        if output.size > 0 for row in output if len(row) >= 5
    ))
    
    # Calculate ID mapping quality
    total_assignments = sum(len(frame_dict) for frame_dict in gt_to_pred_history.values())
    id_mapping_quality = 1.0 - (id_switches / max(total_assignments, 1))
    
    total_frames = len(all_frames)
    id_switch_rate = id_switches / max(total_frames, 1)
    fragmentation_rate = fragmentations / max(unique_gt_ids, 1)
    
    return {
        'id_switches': id_switches,
        'id_switch_rate': id_switch_rate,
        'fragmentations': fragmentations,
        'fragmentation_rate': fragmentation_rate,
        'unique_gt_ids': unique_gt_ids,
        'unique_pred_ids': unique_pred_ids,
        'id_mapping_quality': id_mapping_quality
    }


def _calculate_trajectory_metrics(
    raw_outputs: RawTrackerOutputs,
    ground_truth_data: GroundTruthData,
    active_camera_ids: List[str]
) -> Dict[str, float]:
    """Calculate trajectory-related quality metrics"""
    
    # Calculate trajectory lengths for GT and predictions
    gt_trajectories = defaultdict(list)  # {gt_id: [frame_indices]}
    pred_trajectories = defaultdict(list)  # {pred_id: [frame_indices]}
    
    # Collect GT trajectories
    for (frame_idx, cam_id), gt_data in ground_truth_data.items():
        for obj_id, _, _, _, _ in gt_data:
            gt_trajectories[obj_id].append(frame_idx)
    
    # Collect prediction trajectories
    for (frame_idx, cam_id), output in raw_outputs.items():
        if output.size > 0:
            for row in output:
                if len(row) >= 5:
                    pred_id = int(row[4])
                    pred_trajectories[pred_id].append(frame_idx)
    
    # Calculate average trajectory lengths
    gt_lengths = [len(set(frames)) for frames in gt_trajectories.values()]
    pred_lengths = [len(set(frames)) for frames in pred_trajectories.values()]
    
    avg_gt_trajectory_length = np.mean(gt_lengths) if gt_lengths else 0.0
    avg_pred_trajectory_length = np.mean(pred_lengths) if pred_lengths else 0.0
    avg_trajectory_length = (avg_gt_trajectory_length + avg_pred_trajectory_length) / 2
    
    # Calculate trajectory consistency (fewer gaps = higher consistency)
    total_gaps = 0
    total_trajectories = 0
    
    for frames in pred_trajectories.values():
        if len(frames) > 1:
            frames_sorted = sorted(set(frames))
            gaps = sum(1 for i in range(1, len(frames_sorted)) 
                      if frames_sorted[i] - frames_sorted[i-1] > 1)
            total_gaps += gaps
            total_trajectories += 1
    
    trajectory_consistency_score = 1.0 - (total_gaps / max(total_trajectories, 1))
    
    # Temporal stability (consistency of detections over time)
    frame_detection_counts = []
    all_frames = sorted(set(frame_idx for frame_idx, _ in raw_outputs.keys()))
    
    for frame_idx in all_frames:
        total_detections = 0
        for cam_id in active_camera_ids:
            output = raw_outputs.get((frame_idx, cam_id))
            if output is not None and output.size > 0:
                total_detections += len(output)
        frame_detection_counts.append(total_detections)
    
    temporal_stability_score = 1.0 - (np.std(frame_detection_counts) / (np.mean(frame_detection_counts) + 1e-6))
    temporal_stability_score = max(0.0, min(1.0, temporal_stability_score))  # Clamp to [0, 1]
    
    return {
        'avg_trajectory_length': avg_trajectory_length,
        'trajectory_consistency_score': trajectory_consistency_score,
        'temporal_stability_score': temporal_stability_score
    }


def _calculate_per_camera_metrics(
    raw_outputs: RawTrackerOutputs,
    ground_truth_data: GroundTruthData,
    active_camera_ids: List[str]
) -> Dict[str, Dict[str, float]]:
    """Calculate metrics for each camera separately"""
    per_camera_metrics = {}
    
    for cam_id in active_camera_ids:
        # Filter data for this camera
        cam_outputs = {k: v for k, v in raw_outputs.items() if k[1] == cam_id}
        cam_gt_data = {k: v for k, v in ground_truth_data.items() if k[1] == cam_id}
        
        if not cam_outputs and not cam_gt_data:
            per_camera_metrics[cam_id] = {
                'detections': 0, 'avg_detections_per_frame': 0.0,
                'unique_ids': 0, 'trajectory_count': 0
            }
            continue
        
        # Count detections
        total_detections = sum(len(output) for output in cam_outputs.values() if output.size > 0)
        total_frames = len(cam_outputs)
        avg_detections_per_frame = total_detections / max(total_frames, 1)
        
        # Count unique IDs
        unique_ids = len(set(
            int(row[4]) for output in cam_outputs.values()
            if output.size > 0 for row in output if len(row) >= 5
        ))
        
        # Count GT trajectories
        gt_ids_in_camera = set(
            obj_id for gt_data in cam_gt_data.values()
            for obj_id, _, _, _, _ in gt_data
        )
        trajectory_count = len(gt_ids_in_camera)
        
        per_camera_metrics[cam_id] = {
            'detections': total_detections,
            'avg_detections_per_frame': avg_detections_per_frame,
            'unique_ids': unique_ids,
            'trajectory_count': trajectory_count
        }
    
    return per_camera_metrics


def _match_gt_to_predictions_frame(
    frame_idx: int,
    raw_outputs: RawTrackerOutputs,
    ground_truth_data: GroundTruthData,
    active_camera_ids: List[str],
    iou_threshold: float = 0.5
) -> Dict[int, int]:
    """Match ground truth to predictions for a single frame using IoU"""
    
    gt_boxes = {}  # {gt_id: (x1, y1, x2, y2)}
    pred_boxes = {}  # {pred_id: (x1, y1, x2, y2)}
    
    # Collect GT boxes for this frame across all cameras
    for cam_id in active_camera_ids:
        gt_data = ground_truth_data.get((frame_idx, cam_id), [])
        for obj_id, cx, cy, w, h in gt_data:
            if w > 0 and h > 0:
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2
                gt_boxes[obj_id] = (x1, y1, x2, y2)
    
    # Collect prediction boxes for this frame across all cameras
    for cam_id in active_camera_ids:
        output = raw_outputs.get((frame_idx, cam_id))
        if output is not None and output.size > 0:
            for row in output:
                if len(row) >= 5:
                    x1, y1, x2, y2, pred_id = row[:5]
                    pred_boxes[int(pred_id)] = (x1, y1, x2, y2)
    
    # Match using IoU
    matches = {}
    if not gt_boxes or not pred_boxes:
        return matches
    
    # Simple greedy matching (could be improved with Hungarian algorithm)
    used_pred_ids = set()
    for gt_id, gt_box in gt_boxes.items():
        best_iou = 0.0
        best_pred_id = None
        
        for pred_id, pred_box in pred_boxes.items():
            if pred_id in used_pred_ids:
                continue
            
            iou = _calculate_iou(gt_box, pred_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pred_id = pred_id
        
        if best_pred_id is not None:
            matches[gt_id] = best_pred_id
            used_pred_ids.add(best_pred_id)
    
    return matches


def _calculate_iou(box1: Tuple[float, float, float, float], 
                  box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def analyze_tracker_specific_performance(
    tracker_results: Dict[str, Tuple[RawTrackerOutputs, TrackingQualityAnalysis]],
    ground_truth_data: GroundTruthData,
    active_camera_ids: List[str]
) -> Dict[str, Any]:
    """
    Analyze and compare performance across different tracker types.
    
    Args:
        tracker_results: Dict mapping tracker names to their (outputs, analysis) tuples
        ground_truth_data: Ground truth data
        active_camera_ids: List of active camera IDs
    
    Returns:
        Dictionary containing comparative analysis results
    """
    logger.info(f"Analyzing performance across {len(tracker_results)} trackers...")
    
    analysis = {
        'tracker_comparison': {},
        'best_performers': {},
        'failure_analysis': {},
        'recommendations': {}
    }
    
    # Compare trackers on key metrics
    metrics_to_compare = ['mota', 'motp', 'idf1', 'id_switches', 'trajectory_consistency_score']
    
    for metric in metrics_to_compare:
        metric_values = {}
        for tracker_name, (_, quality_analysis) in tracker_results.items():
            metric_values[tracker_name] = getattr(quality_analysis, metric, 0.0)
        
        # Find best and worst performers
        if metric in ['id_switches']:  # Lower is better
            best_tracker = min(metric_values.items(), key=lambda x: x[1])
            worst_tracker = max(metric_values.items(), key=lambda x: x[1])
        else:  # Higher is better
            best_tracker = max(metric_values.items(), key=lambda x: x[1])
            worst_tracker = min(metric_values.items(), key=lambda x: x[1])
        
        analysis['tracker_comparison'][metric] = metric_values
        analysis['best_performers'][metric] = {
            'best': {'tracker': best_tracker[0], 'value': best_tracker[1]},
            'worst': {'tracker': worst_tracker[0], 'value': worst_tracker[1]}
        }
    
    # Analyze failure patterns by tracker
    for tracker_name, (raw_outputs, quality_analysis) in tracker_results.items():
        failure_stats = {
            'id_switches_normalized': quality_analysis.id_switches / max(len(raw_outputs), 1),
            'fragmentation_rate': quality_analysis.fragmentation_rate,
            'false_positive_rate': quality_analysis.false_positives / max(len(raw_outputs), 1),
            'false_negative_rate': quality_analysis.false_negatives / max(len(raw_outputs), 1)
        }
        analysis['failure_analysis'][tracker_name] = failure_stats
    
    # Generate recommendations
    recommendations = []
    
    # Overall best tracker
    overall_scores = {}
    for tracker_name, (_, quality_analysis) in tracker_results.items():
        # Weighted combination of key metrics
        score = (
            quality_analysis.mota * 0.3 +
            quality_analysis.idf1 * 0.3 +
            quality_analysis.trajectory_consistency_score * 0.2 +
            (1.0 - quality_analysis.id_switch_rate) * 0.2
        )
        overall_scores[tracker_name] = score
    
    best_overall = max(overall_scores.items(), key=lambda x: x[1])
    recommendations.append(f"Overall best performer: {best_overall[0]} (score: {best_overall[1]:.3f})")
    
    # Specific use case recommendations
    best_identity = analysis['best_performers']['idf1']['best']['tracker']
    recommendations.append(f"Best for identity preservation: {best_identity}")
    
    best_accuracy = analysis['best_performers']['mota']['best']['tracker']
    recommendations.append(f"Best for detection accuracy: {best_accuracy}")
    
    analysis['recommendations']['summary'] = recommendations
    analysis['recommendations']['overall_ranking'] = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    
    return analysis