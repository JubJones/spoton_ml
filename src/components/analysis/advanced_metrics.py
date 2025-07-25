"""
Advanced Detection Metrics Collection System for Phase 1.3.

This module implements comprehensive detection performance measurement including:
1. Fine-grained mAP analysis (by distance, size, occlusion)
2. Precision-recall curves per scenario type
3. Confidence score distribution analysis
4. Detection latency and throughput metrics
5. Advanced statistical measures
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import cv2
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DetectionInstance:
    """Single detection instance with comprehensive metadata."""
    frame_idx: int
    scene_id: str
    camera_id: str
    person_id: int
    bbox_xyxy: List[float]  # Ground truth or prediction box
    confidence: Optional[float]  # None for ground truth
    is_ground_truth: bool
    
    # Context metadata
    object_size: str  # 'small', 'medium', 'large'
    distance_category: str  # 'close', 'medium', 'far'
    occlusion_level: str  # 'none', 'partial', 'heavy'
    lighting_condition: str  # 'day', 'night', 'transition'
    crowd_density: str  # 'low', 'medium', 'high'
    
    # Quality metrics
    bbox_area: float
    aspect_ratio: float
    edge_distance: float  # Distance from image edge
    blur_score: float
    

@dataclass
class PrecisionRecallMetrics:
    """Precision-recall analysis for specific scenario."""
    scenario_name: str
    precision_values: List[float]
    recall_values: List[float]
    average_precision: float
    f1_scores: List[float]
    best_f1_score: float
    optimal_threshold: float
    

@dataclass
class ConfidenceAnalysis:
    """Confidence score distribution analysis."""
    model_name: str
    true_positive_confidences: List[float]
    false_positive_confidences: List[float]
    confidence_histogram_bins: List[float]
    confidence_histogram_counts: List[int]
    optimal_threshold: float
    calibration_error: float  # Expected Calibration Error
    

@dataclass
class PerformanceProfileMetrics:
    """Detailed performance profiling metrics."""
    model_name: str
    
    # Latency metrics
    inference_times: List[float]
    preprocessing_times: List[float] 
    postprocessing_times: List[float]
    total_processing_times: List[float]
    
    # Throughput metrics
    frames_per_second: float
    detections_per_second: float
    
    # Statistical measures
    mean_inference_time: float
    std_inference_time: float
    p95_inference_time: float
    p99_inference_time: float
    

@dataclass
class AdvancedmAPMetrics:
    """Fine-grained mAP analysis across different conditions."""
    overall_map_50: float
    overall_map_75: float
    overall_map_50_95: float
    
    # Size-based mAP
    map_small_objects: float  # Area < 32²
    map_medium_objects: float  # 32² < Area < 96²
    map_large_objects: float  # Area > 96²
    
    # Distance-based mAP
    map_close_distance: float
    map_medium_distance: float
    map_far_distance: float
    
    # Occlusion-based mAP
    map_no_occlusion: float
    map_partial_occlusion: float
    map_heavy_occlusion: float
    
    # Lighting-based mAP
    map_day_lighting: float
    map_night_lighting: float
    map_transition_lighting: float
    
    # Crowd density-based mAP
    map_low_density: float
    map_medium_density: float
    map_high_density: float


class InstanceAnalyzer:
    """Analyzes detection instances and assigns metadata."""
    
    def __init__(self):
        self.size_thresholds = {'small': 32*32, 'large': 96*96}
        self.distance_thresholds = {'close': 0.3, 'far': 0.7}
        self.brightness_thresholds = {'day': 120, 'night': 80}
        self.density_thresholds = {'low': 2, 'high': 8}
        
    def analyze_detection_instance(self, bbox_xyxy: List[float], 
                                 frame: np.ndarray,
                                 frame_idx: int,
                                 scene_id: str,
                                 camera_id: str,
                                 person_id: int,
                                 confidence: Optional[float] = None,
                                 is_ground_truth: bool = True,
                                 num_persons_in_frame: int = 1) -> DetectionInstance:
        """Analyze single detection instance and extract metadata."""
        
        x1, y1, x2, y2 = bbox_xyxy
        height, width = frame.shape[:2]
        
        # Calculate basic metrics
        bbox_width = x2 - x1
        bbox_height = y2 - y1  
        bbox_area = bbox_width * bbox_height
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
        
        # Calculate distance from edges
        edge_distance = min(x1, y1, width - x2, height - y2) / min(width, height)
        
        # Analyze object size
        object_size = self._categorize_object_size(bbox_area)
        
        # Analyze distance (using bbox height as proxy)
        normalized_height = bbox_height / height
        distance_category = self._categorize_distance(normalized_height)
        
        # Analyze lighting condition
        lighting_condition = self._analyze_lighting(frame)
        
        # Analyze crowd density
        crowd_density = self._categorize_crowd_density(num_persons_in_frame, width * height)
        
        # Analyze occlusion (simplified - would need more sophisticated analysis)
        occlusion_level = self._estimate_occlusion(bbox_xyxy, frame)
        
        # Calculate blur score
        blur_score = self._calculate_blur_score(frame, bbox_xyxy)
        
        return DetectionInstance(
            frame_idx=frame_idx,
            scene_id=scene_id,
            camera_id=camera_id,
            person_id=person_id,
            bbox_xyxy=bbox_xyxy,
            confidence=confidence,
            is_ground_truth=is_ground_truth,
            object_size=object_size,
            distance_category=distance_category,
            occlusion_level=occlusion_level,
            lighting_condition=lighting_condition,
            crowd_density=crowd_density,
            bbox_area=bbox_area,
            aspect_ratio=aspect_ratio,
            edge_distance=edge_distance,
            blur_score=blur_score
        )
    
    def _categorize_object_size(self, area: float) -> str:
        """Categorize object by size following COCO standards."""
        if area < self.size_thresholds['small']:
            return 'small'
        elif area < self.size_thresholds['large']:
            return 'medium'
        else:
            return 'large'
    
    def _categorize_distance(self, normalized_height: float) -> str:
        """Categorize distance based on normalized bbox height."""
        if normalized_height > self.distance_thresholds['close']:
            return 'close'
        elif normalized_height < self.distance_thresholds['far']:
            return 'far'
        else:
            return 'medium'
    
    def _analyze_lighting(self, frame: np.ndarray) -> str:
        """Analyze lighting condition."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness > self.brightness_thresholds['day']:
            return 'day'
        elif avg_brightness < self.brightness_thresholds['night']:
            return 'night'
        else:
            return 'transition'
    
    def _categorize_crowd_density(self, num_persons: int, image_area: int) -> str:
        """Categorize crowd density."""
        density_score = num_persons / (image_area / 1_000_000)
        
        if density_score <= self.density_thresholds['low']:
            return 'low'
        elif density_score >= self.density_thresholds['high']:
            return 'high'
        else:
            return 'medium'
    
    def _estimate_occlusion(self, bbox_xyxy: List[float], frame: np.ndarray) -> str:
        """Estimate occlusion level (simplified approach)."""
        # This is a simplified approach - in practice would use more sophisticated methods
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        
        # Extract the region of interest
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 'heavy'
        
        # Simple edge-based analysis
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        if edge_ratio < 0.05:
            return 'heavy'
        elif edge_ratio < 0.15:
            return 'partial'
        else:
            return 'none'
    
    def _calculate_blur_score(self, frame: np.ndarray, bbox_xyxy: List[float]) -> float:
        """Calculate blur score using Laplacian variance."""
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.0
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        
        # Normalize to 0-1 scale
        blur_score = min(laplacian_var / 500.0, 1.0)
        return blur_score


class mAPCalculator:
    """Calculates fine-grained mAP metrics across different conditions."""
    
    def __init__(self, iou_thresholds: List[float] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_ap_for_category(self, gt_instances: List[DetectionInstance],
                                pred_instances: List[DetectionInstance],
                                iou_threshold: float = 0.5) -> float:
        """Calculate Average Precision for a specific category."""
        if not gt_instances:
            return 0.0
        
        # Sort predictions by confidence (descending)
        pred_instances_sorted = sorted(
            [p for p in pred_instances if not p.is_ground_truth],
            key=lambda x: x.confidence or 0.0, 
            reverse=True
        )
        
        if not pred_instances_sorted:
            return 0.0
        
        # Track which ground truth instances have been matched
        gt_matched = [False] * len(gt_instances)
        
        # Calculate precision and recall at each prediction
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        
        for pred in pred_instances_sorted:
            # Find best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_instances):
                if gt_matched[gt_idx]:
                    continue
                
                iou = self.calculate_iou(pred.bbox_xyxy, gt.bbox_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if it's a true positive
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / len(gt_instances) if len(gt_instances) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate Average Precision using the precision-recall curve
        if not precisions or not recalls:
            return 0.0
        
        # Add endpoints
        precisions = [1.0] + precisions + [0.0]
        recalls = [0.0] + recalls + [1.0]
        
        # Make precision monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # Calculate AP using trapezoidal rule
        ap = 0.0
        for i in range(len(recalls) - 1):
            ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]
        
        return ap
    
    def calculate_advanced_map(self, gt_instances: List[DetectionInstance],
                             pred_instances: List[DetectionInstance]) -> AdvancedmAPMetrics:
        """Calculate comprehensive mAP metrics across all conditions."""
        
        # Overall mAP at different IoU thresholds
        map_50 = self.calculate_ap_for_category(gt_instances, pred_instances, 0.5)
        map_75 = self.calculate_ap_for_category(gt_instances, pred_instances, 0.75)
        
        # Calculate mAP@[0.5:0.95]
        aps = []
        for iou_thresh in self.iou_thresholds:
            ap = self.calculate_ap_for_category(gt_instances, pred_instances, iou_thresh)
            aps.append(ap)
        map_50_95 = np.mean(aps)
        
        # Size-based mAP
        map_small = self._calculate_conditional_map(
            gt_instances, pred_instances, 'object_size', 'small'
        )
        map_medium = self._calculate_conditional_map(
            gt_instances, pred_instances, 'object_size', 'medium'
        )
        map_large = self._calculate_conditional_map(
            gt_instances, pred_instances, 'object_size', 'large'
        )
        
        # Distance-based mAP
        map_close = self._calculate_conditional_map(
            gt_instances, pred_instances, 'distance_category', 'close'
        )
        map_medium_dist = self._calculate_conditional_map(
            gt_instances, pred_instances, 'distance_category', 'medium'
        )
        map_far = self._calculate_conditional_map(
            gt_instances, pred_instances, 'distance_category', 'far'
        )
        
        # Occlusion-based mAP
        map_no_occ = self._calculate_conditional_map(
            gt_instances, pred_instances, 'occlusion_level', 'none'
        )
        map_partial_occ = self._calculate_conditional_map(
            gt_instances, pred_instances, 'occlusion_level', 'partial'
        )
        map_heavy_occ = self._calculate_conditional_map(
            gt_instances, pred_instances, 'occlusion_level', 'heavy'
        )
        
        # Lighting-based mAP
        map_day = self._calculate_conditional_map(
            gt_instances, pred_instances, 'lighting_condition', 'day'
        )
        map_night = self._calculate_conditional_map(
            gt_instances, pred_instances, 'lighting_condition', 'night'
        )
        map_transition = self._calculate_conditional_map(
            gt_instances, pred_instances, 'lighting_condition', 'transition'
        )
        
        # Density-based mAP
        map_low_density = self._calculate_conditional_map(
            gt_instances, pred_instances, 'crowd_density', 'low'
        )
        map_medium_density = self._calculate_conditional_map(
            gt_instances, pred_instances, 'crowd_density', 'medium'
        )
        map_high_density = self._calculate_conditional_map(
            gt_instances, pred_instances, 'crowd_density', 'high'
        )
        
        return AdvancedmAPMetrics(
            overall_map_50=map_50,
            overall_map_75=map_75,
            overall_map_50_95=map_50_95,
            map_small_objects=map_small,
            map_medium_objects=map_medium,
            map_large_objects=map_large,
            map_close_distance=map_close,
            map_medium_distance=map_medium_dist,
            map_far_distance=map_far,
            map_no_occlusion=map_no_occ,
            map_partial_occlusion=map_partial_occ,
            map_heavy_occlusion=map_heavy_occ,
            map_day_lighting=map_day,
            map_night_lighting=map_night,
            map_transition_lighting=map_transition,
            map_low_density=map_low_density,
            map_medium_density=map_medium_density,
            map_high_density=map_high_density
        )
    
    def _calculate_conditional_map(self, gt_instances: List[DetectionInstance],
                                 pred_instances: List[DetectionInstance],
                                 condition_attr: str,
                                 condition_value: str) -> float:
        """Calculate mAP for instances matching specific condition."""
        filtered_gt = [inst for inst in gt_instances 
                      if getattr(inst, condition_attr) == condition_value]
        filtered_pred = [inst for inst in pred_instances 
                        if getattr(inst, condition_attr) == condition_value]
        
        if not filtered_gt:
            return 0.0
        
        return self.calculate_ap_for_category(filtered_gt, filtered_pred)


class ConfidenceAnalyzer:
    """Analyzes confidence score distributions and calibration."""
    
    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
        
    def analyze_confidence_distribution(self, gt_instances: List[DetectionInstance],
                                      pred_instances: List[DetectionInstance],
                                      model_name: str,
                                      iou_threshold: float = 0.5) -> ConfidenceAnalysis:
        """Analyze confidence score distribution and calibration."""
        
        # Separate predictions by whether they match ground truth
        tp_confidences = []
        fp_confidences = []
        
        # Create mapping of ground truth for quick lookup
        gt_boxes_dict = defaultdict(list)
        for gt in gt_instances:
            key = (gt.frame_idx, gt.scene_id, gt.camera_id)
            gt_boxes_dict[key].append(gt)
        
        # Classify predictions as TP or FP
        for pred in [p for p in pred_instances if not p.is_ground_truth and p.confidence is not None]:
            key = (pred.frame_idx, pred.scene_id, pred.camera_id)
            gt_list = gt_boxes_dict.get(key, [])
            
            is_tp = False
            for gt in gt_list:
                iou = self._calculate_iou(pred.bbox_xyxy, gt.bbox_xyxy)
                if iou >= iou_threshold:
                    is_tp = True
                    break
            
            if is_tp:
                tp_confidences.append(pred.confidence)
            else:
                fp_confidences.append(pred.confidence)
        
        # Create confidence histogram
        all_confidences = tp_confidences + fp_confidences
        if all_confidences:
            hist_counts, hist_bins = np.histogram(all_confidences, bins=self.num_bins, range=(0, 1))
            hist_bins = hist_bins[:-1]  # Remove last bin edge
        else:
            hist_counts, hist_bins = [], []
        
        # Find optimal threshold
        optimal_threshold = self._find_optimal_threshold(tp_confidences, fp_confidences)
        
        # Calculate calibration error
        calibration_error = self._calculate_calibration_error(tp_confidences, fp_confidences)
        
        return ConfidenceAnalysis(
            model_name=model_name,
            true_positive_confidences=tp_confidences,
            false_positive_confidences=fp_confidences,
            confidence_histogram_bins=hist_bins.tolist(),
            confidence_histogram_counts=hist_counts.tolist(),
            optimal_threshold=optimal_threshold,
            calibration_error=calibration_error
        )
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _find_optimal_threshold(self, tp_confidences: List[float], 
                              fp_confidences: List[float]) -> float:
        """Find optimal confidence threshold based on F1 score."""
        if not tp_confidences and not fp_confidences:
            return 0.5
        
        # Create labels (1 for TP, 0 for FP)
        y_true = [1] * len(tp_confidences) + [0] * len(fp_confidences)
        y_scores = tp_confidences + fp_confidences
        
        if not y_scores:
            return 0.5
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Calculate F1 scores
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Find threshold with best F1 score
        if len(f1_scores) > 0:
            best_idx = np.argmax(f1_scores)
            if best_idx < len(thresholds):
                return thresholds[best_idx]
        
        return 0.5
    
    def _calculate_calibration_error(self, tp_confidences: List[float], 
                                   fp_confidences: List[float]) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        if not tp_confidences and not fp_confidences:
            return 0.0
        
        # Combine all predictions with their correctness
        all_predictions = [(conf, True) for conf in tp_confidences] + [(conf, False) for conf in fp_confidences]
        
        if not all_predictions:
            return 0.0
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x[0])
        
        # Calculate ECE using equal-width binning
        ece = 0.0
        n_samples = len(all_predictions)
        
        for i in range(self.num_bins):
            bin_start = i / self.num_bins
            bin_end = (i + 1) / self.num_bins
            
            # Get predictions in this bin
            bin_predictions = [p for p in all_predictions if bin_start <= p[0] < bin_end]
            
            if not bin_predictions:
                continue
            
            # Calculate accuracy and confidence for this bin
            accuracy = sum(1 for _, correct in bin_predictions if correct) / len(bin_predictions)
            avg_confidence = sum(conf for conf, _ in bin_predictions) / len(bin_predictions)
            
            # Add to ECE
            bin_weight = len(bin_predictions) / n_samples
            ece += bin_weight * abs(accuracy - avg_confidence)
        
        return ece


class PerformanceProfiler:
    """Profiles detection model performance and latency."""
    
    def __init__(self):
        self.timing_data = defaultdict(list)
        
    def record_timing(self, model_name: str, 
                     inference_time: float,
                     preprocessing_time: float = 0.0,
                     postprocessing_time: float = 0.0):
        """Record timing data for a model."""
        total_time = inference_time + preprocessing_time + postprocessing_time
        
        self.timing_data[model_name].append({
            'inference_time': inference_time,
            'preprocessing_time': preprocessing_time,
            'postprocessing_time': postprocessing_time,
            'total_time': total_time
        })
    
    def generate_performance_profile(self, model_name: str, 
                                   num_detections: int = 0) -> PerformanceProfileMetrics:
        """Generate comprehensive performance profile for a model."""
        if model_name not in self.timing_data:
            return PerformanceProfileMetrics(
                model_name=model_name,
                inference_times=[], preprocessing_times=[], 
                postprocessing_times=[], total_processing_times=[],
                frames_per_second=0.0, detections_per_second=0.0,
                mean_inference_time=0.0, std_inference_time=0.0,
                p95_inference_time=0.0, p99_inference_time=0.0
            )
        
        timings = self.timing_data[model_name]
        
        # Extract timing arrays
        inference_times = [t['inference_time'] for t in timings]
        preprocessing_times = [t['preprocessing_time'] for t in timings]
        postprocessing_times = [t['postprocessing_time'] for t in timings]
        total_times = [t['total_time'] for t in timings]
        
        # Calculate statistics
        mean_inference = np.mean(inference_times) if inference_times else 0.0
        std_inference = np.std(inference_times) if inference_times else 0.0
        p95_inference = np.percentile(inference_times, 95) if inference_times else 0.0
        p99_inference = np.percentile(inference_times, 99) if inference_times else 0.0
        
        # Calculate throughput
        mean_total_time = np.mean(total_times) if total_times else 0.0
        fps = 1.0 / mean_total_time if mean_total_time > 0 else 0.0
        
        # Detections per second (rough estimate)
        avg_detections_per_frame = num_detections / len(timings) if timings else 0
        detections_per_sec = fps * avg_detections_per_frame
        
        return PerformanceProfileMetrics(
            model_name=model_name,
            inference_times=inference_times,
            preprocessing_times=preprocessing_times,
            postprocessing_times=postprocessing_times,
            total_processing_times=total_times,
            frames_per_second=fps,
            detections_per_second=detections_per_sec,
            mean_inference_time=mean_inference,
            std_inference_time=std_inference,
            p95_inference_time=p95_inference,
            p99_inference_time=p99_inference
        )


class AdvancedMetricsCollector:
    """Main collector for advanced detection metrics."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.instance_analyzer = InstanceAnalyzer()
        self.map_calculator = mAPCalculator()
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.performance_profiler = PerformanceProfiler()
        
        # Storage
        self.gt_instances: List[DetectionInstance] = []
        self.pred_instances: List[DetectionInstance] = []
        
    def add_detection_results(self, frame: np.ndarray,
                            frame_idx: int,
                            scene_id: str,
                            camera_id: str,
                            ground_truth: Dict[str, torch.Tensor],
                            predictions: Dict[str, torch.Tensor],
                            model_name: str,
                            timing_data: Optional[Dict[str, float]] = None):
        """Add detection results for analysis."""
        
        # Record timing data if provided
        if timing_data:
            self.performance_profiler.record_timing(
                model_name,
                timing_data.get('inference_time', 0.0),
                timing_data.get('preprocessing_time', 0.0),
                timing_data.get('postprocessing_time', 0.0)
            )
        
        # Process ground truth instances
        gt_boxes = ground_truth.get("boxes", torch.empty(0, 4))
        gt_labels = ground_truth.get("labels", torch.empty(0))
        
        num_persons = len(gt_labels)
        
        for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            gt_instance = self.instance_analyzer.analyze_detection_instance(
                bbox_xyxy=box.cpu().numpy().tolist(),
                frame=frame,
                frame_idx=frame_idx,
                scene_id=scene_id,
                camera_id=camera_id,
                person_id=int(label),
                confidence=None,
                is_ground_truth=True,
                num_persons_in_frame=num_persons
            )
            self.gt_instances.append(gt_instance)
        
        # Process prediction instances
        pred_boxes = predictions.get("boxes", torch.empty(0, 4))
        pred_scores = predictions.get("scores", torch.empty(0))
        
        for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
            # Convert prediction box format if needed (assuming xyxy)
            pred_instance = self.instance_analyzer.analyze_detection_instance(
                bbox_xyxy=box.cpu().numpy().tolist(),
                frame=frame,
                frame_idx=frame_idx,
                scene_id=scene_id,
                camera_id=camera_id,
                person_id=-1,  # Unknown person ID for predictions
                confidence=float(score),
                is_ground_truth=False,
                num_persons_in_frame=num_persons
            )
            self.pred_instances.append(pred_instance)
    
    def generate_comprehensive_metrics(self, model_name: str) -> Dict[str, Any]:
        """Generate all advanced metrics."""
        logger.info(f"Generating comprehensive metrics for {model_name}...")
        
        results = {}
        
        # Calculate fine-grained mAP
        map_metrics = self.map_calculator.calculate_advanced_map(
            self.gt_instances, self.pred_instances
        )
        results['advanced_map'] = asdict(map_metrics)
        
        # Analyze confidence distribution
        confidence_analysis = self.confidence_analyzer.analyze_confidence_distribution(
            self.gt_instances, self.pred_instances, model_name
        )
        results['confidence_analysis'] = asdict(confidence_analysis)
        
        # Generate performance profile
        num_detections = len([p for p in self.pred_instances if not p.is_ground_truth])
        perf_profile = self.performance_profiler.generate_performance_profile(
            model_name, num_detections
        )
        results['performance_profile'] = asdict(perf_profile)
        
        # Generate precision-recall curves for different scenarios
        pr_curves = self._generate_scenario_pr_curves()
        results['precision_recall_curves'] = pr_curves
        
        return results
    
    def _generate_scenario_pr_curves(self) -> Dict[str, Dict[str, List[float]]]:
        """Generate precision-recall curves for different scenarios."""
        scenarios = {
            'overall': (self.gt_instances, self.pred_instances),
            'small_objects': self._filter_by_attribute('object_size', 'small'),
            'medium_objects': self._filter_by_attribute('object_size', 'medium'),
            'large_objects': self._filter_by_attribute('object_size', 'large'),
            'day_lighting': self._filter_by_attribute('lighting_condition', 'day'),
            'night_lighting': self._filter_by_attribute('lighting_condition', 'night'),
            'low_density': self._filter_by_attribute('crowd_density', 'low'),
            'high_density': self._filter_by_attribute('crowd_density', 'high')
        }
        
        pr_curves = {}
        
        for scenario_name, (gt_instances, pred_instances) in scenarios.items():
            if not gt_instances:
                continue
            
            # Calculate precision-recall curve
            pr_curve = self._calculate_pr_curve(gt_instances, pred_instances)
            if pr_curve:
                pr_curves[scenario_name] = {
                    'precision': pr_curve['precision'],
                    'recall': pr_curve['recall'],
                    'average_precision': pr_curve['average_precision']
                }
        
        return pr_curves
    
    def _filter_by_attribute(self, attr_name: str, attr_value: str) -> Tuple[List[DetectionInstance], List[DetectionInstance]]:
        """Filter instances by specific attribute value."""
        filtered_gt = [inst for inst in self.gt_instances if getattr(inst, attr_name) == attr_value]
        filtered_pred = [inst for inst in self.pred_instances if getattr(inst, attr_name) == attr_value]
        return filtered_gt, filtered_pred
    
    def _calculate_pr_curve(self, gt_instances: List[DetectionInstance],
                          pred_instances: List[DetectionInstance]) -> Optional[Dict[str, Any]]:
        """Calculate precision-recall curve for given instances."""
        if not gt_instances:
            return None
        
        pred_only = [p for p in pred_instances if not p.is_ground_truth and p.confidence is not None]
        if not pred_only:
            return None
        
        # Create binary classification data
        y_true = []
        y_scores = []
        
        # Create ground truth lookup
        gt_lookup = defaultdict(list)
        for gt in gt_instances:
            key = (gt.frame_idx, gt.scene_id, gt.camera_id)
            gt_lookup[key].append(gt)
        
        # Match predictions to ground truth
        for pred in pred_only:
            key = (pred.frame_idx, pred.scene_id, pred.camera_id)
            gt_list = gt_lookup.get(key, [])
            
            is_match = False
            for gt in gt_list:
                iou = self.map_calculator.calculate_iou(pred.bbox_xyxy, gt.bbox_xyxy)
                if iou >= 0.5:  # IoU threshold
                    is_match = True
                    break
            
            y_true.append(1 if is_match else 0)
            y_scores.append(pred.confidence)
        
        if not y_true or not y_scores:
            return None
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'average_precision': ap
        }
    
    def save_comprehensive_report(self, model_name: str) -> Path:
        """Save comprehensive metrics report."""
        metrics = self.generate_comprehensive_metrics(model_name)
        
        report_data = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'total_gt_instances': len(self.gt_instances),
                'total_pred_instances': len([p for p in self.pred_instances if not p.is_ground_truth]),
                'analysis_type': 'Advanced Detection Metrics Collection'
            },
            'metrics': metrics
        }
        
        report_path = self.output_dir / f'advanced_metrics_{model_name}.json'
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return report_path
    
    def generate_visualization_plots(self, model_name: str) -> List[Path]:
        """Generate visualization plots for advanced metrics."""
        plot_paths = []
        
        try:
            metrics = self.generate_comprehensive_metrics(model_name)
            
            # 1. mAP comparison chart
            map_data = metrics['advanced_map']
            fig, ax = plt.subplots(figsize=(12, 8))
            
            categories = ['Overall', 'Small', 'Medium', 'Large', 'Close', 'Medium Dist', 'Far', 
                         'No Occlusion', 'Partial Occlusion', 'Heavy Occlusion']
            values = [
                map_data['overall_map_50'], map_data['map_small_objects'], 
                map_data['map_medium_objects'], map_data['map_large_objects'],
                map_data['map_close_distance'], map_data['map_medium_distance'],
                map_data['map_far_distance'], map_data['map_no_occlusion'],
                map_data['map_partial_occlusion'], map_data['map_heavy_occlusion']
            ]
            
            bars = ax.bar(categories, values, color='skyblue')
            ax.set_title(f'Fine-grained mAP Analysis - {model_name}')
            ax.set_ylabel('mAP@0.5')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = self.output_dir / f'map_analysis_{model_name}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
            
            # 2. Confidence distribution plot
            conf_data = metrics['confidence_analysis']
            if conf_data['true_positive_confidences'] or conf_data['false_positive_confidences']:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Histogram
                if conf_data['true_positive_confidences']:
                    ax1.hist(conf_data['true_positive_confidences'], bins=20, 
                            alpha=0.7, label='True Positives', color='green')
                if conf_data['false_positive_confidences']:
                    ax1.hist(conf_data['false_positive_confidences'], bins=20, 
                            alpha=0.7, label='False Positives', color='red')
                
                ax1.set_title('Confidence Score Distribution')
                ax1.set_xlabel('Confidence Score')
                ax1.set_ylabel('Count')
                ax1.legend()
                ax1.axvline(conf_data['optimal_threshold'], color='black', 
                           linestyle='--', label=f"Optimal Threshold: {conf_data['optimal_threshold']:.3f}")
                
                # Calibration plot (simplified)
                ax2.text(0.5, 0.5, f"Calibration Error: {conf_data['calibration_error']:.3f}", 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                ax2.set_title('Model Calibration')
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                
                plt.tight_layout()
                plot_path = self.output_dir / f'confidence_analysis_{model_name}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths.append(plot_path)
            
        except Exception as e:
            logger.error(f"Error generating visualization plots: {e}")
        
        return plot_paths