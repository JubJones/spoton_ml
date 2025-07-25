"""
Enhanced Detection Analysis Framework for Phase 1: Multi-dimensional failure analysis.

This module implements comprehensive detection failure analysis including:
1. Scene-aware failure detection (lighting, crowd density, occlusion, distance)
2. Temporal failure pattern analysis 
3. Multi-layered failure visualization
4. Statistical failure analysis and reporting
5. Cross-model comparison capabilities
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

logger = logging.getLogger(__name__)


@dataclass
class SceneContext:
    """Context information for scene analysis."""
    lighting_condition: str  # 'day', 'night', 'transition'
    crowd_density: str      # 'low', 'medium', 'high'
    occlusion_level: str    # 'none', 'partial', 'heavy'
    average_distance: float # Average distance of persons from camera
    frame_quality: float    # Image quality score (0-1)
    

@dataclass
class DetectionFailure:
    """Comprehensive detection failure record."""
    frame_idx: int
    person_id: int
    scene_id: str
    camera_id: str
    timestamp: Optional[str]
    ground_truth_box: List[float]  # [x1, y1, x2, y2]
    context: SceneContext
    is_first_occurrence: bool
    consecutive_frames_missed: int
    confidence_scores: List[float]  # Confidence scores of nearby detections
    nearest_detection_distance: Optional[float]
    

@dataclass
class TemporalPattern:
    """Temporal failure pattern analysis."""
    person_id: int
    total_frames: int
    detected_frames: int
    missed_frames: int
    detection_consistency: float  # Percentage of frames detected
    flickering_instances: int     # Number of detection/miss alternations
    longest_miss_streak: int      # Longest consecutive misses
    stability_score: float        # Overall temporal stability (0-1)


@dataclass
class FailureStatistics:
    """Statistical analysis of detection failures."""
    total_failures: int
    failures_by_lighting: Dict[str, int]
    failures_by_density: Dict[str, int]
    failures_by_occlusion: Dict[str, int]
    failures_by_distance_range: Dict[str, int]
    per_camera_failure_rates: Dict[str, float]
    per_person_failure_rates: Dict[int, float]
    temporal_patterns: List[TemporalPattern]


class SceneAnalyzer:
    """Analyzes scene context for failure classification."""
    
    def __init__(self):
        self.brightness_thresholds = {'day': 120, 'night': 80}
        self.density_thresholds = {'low': 2, 'high': 8}
        
    def analyze_lighting_condition(self, image: np.ndarray) -> str:
        """Determine lighting condition from image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness > self.brightness_thresholds['day']:
            return 'day'
        elif avg_brightness < self.brightness_thresholds['night']:
            return 'night'
        else:
            return 'transition'
    
    def analyze_crowd_density(self, num_persons: int, image_area: int) -> str:
        """Determine crowd density based on person count and image area."""
        # Normalize by image area (people per megapixel)
        density_score = num_persons / (image_area / 1_000_000)
        
        if density_score <= self.density_thresholds['low']:
            return 'low'
        elif density_score >= self.density_thresholds['high']:
            return 'high'
        else:
            return 'medium'
    
    def analyze_occlusion_level(self, gt_boxes: torch.Tensor) -> str:
        """Estimate occlusion level based on bounding box overlaps."""
        if len(gt_boxes) < 2:
            return 'none'
        
        total_overlap = 0
        total_pairs = 0
        
        for i in range(len(gt_boxes)):
            for j in range(i + 1, len(gt_boxes)):
                box1, box2 = gt_boxes[i], gt_boxes[j]
                intersection = self._calculate_intersection(box1, box2)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                
                overlap_ratio = intersection / min(area1, area2)
                total_overlap += overlap_ratio
                total_pairs += 1
        
        if total_pairs == 0:
            return 'none'
        
        avg_overlap = total_overlap / total_pairs
        if avg_overlap < 0.1:
            return 'none'
        elif avg_overlap < 0.3:
            return 'partial'
        else:
            return 'heavy'
    
    def _calculate_intersection(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Calculate intersection area between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return (x2 - x1) * (y2 - y1)
    
    def calculate_average_distance(self, gt_boxes: torch.Tensor, image_height: int) -> float:
        """Estimate average distance based on bounding box sizes."""
        if len(gt_boxes) == 0:
            return 0.0
        
        # Use box height as proxy for distance (larger = closer)
        heights = [box[3] - box[1] for box in gt_boxes]
        avg_height = sum(heights) / len(heights)
        
        # Normalize by image height and invert (smaller height = greater distance)
        normalized_height = avg_height / image_height
        distance_score = 1.0 - normalized_height  # 0 = very close, 1 = very far
        
        return float(distance_score)
    
    def calculate_image_quality(self, image: np.ndarray) -> float:
        """Calculate image quality score using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 scale (higher variance = better quality)
        # Typical range is 0-1000, but can vary significantly
        quality_score = min(laplacian_var / 500.0, 1.0)
        return quality_score
    
    def analyze_scene_context(self, image: np.ndarray, gt_boxes: torch.Tensor, 
                            num_persons: int) -> SceneContext:
        """Comprehensive scene context analysis."""
        height, width = image.shape[:2]
        image_area = height * width
        
        return SceneContext(
            lighting_condition=self.analyze_lighting_condition(image),
            crowd_density=self.analyze_crowd_density(num_persons, image_area),
            occlusion_level=self.analyze_occlusion_level(gt_boxes),
            average_distance=self.calculate_average_distance(gt_boxes, height),
            frame_quality=self.calculate_image_quality(image)
        )


class TemporalAnalyzer:
    """Analyzes temporal patterns in detection failures."""
    
    def __init__(self):
        self.person_histories: Dict[int, List[bool]] = defaultdict(list)
        self.frame_timestamps: List[Optional[str]] = []
    
    def update_detection_history(self, frame_idx: int, detected_ids: Set[int], 
                               all_gt_ids: Set[int], timestamp: Optional[str] = None):
        """Update detection history for temporal analysis."""
        # Ensure we have enough entries
        while len(self.frame_timestamps) <= frame_idx:
            self.frame_timestamps.append(None)
        
        self.frame_timestamps[frame_idx] = timestamp
        
        for person_id in all_gt_ids:
            # Ensure history list is long enough
            while len(self.person_histories[person_id]) <= frame_idx:
                self.person_histories[person_id].append(False)
            
            self.person_histories[person_id][frame_idx] = person_id in detected_ids
    
    def analyze_temporal_patterns(self) -> List[TemporalPattern]:
        """Analyze temporal patterns for all tracked persons."""
        patterns = []
        
        for person_id, history in self.person_histories.items():
            if not history:
                continue
            
            total_frames = len(history)
            detected_frames = sum(history)
            missed_frames = total_frames - detected_frames
            
            if total_frames == 0:
                continue
            
            detection_consistency = detected_frames / total_frames
            flickering_instances = self._count_flickering(history)
            longest_miss_streak = self._longest_streak(history, False)
            stability_score = self._calculate_stability_score(history)
            
            patterns.append(TemporalPattern(
                person_id=person_id,
                total_frames=total_frames,
                detected_frames=detected_frames,
                missed_frames=missed_frames,
                detection_consistency=detection_consistency,
                flickering_instances=flickering_instances,
                longest_miss_streak=longest_miss_streak,
                stability_score=stability_score
            ))
        
        return patterns
    
    def _count_flickering(self, history: List[bool]) -> int:
        """Count detection/miss alternations (flickering)."""
        if len(history) < 2:
            return 0
        
        flickering = 0
        for i in range(1, len(history)):
            if history[i] != history[i-1]:
                flickering += 1
        
        return flickering
    
    def _longest_streak(self, history: List[bool], value: bool) -> int:
        """Find longest consecutive streak of specified value."""
        if not history:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for detection in history:
            if detection == value:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_stability_score(self, history: List[bool]) -> float:
        """Calculate overall temporal stability score."""
        if len(history) <= 1:
            return 1.0
        
        # Factors: consistency, low flickering, no long miss streaks
        consistency = sum(history) / len(history)
        
        # Penalize flickering
        flickering_penalty = self._count_flickering(history) / len(history)
        
        # Penalize long miss streaks
        longest_miss = self._longest_streak(history, False)
        miss_penalty = min(longest_miss / len(history), 0.5)
        
        stability = consistency * (1 - flickering_penalty - miss_penalty)
        return max(0.0, min(1.0, stability))


class EnhancedVisualizationGenerator:
    """Generates multi-layered failure visualizations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_failure_visualization(self, image: np.ndarray, failure: DetectionFailure,
                                     nearest_detections: Optional[List[Tuple[List[float], float]]] = None,
                                     save_path: Optional[Path] = None) -> Path:
        """Generate comprehensive failure visualization."""
        img_vis = image.copy()
        height, width = img_vis.shape[:2]
        
        # Colors
        gt_color = (0, 255, 0)      # Green for ground truth
        miss_color = (0, 0, 255)    # Red for missed detection
        near_color = (255, 255, 0)  # Cyan for nearby detections
        
        # Draw ground truth box (missed)
        x1, y1, x2, y2 = map(int, failure.ground_truth_box)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), miss_color, 3)
        
        # Add failure information text
        info_lines = [
            f"MISSED: Person ID {failure.person_id}",
            f"Frame: {failure.frame_idx}",
            f"Context: {failure.context.lighting_condition}, {failure.context.crowd_density} density",
            f"Occlusion: {failure.context.occlusion_level}",
            f"Consecutive misses: {failure.consecutive_frames_missed}",
            f"Quality: {failure.context.frame_quality:.2f}"
        ]
        
        # Add contextual information overlay
        y_offset = 30
        for line in info_lines:
            cv2.putText(img_vis, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            cv2.putText(img_vis, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 0), 1)
            y_offset += 25
        
        # Draw nearby detections for comparison
        if nearest_detections:
            for det_box, confidence in nearest_detections:
                dx1, dy1, dx2, dy2 = map(int, det_box)
                cv2.rectangle(img_vis, (dx1, dy1), (dx2, dy2), near_color, 2)
                cv2.putText(img_vis, f"Det: {confidence:.2f}", (dx1, dy1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, near_color, 1)
        
        # Save visualization
        if save_path is None:
            filename = f"failure_{failure.scene_id}_{failure.camera_id}_frame{failure.frame_idx}_id{failure.person_id}.png"
            save_path = self.output_dir / filename
        
        cv2.imwrite(str(save_path), img_vis)
        return save_path
    
    def generate_heatmap(self, failures: List[DetectionFailure], image_shape: Tuple[int, int], 
                        camera_id: str) -> Path:
        """Generate failure heatmap for camera view."""
        height, width = image_shape
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Create density map of failure locations
        for failure in failures:
            x1, y1, x2, y2 = failure.ground_truth_box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Add Gaussian kernel around failure center
            kernel_size = min(max(int((x2 - x1 + y2 - y1) / 4), 20), 100)
            y_min = max(0, center_y - kernel_size)
            y_max = min(height, center_y + kernel_size)
            x_min = max(0, center_x - kernel_size)
            x_max = min(width, center_x + kernel_size)
            
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if distance <= kernel_size:
                        weight = np.exp(-(distance**2) / (2 * (kernel_size/3)**2))
                        heatmap[y, x] += weight
        
        # Normalize and convert to colormap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Save heatmap
        heatmap_path = self.output_dir / f"failure_heatmap_{camera_id}.png"
        cv2.imwrite(str(heatmap_path), heatmap_colored)
        
        return heatmap_path


class StatisticalAnalyzer:
    """Generates statistical analysis and reports."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_failure_statistics(self, failures: List[DetectionFailure], 
                                  temporal_patterns: List[TemporalPattern]) -> FailureStatistics:
        """Generate comprehensive failure statistics."""
        if not failures:
            return FailureStatistics(
                total_failures=0,
                failures_by_lighting={},
                failures_by_density={},
                failures_by_occlusion={},
                failures_by_distance_range={},
                per_camera_failure_rates={},
                per_person_failure_rates={},
                temporal_patterns=temporal_patterns
            )
        
        # Count failures by different factors
        lighting_counts = defaultdict(int)
        density_counts = defaultdict(int)
        occlusion_counts = defaultdict(int)
        distance_counts = defaultdict(int)
        camera_counts = defaultdict(int)
        person_counts = defaultdict(int)
        
        for failure in failures:
            lighting_counts[failure.context.lighting_condition] += 1
            density_counts[failure.context.crowd_density] += 1
            occlusion_counts[failure.context.occlusion_level] += 1
            
            # Distance ranges
            distance = failure.context.average_distance
            if distance < 0.3:
                distance_range = 'close'
            elif distance < 0.7:
                distance_range = 'medium'
            else:
                distance_range = 'far'
            distance_counts[distance_range] += 1
            
            camera_counts[failure.camera_id] += 1
            person_counts[failure.person_id] += 1
        
        # Calculate rates (assuming total frames/persons are available)
        total_cameras = len(set(f.camera_id for f in failures))
        total_persons = len(set(f.person_id for f in failures))
        
        camera_failure_rates = {}
        for camera_id, count in camera_counts.items():
            # This would need total frames per camera for accurate rate
            camera_failure_rates[camera_id] = count / len(failures)
        
        person_failure_rates = {}
        for person_id, count in person_counts.items():
            # This would need total appearances per person for accurate rate
            person_failure_rates[person_id] = count / len(failures)
        
        return FailureStatistics(
            total_failures=len(failures),
            failures_by_lighting=dict(lighting_counts),
            failures_by_density=dict(density_counts),
            failures_by_occlusion=dict(occlusion_counts),
            failures_by_distance_range=dict(distance_counts),
            per_camera_failure_rates=camera_failure_rates,
            per_person_failure_rates=person_failure_rates,
            temporal_patterns=temporal_patterns
        )
    
    def generate_statistical_plots(self, statistics: FailureStatistics) -> List[Path]:
        """Generate statistical visualization plots."""
        plot_paths = []
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Failure distribution by lighting condition
        if statistics.failures_by_lighting:
            fig, ax = plt.subplots(figsize=(10, 6))
            lighting_data = statistics.failures_by_lighting
            bars = ax.bar(lighting_data.keys(), lighting_data.values())
            ax.set_title('Detection Failures by Lighting Condition')
            ax.set_xlabel('Lighting Condition')
            ax.set_ylabel('Number of Failures')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom')
            
            plot_path = self.output_dir / 'failures_by_lighting.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # 2. Failure distribution by crowd density
        if statistics.failures_by_density:
            fig, ax = plt.subplots(figsize=(10, 6))
            density_data = statistics.failures_by_density
            bars = ax.bar(density_data.keys(), density_data.values(), color='orange')
            ax.set_title('Detection Failures by Crowd Density')
            ax.set_xlabel('Crowd Density')
            ax.set_ylabel('Number of Failures')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom')
            
            plot_path = self.output_dir / 'failures_by_density.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # 3. Per-camera failure rates
        if statistics.per_camera_failure_rates:
            fig, ax = plt.subplots(figsize=(12, 6))
            camera_data = statistics.per_camera_failure_rates
            bars = ax.bar(camera_data.keys(), camera_data.values(), color='red')
            ax.set_title('Failure Rates by Camera')
            ax.set_xlabel('Camera ID')
            ax.set_ylabel('Failure Rate')
            ax.tick_params(axis='x', rotation=45)
            
            plot_path = self.output_dir / 'failure_rates_by_camera.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        # 4. Temporal stability distribution
        if statistics.temporal_patterns:
            stability_scores = [p.stability_score for p in statistics.temporal_patterns]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(stability_scores, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            ax.set_title('Distribution of Temporal Stability Scores')
            ax.set_xlabel('Stability Score')
            ax.set_ylabel('Number of Persons')
            ax.axvline(np.mean(stability_scores), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(stability_scores):.2f}')
            ax.legend()
            
            plot_path = self.output_dir / 'temporal_stability_distribution.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
        
        return plot_paths
    
    def save_statistics_report(self, statistics: FailureStatistics) -> Path:
        """Save detailed statistics report as JSON."""
        report_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_failures': statistics.total_failures,
                'unique_persons_with_failures': len(statistics.per_person_failure_rates),
                'cameras_analyzed': len(statistics.per_camera_failure_rates)
            },
            'failure_breakdown': {
                'by_lighting': statistics.failures_by_lighting,
                'by_density': statistics.failures_by_density,
                'by_occlusion': statistics.failures_by_occlusion,
                'by_distance_range': statistics.failures_by_distance_range
            },
            'temporal_analysis': {
                'patterns': [asdict(pattern) for pattern in statistics.temporal_patterns],
                'summary_stats': {
                    'avg_detection_consistency': np.mean([p.detection_consistency for p in statistics.temporal_patterns]) if statistics.temporal_patterns else 0,
                    'avg_stability_score': np.mean([p.stability_score for p in statistics.temporal_patterns]) if statistics.temporal_patterns else 0,
                    'high_flickering_persons': len([p for p in statistics.temporal_patterns if p.flickering_instances > 10])
                }
            }
        }
        
        report_path = self.output_dir / 'failure_statistics_report.json'
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return report_path


class EnhancedDetectionAnalyzer:
    """Main enhanced detection analyzer integrating all components."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.scene_analyzer = SceneAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.visualizer = EnhancedVisualizationGenerator(output_dir / 'visualizations')
        self.stats_analyzer = StatisticalAnalyzer(output_dir / 'statistics')
        
        # Storage for analysis results
        self.failures: List[DetectionFailure] = []
        self.frame_count = 0
        
    def analyze_frame(self, frame_idx: int, image: np.ndarray, predictions: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor], scene_id: str, camera_id: str,
                     timestamp: Optional[str] = None) -> List[DetectionFailure]:
        """Analyze a single frame for detection failures with context."""
        self.frame_count += 1
        frame_failures = []
        
        # Extract data
        pred_boxes = predictions.get("boxes", torch.empty(0, 4))
        pred_scores = predictions.get("scores", torch.empty(0))
        gt_boxes = targets.get("boxes", torch.empty(0, 4))
        gt_ids = targets.get("labels", torch.empty(0))
        
        if len(gt_boxes) == 0:
            return frame_failures
        
        # Analyze scene context
        scene_context = self.scene_analyzer.analyze_scene_context(
            image, gt_boxes, len(gt_ids)
        )
        
        # Determine detected and missed IDs
        detected_ids, missed_ids = self._get_detection_status(
            pred_boxes, gt_boxes, gt_ids
        )
        
        # Update temporal tracking
        all_gt_ids = set(gt_ids.tolist())
        self.temporal_analyzer.update_detection_history(
            frame_idx, detected_ids, all_gt_ids, timestamp
        )
        
        # Create failure records for missed detections
        for person_id in missed_ids:
            # Get ground truth box for this person
            person_mask = gt_ids == person_id
            if not torch.any(person_mask):
                continue
                
            gt_box = gt_boxes[person_mask][0]  # Take first if multiple
            
            # Find nearest detections for context
            nearest_detections = self._find_nearest_detections(
                gt_box, pred_boxes, pred_scores
            )
            
            # Check if this is first occurrence for this person
            is_first_occurrence = person_id not in [f.person_id for f in self.failures]
            
            # Calculate consecutive frames missed (simplified)
            consecutive_missed = self._calculate_consecutive_misses(person_id, frame_idx)
            
            failure = DetectionFailure(
                frame_idx=frame_idx,
                person_id=int(person_id),
                scene_id=scene_id,
                camera_id=camera_id,
                timestamp=timestamp,
                ground_truth_box=gt_box.tolist(),
                context=scene_context,
                is_first_occurrence=is_first_occurrence,
                consecutive_frames_missed=consecutive_missed,
                confidence_scores=pred_scores.tolist() if len(pred_scores) > 0 else [],
                nearest_detection_distance=self._calculate_nearest_distance(gt_box, pred_boxes)
            )
            
            frame_failures.append(failure)
            self.failures.append(failure)
        
        return frame_failures
    
    def _get_detection_status(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, 
                            gt_ids: torch.Tensor, iou_threshold: float = 0.5) -> Tuple[Set[int], Set[int]]:
        """Determine which ground truth IDs were detected vs missed."""
        if len(gt_boxes) == 0:
            return set(), set()
        
        all_gt_ids = set(gt_ids.tolist())
        detected_ids = set()
        
        if len(pred_boxes) > 0:
            # Calculate IoU matrix
            from torchvision.ops import box_iou
            iou_matrix = box_iou(gt_boxes, pred_boxes)
            
            if iou_matrix.shape[1] > 0:
                max_ious, _ = torch.max(iou_matrix, dim=1)
                matched_mask = max_ious >= iou_threshold
                detected_ids.update(gt_ids[matched_mask].tolist())
        
        missed_ids = all_gt_ids - detected_ids
        return detected_ids, missed_ids
    
    def _find_nearest_detections(self, gt_box: torch.Tensor, pred_boxes: torch.Tensor, 
                               pred_scores: torch.Tensor, max_detections: int = 3) -> List[Tuple[List[float], float]]:
        """Find nearest predictions to a ground truth box."""
        if len(pred_boxes) == 0:
            return []
        
        # Calculate distances (center-to-center)
        gt_center = [(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2]
        distances = []
        
        for i, pred_box in enumerate(pred_boxes):
            pred_center = [(pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2]
            distance = ((gt_center[0] - pred_center[0]) ** 2 + (gt_center[1] - pred_center[1]) ** 2) ** 0.5
            distances.append((distance, i))
        
        # Sort by distance and return top N
        distances.sort()
        nearest = []
        for _, idx in distances[:max_detections]:
            if idx < len(pred_scores):
                nearest.append((pred_boxes[idx].tolist(), float(pred_scores[idx])))
        
        return nearest
    
    def _calculate_consecutive_misses(self, person_id: int, current_frame: int) -> int:
        """Calculate consecutive frames this person has been missed."""
        if person_id not in self.temporal_analyzer.person_histories:
            return 1
        
        history = self.temporal_analyzer.person_histories[person_id]
        consecutive = 0
        
        # Count backwards from current frame
        for i in range(len(history) - 1, -1, -1):
            if not history[i]:  # Missed detection
                consecutive += 1
            else:
                break
        
        return consecutive + 1  # Include current frame
    
    def _calculate_nearest_distance(self, gt_box: torch.Tensor, pred_boxes: torch.Tensor) -> Optional[float]:
        """Calculate distance to nearest prediction."""
        if len(pred_boxes) == 0:
            return None
        
        gt_center = [(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2]
        min_distance = float('inf')
        
        for pred_box in pred_boxes:
            pred_center = [(pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2]
            distance = ((gt_center[0] - pred_center[0]) ** 2 + (gt_center[1] - pred_center[1]) ** 2) ** 0.5
            min_distance = min(min_distance, distance)
        
        return float(min_distance) if min_distance != float('inf') else None
    
    def generate_comprehensive_report(self) -> Dict[str, Path]:
        """Generate comprehensive analysis report with all components."""
        logger.info("Generating comprehensive enhanced detection analysis report...")
        
        # Generate temporal patterns
        temporal_patterns = self.temporal_analyzer.analyze_temporal_patterns()
        
        # Generate statistics
        statistics = self.stats_analyzer.generate_failure_statistics(
            self.failures, temporal_patterns
        )
        
        # Generate visualizations
        generated_files = {}
        
        # Statistical plots
        plot_paths = self.stats_analyzer.generate_statistical_plots(statistics)
        for i, path in enumerate(plot_paths):
            generated_files[f'statistical_plot_{i}'] = path
        
        # Statistics report
        stats_report_path = self.stats_analyzer.save_statistics_report(statistics)
        generated_files['statistics_report'] = stats_report_path
        
        # Generate failure heatmaps per camera
        cameras = set(f.camera_id for f in self.failures)
        for camera_id in cameras:
            camera_failures = [f for f in self.failures if f.camera_id == camera_id]
            if camera_failures:
                # Use representative image shape (would need actual image for precise shape)
                heatmap_path = self.visualizer.generate_heatmap(
                    camera_failures, (1080, 1920), camera_id  # Default HD resolution
                )
                generated_files[f'heatmap_{camera_id}'] = heatmap_path
        
        # Generate sample failure visualizations
        unique_failures = {}
        for failure in self.failures:
            key = (failure.person_id, failure.scene_id, failure.camera_id)
            if key not in unique_failures:
                unique_failures[key] = failure
        
        # Save summary report
        summary_report = self._generate_summary_report(statistics, generated_files)
        summary_path = self.output_dir / 'enhanced_analysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        generated_files['summary_report'] = summary_path
        
        logger.info(f"Enhanced detection analysis complete. Generated {len(generated_files)} files.")
        return generated_files
    
    def _generate_summary_report(self, statistics: FailureStatistics, 
                               generated_files: Dict[str, Path]) -> Dict[str, Any]:
        """Generate executive summary report."""
        return {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_frames_analyzed': self.frame_count,
                'total_failures_found': len(self.failures),
                'analysis_type': 'Enhanced Detection Analysis - Phase 1'
            },
            'key_findings': {
                'most_problematic_lighting': max(statistics.failures_by_lighting, 
                                               key=statistics.failures_by_lighting.get) if statistics.failures_by_lighting else None,
                'most_problematic_density': max(statistics.failures_by_density,
                                              key=statistics.failures_by_density.get) if statistics.failures_by_density else None,
                'most_problematic_camera': max(statistics.per_camera_failure_rates,
                                             key=statistics.per_camera_failure_rates.get) if statistics.per_camera_failure_rates else None,
                'average_temporal_stability': np.mean([p.stability_score for p in statistics.temporal_patterns]) if statistics.temporal_patterns else 0
            },
            'recommendations': self._generate_recommendations(statistics),
            'generated_files': {key: str(path) for key, path in generated_files.items()}
        }
    
    def _generate_recommendations(self, statistics: FailureStatistics) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if statistics.failures_by_lighting:
            worst_lighting = max(statistics.failures_by_lighting, key=statistics.failures_by_lighting.get)
            recommendations.append(f"Focus on improving detection under {worst_lighting} lighting conditions")
        
        if statistics.failures_by_density:
            worst_density = max(statistics.failures_by_density, key=statistics.failures_by_density.get)
            recommendations.append(f"Address detection issues in {worst_density} crowd density scenarios")
        
        if statistics.temporal_patterns:
            low_stability_persons = [p for p in statistics.temporal_patterns if p.stability_score < 0.5]
            if low_stability_persons:
                recommendations.append(f"Investigate {len(low_stability_persons)} persons with poor temporal stability")
        
        if statistics.per_camera_failure_rates:
            worst_camera = max(statistics.per_camera_failure_rates, key=statistics.per_camera_failure_rates.get)
            recommendations.append(f"Review camera {worst_camera} setup and calibration")
        
        return recommendations