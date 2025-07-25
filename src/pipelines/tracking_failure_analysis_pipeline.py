# FILE: src/pipelines/tracking_failure_analysis_pipeline.py
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Set, NamedTuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Import pandas and motmetrics
try:
    import pandas as pd
    import motmetrics as mm
    MOTMETRICS_AVAILABLE = True
except ImportError:
    pd = None
    mm = None
    MOTMETRICS_AVAILABLE = False

# BoxMOT Imports
try:
    from boxmot.trackers.strongsort.strongsort import StrongSort
    from boxmot.trackers.botsort.botsort import BotSort
    from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
    from boxmot.trackers.ocsort.ocsort import OcSort
    from boxmot.trackers.boosttrack.boosttrack import BoostTrack
    from boxmot.trackers.basetracker import BaseTracker
    BOXMOT_AVAILABLE = True
    TRACKER_CLASSES: Dict[str, type] = {
        'strongsort': StrongSort,
        'botsort': BotSort,
        'deepocsort': DeepOcSort,
        'ocsort': OcSort,
        'boosttrack': BoostTrack,
    }
except ImportError:
    BOXMOT_AVAILABLE = False
    TRACKER_CLASSES = {}

# Local Imports
try:
    from src.components.data.loader import FrameDataLoader
    from src.components.evaluation.metrics import load_ground_truth, GroundTruthData
    from src.utils.reid_device_utils import get_reid_device_specifier_string
    try: from src.alias_types import CameraID
    except ImportError: CameraID = str
except ImportError:
    import sys
    _project_root = Path(__file__).parent.parent.parent
    if str(_project_root) not in sys.path: sys.path.insert(0, str(_project_root))
    from data.loader import FrameDataLoader
    from evaluation.metrics import load_ground_truth, GroundTruthData
    from utils.reid_device_utils import get_reid_device_specifier_string
    CameraID = str

logger = logging.getLogger(__name__)

# Type Definitions
@dataclass
class TrackingFailure:
    """Represents a tracking failure instance"""
    frame_idx: int
    camera_id: str
    failure_type: str
    gt_id: int
    predicted_id: Optional[int]
    gt_bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    predicted_bbox: Optional[Tuple[float, float, float, float]]
    confidence: float
    metadata: Dict[str, Any]

@dataclass 
class IDSwitchEvent:
    """Represents an ID switching event"""
    frame_idx: int
    camera_id: str
    gt_id: int
    old_predicted_id: int
    new_predicted_id: int
    bbox: Tuple[float, float, float, float]
    switch_confidence: float

@dataclass
class TrajectoryGap:
    """Represents a gap in trajectory continuity"""
    gt_id: int
    camera_id: str
    start_frame: int
    end_frame: int
    gap_length: int
    last_bbox: Tuple[float, float, float, float]
    next_bbox: Tuple[float, float, float, float]

@dataclass
class TrackingQualityMetrics:
    """Extended tracking quality metrics"""
    # Standard MOT metrics
    mota: float
    motp: float
    idf1: float
    idp: float
    idr: float
    
    # Failure analysis metrics
    id_switches: int
    fragmentations: int
    false_positives: int
    false_negatives: int
    trajectory_gaps: int
    
    # Quality scores
    trajectory_consistency_score: float
    identity_preservation_score: float
    temporal_stability_score: float
    
    # Performance metrics
    avg_processing_time_ms: float
    frames_per_second: float

class TrackingFailureAnalysisPipeline:
    """
    Advanced tracking failure analysis pipeline that extends the basic tracking pipeline
    with comprehensive failure detection, analysis, and visualization capabilities.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device, project_root: Path):
        """Initialize the tracking failure analysis pipeline"""
        if not BOXMOT_AVAILABLE:
            raise ImportError("BoxMOT library is required but not available")
            
        self.config = config
        self.preferred_device = device
        self.project_root = project_root
        self.tracker_instances: Dict[CameraID, BaseTracker] = {}
        self.data_loader: Optional[FrameDataLoader] = None
        self.ground_truth_data: Optional[GroundTruthData] = None
        
        # Analysis-specific attributes
        self.raw_tracker_outputs: Dict[Tuple[int, str], np.ndarray] = {}
        self.tracking_failures: List[TrackingFailure] = []
        self.id_switches: List[IDSwitchEvent] = []
        self.trajectory_gaps: List[TrajectoryGap] = []
        self.quality_metrics: TrackingQualityMetrics = None
        
        # Tracking state for analysis
        self.gt_id_to_pred_id_history: Dict[int, Dict[int, int]] = defaultdict(dict)  # {gt_id: {frame: pred_id}}
        self.pred_id_to_gt_id_mapping: Dict[int, Dict[int, int]] = defaultdict(dict)  # {pred_id: {frame: gt_id}}
        self.last_seen_frame: Dict[Tuple[int, str], int] = {}  # {(gt_id, cam_id): last_frame}
        
        # Configuration extraction
        self.tracker_config = config.get("tracker", {})
        self.reid_config = config.get("reid_model", {})
        self.data_config = config.get("data", {})
        self.analysis_config = config.get("tracking_analysis", {})
        
        self.tracker_type = self.tracker_config.get("type", "").lower()
        self.person_class_id = config.get("evaluation", {}).get("person_class_id", 0)
        
        # Analysis parameters
        self.id_switch_threshold = self.analysis_config.get("id_switch_threshold", 0.5)
        self.trajectory_gap_threshold = self.analysis_config.get("trajectory_gap_threshold", 5)
        self.min_trajectory_length = self.analysis_config.get("min_trajectory_length", 10)
        
        # State flags
        self.initialized = False
        self.processed = False
        self.analyzed = False

    def initialize_components(self) -> bool:
        """Initialize data loader, ground truth, and tracker components"""
        run_name = f"TrackAnalysis_{self.tracker_type}"
        logger.info(f"[{run_name}] Initializing tracking failure analysis components...")
        
        try:
            # Initialize data loader
            self.data_loader = FrameDataLoader(self.config)
            if not self.data_loader.active_camera_ids or len(self.data_loader) == 0:
                raise ValueError("Data loader has no active cameras or frames")
            
            logger.info(f"[{run_name}] Processing {len(self.data_loader)} frames across "
                       f"{len(self.data_loader.active_camera_ids)} cameras")
            
            # Load ground truth
            self.ground_truth_data, _ = load_ground_truth(
                self.data_loader.scene_path,
                self.data_loader.active_camera_ids,
                self.data_loader.image_filenames,
                self.person_class_id
            )
            
            if not self.ground_truth_data:
                raise FileNotFoundError("Ground truth data could not be loaded")
            
            logger.info(f"[{run_name}] Ground truth loaded for {len(self.ground_truth_data)} frame-camera pairs")
            
            # Initialize trackers
            self._initialize_trackers(run_name)
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"[{run_name}] Failed to initialize components: {e}", exc_info=True)
            self.initialized = False
            return False
    
    def _initialize_trackers(self, run_name: str):
        """Initialize BoxMOT trackers for each camera"""
        if self.tracker_type not in TRACKER_CLASSES:
            raise ValueError(f"Unsupported tracker: {self.tracker_type}")
        
        TrackerClass = TRACKER_CLASSES[self.tracker_type]
        
        for cam_id in self.data_loader.active_camera_ids:
            logger.info(f"[{run_name}] Initializing tracker for camera {cam_id}")
            
            tracker_args = {}
            
            # Device configuration
            reid_device = get_reid_device_specifier_string(self.preferred_device)
            init_signature = getattr(TrackerClass.__init__, '__code__', None)
            allowed_args = init_signature.co_varnames if init_signature else []
            
            if 'device' in allowed_args:
                tracker_args['device'] = reid_device
            
            # Re-ID model configuration
            if self.tracker_type in ['strongsort', 'botsort', 'deepocsort', 'boosttrack']:
                reid_weights = self._get_reid_weights_path()
                if reid_weights and 'reid_weights' in allowed_args:
                    tracker_args['reid_weights'] = reid_weights
                if 'half' in allowed_args:
                    tracker_args['half'] = self.preferred_device.type == 'cuda'
            
            # Additional tracker arguments
            for arg_name, arg_val in self.tracker_config.items():
                if arg_name != 'type' and arg_name in allowed_args and arg_name not in tracker_args:
                    tracker_args[arg_name] = arg_val
            
            # Instantiate tracker
            instance = TrackerClass(**tracker_args)
            self.tracker_instances[cam_id] = instance
            
            logger.info(f"[{run_name}] Tracker initialized for camera {cam_id}")
    
    def _get_reid_weights_path(self) -> Optional[Path]:
        """Get Re-ID weights path from configuration"""
        reid_weights_rel = self.reid_config.get("weights_path")
        if not reid_weights_rel:
            return None
        
        weights_base_dir = self.project_root / self.data_config.get("weights_base_dir", "weights/reid")
        potential_path = weights_base_dir / reid_weights_rel
        
        if potential_path.is_file():
            return potential_path.resolve()
        else:
            logger.warning(f"Re-ID weights not found at {potential_path}")
            return Path(reid_weights_rel)
    
    def process_frames_with_analysis(self) -> bool:
        """Process frames while collecting detailed tracking information for failure analysis"""
        if not self.initialized:
            logger.error("Pipeline not initialized")
            return False
        
        run_name = f"TrackAnalysis_{self.tracker_type}"
        logger.info(f"[{run_name}] Starting frame processing with failure analysis...")
        
        self.raw_tracker_outputs = {}
        self.tracking_failures = []
        self.id_switches = []
        self.trajectory_gaps = []
        
        frame_times = []
        total_frames = len(self.data_loader) * len(self.data_loader.active_camera_ids)
        
        with tqdm(total=total_frames, desc=f"Tracking Analysis ({run_name})") as pbar:
            try:
                for frame_idx, cam_id, filename, frame_bgr in self.data_loader:
                    if frame_bgr is None:
                        pbar.update(1)
                        continue
                    
                    start_time = time.perf_counter()
                    
                    # Process frame
                    success = self._process_single_frame(frame_idx, cam_id, frame_bgr)
                    
                    end_time = time.perf_counter()
                    frame_times.append((end_time - start_time) * 1000)
                    
                    pbar.update(1)
                    
                    if not success:
                        logger.warning(f"Failed to process frame {frame_idx} for camera {cam_id}")
                
                self.processed = True
                logger.info(f"[{run_name}] Frame processing completed")
                return True
                
            except Exception as e:
                logger.error(f"[{run_name}] Error during frame processing: {e}", exc_info=True)
                return False
    
    def _process_single_frame(self, frame_idx: int, cam_id: str, frame_bgr: np.ndarray) -> bool:
        """Process a single frame and collect tracking data"""
        try:
            tracker = self.tracker_instances.get(cam_id)
            if not tracker:
                return False
            
            h_img, w_img = frame_bgr.shape[:2]
            
            # Get ground truth for this frame
            gt_data = self.ground_truth_data.get((frame_idx, cam_id), [])
            detections = self._prepare_detections(gt_data, w_img, h_img)
            
            # Run tracker
            tracker_output = tracker.update(detections, frame_bgr)
            
            # Store raw output
            if tracker_output is not None and tracker_output.size > 0:
                self.raw_tracker_outputs[(frame_idx, cam_id)] = tracker_output
            else:
                self.raw_tracker_outputs[(frame_idx, cam_id)] = np.empty((0, 7))
            
            # Analyze this frame for failures
            self._analyze_frame_failures(frame_idx, cam_id, gt_data, tracker_output)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}, camera {cam_id}: {e}")
            return False
    
    def _prepare_detections(self, gt_data: List, w_img: int, h_img: int) -> np.ndarray:
        """Prepare ground truth detections for tracker input"""
        if not gt_data:
            return np.empty((0, 6))
        
        valid_boxes = []
        for _, cx, cy, w_gt, h_gt in gt_data:
            if w_gt <= 0 or h_gt <= 0:
                continue
            
            x1 = max(0.0, cx - w_gt / 2)
            y1 = max(0.0, cy - h_gt / 2)
            x2 = min(float(w_img), cx + w_gt / 2)
            y2 = min(float(h_img), cy + h_gt / 2)
            
            if x2 > x1 and y2 > y1:
                valid_boxes.append([x1, y1, x2, y2])
        
        if not valid_boxes:
            return np.empty((0, 6))
        
        detections_np = np.array(valid_boxes)
        confidences = np.ones((len(valid_boxes), 1))
        class_ids = np.full((len(valid_boxes), 1), self.person_class_id)
        
        return np.hstack((detections_np, confidences, class_ids))
    
    def _analyze_frame_failures(self, frame_idx: int, cam_id: str, gt_data: List, tracker_output: np.ndarray):
        """Analyze tracking failures for a single frame"""
        # Create ground truth and prediction mappings
        gt_boxes = {}  # {gt_id: bbox}
        pred_boxes = {}  # {pred_id: bbox}
        
        # Extract ground truth
        for gt_id, cx, cy, w, h in gt_data:
            if w > 0 and h > 0:
                x1, y1 = cx - w/2, cy - h/2
                x2, y2 = cx + w/2, cy + h/2
                gt_boxes[gt_id] = (x1, y1, x2, y2)
        
        # Extract predictions
        if tracker_output is not None and tracker_output.size > 0:
            for row in tracker_output:
                if len(row) >= 5:
                    x1, y1, x2, y2, pred_id = row[:5]
                    pred_boxes[int(pred_id)] = (x1, y1, x2, y2)
        
        # Analyze ID consistency and switches
        self._detect_id_switches(frame_idx, cam_id, gt_boxes, pred_boxes)
        
        # Detect trajectory gaps
        self._detect_trajectory_gaps(frame_idx, cam_id, gt_boxes)
        
        # Record tracking failures
        self._record_tracking_failures(frame_idx, cam_id, gt_boxes, pred_boxes)
    
    def _detect_id_switches(self, frame_idx: int, cam_id: str, gt_boxes: Dict, pred_boxes: Dict):
        """Detect identity switching events"""
        # Match GT to predictions using IoU
        matches = self._match_gt_to_predictions(gt_boxes, pred_boxes)
        
        for gt_id, pred_id in matches.items():
            # Check if this GT ID had a different prediction ID in recent frames
            if gt_id in self.gt_id_to_pred_id_history:
                recent_frames = [f for f in self.gt_id_to_pred_id_history[gt_id].keys() 
                               if frame_idx - f <= 5]  # Look at last 5 frames
                
                if recent_frames:
                    recent_pred_ids = [self.gt_id_to_pred_id_history[gt_id][f] for f in recent_frames]
                    most_common_pred_id = Counter(recent_pred_ids).most_common(1)[0][0]
                    
                    if pred_id != most_common_pred_id:
                        # ID switch detected
                        switch_event = IDSwitchEvent(
                            frame_idx=frame_idx,
                            camera_id=cam_id,
                            gt_id=gt_id,
                            old_predicted_id=most_common_pred_id,
                            new_predicted_id=pred_id,
                            bbox=gt_boxes[gt_id],
                            switch_confidence=1.0  # Could be improved with actual confidence
                        )
                        self.id_switches.append(switch_event)
            
            # Update history
            self.gt_id_to_pred_id_history[gt_id][frame_idx] = pred_id
    
    def _detect_trajectory_gaps(self, frame_idx: int, cam_id: str, gt_boxes: Dict):
        """Detect gaps in trajectory continuity"""
        for gt_id in gt_boxes:
            key = (gt_id, cam_id)
            
            if key in self.last_seen_frame:
                last_frame = self.last_seen_frame[key]
                gap_length = frame_idx - last_frame - 1
                
                if gap_length >= self.trajectory_gap_threshold:
                    gap = TrajectoryGap(
                        gt_id=gt_id,
                        camera_id=cam_id,
                        start_frame=last_frame,
                        end_frame=frame_idx,
                        gap_length=gap_length,
                        last_bbox=gt_boxes.get(gt_id, (0, 0, 0, 0)),
                        next_bbox=gt_boxes[gt_id]
                    )
                    self.trajectory_gaps.append(gap)
            
            self.last_seen_frame[key] = frame_idx
    
    def _match_gt_to_predictions(self, gt_boxes: Dict, pred_boxes: Dict, iou_threshold: float = 0.5) -> Dict[int, int]:
        """Match ground truth boxes to predictions using IoU"""
        matches = {}
        
        if not gt_boxes or not pred_boxes:
            return matches
        
        # Calculate IoU matrix
        gt_ids = list(gt_boxes.keys())
        pred_ids = list(pred_boxes.keys())
        
        iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
        
        for i, gt_id in enumerate(gt_ids):
            for j, pred_id in enumerate(pred_ids):
                iou = self._calculate_iou(gt_boxes[gt_id], pred_boxes[pred_id])
                iou_matrix[i, j] = iou
        
        # Hungarian algorithm could be used here for optimal matching
        # For now, use greedy matching
        used_pred_ids = set()
        for i, gt_id in enumerate(gt_ids):
            best_j = np.argmax(iou_matrix[i])
            best_iou = iou_matrix[i, best_j]
            
            if best_iou >= iou_threshold and pred_ids[best_j] not in used_pred_ids:
                matches[gt_id] = pred_ids[best_j]
                used_pred_ids.add(pred_ids[best_j])
        
        return matches
    
    def _calculate_iou(self, box1: Tuple[float, float, float, float], 
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
    
    def _record_tracking_failures(self, frame_idx: int, cam_id: str, gt_boxes: Dict, pred_boxes: Dict):
        """Record various types of tracking failures"""
        matches = self._match_gt_to_predictions(gt_boxes, pred_boxes)
        
        # False negatives (missed detections)
        for gt_id, gt_bbox in gt_boxes.items():
            if gt_id not in matches:
                failure = TrackingFailure(
                    frame_idx=frame_idx,
                    camera_id=cam_id,
                    failure_type="false_negative",
                    gt_id=gt_id,
                    predicted_id=None,
                    gt_bbox=gt_bbox,
                    predicted_bbox=None,
                    confidence=0.0,
                    metadata={"reason": "no_matching_prediction"}
                )
                self.tracking_failures.append(failure)
        
        # False positives (extra predictions)
        matched_pred_ids = set(matches.values())
        for pred_id, pred_bbox in pred_boxes.items():
            if pred_id not in matched_pred_ids:
                failure = TrackingFailure(
                    frame_idx=frame_idx,
                    camera_id=cam_id,
                    failure_type="false_positive",
                    gt_id=-1,  # No corresponding GT
                    predicted_id=pred_id,
                    gt_bbox=None,
                    predicted_bbox=pred_bbox,
                    confidence=1.0,  # Could extract actual confidence
                    metadata={"reason": "no_matching_gt"}
                )
                self.tracking_failures.append(failure)
    
    def calculate_enhanced_metrics(self) -> bool:
        """Calculate enhanced tracking quality metrics"""
        if not self.processed:
            logger.error("Cannot calculate metrics: frame processing not completed")
            return False
        
        logger.info("Calculating enhanced tracking quality metrics...")
        
        try:
            # Calculate standard MOT metrics if motmetrics is available
            standard_metrics = self._calculate_standard_mot_metrics()
            
            # Calculate failure-specific metrics
            failure_metrics = self._calculate_failure_metrics()
            
            # Calculate quality scores
            quality_scores = self._calculate_quality_scores()
            
            # Combine all metrics
            self.quality_metrics = TrackingQualityMetrics(
                **standard_metrics,
                **failure_metrics,
                **quality_scores
            )
            
            self.analyzed = True
            logger.info("Enhanced tracking metrics calculated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {e}", exc_info=True)
            return False
    
    def _calculate_standard_mot_metrics(self) -> Dict[str, float]:
        """Calculate standard MOT metrics using motmetrics"""
        default_metrics = {
            'mota': 0.0, 'motp': 0.0, 'idf1': 0.0, 
            'idp': 0.0, 'idr': 0.0
        }
        
        if not MOTMETRICS_AVAILABLE:
            logger.warning("motmetrics not available, returning default values")
            return default_metrics
        
        try:
            acc = mm.MOTAccumulator(auto_id=True)
            
            # Process each frame for motmetrics
            for (frame_idx, cam_id), tracker_output in self.raw_tracker_outputs.items():
                gt_data = self.ground_truth_data.get((frame_idx, cam_id), [])
                
                # Prepare GT data
                gt_ids = []
                gt_boxes = []
                for gt_id, cx, cy, w, h in gt_data:
                    if w > 0 and h > 0:
                        gt_ids.append(gt_id)
                        x1, y1 = cx - w/2, cy - h/2
                        gt_boxes.append([x1, y1, w, h])
                
                # Prepare hypothesis data
                hyp_ids = []
                hyp_boxes = []
                if tracker_output.size > 0:
                    for row in tracker_output:
                        if len(row) >= 5:
                            x1, y1, x2, y2, track_id = row[:5]
                            w, h = x2 - x1, y2 - y1
                            if w > 0 and h > 0:
                                hyp_ids.append(int(track_id))
                                hyp_boxes.append([x1, y1, w, h])
                
                # Update accumulator
                if gt_boxes and hyp_boxes:
                    distances = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=0.5)
                    acc.update(gt_ids, hyp_ids, distances)
                elif gt_ids:
                    acc.update(gt_ids, [], [])
                elif hyp_ids:
                    acc.update([], hyp_ids, [])
            
            # Compute metrics
            mh = mm.metrics.create()
            summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'idp', 'idr'], 
                               name='summary')
            
            if not summary.empty:
                result = {}
                for metric in ['mota', 'motp', 'idf1', 'idp', 'idr']:
                    try:
                        value = summary.loc['summary', metric]
                        result[metric] = float(value) if not pd.isna(value) else 0.0
                    except (KeyError, IndexError):
                        result[metric] = 0.0
                return result
            else:
                return default_metrics
                
        except Exception as e:
            logger.warning(f"Error calculating standard MOT metrics: {e}")
            return default_metrics
    
    def _calculate_failure_metrics(self) -> Dict[str, Any]:
        """Calculate failure-specific metrics"""
        return {
            'id_switches': len(self.id_switches),
            'fragmentations': len(self.trajectory_gaps),
            'false_positives': len([f for f in self.tracking_failures if f.failure_type == 'false_positive']),
            'false_negatives': len([f for f in self.tracking_failures if f.failure_type == 'false_negative']),
            'trajectory_gaps': len(self.trajectory_gaps),
            'avg_processing_time_ms': 0.0,  # Would be calculated from actual timing data
            'frames_per_second': 0.0  # Would be calculated from actual timing data
        }
    
    def _calculate_quality_scores(self) -> Dict[str, float]:
        """Calculate quality scores based on failure analysis"""
        total_frames = len(self.raw_tracker_outputs)
        
        if total_frames == 0:
            return {
                'trajectory_consistency_score': 0.0,
                'identity_preservation_score': 0.0,
                'temporal_stability_score': 0.0
            }
        
        # Trajectory consistency score (lower gaps = higher score)
        gap_penalty = len(self.trajectory_gaps) / max(total_frames, 1)
        trajectory_consistency = max(0.0, 1.0 - gap_penalty)
        
        # Identity preservation score (fewer switches = higher score)
        switch_penalty = len(self.id_switches) / max(total_frames, 1)
        identity_preservation = max(0.0, 1.0 - switch_penalty)
        
        # Temporal stability score (fewer false positives/negatives = higher score)
        false_positives = len([f for f in self.tracking_failures if f.failure_type == 'false_positive'])
        false_negatives = len([f for f in self.tracking_failures if f.failure_type == 'false_negative'])
        stability_penalty = (false_positives + false_negatives) / max(total_frames, 1)
        temporal_stability = max(0.0, 1.0 - stability_penalty)
        
        return {
            'trajectory_consistency_score': trajectory_consistency,
            'identity_preservation_score': identity_preservation,
            'temporal_stability_score': temporal_stability
        }
    
    def generate_failure_analysis_report(self, output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive failure analysis report"""
        if not self.analyzed:
            logger.error("Cannot generate report: analysis not completed")
            return {}
        
        output_dir.mkdir(parents=True, exist_ok=True)
        report_data = {}
        
        try:
            # Generate summary statistics
            summary_stats = self._generate_summary_statistics()
            report_data['summary'] = summary_stats
            
            # Generate failure visualizations
            viz_paths = self._generate_failure_visualizations(output_dir)
            report_data['visualizations'] = viz_paths
            
            # Generate detailed failure lists
            failure_details = self._generate_failure_details()
            report_data['failures'] = failure_details
            
            # Save comprehensive report
            report_path = output_dir / "tracking_failure_analysis_report.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Failure analysis report saved to {report_path}")
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating failure analysis report: {e}", exc_info=True)
            return {}
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the analysis"""
        return {
            'tracker_type': self.tracker_type,
            'total_frames_processed': len(self.raw_tracker_outputs),
            'total_cameras': len(self.data_loader.active_camera_ids),
            'tracking_failures': {
                'total': len(self.tracking_failures),
                'false_positives': len([f for f in self.tracking_failures if f.failure_type == 'false_positive']),
                'false_negatives': len([f for f in self.tracking_failures if f.failure_type == 'false_negative'])
            },
            'id_switches': {
                'total': len(self.id_switches),
                'by_camera': dict(Counter(switch.camera_id for switch in self.id_switches))
            },
            'trajectory_gaps': {
                'total': len(self.trajectory_gaps),
                'avg_gap_length': np.mean([gap.gap_length for gap in self.trajectory_gaps]) if self.trajectory_gaps else 0.0,
                'max_gap_length': max([gap.gap_length for gap in self.trajectory_gaps]) if self.trajectory_gaps else 0
            },
            'quality_metrics': {
                'mota': self.quality_metrics.mota,
                'motp': self.quality_metrics.motp,
                'idf1': self.quality_metrics.idf1,
                'trajectory_consistency_score': self.quality_metrics.trajectory_consistency_score,
                'identity_preservation_score': self.quality_metrics.identity_preservation_score,
                'temporal_stability_score': self.quality_metrics.temporal_stability_score
            }
        }
    
    def _generate_failure_visualizations(self, output_dir: Path) -> Dict[str, str]:
        """Generate failure visualization plots"""
        viz_paths = {}
        
        try:
            # ID switches over time
            if self.id_switches:
                plt.figure(figsize=(12, 6))
                switch_frames = [switch.frame_idx for switch in self.id_switches]
                plt.hist(switch_frames, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('Frame Index')
                plt.ylabel('Number of ID Switches')
                plt.title('ID Switches Distribution Over Time')
                plt.grid(True, alpha=0.3)
                
                switch_plot_path = output_dir / "id_switches_distribution.png"
                plt.savefig(switch_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths['id_switches_distribution'] = str(switch_plot_path)
            
            # Trajectory gaps analysis
            if self.trajectory_gaps:
                plt.figure(figsize=(10, 6))
                gap_lengths = [gap.gap_length for gap in self.trajectory_gaps]
                plt.hist(gap_lengths, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Gap Length (frames)')
                plt.ylabel('Number of Gaps')
                plt.title('Trajectory Gap Length Distribution')
                plt.grid(True, alpha=0.3)
                
                gaps_plot_path = output_dir / "trajectory_gaps_distribution.png"
                plt.savefig(gaps_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths['trajectory_gaps_distribution'] = str(gaps_plot_path)
            
            # Failure types by camera
            failure_by_camera = defaultdict(lambda: defaultdict(int))
            for failure in self.tracking_failures:
                failure_by_camera[failure.camera_id][failure.failure_type] += 1
            
            if failure_by_camera:
                cameras = list(failure_by_camera.keys())
                failure_types = ['false_positive', 'false_negative']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(cameras))
                width = 0.35
                
                fp_counts = [failure_by_camera[cam]['false_positive'] for cam in cameras]
                fn_counts = [failure_by_camera[cam]['false_negative'] for cam in cameras]
                
                ax.bar(x - width/2, fp_counts, width, label='False Positives', alpha=0.8)
                ax.bar(x + width/2, fn_counts, width, label='False Negatives', alpha=0.8)
                
                ax.set_xlabel('Camera ID')
                ax.set_ylabel('Number of Failures')
                ax.set_title('Tracking Failures by Camera')
                ax.set_xticks(x)
                ax.set_xticklabels(cameras)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                camera_failures_path = output_dir / "failures_by_camera.png"
                plt.savefig(camera_failures_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths['failures_by_camera'] = str(camera_failures_path)
            
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
        
        return viz_paths
    
    def _generate_failure_details(self) -> Dict[str, List[Dict]]:
        """Generate detailed failure information"""
        return {
            'tracking_failures': [
                {
                    'frame_idx': f.frame_idx,
                    'camera_id': f.camera_id,
                    'failure_type': f.failure_type,
                    'gt_id': f.gt_id,
                    'predicted_id': f.predicted_id,
                    'gt_bbox': f.gt_bbox,
                    'predicted_bbox': f.predicted_bbox,
                    'confidence': f.confidence,
                    'metadata': f.metadata
                }
                for f in self.tracking_failures[:100]  # Limit to first 100 for report size
            ],
            'id_switches': [
                {
                    'frame_idx': s.frame_idx,
                    'camera_id': s.camera_id,
                    'gt_id': s.gt_id,
                    'old_predicted_id': s.old_predicted_id,
                    'new_predicted_id': s.new_predicted_id,
                    'bbox': s.bbox,
                    'switch_confidence': s.switch_confidence
                }
                for s in self.id_switches[:50]  # Limit to first 50
            ],
            'trajectory_gaps': [
                {
                    'gt_id': g.gt_id,
                    'camera_id': g.camera_id,
                    'start_frame': g.start_frame,
                    'end_frame': g.end_frame,
                    'gap_length': g.gap_length,
                    'last_bbox': g.last_bbox,
                    'next_bbox': g.next_bbox
                }
                for g in self.trajectory_gaps[:50]  # Limit to first 50
            ]
        }
    
    def run_complete_analysis(self) -> Tuple[bool, Dict[str, Any]]:
        """Run the complete tracking failure analysis pipeline"""
        run_name = f"TrackAnalysis_{self.tracker_type}"
        logger.info(f"[{run_name}] Starting complete tracking failure analysis...")
        
        try:
            # Initialize components
            if not self.initialize_components():
                logger.error(f"[{run_name}] Component initialization failed")
                return False, {}
            
            # Process frames with analysis
            if not self.process_frames_with_analysis():
                logger.error(f"[{run_name}] Frame processing failed")
                return False, {}
            
            # Calculate enhanced metrics
            if not self.calculate_enhanced_metrics():
                logger.error(f"[{run_name}] Metrics calculation failed")
                return False, {}
            
            # Generate analysis report
            output_dir = self.project_root / "analysis_outputs" / f"tracking_analysis_{self.tracker_type}"
            report_data = self.generate_failure_analysis_report(output_dir)
            
            logger.info(f"[{run_name}] Complete tracking failure analysis finished successfully")
            
            # Return summary metrics
            summary_metrics = {
                'quality_metrics': self.quality_metrics.__dict__ if self.quality_metrics else {},
                'failure_counts': {
                    'total_failures': len(self.tracking_failures),
                    'id_switches': len(self.id_switches),
                    'trajectory_gaps': len(self.trajectory_gaps),
                    'false_positives': len([f for f in self.tracking_failures if f.failure_type == 'false_positive']),
                    'false_negatives': len([f for f in self.tracking_failures if f.failure_type == 'false_negative'])
                },
                'report_path': str(output_dir),
                'analysis_completed': True
            }
            
            return True, summary_metrics
            
        except Exception as e:
            logger.error(f"[{run_name}] Error during complete analysis: {e}", exc_info=True)
            return False, {}