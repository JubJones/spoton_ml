# FILE: src/pipelines/tracking_reid_pipeline.py
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Type, Mapping, Set # Added Set

import cv2
import numpy as np
import torch
from tqdm import tqdm

# --- Optional: Import pandas and motmetrics ---
try:
    import pandas as pd
    import motmetrics as mm
    MOTMETRICS_AVAILABLE = True
except ImportError:
    pd = None
    mm = None
    MOTMETRICS_AVAILABLE = False
    logging.warning("`motmetrics` or `pandas` not found. Install them (`pip install motmetrics pandas`) to calculate standard MOT metrics.")
# --- End Optional Import ---


# --- Corrected BoxMOT Imports ---
try:
    from boxmot.trackers.strongsort.strongsort import StrongSort
    from boxmot.trackers.botsort.botsort import BotSort
    from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
    from boxmot.trackers.ocsort.ocsort import OcSort
    from boxmot.trackers.boosttrack.boosttrack import BoostTrack
    from boxmot.trackers.imprassoc.imprassoctrack import ImprAssocTrack
    from boxmot.trackers.basetracker import BaseTracker

    BOXMOT_AVAILABLE = True
    # Map tracker type strings (lowercase, from config) to the imported BoxMOT classes
    TRACKER_CLASSES: Dict[str, Type[BaseTracker]] = {
        'strongsort': StrongSort,
        'botsort': BotSort,
        'deepocsort': DeepOcSort,
        'ocsort': OcSort,
        'boosttrack': BoostTrack,
        'imprassoc': ImprAssocTrack,
    }
except ImportError as e:
    logging.critical(f"Failed to import BoxMOT components. Tracking functionality unavailable. Error: {e}")
    BOXMOT_AVAILABLE = False
    StrongSort, BotSort, DeepOcSort, OcSort, ByteTrack, BoostTrack, ImprAssocTrack, BaseTracker = (
        None, None, None, None, None, None, None, None
    )
    TRACKER_CLASSES = {}


# --- Local Imports ---
# Assume these imports are correct relative to the project structure
try:
    from src.data.loader import FrameDataLoader # Reuse detection loader
    from src.evaluation.metrics import load_ground_truth, GroundTruthData # Reuse GT loader
    from src.evaluation.tracking_metrics import calculate_tracking_summary # Keep basic summary for now
    from src.utils.reid_device_utils import get_reid_device_specifier_string
    # Try to import CameraID, define locally if not found
    try: from src.alias_types import CameraID
    except ImportError: CameraID = str
except ImportError:
    # Fallback for potentially different execution contexts
    import sys
    _project_root = Path(__file__).parent.parent.parent
    if str(_project_root) not in sys.path: sys.path.insert(0, str(_project_root))
    from data.loader import FrameDataLoader
    from evaluation.metrics import load_ground_truth, GroundTruthData
    from evaluation.tracking_metrics import calculate_tracking_summary
    from utils.reid_device_utils import get_reid_device_specifier_string
    # Define CameraID if not imported
    CameraID = str

logger = logging.getLogger(__name__)

# Define a type hint for the summary dictionary
TrackingResultSummary = Dict[str, Any]


class TrackingReidPipeline:
    """
    Encapsulates the logic for running a BoxMOT tracker with a specified Re-ID model,
    using ground truth bounding boxes as input. Manages separate tracker instances per camera
    and calculates standard MOT metrics.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device, project_root: Path):
        """
        Initializes the pipeline with configuration, preferred device, and project root.

        Args:
            config: The run configuration dictionary.
            device: The preferred torch.device (might be overridden by tracker needs).
            project_root: The root directory of the project.
        """
        if not BOXMOT_AVAILABLE:
            raise ImportError("BoxMOT library is required for TrackingReidPipeline but not found or failed to import.")

        self.config = config
        self.preferred_device = device # Store the preference
        self.project_root = project_root
        self.tracker_instances: Dict[CameraID, BaseTracker] = {}
        self.data_loader: Optional[FrameDataLoader] = None
        self.ground_truth_data: Optional[GroundTruthData] = None
        self.person_class_id = config.get("evaluation", {}).get("person_class_id", 0) # For GT association

        self.raw_tracker_outputs: Dict[Tuple[int, str], np.ndarray] = {} # {(frame_idx, cam_id): tracker_output_array}
        self.summary_metrics: TrackingResultSummary = {}
        self.actual_tracker_devices: Dict[CameraID, Any] = {}

        self.initialized = False
        self.processed = False
        self.metrics_calculated = False

        # Extract config sections
        self.tracker_config = config.get("tracker", {})
        self.reid_config = config.get("reid_model", {})
        self.data_config = config.get("data", {})
        self.env_config = config.get("environment", {})

        self.tracker_type = self.tracker_config.get("type", "").lower()
        self.reid_model_type = self.reid_config.get("model_type", "default")
        self.reid_weights_path_rel = self.reid_config.get("weights_path") # Relative path or name

    def initialize_components(self) -> bool:
        """Initializes data loader, ground truth loader, and BoxMOT trackers (one per camera)."""
        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        logger.info(f"[{run_name_tag}] Initializing Tracking+ReID pipeline components...")

        try:
            # 1. Initialize Data Loader
            logger.info(f"[{run_name_tag}] Initializing frame data loader...")
            self.data_loader = FrameDataLoader(self.config)
            if not self.data_loader.active_camera_ids:
                 raise ValueError("Data loader found 0 active cameras to process.")
            if len(self.data_loader) == 0:
                raise ValueError("Data loader found 0 frame indices to process.")
            logger.info(
                f"[{run_name_tag}] Data loader initialized. Processing {len(self.data_loader)} frame indices across "
                f"{len(self.data_loader.active_camera_ids)} active cameras: {self.data_loader.active_camera_ids}."
            )

            # 2. Load Ground Truth Data
            logger.info(f"[{run_name_tag}] Loading ground truth data (gt.txt)...")
            self.ground_truth_data, _ = load_ground_truth(
                self.data_loader.scene_path,
                self.data_loader.active_camera_ids,
                self.data_loader.image_filenames,
                self.person_class_id
            )
            if self.ground_truth_data is None or not self.ground_truth_data:
                raise FileNotFoundError("Ground truth (gt.txt) could not be loaded or is empty. Cannot proceed.")
            logger.info(f"[{run_name_tag}] Ground truth loaded successfully. Found GT for {len(self.ground_truth_data)} (frame, cam) pairs.")

            # 3. Initialize BoxMOT Trackers (One per camera)
            logger.info(f"[{run_name_tag}] Initializing BoxMOT tracker '{self.tracker_type}' for each active camera...")
            if self.tracker_type not in TRACKER_CLASSES:
                raise ValueError(f"Unsupported tracker type: '{self.tracker_type}'. Available: {list(TRACKER_CLASSES.keys())}")

            TrackerClass = TRACKER_CLASSES[self.tracker_type]
            self.tracker_instances = {}
            self.actual_tracker_devices = {}

            for cam_id in self.data_loader.active_camera_ids:
                logger.info(f"[{run_name_tag}] Initializing tracker for Camera: {cam_id}")
                tracker_args: Dict[str, Any] = {}

                # -- Device --
                reid_device_specifier = get_reid_device_specifier_string(self.preferred_device)
                if 'device' in TrackerClass.__init__.__code__.co_varnames:
                     tracker_args['device'] = reid_device_specifier
                else: logger.warning(f"[{run_name_tag}][{cam_id}] Tracker {TrackerClass.__name__} might not accept 'device' arg.")
                logger.info(f"[{run_name_tag}][{cam_id}] Requesting tracker device: '{reid_device_specifier}'")

                # -- Re-ID Model --
                if self.tracker_type in ['strongsort', 'botsort', 'deepocsort', 'boosttrack', 'imprassoc']:
                    # (Re-ID weight logic remains the same)
                    reid_weights_identifier: Optional[Path] = None
                    if self.reid_weights_path_rel:
                        weights_base_dir_str = self.data_config.get("weights_base_dir", "weights/reid")
                        weights_base_dir = self.project_root / weights_base_dir_str
                        potential_path = weights_base_dir / self.reid_weights_path_rel
                        if potential_path.is_file():
                            reid_weights_identifier = potential_path.resolve()
                            logger.info(f"[{run_name_tag}][{cam_id}] Using local ReID weights: {reid_weights_identifier}")
                        else:
                            logger.warning(f"[{run_name_tag}][{cam_id}] Local ReID weights not found at {potential_path}. Using identifier '{self.reid_model_type}'. BoxMOT might download.")
                            reid_weights_identifier = Path(self.reid_model_type)
                    else:
                        reid_weights_identifier = Path(self.reid_model_type)
                        logger.info(f"[{run_name_tag}][{cam_id}] No ReID path. Using identifier: '{reid_weights_identifier}'. BoxMOT might download.")

                    reid_param_name = None
                    if 'reid_weights' in TrackerClass.__init__.__code__.co_varnames: reid_param_name = 'reid_weights'
                    elif 'model_weights' in TrackerClass.__init__.__code__.co_varnames: reid_param_name = 'model_weights'

                    if reid_param_name and reid_weights_identifier:
                        tracker_args[reid_param_name] = reid_weights_identifier
                        if 'half' in TrackerClass.__init__.__code__.co_varnames:
                            use_half = self.preferred_device.type == 'cuda'
                            tracker_args['half'] = use_half
                            logger.info(f"[{run_name_tag}][{cam_id}] Setting 'half' precision: {use_half}")
                        if 'per_class' in TrackerClass.__init__.__code__.co_varnames:
                            tracker_args['per_class'] = self.config.get('tracker', {}).get('per_class', False)
                    elif not reid_param_name: logger.warning(f"[{run_name_tag}][{cam_id}] Could not determine ReID weights param name for {TrackerClass.__name__}.")
                    else: logger.warning(f"[{run_name_tag}][{cam_id}] ReID specified but no valid path/ID found. Tracker might use default.")
                else: logger.info(f"[{run_name_tag}][{cam_id}] Tracker type '{self.tracker_type}' typically doesn't use ReID model in constructor.")

                # -- Other Potential Args --
                allowed_args = TrackerClass.__init__.__code__.co_varnames
                for arg_name, arg_val in self.tracker_config.items():
                    if arg_name != 'type' and arg_name in allowed_args and arg_name not in tracker_args:
                        tracker_args[arg_name] = arg_val
                        logger.info(f"[{run_name_tag}][{cam_id}] Setting tracker arg '{arg_name}': {arg_val}")

                # Instantiate the tracker for this camera
                logger.info(f"[{run_name_tag}][{cam_id}] Instantiating {TrackerClass.__name__}...")
                log_args = {k: str(v) if isinstance(v, Path) else v for k, v in tracker_args.items()}
                logger.debug(f"[{run_name_tag}][{cam_id}] Args: {log_args}")
                instance = TrackerClass(**tracker_args) # Errors here should be caught by caller
                self.tracker_instances[cam_id] = instance

                # Store the actual device used
                actual_device_instance = None
                if hasattr(instance, 'device'): actual_device_instance = instance.device
                elif hasattr(instance, 'dev'): actual_device_instance = instance.dev
                else: actual_device_instance = tracker_args.get('device', self.preferred_device)
                self.actual_tracker_devices[cam_id] = actual_device_instance
                logger.info(f"[{run_name_tag}][{cam_id}] Instance created. Reported device: {actual_device_instance}")

            logger.info(f"[{run_name_tag}] BoxMOT tracker '{self.tracker_type}' initialized successfully for {len(self.tracker_instances)} active cameras.")
            self.initialized = True
            return True

        except (FileNotFoundError, ValueError, RuntimeError, ImportError, TypeError, Exception) as e:
            logger.critical(f"[{run_name_tag}] Failed to initialize Tracking+ReID pipeline components: {e}", exc_info=True)
            self.initialized = False
            raise e # Re-raise for the caller

    def process_frames(self) -> bool:
        """Processes frames sequentially, feeding GT boxes to the appropriate camera's tracker."""
        # (Frame processing loop remains largely the same as the previous version
        # - it correctly gets tracker_instance per cam_id and handles errors during update)
        # --- No changes needed in this method from the previous version ---
        if not self.initialized or not self.data_loader or not self.tracker_instances or self.ground_truth_data is None:
            logger.error("Cannot process frames: Pipeline components not initialized, trackers missing, or GT missing.")
            return False

        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        logger.info(f"[{run_name_tag}] Starting frame processing loop...")

        self.raw_tracker_outputs = {}
        frame_processing_times = []
        total_gt_boxes_processed = 0
        total_tracks_output = 0
        processed_indices: Set[int] = set() # Track unique frame indices processed

        start_time_total = time.perf_counter()
        num_frames_indices = len(self.data_loader)
        total_frames_to_process_approx = num_frames_indices * len(self.data_loader.active_camera_ids)

        pbar = tqdm(total=total_frames_to_process_approx, desc=f"Tracking ({run_name_tag})")
        frames_processed_count = 0

        try:
            for frame_idx, cam_id, filename, frame_bgr in self.data_loader:
                frames_processed_count += 1
                pbar.update(1)

                if frame_bgr is None:
                    logger.debug(f"[{cam_id}] Skipping frame {frame_idx} due to load error.")
                    continue

                tracker_instance = self.tracker_instances.get(cam_id)
                if not tracker_instance:
                    logger.warning(f"[{cam_id}] No tracker instance for frame {frame_idx}. Skipping.")
                    continue

                processed_indices.add(frame_idx)
                frame_start_time = time.perf_counter()

                # --- Get Ground Truth ---
                gt_for_frame_tuples = self.ground_truth_data.get((frame_idx, cam_id), [])
                if not gt_for_frame_tuples:
                     detections_for_tracker = np.empty((0, 6))
                     num_gt_boxes_frame = 0
                else:
                    num_gt_boxes_frame = len(gt_for_frame_tuples)
                    total_gt_boxes_processed += num_gt_boxes_frame
                    boxes_xyxy = []
                    for _, cx, cy, w, h in gt_for_frame_tuples:
                        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                        if w > 0 and h > 0: boxes_xyxy.append([x1, y1, x2, y2])

                    if not boxes_xyxy: detections_for_tracker = np.empty((0, 6))
                    else:
                        detections_np = np.array(boxes_xyxy)
                        confidences = np.ones((len(boxes_xyxy), 1))
                        class_ids = np.full((len(boxes_xyxy), 1), self.person_class_id)
                        detections_for_tracker = np.hstack((detections_np, confidences, class_ids))

                # --- Update Tracker ---
                tracker_output: Optional[np.ndarray] = None
                try:
                    tracker_output = tracker_instance.update(detections_for_tracker, frame_bgr)
                except cv2.error as cv_err:
                    logger.error(f"[{run_name_tag}][{cam_id}] OpenCV error during update (Frame {frame_idx}): {cv_err}", exc_info=False)
                    tracker_output = np.empty((0, 7))
                except Exception as update_err:
                     logger.error(f"[{run_name_tag}][{cam_id}] Generic error during update (Frame {frame_idx}): {update_err}", exc_info=True)
                     tracker_output = np.empty((0, 7))

                frame_end_time = time.perf_counter()
                frame_processing_times.append((frame_end_time - frame_start_time) * 1000)

                # --- Determine output shape ---
                default_cols = 7
                output_shape_cols = default_cols
                if tracker_output is not None and isinstance(tracker_output, np.ndarray) and tracker_output.ndim == 2:
                    if tracker_output.shape[1] > 0: output_shape_cols = tracker_output.shape[1]

                # --- Store Tracker Output ---
                if tracker_output is not None and isinstance(tracker_output, np.ndarray) and tracker_output.size > 0:
                    if tracker_output.ndim == 2 and tracker_output.shape[1] == output_shape_cols:
                        self.raw_tracker_outputs[(frame_idx, cam_id)] = tracker_output
                        total_tracks_output += len(tracker_output)
                        pbar.set_postfix({"GT": num_gt_boxes_frame, "Tracks": len(tracker_output)})
                    else:
                        logger.warning(f"[{run_name_tag}][{cam_id}] Frame {frame_idx}: Output shape {tracker_output.shape} != expected cols {output_shape_cols}. Storing empty.")
                        self.raw_tracker_outputs[(frame_idx, cam_id)] = np.empty((0, output_shape_cols))
                        pbar.set_postfix({"GT": num_gt_boxes_frame, "Tracks": 0})
                else:
                    self.raw_tracker_outputs[(frame_idx, cam_id)] = np.empty((0, output_shape_cols))
                    pbar.set_postfix({"GT": num_gt_boxes_frame, "Tracks": 0})

            # --- Finalize ---
            pbar.close()
            end_time_total = time.perf_counter()
            total_processing_time_sec = end_time_total - start_time_total

            self.summary_metrics['perf_total_frames_processed'] = frames_processed_count
            self.summary_metrics['perf_unique_frame_indices_processed'] = len(processed_indices)
            self.summary_metrics['perf_total_processing_time_sec'] = round(total_processing_time_sec, 2)
            self.summary_metrics['perf_avg_frame_processing_time_ms'] = round(np.mean(frame_processing_times), 2) if frame_processing_times else 0
            self.summary_metrics['perf_processing_fps'] = round(frames_processed_count / total_processing_time_sec, 2) if total_processing_time_sec > 0 else 0
            self.summary_metrics['input_total_gt_boxes_fed'] = total_gt_boxes_processed
            self.summary_metrics['output_total_tracked_instances'] = total_tracks_output # Sum of len() over frames

            self.processed = True
            logger.info(f"[{run_name_tag}] --- Frame Processing Finished ---")
            logger.info(f"[{run_name_tag}] Processed {frames_processed_count} frames in {total_processing_time_sec:.2f} seconds.")
            logger.info(f"[{run_name_tag}] Total GT boxes fed: {total_gt_boxes_processed}, Total tracks output: {total_tracks_output}")
            return True

        except Exception as e:
            if 'pbar' in locals() and pbar: pbar.close()
            logger.error(f"[{run_name_tag}] Error during frame processing loop: {e}", exc_info=True)
            self.processed = False
            # Store partial performance results if any
            end_time_total = time.perf_counter()
            self.summary_metrics['perf_total_frames_processed'] = frames_processed_count
            self.summary_metrics['perf_unique_frame_indices_processed'] = len(processed_indices)
            self.summary_metrics['perf_total_processing_time_sec'] = round(end_time_total - start_time_total, 2)
            return False

    def calculate_metrics(self) -> bool:
        """
        Calculates summary and standard MOT metrics using motmetrics.
        """
        if not self.processed:
            logger.error("Cannot calculate metrics: Frame processing did not complete successfully.")
            return False
        if not self.data_loader or not self.ground_truth_data or not self.raw_tracker_outputs:
             logger.error("Cannot calculate metrics: Missing data loader, GT data, or tracker outputs.")
             return False

        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        logger.info(f"[{run_name_tag}] Calculating tracking metrics...")

        # --- Basic Summary (from tracking_metrics.py - keep for now) ---
        try:
            basic_summary = calculate_tracking_summary(self.raw_tracker_outputs)
            self.summary_metrics.update(basic_summary)
        except Exception as e:
             logger.warning(f"[{run_name_tag}] Error calculating basic summary metrics: {e}", exc_info=False)

        # --- MOT Metrics Calculation ---
        if not MOTMETRICS_AVAILABLE:
            logger.warning("`motmetrics` library not available. Skipping standard MOT metric calculation.")
            self.metrics_calculated = True # Mark as calculated (even if partially)
            return True # Allow proceeding with basic summary + perf metrics

        logger.info(f"[{run_name_tag}] Preparing data for motmetrics...")
        acc = mm.MOTAccumulator(auto_id=True)
        processed_frame_indices = set(idx for idx, cam in self.raw_tracker_outputs.keys())
        gt_frame_indices = set(idx for idx, cam in self.ground_truth_data.keys())
        # Evaluate only on frames where both GT and tracker output might exist
        eval_frame_indices = sorted(list(processed_frame_indices.union(gt_frame_indices)))
        if not eval_frame_indices:
            logger.warning("No frame indices found for evaluation (GT or Hyp). Cannot calculate MOT metrics.")
            self.metrics_calculated = True
            return True

        max_frame_idx_processed = max(eval_frame_indices) if eval_frame_indices else -1
        logger.info(f"Evaluating MOT metrics across {len(eval_frame_indices)} unique frame indices (up to {max_frame_idx_processed})...")

        # Convert GT and Hyp data frame by frame
        gt_data_list = []
        hyp_data_list = []

        # Define expected columns for motmetrics DataFrame
        mot_cols = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility']

        for frame_idx in eval_frame_indices:
            frame_gt_ids = []
            frame_gt_boxes = []
            frame_hyp_ids = []
            frame_hyp_boxes = []
            frame_hyp_confs = []

            # Aggregate GT for this frame index across all cameras
            # IMPORTANT: motmetrics typically evaluates a single sequence.
            # Evaluating across cameras simultaneously like this assumes IDs are globally unique
            # OR that distances between objects in different cameras are infinite.
            # This simple aggregation might inflate FN/FP if objects appear in multiple cameras
            # under different GT IDs without proper cross-camera association (which this pipeline doesn't do).
            # For a true multi-camera evaluation, a different setup is needed.
            # Here, we proceed assuming we want a "scene-level" metric based on GT presence.
            gt_objs_in_frame: Dict[int, Tuple[float,float,float,float]] = {} # Store GT obj_id -> bbox
            for cam_id in self.data_loader.active_camera_ids:
                gt_tuples = self.ground_truth_data.get((frame_idx, cam_id), [])
                for obj_id, cx, cy, w, h in gt_tuples:
                     if w > 0 and h > 0:
                        x1, y1 = cx - w / 2, cy - h / 2
                        # Add to list for motmetrics dataframe format
                        gt_data_list.append({
                            'FrameId': frame_idx + 1, # motmetrics often uses 1-based indexing
                            'Id': obj_id,
                            'X': x1, 'Y': y1, 'Width': w, 'Height': h,
                            'Confidence': 1.0, # GT confidence is 1
                            'ClassId': self.person_class_id,
                            'Visibility': 1.0
                        })
                        # Track for distance calculation
                        frame_gt_ids.append(obj_id)
                        frame_gt_boxes.append([x1, y1, w, h]) # Use x, y, w, h for iou_matrix


            # Aggregate Hypotheses for this frame index across all cameras
            for cam_id in self.data_loader.active_camera_ids:
                tracker_output = self.raw_tracker_outputs.get((frame_idx, cam_id))
                if tracker_output is not None and tracker_output.size > 0:
                    for row in tracker_output:
                        # Expected format: x1, y1, x2, y2, track_id, conf, cls, ...
                        if len(row) >= 7:
                            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                            track_id = int(row[4])
                            conf = float(row[5])
                            cls_id = int(row[6])
                            w, h = x2 - x1, y2 - y1
                            if w > 0 and h > 0:
                                # Add to list for motmetrics dataframe format
                                hyp_data_list.append({
                                    'FrameId': frame_idx + 1, # Use 1-based indexing consistent with GT
                                    'Id': track_id,
                                    'X': x1, 'Y': y1, 'Width': w, 'Height': h,
                                    'Confidence': conf,
                                    'ClassId': cls_id,
                                    'Visibility': 1.0 # Assume visible
                                })
                                # Track for distance calculation
                                frame_hyp_ids.append(track_id)
                                frame_hyp_boxes.append([x1, y1, w, h]) # Use x, y, w, h


            # --- Update Accumulator for this frame ---
            # Calculate IoU distance matrix for this frame (only if both GT and Hyp exist)
            if frame_gt_boxes and frame_hyp_boxes:
                # distance = 1 - iou
                # motmetrics expects distance, lower is better.
                # Input boxes are [x, y, w, h]
                iou_dists = mm.distances.iou_matrix(frame_gt_boxes, frame_hyp_boxes, max_iou=1.0)
                # Update accumulator
                acc.update(
                    frame_gt_ids,           # Ground truth IDs for this frame
                    frame_hyp_ids,          # Tracker IDs for this frame
                    iou_dists               # Distance matrix
                    # frameid=frame_idx + 1 # Optional: Can specify frameid here too
                )
            elif frame_gt_ids:
                # Only GT exists, update with empty hypothesis
                 acc.update(frame_gt_ids, [], [])
            elif frame_hyp_ids:
                 # Only Hyp exists, update with empty GT
                 acc.update([], frame_hyp_ids, [])
            # If neither exists, acc.update([], [], []) is implicitly handled or skipped

        logger.info(f"[{run_name_tag}] motmetrics accumulator updated for {len(eval_frame_indices)} frames.")

        # --- Compute and Log Metrics ---
        try:
            mh = mm.metrics.create()
            # Compute metrics using the standard MOTChallenge set
            summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')

            # Format metrics for logging
            # Convert pandas series/df to dict for easier logging
            summary_dict = summary.to_dict(orient='index')
            if 'acc' in summary_dict: # Check if the summary name 'acc' exists
                mot_metrics_results = summary_dict['acc']
                logger.info(f"[{run_name_tag}] MOT Metrics Calculation Complete:")
                # Log common metrics
                for metric_name, value in mot_metrics_results.items():
                     # Clean key name for MLflow
                     mlflow_key = f"mot_{metric_name.lower().replace('%', '_pct')}"
                     # Store in summary dict
                     self.summary_metrics[mlflow_key] = value
                     logger.info(f"  {metric_name:<20}: {value:.4f}" if isinstance(value, float) else f"  {metric_name:<20}: {value}")

            else:
                logger.warning(f"[{run_name_tag}] Could not find summary results under name 'acc' in motmetrics output.")


            # Optionally compute HOTA metrics separately if needed
            # hota_summary = mh.compute(acc, metrics=['hota', 'detah', 'assah'], name='hota_acc')
            # logger.info(f"[{run_name_tag}] HOTA Metrics:\n{hota_summary}")
            # ... add hota metrics to self.summary_metrics ...

        except Exception as mot_err:
            logger.error(f"[{run_name_tag}] Failed during motmetrics computation: {mot_err}", exc_info=True)
            # Add placeholders if calculation fails
            placeholder_metrics = ['mota', 'idf1', 'motp', 'num_switches', 'mostly_tracked', 'mostly_lost', 'num_false_positives', 'num_misses']
            for key in placeholder_metrics:
                self.summary_metrics[f"mot_{key}"] = 0.0 # Or np.nan


        self.metrics_calculated = True
        logger.info(f"[{run_name_tag}] Finished calculating all tracking metrics.")
        return True

    def run(self) -> Tuple[bool, TrackingResultSummary]:
        """
        Executes the full Tracking+ReID pipeline, calculating MOT metrics.

        Returns:
            Tuple[bool, TrackingResultSummary]:
                - Success status (bool).
                - Dictionary containing summary and MOT metrics.
        """
        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        success = False
        try:
            # Initialization errors are caught by the caller
            if not self.initialize_components():
                 logger.error(f"[{run_name_tag}] Initialization check failed unexpectedly after call.")
                 return False, self.summary_metrics

            if not self.process_frames():
                self.calculate_metrics() # Attempt metrics even on partial processing
                logger.warning(f"[{run_name_tag}] Frame processing failed. Metrics calculated on partial data.")
                return False, self.summary_metrics

            if not self.calculate_metrics():
                 logger.warning(f"[{run_name_tag}] Metrics calculation failed after successful processing.")
                 # Return success=False but include performance/basic summary metrics
                 return False, self.summary_metrics

            success = True

        except Exception as e:
            logger.critical(f"[{run_name_tag}] Unexpected error during Tracking+ReID pipeline execution: {e}", exc_info=True)
            success = False
            if not self.summary_metrics: self.summary_metrics = {} # Ensure dict exists

        self.dump_cache()

        logger.info(f"[{run_name_tag}] Tracking+ReID Pipeline Run completed. Success: {success}")
        return success, self.summary_metrics

    def dump_cache(self):
        """Saves CMC cache for each tracker instance if applicable."""
        # (Method remains the same as previous version)
        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        if not self.tracker_instances: return

        for cam_id, tracker_instance in self.tracker_instances.items():
            if tracker_instance is None: continue
            cache_saved = False
            if hasattr(tracker_instance, 'cmc') and hasattr(tracker_instance.cmc, 'save_cache'):
                try:
                    tracker_instance.cmc.save_cache(); logger.info(f"[{run_name_tag}][{cam_id}] Saved CMC cache."); cache_saved = True
                except Exception as e: logger.warning(f"[{run_name_tag}][{cam_id}] Failed to save CMC cache: {e}")
            elif hasattr(tracker_instance, 'save_cache'):
                try:
                    tracker_instance.save_cache(); logger.info(f"[{run_name_tag}][{cam_id}] Saved tracker cache (direct call)."); cache_saved = True
                except Exception as e: logger.warning(f"[{run_name_tag}][{cam_id}] Failed to save tracker cache (direct call): {e}")
