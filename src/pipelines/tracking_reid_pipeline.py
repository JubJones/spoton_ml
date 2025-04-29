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
    # *** MODIFICATION: Define only the metrics needed for IDF1 ***
    # Requesting 'idf1' should automatically compute 'idp' and 'idr' as dependencies.
    REQUESTED_MOT_METRICS = ['idf1', 'idp', 'idr', 'num_matches', 'num_false_positives', 'num_misses']
    MOTMETRICS_AVAILABLE = True
except ImportError:
    pd = None
    mm = None
    REQUESTED_MOT_METRICS = []
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
    StrongSort, BotSort, DeepOcSort, OcSort, BoostTrack, ImprAssocTrack, BaseTracker = (
        None, None, None, None, None, None, None
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
    using ground truth bounding boxes as input. Calculates IDF1 tracking metric.
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
                # Check __init__ signature for device argument
                init_signature = getattr(TrackerClass.__init__, '__code__', None)
                allowed_args = init_signature.co_varnames if init_signature else []

                if 'device' in allowed_args:
                     tracker_args['device'] = reid_device_specifier
                else: logger.warning(f"[{run_name_tag}][{cam_id}] Tracker {TrackerClass.__name__} might not accept 'device' arg.")
                logger.info(f"[{run_name_tag}][{cam_id}] Requesting tracker device: '{reid_device_specifier}'")


                # -- Re-ID Model --
                if self.tracker_type in ['strongsort', 'botsort', 'deepocsort', 'boosttrack', 'imprassoc']:
                    reid_weights_identifier: Optional[Path] = None
                    if self.reid_weights_path_rel:
                        weights_base_dir_str = self.data_config.get("weights_base_dir", "weights/reid")
                        weights_base_dir = self.project_root / weights_base_dir_str
                        potential_path = weights_base_dir / self.reid_weights_path_rel
                        if potential_path.is_file():
                            reid_weights_identifier = potential_path.resolve()
                            logger.info(f"[{run_name_tag}][{cam_id}] Using local ReID weights: {reid_weights_identifier}")
                        else:
                            logger.warning(f"[{run_name_tag}][{cam_id}] Local ReID weights not found at {potential_path}. Using identifier '{self.reid_weights_path_rel}'. BoxMOT might download.")
                            reid_weights_identifier = Path(self.reid_weights_path_rel) # Treat as identifier if not found
                    else:
                        # Use model type as identifier if path is not given
                        reid_weights_identifier = Path(self.reid_model_type)
                        logger.info(f"[{run_name_tag}][{cam_id}] No ReID path. Using identifier: '{reid_weights_identifier}'. BoxMOT might download.")

                    reid_param_name = None
                    if 'reid_weights' in allowed_args: reid_param_name = 'reid_weights'
                    elif 'model_weights' in allowed_args: reid_param_name = 'model_weights'

                    if reid_param_name and reid_weights_identifier:
                        tracker_args[reid_param_name] = reid_weights_identifier
                        if 'half' in allowed_args:
                            use_half = self.preferred_device.type == 'cuda'
                            tracker_args['half'] = use_half
                            logger.info(f"[{run_name_tag}][{cam_id}] Setting 'half' precision: {use_half}")
                        if 'per_class' in allowed_args:
                            tracker_args['per_class'] = self.config.get('tracker', {}).get('per_class', False)
                    elif not reid_param_name: logger.warning(f"[{run_name_tag}][{cam_id}] Could not determine ReID weights param name for {TrackerClass.__name__}.")
                    else: logger.warning(f"[{run_name_tag}][{cam_id}] ReID specified but no valid path/ID found. Tracker might use default.")
                else: logger.info(f"[{run_name_tag}][{cam_id}] Tracker type '{self.tracker_type}' typically doesn't use ReID model in constructor.")

                # -- Other Potential Args --
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
        """Processes frames sequentially, feeding GT boxes (clipped to image bounds) to the appropriate camera's tracker."""
        if not self.initialized or not self.data_loader or not self.tracker_instances or self.ground_truth_data is None:
            logger.error("Cannot process frames: Pipeline components not initialized, trackers missing, or GT missing.")
            return False

        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        logger.info(f"[{run_name_tag}] Starting frame processing loop...")

        self.raw_tracker_outputs = {}
        frame_processing_times = []
        total_gt_boxes_processed = 0 # Count of *valid* GT boxes fed
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
                h_img, w_img, _ = frame_bgr.shape # Get image dimensions for clipping

                # --- Get Ground Truth and PREPARE/VALIDATE Detections for Tracker ---
                gt_for_frame_tuples = self.ground_truth_data.get((frame_idx, cam_id), [])
                valid_boxes_xyxy = []
                num_gt_boxes_frame = 0 # Count of valid GT boxes for this frame

                if not gt_for_frame_tuples:
                     detections_for_tracker = np.empty((0, 6)) # No GT for this frame/cam
                else:
                    for _, cx, cy, w_gt, h_gt in gt_for_frame_tuples:
                         if w_gt <= 0 or h_gt <= 0: continue # Skip invalid dimensions early

                         x1 = cx - w_gt / 2
                         y1 = cy - h_gt / 2
                         x2 = cx + w_gt / 2
                         y2 = cy + h_gt / 2

                         # --- Clip coordinates to image boundaries ---
                         x1_clipped = max(0.0, x1)
                         y1_clipped = max(0.0, y1)
                         x2_clipped = min(float(w_img), x2) # Use float for consistency
                         y2_clipped = min(float(h_img), y2)

                         # Recalculate width and height AFTER clipping
                         w_clipped = x2_clipped - x1_clipped
                         h_clipped = y2_clipped - y1_clipped

                         # Only add the box if it still has positive dimensions after clipping
                         if w_clipped > 0 and h_clipped > 0:
                             valid_boxes_xyxy.append([x1_clipped, y1_clipped, x2_clipped, y2_clipped])
                         else:
                             # Log only if original box had area, to avoid spamming for GT outside FoV
                             if w_gt > 0 and h_gt > 0:
                                 logger.debug(f"[{run_name_tag}][{cam_id}] Frame {frame_idx}: GT box ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) resulted in zero area after clipping to ({w_img}x{h_img}). Skipping.")

                    num_gt_boxes_frame = len(valid_boxes_xyxy) # Count *valid* boxes
                    total_gt_boxes_processed += num_gt_boxes_frame # Accumulate valid count

                    if not valid_boxes_xyxy:
                        detections_for_tracker = np.empty((0, 6)) # No *valid* GT boxes
                    else:
                        # Prepare array for tracker: [x1, y1, x2, y2, conf, cls_id]
                        detections_np = np.array(valid_boxes_xyxy) # Use CLIPPED boxes
                        confidences = np.ones((len(valid_boxes_xyxy), 1)) # Assume GT confidence = 1
                        class_ids = np.full((len(valid_boxes_xyxy), 1), self.person_class_id)
                        detections_for_tracker = np.hstack((detections_np, confidences, class_ids))


                # --- Update Tracker ---
                tracker_output: Optional[np.ndarray] = None
                try:
                    # Input is [N, 6] -> [x1, y1, x2, y2, conf, cls]
                    tracker_output = tracker_instance.update(detections_for_tracker, frame_bgr)
                    # Output is usually [M, 7] -> [x1, y1, x2, y2, track_id, conf, cls]
                except cv2.error as cv_err:
                    # Catch the specific error if it still occurs (e.g., due to internal tracker issues)
                    logger.error(f"[{run_name_tag}][{cam_id}] OpenCV error during update (Frame {frame_idx}): {cv_err}", exc_info=False)
                    logger.error(f"Input detections shape: {detections_for_tracker.shape}")
                    tracker_output = np.empty((0, 7)) # Return empty on error
                except Exception as update_err:
                     logger.error(f"[{run_name_tag}][{cam_id}] Generic error during update (Frame {frame_idx}): {update_err}", exc_info=True)
                     logger.error(f"Input detections shape: {detections_for_tracker.shape}")
                     tracker_output = np.empty((0, 7))

                frame_end_time = time.perf_counter()
                frame_processing_times.append((frame_end_time - frame_start_time) * 1000)

                # --- Determine output shape (assuming standard 7 columns or padding) ---
                default_cols = 7 # x1, y1, x2, y2, id, conf, cls

                # --- Store Tracker Output ---
                num_tracks_frame = 0
                if tracker_output is not None and isinstance(tracker_output, np.ndarray) and tracker_output.size > 0:
                    if tracker_output.ndim == 2 and tracker_output.shape[1] >= 5: # Need at least x1,y1,x2,y2,id
                        # Ensure array has expected number of columns, pad if necessary (e.g., missing class)
                        if tracker_output.shape[1] < default_cols:
                            padded_output = np.full((tracker_output.shape[0], default_cols), np.nan)
                            padded_output[:, :tracker_output.shape[1]] = tracker_output
                            # Fill default class/conf if missing?
                            if tracker_output.shape[1] < 7: padded_output[:, 6] = self.person_class_id # Default class
                            if tracker_output.shape[1] < 6: padded_output[:, 5] = 1.0 # Default conf
                            self.raw_tracker_outputs[(frame_idx, cam_id)] = padded_output
                            num_tracks_frame = len(padded_output)
                        else:
                             # Slice just in case tracker returns extra columns
                            self.raw_tracker_outputs[(frame_idx, cam_id)] = tracker_output[:, :default_cols]
                            num_tracks_frame = len(tracker_output)

                        total_tracks_output += num_tracks_frame
                    else:
                        logger.warning(f"[{run_name_tag}][{cam_id}] Frame {frame_idx}: Output shape {tracker_output.shape} invalid (< 5 cols or not 2D). Storing empty.")
                        self.raw_tracker_outputs[(frame_idx, cam_id)] = np.empty((0, default_cols))
                else:
                    self.raw_tracker_outputs[(frame_idx, cam_id)] = np.empty((0, default_cols))

                pbar.set_postfix({"GT_Valid": num_gt_boxes_frame, "Tracks": num_tracks_frame})

            # --- Finalize ---
            pbar.close()
            end_time_total = time.perf_counter()
            total_processing_time_sec = end_time_total - start_time_total

            self.summary_metrics['perf_total_frames_processed'] = frames_processed_count
            self.summary_metrics['perf_unique_frame_indices_processed'] = len(processed_indices)
            self.summary_metrics['perf_total_processing_time_sec'] = round(total_processing_time_sec, 2)
            self.summary_metrics['perf_avg_frame_processing_time_ms'] = round(np.mean(frame_processing_times), 2) if frame_processing_times else 0
            self.summary_metrics['perf_processing_fps'] = round(frames_processed_count / total_processing_time_sec, 2) if total_processing_time_sec > 0 else 0
            self.summary_metrics['input_total_gt_boxes_fed'] = total_gt_boxes_processed # Count of valid boxes fed
            self.summary_metrics['output_total_tracked_instances'] = total_tracks_output # Sum of len() over frames

            self.processed = True
            logger.info(f"[{run_name_tag}] --- Frame Processing Finished ---")
            logger.info(f"[{run_name_tag}] Processed {frames_processed_count} frames in {total_processing_time_sec:.2f} seconds.")
            logger.info(f"[{run_name_tag}] Total valid GT boxes fed: {total_gt_boxes_processed}, Total tracks output: {total_tracks_output}")
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
        Calculates summary and IDF1 tracking metrics using motmetrics.
        """
        if not self.processed:
            logger.error("Cannot calculate metrics: Frame processing did not complete successfully.")
            return False
        if not self.data_loader or not self.ground_truth_data or not self.raw_tracker_outputs:
             logger.error("Cannot calculate metrics: Missing data loader, GT data, or tracker outputs.")
             return False

        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        logger.info(f"[{run_name_tag}] Calculating tracking metrics (IDF1 focus)...")

        # --- Basic Summary (from tracking_metrics.py - keep for now) ---
        try:
            basic_summary = calculate_tracking_summary(self.raw_tracker_outputs)
            self.summary_metrics.update(basic_summary)
        except Exception as e:
             logger.warning(f"[{run_name_tag}] Error calculating basic summary metrics: {e}", exc_info=False)

        # --- MOT Metrics Calculation (IDF1 focus) ---
        if not MOTMETRICS_AVAILABLE:
            logger.warning("`motmetrics` library not available. Skipping IDF1 calculation.")
            self.metrics_calculated = True # Mark as calculated (even if partially)
            return True # Allow proceeding with basic summary + perf metrics

        logger.info(f"[{run_name_tag}] Preparing data for motmetrics (IDF1)...")
        acc = mm.MOTAccumulator(auto_id=True)
        processed_frame_indices = set(idx for idx, cam in self.raw_tracker_outputs.keys())
        gt_frame_indices = set(idx for idx, cam in self.ground_truth_data.keys())
        eval_frame_indices = sorted(list(processed_frame_indices.union(gt_frame_indices)))
        if not eval_frame_indices:
            logger.warning("No frame indices found for evaluation (GT or Hyp). Cannot calculate IDF1.")
            self.metrics_calculated = True
            return True

        max_frame_idx_processed = max(eval_frame_indices) if eval_frame_indices else -1
        logger.info(f"Evaluating IDF1 metrics across {len(eval_frame_indices)} unique frame indices (up to {max_frame_idx_processed})...")

        # --- Use tqdm for metric calculation loop ---
        pbar_metrics = tqdm(eval_frame_indices, desc=f"Accumulating IDF1 ({run_name_tag})")

        for frame_idx in pbar_metrics:
            frame_gt_ids = []
            frame_gt_boxes = []
            frame_hyp_ids = []
            frame_hyp_boxes = []

            # Aggregate GT for this frame index across all cameras
            for cam_id in self.data_loader.active_camera_ids:
                gt_tuples = self.ground_truth_data.get((frame_idx, cam_id), [])
                for obj_id, cx, cy, w, h in gt_tuples:
                     if w > 0 and h > 0:
                        x1, y1 = cx - w / 2, cy - h / 2
                        frame_gt_ids.append(obj_id)
                        frame_gt_boxes.append([x1, y1, w, h]) # Use x, y, w, h for iou_matrix

            # Aggregate Hypotheses for this frame index across all cameras
            for cam_id in self.data_loader.active_camera_ids:
                tracker_output = self.raw_tracker_outputs.get((frame_idx, cam_id))
                if tracker_output is not None and tracker_output.size > 0:
                    # Expected format: x1, y1, x2, y2, track_id, conf, cls
                    for row in tracker_output:
                        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                        track_id = int(row[4])
                        w, h = x2 - x1, y2 - y1
                        if w > 0 and h > 0:
                            frame_hyp_ids.append(track_id)
                            frame_hyp_boxes.append([x1, y1, w, h]) # Use x, y, w, h

            # --- Update Accumulator for this frame ---
            if frame_gt_boxes and frame_hyp_boxes:
                iou_dists = mm.distances.iou_matrix(frame_gt_boxes, frame_hyp_boxes, max_iou=0.5) # Standard 0.5 IoU threshold
                acc.update(frame_gt_ids, frame_hyp_ids, iou_dists)
            elif frame_gt_ids:
                acc.update(frame_gt_ids, [], [])
            elif frame_hyp_ids:
                acc.update([], frame_hyp_ids, [])

        pbar_metrics.close()
        logger.info(f"[{run_name_tag}] motmetrics accumulator updated for {len(eval_frame_indices)} frames.")

        # --- Compute and Log Metrics ---
        try:
            mh = mm.metrics.create()
            # *** MODIFICATION: Request only the metrics needed for IDF1 ***
            logger.info(f"[{run_name_tag}] Computing IDF1 related metrics...")
            summary = mh.compute(acc, metrics=REQUESTED_MOT_METRICS, name='idf1_summary')

            # Format metrics for logging
            summary_dict = summary.to_dict(orient='index')
            if 'idf1_summary' in summary_dict: # Check if the summary name exists
                idf1_metrics_results = summary_dict['idf1_summary']
                logger.info(f"[{run_name_tag}] IDF1 Metrics Calculation Complete:")

                # Log computed IDF1 metrics
                for metric_name in REQUESTED_MOT_METRICS:
                    if metric_name in idf1_metrics_results:
                        value = idf1_metrics_results[metric_name]
                        # Clean key name for MLflow (use 'mot_' prefix for consistency)
                        mlflow_key = f"mot_{metric_name.lower().replace('%', '_pct')}"
                        # Store in summary dict
                        self.summary_metrics[mlflow_key] = value
                        log_value = f"{value:.4f}" if isinstance(value, (float, np.float_)) else str(value)
                        logger.info(f"  {metric_name:<20}: {log_value}")
                    else:
                        logger.warning(f"Metric '{metric_name}' requested but not found in results.")
                        self.summary_metrics[f"mot_{metric_name.lower()}"] = 0.0 # Placeholder

            else:
                logger.warning(f"[{run_name_tag}] Could not find results under name 'idf1_summary' in motmetrics output.")
                # Add placeholders if calculation fails unexpectedly
                for key in REQUESTED_MOT_METRICS: self.summary_metrics[f"mot_{key.lower()}"] = 0.0

        except Exception as mot_err:
            logger.error(f"[{run_name_tag}] Failed during motmetrics IDF1 computation: {mot_err}", exc_info=True)
            # Add placeholders if calculation fails
            for key in REQUESTED_MOT_METRICS:
                self.summary_metrics[f"mot_{key.lower()}"] = 0.0

        self.metrics_calculated = True
        logger.info(f"[{run_name_tag}] Finished calculating tracking metrics (IDF1 focus).")
        return True

    def run(self) -> Tuple[bool, TrackingResultSummary]:
        """
        Executes the full Tracking+ReID pipeline, calculating IDF1 metric.

        Returns:
            Tuple[bool, TrackingResultSummary]:
                - Success status (bool).
                - Dictionary containing summary and IDF1 metrics.
        """
        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        success = False
        try:
            # Initialization errors are caught by the caller
            if not self.initialize_components():
                 logger.error(f"[{run_name_tag}] Initialization check failed unexpectedly after call.")
                 return False, self.summary_metrics

            if not self.process_frames():
                # Attempt metrics even on partial processing, might have perf metrics
                metrics_success = self.calculate_metrics()
                logger.warning(f"[{run_name_tag}] Frame processing failed. Metrics calculation on partial data: {'Success' if metrics_success else 'Failed'}")
                return False, self.summary_metrics

            if not self.calculate_metrics():
                 logger.warning(f"[{run_name_tag}] Metrics calculation failed after successful processing.")
                 # Return success=False but include performance/basic summary metrics
                 return False, self.summary_metrics

            success = True

        except Exception as e:
            logger.critical(f"[{run_name_tag}] Unexpected error during Tracking+ReID pipeline execution: {e}", exc_info=True)
            success = False
            # Ensure summary_metrics is a dict even if initialization failed early
            if not hasattr(self, 'summary_metrics') or self.summary_metrics is None:
                 self.summary_metrics = {}
            elif not isinstance(self.summary_metrics, dict):
                 self.summary_metrics = {} # Reset if it became something else

        self.dump_cache()

        logger.info(f"[{run_name_tag}] Tracking+ReID Pipeline Run completed. Success: {success}")
        return success, self.summary_metrics

    def dump_cache(self):
        """Saves CMC cache for each tracker instance if applicable."""
        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        if not hasattr(self, 'tracker_instances') or not self.tracker_instances: return

        for cam_id, tracker_instance in self.tracker_instances.items():
            if tracker_instance is None: continue
            cache_saved = False
            # Check specific cache types known in BoxMOT
            if hasattr(tracker_instance, 'cmc') and hasattr(tracker_instance.cmc, 'save_cache'):
                try:
                    tracker_instance.cmc.save_cache()
                    logger.info(f"[{run_name_tag}][{cam_id}] Saved CMC cache.")
                    cache_saved = True
                except Exception as e: logger.warning(f"[{run_name_tag}][{cam_id}] Failed to save CMC cache: {e}")
            # Check BoostTrack specific cache
            elif hasattr(tracker_instance, 'tracker') and hasattr(tracker_instance.tracker, 'save_cache'): # For BoostTrack
                 try:
                     tracker_instance.tracker.save_cache()
                     logger.info(f"[{run_name_tag}][{cam_id}] Saved BoostTrack internal tracker cache.")
                     cache_saved = True
                 except Exception as e: logger.warning(f"[{run_name_tag}][{cam_id}] Failed to save BoostTrack cache: {e}")
            # Generic check (might catch others or internal methods)
            elif hasattr(tracker_instance, 'save_cache'):
                try:
                    tracker_instance.save_cache()
                    logger.info(f"[{run_name_tag}][{cam_id}] Saved tracker cache (generic call).")
                    cache_saved = True
                except Exception as e: logger.warning(f"[{run_name_tag}][{cam_id}] Failed to save tracker cache (generic call): {e}")

            # if not cache_saved: logger.debug(f"[{run_name_tag}][{cam_id}] No specific cache save method found for {type(tracker_instance)}.")