import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Type

import cv2
import numpy as np
import torch
from tqdm import tqdm


# --- Corrected BoxMOT Imports ---
try:
    from boxmot.trackers.strongsort.strongsort import StrongSORT
    from boxmot.trackers.botsort.botsort import BoTSORT
    from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
    from boxmot.trackers.ocsort.ocsort import OcSort
    from boxmot.trackers.bytetrack.bytetrack import ByteTrack
    from boxmot.trackers.boosttrack.boosttrack import BoostTrack
    from boxmot.trackers.imprassoc.imprassoctrack import ImprAssocTrack

    # Import base class for type hinting if needed
    from boxmot.trackers.basetracker import BaseTracker

    BOXMOT_AVAILABLE = True
    # Map tracker type strings (lowercase, from config) to the imported BoxMOT classes
    TRACKER_CLASSES: Dict[str, Type[BaseTracker]] = {
        'strongsort': StrongSORT,
        'botsort': BoTSORT,
        'deepocsort': DeepOcSort,
        'ocsort': OcSort,
        'bytetrack': ByteTrack,
        'boosttrack': BoostTrack,
        'imprassoc': ImprAssocTrack,
    }
except ImportError as e:
    logging.critical(f"Failed to import BoxMOT components. Tracking functionality unavailable. Error: {e}")
    BOXMOT_AVAILABLE = False
    StrongSORT, BoTSORT, DeepOcSort, OcSort, ByteTrack, BoostTrack, ImprAssocTrack, BaseTracker = (
        None, None, None, None, None, None, None, None
    )
    TRACKER_CLASSES = {}


# --- Local Imports ---
# Assume these imports are correct relative to the project structure
try:
    from src.data.loader import FrameDataLoader # Reuse detection loader
    from src.evaluation.metrics import load_ground_truth, GroundTruthData # Reuse GT loader
    from src.evaluation.tracking_metrics import calculate_tracking_summary # Use new basic tracking metrics
    # Assuming the utility function was moved/created as suggested previously
    from src.utils.reid_device_utils import get_reid_device_specifier_string
except ImportError:
    # Fallback for potentially different execution contexts
    import sys
    _project_root = Path(__file__).parent.parent.parent
    if str(_project_root) not in sys.path: sys.path.insert(0, str(_project_root))
    from data.loader import FrameDataLoader
    from evaluation.metrics import load_ground_truth, GroundTruthData
    from evaluation.tracking_metrics import calculate_tracking_summary
    from utils.reid_device_utils import get_reid_device_specifier_string

logger = logging.getLogger(__name__)

# Define a type hint for the summary dictionary
TrackingResultSummary = Dict[str, Any]


class TrackingReidPipeline:
    """
    Encapsulates the logic for running a BoxMOT tracker with a specified Re-ID model,
    using ground truth bounding boxes as input.
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
        self.tracker_instance: Optional[BaseTracker] = None # Type hint with BaseTracker
        self.data_loader: Optional[FrameDataLoader] = None
        self.ground_truth_data: Optional[GroundTruthData] = None
        self.person_class_id = config.get("evaluation", {}).get("person_class_id", 0) # For GT association

        self.raw_tracker_outputs: Dict[Tuple[int, str], np.ndarray] = {} # {(frame_idx, cam_id): tracker_output_array}
        self.summary_metrics: TrackingResultSummary = {}
        self.actual_tracker_device: Optional[Any] = None # Device reported by tracker (can be str or torch.device)

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
        """Initializes data loader, ground truth loader, and the BoxMOT tracker."""
        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        logger.info(f"[{run_name_tag}] Initializing Tracking+ReID pipeline components...")

        try:
            # 1. Initialize Data Loader (Reusing detection loader)
            logger.info(f"[{run_name_tag}] Initializing frame data loader...")
            self.data_loader = FrameDataLoader(self.config)
            if len(self.data_loader) == 0:
                raise ValueError("Data loader found 0 frame indices to process.")
            logger.info(
                f"[{run_name_tag}] Data loader initialized. Processing {len(self.data_loader)} frame indices across "
                f"{len(self.data_loader.active_camera_ids)} cameras: {self.data_loader.active_camera_ids}."
            )

            # 2. Load Ground Truth Data
            logger.info(f"[{run_name_tag}] Loading ground truth data (gt.txt)...")
            # Pass person_class_id if needed by loader (currently uses it for logging only)
            self.ground_truth_data, _ = load_ground_truth(
                self.data_loader.scene_path,
                self.data_loader.active_camera_ids,
                self.data_loader.image_filenames,
                self.person_class_id
            )
            if self.ground_truth_data is None:
                # This pipeline *requires* GT for input, so it's an error.
                raise FileNotFoundError("Ground truth (gt.txt) could not be loaded or is empty. Cannot proceed.")
            logger.info(f"[{run_name_tag}] Ground truth loaded successfully. Found GT for {len(self.ground_truth_data)} (frame, cam) pairs.")

            # 3. Initialize BoxMOT Tracker
            logger.info(f"[{run_name_tag}] Initializing BoxMOT tracker: {self.tracker_type}")
            if self.tracker_type not in TRACKER_CLASSES:
                raise ValueError(f"Unsupported tracker type: '{self.tracker_type}'. "
                                 f"Available and imported: {list(TRACKER_CLASSES.keys())}")

            TrackerClass = TRACKER_CLASSES[self.tracker_type]

            # Prepare arguments for the tracker constructor
            # Arguments might vary between trackers, check BoxMOT docs for specifics
            tracker_args: Dict[str, Any] = {}

            # -- Device --
            # Convert preferred torch device to BoxMOT string specifier ('0', 'mps', 'cpu')
            reid_device_specifier = get_reid_device_specifier_string(self.preferred_device)
            # Check if TrackerClass expects 'device' argument (some might not)
            # This requires inspecting the class signature or documentation.
            # For now, assume common trackers accept it.
            if 'device' in TrackerClass.__init__.__code__.co_varnames:
                 tracker_args['device'] = reid_device_specifier
            else:
                 logger.warning(f"Tracker class {TrackerClass.__name__} might not accept 'device' argument. Check BoxMOT source/docs.")
            logger.info(f"[{run_name_tag}] Requesting tracker device: '{reid_device_specifier}' "
                        f"(from preferred: {self.preferred_device})")

            # -- Re-ID Model --
            # Only pass ReID weights to trackers known to use them
            if self.tracker_type in ['strongsort', 'botsort', 'deepocsort', 'boosttrack', 'imprassoc']:
                # Construct the full path to the Re-ID weights if provided
                reid_weights_identifier: Optional[Path] = None
                if self.reid_weights_path_rel:
                    weights_base_dir_str = self.data_config.get("weights_base_dir", "weights/reid")
                    weights_base_dir = self.project_root / weights_base_dir_str
                    potential_path = weights_base_dir / self.reid_weights_path_rel
                    if potential_path.is_file():
                        reid_weights_identifier = potential_path.resolve()
                        logger.info(f"[{run_name_tag}] Found ReID weights file: {reid_weights_identifier}")
                    else:
                        logger.warning(f"[{run_name_tag}] ReID weights file not found at {potential_path}. "
                                    f"Tracker will use specified type '{self.reid_model_type}' or its default.")
                        # BoxMOT might handle type name as identifier for download
                        reid_weights_identifier = Path(self.reid_model_type)
                else:
                    # If no path, use the type name (e.g., 'osnet_x1_0') - BoxMOT might download this
                    reid_weights_identifier = Path(self.reid_model_type)
                    logger.info(f"[{run_name_tag}] No specific ReID weights path. Using type/identifier: '{reid_weights_identifier}'")

                # Check BoxMOT docs for exact parameter name ('reid_weights', 'model_weights', etc.)
                # Common patterns:
                reid_param_name = None
                if 'reid_weights' in TrackerClass.__init__.__code__.co_varnames:
                     reid_param_name = 'reid_weights'
                elif 'model_weights' in TrackerClass.__init__.__code__.co_varnames:
                     reid_param_name = 'model_weights'

                if reid_param_name and reid_weights_identifier:
                     tracker_args[reid_param_name] = reid_weights_identifier
                     # Set 'half' if the tracker supports it (often related to ReID)
                     if 'half' in TrackerClass.__init__.__code__.co_varnames:
                          # Use half precision based on resolved device type (optional)
                          use_half = self.preferred_device.type == 'cuda' # Example: Only use half on CUDA
                          tracker_args['half'] = use_half
                          logger.info(f"[{run_name_tag}] Setting tracker 'half' precision: {use_half}")
                     # Set 'per_class' if needed (usually False for this setup)
                     if 'per_class' in TrackerClass.__init__.__code__.co_varnames:
                          tracker_args['per_class'] = self.config.get('tracker', {}).get('per_class', False)

                elif not reid_param_name:
                     logger.warning(f"Could not determine ReID weights parameter name for {TrackerClass.__name__}.")
                else: # reid_param_name exists but reid_weights_identifier is None/empty
                     logger.warning(f"[{run_name_tag}] ReID specified in config but no valid path/identifier found for {self.tracker_type}. Tracker might fail or use default ReID.")

            else:
                 logger.info(f"[{run_name_tag}] Tracker type '{self.tracker_type}' typically does not use ReID models directly in constructor.")


            # -- Other Potential Args (Add as needed based on BoxMOT tracker docs/config) --
            # Example: max_age, min_hits, iou_threshold etc. might be settable here
            # Check config['tracker'] for specific overrides
            allowed_args = TrackerClass.__init__.__code__.co_varnames
            for arg_name, arg_val in self.tracker_config.items():
                 if arg_name != 'type' and arg_name in allowed_args and arg_name not in tracker_args:
                      tracker_args[arg_name] = arg_val
                      logger.info(f"[{run_name_tag}] Setting tracker arg '{arg_name}' from config: {arg_val}")


            # Instantiate the tracker
            logger.info(f"[{run_name_tag}] Instantiating {TrackerClass.__name__} with args: { {k: (str(v) if isinstance(v, Path) else v) for k,v in tracker_args.items()} }") # Log args cleanly
            self.tracker_instance = TrackerClass(**tracker_args)

            # Store the actual device used by the tracker if possible (BoxMOT might expose it)
            # Common attribute names: 'device', 'dev'
            if hasattr(self.tracker_instance, 'device'):
                 self.actual_tracker_device = self.tracker_instance.device
                 logger.info(f"[{run_name_tag}] Tracker reported using device: {self.actual_tracker_device}")
            elif hasattr(self.tracker_instance, 'dev'):
                 self.actual_tracker_device = self.tracker_instance.dev
                 logger.info(f"[{run_name_tag}] Tracker reported using device (dev attr): {self.actual_tracker_device}")
            else:
                 logger.warning(f"[{run_name_tag}] Could not determine actual device used by tracker instance.")
                 # Fallback to the device specifier string passed in args
                 self.actual_tracker_device = tracker_args.get('device', self.preferred_device)


            logger.info(f"[{run_name_tag}] BoxMOT tracker '{self.tracker_type}' initialized successfully.")
            self.initialized = True
            return True

        except (FileNotFoundError, ValueError, RuntimeError, ImportError, Exception) as e:
            logger.critical(f"[{run_name_tag}] Failed to initialize Tracking+ReID pipeline components: {e}", exc_info=True)
            self.initialized = False
            return False

    def process_frames(self) -> bool:
        """Processes frames sequentially, feeding GT boxes to the tracker."""
        if not self.initialized or not self.data_loader or not self.tracker_instance or self.ground_truth_data is None:
            logger.error("Cannot process frames: Pipeline components not initialized or GT missing.")
            return False

        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        logger.info(f"[{run_name_tag}] Starting frame processing loop...")

        self.raw_tracker_outputs = {}
        frame_processing_times = []
        total_gt_boxes_processed = 0
        total_tracks_output = 0
        processed_indices = set()

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
                    logger.debug(f"Skipping frame {frame_idx} for camera {cam_id} due to load error.")
                    continue

                processed_indices.add(frame_idx)
                frame_start_time = time.perf_counter()

                # --- Get Ground Truth for this frame ---
                gt_for_frame_tuples = self.ground_truth_data.get((frame_idx, cam_id), [])
                if not gt_for_frame_tuples:
                     detections_for_tracker = np.empty((0, 6)) # xyxy, conf, cls
                     num_gt_boxes_frame = 0
                else:
                    num_gt_boxes_frame = len(gt_for_frame_tuples)
                    total_gt_boxes_processed += num_gt_boxes_frame
                    # Convert GT tuples [(obj_id, cx, cy, w, h), ...] to tracker format [x1, y1, x2, y2, conf, cls]
                    boxes_xyxy = []
                    for _, cx, cy, w, h in gt_for_frame_tuples:
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        if w > 0 and h > 0: # Filter out invalid boxes from GT
                            boxes_xyxy.append([x1, y1, x2, y2])

                    if not boxes_xyxy:
                        detections_for_tracker = np.empty((0, 6))
                    else:
                        detections_for_tracker = np.array(boxes_xyxy)
                        confidences = np.ones((len(boxes_xyxy), 1)) # Assume confidence 1.0 for GT
                        class_ids = np.full((len(boxes_xyxy), 1), self.person_class_id) # Use configured class ID
                        detections_for_tracker = np.hstack((detections_for_tracker, confidences, class_ids))

                # --- Update Tracker ---
                # BoxMOT .update() expects detections (Nx6 or Nx7 for OBB) and the frame (BGR HWC)
                tracker_output: Optional[np.ndarray] = self.tracker_instance.update(detections_for_tracker, frame_bgr)

                frame_end_time = time.perf_counter()
                frame_processing_times.append((frame_end_time - frame_start_time) * 1000) # Store time in ms

                # --- Store Tracker Output ---
                # Output format is typically [[x1, y1, x2, y2, track_id, conf, cls, Optional[det_ind]], ...]
                # Shape can vary slightly between trackers (e.g., presence of det_ind)
                output_shape_cols = tracker_output.shape[1] if tracker_output is not None else 7 # Default guess if None

                if tracker_output is not None and tracker_output.size > 0:
                    self.raw_tracker_outputs[(frame_idx, cam_id)] = tracker_output
                    total_tracks_output += len(tracker_output)
                    pbar.set_postfix({"GT": num_gt_boxes_frame, "Tracks": len(tracker_output)})
                else:
                    # Store empty array if tracker returns None or empty, maintain consistent shape if possible
                    self.raw_tracker_outputs[(frame_idx, cam_id)] = np.empty((0, output_shape_cols))
                    pbar.set_postfix({"GT": num_gt_boxes_frame, "Tracks": 0})

            # --- Finalize ---
            pbar.close()
            end_time_total = time.perf_counter()
            total_processing_time_sec = end_time_total - start_time_total

            # Store performance stats in summary_metrics
            self.summary_metrics['perf_total_frames_processed'] = frames_processed_count
            self.summary_metrics['perf_unique_frame_indices_processed'] = len(processed_indices)
            self.summary_metrics['perf_total_processing_time_sec'] = round(total_processing_time_sec, 2)
            self.summary_metrics['perf_avg_frame_processing_time_ms'] = round(np.mean(frame_processing_times), 2) if frame_processing_times else 0
            self.summary_metrics['perf_processing_fps'] = round(frames_processed_count / total_processing_time_sec, 2) if total_processing_time_sec > 0 else 0
            self.summary_metrics['input_total_gt_boxes_fed'] = total_gt_boxes_processed
            self.summary_metrics['output_total_tracked_instances'] = total_tracks_output

            self.processed = True
            logger.info(f"[{run_name_tag}] --- Frame Processing Finished ---")
            logger.info(f"[{run_name_tag}] Processed {frames_processed_count} frames in {total_processing_time_sec:.2f} seconds.")
            logger.info(f"[{run_name_tag}] Total GT boxes fed to tracker: {total_gt_boxes_processed}")
            logger.info(f"[{run_name_tag}] Total tracked instances output: {total_tracks_output}")
            return True

        except Exception as e:
            if 'pbar' in locals() and pbar: pbar.close() # Ensure progress bar is closed on error
            logger.error(f"[{run_name_tag}] Error during frame processing loop: {e}", exc_info=True)
            self.processed = False
            # Store partial performance results if any processing occurred
            end_time_total = time.perf_counter()
            self.summary_metrics['perf_total_frames_processed'] = frames_processed_count
            self.summary_metrics['perf_unique_frame_indices_processed'] = len(processed_indices)
            self.summary_metrics['perf_total_processing_time_sec'] = round(end_time_total - start_time_total, 2)
            # ... potentially other partial stats ...
            return False

    def calculate_metrics(self) -> bool:
        """Calculates summary metrics based on the raw tracker outputs."""
        if not self.processed:
            logger.error("Cannot calculate metrics: Frame processing did not complete successfully.")
            return False

        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        logger.info(f"[{run_name_tag}] Calculating tracking summary metrics...")

        try:
            # Use the basic summary function for now
            tracking_summary = calculate_tracking_summary(self.raw_tracker_outputs)
            # Merge with performance metrics already calculated
            self.summary_metrics.update(tracking_summary)
            logger.info(f"[{run_name_tag}] Tracking Summary Metrics: {self.summary_metrics}")
            self.metrics_calculated = True
            return True

        except Exception as e:
            logger.error(f"[{run_name_tag}] Error calculating tracking summary metrics: {e}", exc_info=True)
            self.metrics_calculated = False
            return False

    def run(self) -> Tuple[bool, TrackingResultSummary, Optional[Any]]:
        """
        Executes the full Tracking+ReID pipeline.

        Returns:
            Tuple[bool, TrackingResultSummary, Optional[Any]]:
                - Success status (bool).
                - Dictionary containing summary metrics.
                - The actual device reported by the tracker instance (if available).
                  Could be torch.device or string ('0', 'mps', 'cpu').
        """
        run_name_tag = f"Trk:{self.tracker_type}_ReID:{self.reid_model_type}"
        success = False
        try:
            if not self.initialize_components():
                return False, self.summary_metrics, self.actual_tracker_device

            if not self.process_frames():
                # Attempt metric calculation even on partial processing
                self.calculate_metrics()
                logger.warning(f"[{run_name_tag}] Frame processing failed. Metrics calculation attempted.")
                return False, self.summary_metrics, self.actual_tracker_device

            if not self.calculate_metrics():
                 logger.warning(f"[{run_name_tag}] Metrics calculation failed after processing.")
                 # Return success=False but with potentially partial summary_metrics
                 return False, self.summary_metrics, self.actual_tracker_device

            success = True

        except Exception as e:
            logger.critical(f"[{run_name_tag}] Unexpected error during Tracking+ReID pipeline execution: {e}", exc_info=True)
            success = False
            # Ensure some summary dict exists, even if empty, on critical failure
            if not self.summary_metrics: self.summary_metrics = {}

        logger.info(f"[{run_name_tag}] Tracking+ReID Pipeline Run completed. Success: {success}")
        return success, self.summary_metrics, self.actual_tracker_device

    def dump_cache(self):
        """Saves CMC cache if applicable."""
        if hasattr(self.tracker_instance, 'cmc') and hasattr(self.tracker_instance.cmc, 'save_cache'):
             try:
                 self.tracker_instance.cmc.save_cache()
                 logger.info(f"[{self.tracker_type}] Saved CMC cache.")
             except Exception as e:
                 logger.warning(f"[{self.tracker_type}] Failed to save CMC cache: {e}")
        elif hasattr(self, 'cmc') and hasattr(self.cmc, 'save_cache'): # Fallback if CMC is direct attribute
             try:
                 self.cmc.save_cache()
                 logger.info(f"[{self.tracker_type}] Saved CMC cache (pipeline level).")
             except Exception as e:
                 logger.warning(f"[{self.tracker_type}] Failed to save CMC cache (pipeline level): {e}")