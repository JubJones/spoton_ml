import logging
import time
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List

import torch

from src.data.loader import FrameDataLoader
from src.evaluation.metrics import load_ground_truth, calculate_metrics_with_map, GroundTruthData, xywh_to_xyxy, \
    gt_tuples_to_xyxy
from src.tracking.strategies import get_strategy, DetectionTrackingStrategy

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Encapsulates the core logic for a detection experiment, now using mAP."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initializes the pipeline with configuration and target device.
        """
        self.config = config
        self.device = device  # Device for model inference
        self.data_loader: Optional[FrameDataLoader] = None
        self.detection_strategy: Optional[DetectionTrackingStrategy] = None
        # Store GT data directly
        self.ground_truth_data: Optional[GroundTruthData] = None
        self.person_class_index: int = config.get("model", {}).get("person_class_id", 0)  # Get from config

        # Store raw results and collected preds/targets for mAP
        self.raw_results: Dict[str, Any] = {}
        self.all_predictions_for_map: List[Dict[str, torch.Tensor]] = []
        self.all_targets_for_map: List[Dict[str, torch.Tensor]] = []

        self.calculated_metrics: Dict[str, Any] = {}
        self.initialized = False
        self.processed = False
        self.metrics_calculated = False

    def initialize_components(self) -> bool:
        """Initializes data loader, detection strategy, and loads ground truth."""
        logger.info("Initializing pipeline components...")
        try:
            # Initialize Data Loader
            self.data_loader = FrameDataLoader(self.config)
            if len(self.data_loader) == 0:
                raise ValueError("Data loader found 0 frame indices to process.")
            logger.info(
                f"Data loader initialized. Processing {len(self.data_loader)} frame indices across "
                f"{len(self.data_loader.active_camera_ids)} cameras: {self.data_loader.active_camera_ids}."
            )

            # Initialize Detection Strategy
            model_config = self.config.get("model", {})
            # Get person_class_id from model config for consistency
            self.person_class_index = model_config.get('person_class_id', 0)
            logger.info(f"Using Person Class Index from config: {self.person_class_index}")
            self.detection_strategy = get_strategy(model_config, self.device)
            logger.info(
                f"Using detection strategy: {self.detection_strategy.__class__.__name__} on device: {self.device}")

            # Load Ground Truth Data (required for mAP)
            try:
                # Pass the determined person_class_index to loader (though not strictly used by it currently)
                self.ground_truth_data, _ = load_ground_truth(
                    self.data_loader.scene_path,
                    self.data_loader.active_camera_ids,
                    self.data_loader.image_filenames,
                    self.person_class_index
                )
                if self.ground_truth_data is None:
                    logger.warning("Ground truth could not be loaded. mAP calculation will be skipped.")
                else:
                    logger.info("Ground truth loaded successfully.")

            except Exception as e:
                logger.error(f"Failed to load ground truth data: {e}", exc_info=True)
                self.ground_truth_data = None  # Ensure it's None on error

            self.initialized = True
            logger.info("Pipeline components initialized successfully.")
            return True

        except (FileNotFoundError, ValueError, RuntimeError, ImportError, Exception) as e:
            logger.critical(f"Failed to initialize pipeline components: {e}", exc_info=True)
            self.initialized = False
            return False

    def process_frames(self) -> bool:
        """Processes frames, collects predictions and targets for mAP calculation."""
        if not self.initialized or not self.data_loader or not self.detection_strategy:
            logger.error("Cannot process frames: Pipeline components not initialized.")
            return False

        logger.info("Starting frame processing loop (collecting data for mAP)...")
        # --- Reset collections ---
        self.all_predictions_for_map = []
        self.all_targets_for_map = []
        results = defaultdict(float)  # Still useful for performance stats
        results['frame_counter'] = 0
        results['total_gt_boxes'] = 0  # Count total GT boxes encountered
        processed_indices = set()
        detections_per_camera = defaultdict(int)
        inference_time_per_camera = defaultdict(float)
        frame_count_per_camera = defaultdict(int)
        gt_available = self.ground_truth_data is not None

        start_time_total = time.perf_counter()
        total_frames_to_process = len(self.data_loader) * len(self.data_loader.active_camera_ids)

        try:
            for frame_idx, cam_id, filename, frame_bgr in self.data_loader:
                if frame_bgr is None:
                    logger.debug(f"Skipping frame {frame_idx} for camera {cam_id} due to load error.")
                    # Still need to add empty target if GT exists for this frame to keep lists aligned
                    if gt_available and self.ground_truth_data is not None:
                        gt_for_frame_tuples = self.ground_truth_data.get((frame_idx, cam_id), [])
                        if gt_for_frame_tuples:  # Only add target if GT exists, otherwise skip
                            gt_boxes_xyxy = gt_tuples_to_xyxy(gt_for_frame_tuples)
                            target = {
                                'boxes': torch.empty((0, 4), dtype=torch.float32),
                                # Empty if frame fails but GT exists? Or skip? Let's skip.
                                'labels': torch.empty(0, dtype=torch.long)
                            }
                            # self.all_predictions_for_map.append({'boxes': torch.empty((0,4)), 'scores': torch.empty(0), 'labels': torch.empty(0)}) # Add empty prediction
                            # self.all_targets_for_map.append(target) # Add empty target
                    continue  # Skip processing if frame loading failed

                # --- Frame Accounting ---
                current_frame_count = int(results['frame_counter']) + 1
                results['frame_counter'] = current_frame_count
                frame_count_per_camera[cam_id] += 1
                processed_indices.add(frame_idx)

                # --- Detection ---
                start_time_inference = time.perf_counter()
                if self.detection_strategy is None: raise RuntimeError("Detection strategy None")  # Type check

                # Returns boxes_xywh, placeholder_track_ids, confidences
                boxes_xywh, _, confidences = self.detection_strategy.process_frame(frame_bgr)
                inference_time_ms = (time.perf_counter() - start_time_inference) * 1000

                num_detections_frame = len(boxes_xywh)
                results['total_inference_time_ms'] += inference_time_ms
                results['total_person_detections'] += num_detections_frame
                detections_per_camera[cam_id] += num_detections_frame
                inference_time_per_camera[cam_id] += inference_time_ms

                # --- Prepare Predictions for mAP ---
                # Convert boxes to xyxy format required by torchmetrics
                pred_boxes_xyxy = xywh_to_xyxy(boxes_xywh)
                pred_dict = {
                    'boxes': torch.tensor(pred_boxes_xyxy, dtype=torch.float32),
                    'scores': torch.tensor(confidences, dtype=torch.float32),
                    # Use the configured person_class_index
                    'labels': torch.tensor([self.person_class_index] * num_detections_frame, dtype=torch.long)
                }

                # --- Prepare Targets for mAP ---
                target_dict = {
                    'boxes': torch.empty((0, 4), dtype=torch.float32),  # Default empty if no GT
                    'labels': torch.empty(0, dtype=torch.long)
                }
                num_gt_boxes_frame = 0
                if gt_available and self.ground_truth_data is not None:
                    gt_for_frame_tuples = self.ground_truth_data.get((frame_idx, cam_id), [])
                    num_gt_boxes_frame = len(gt_for_frame_tuples)
                    if num_gt_boxes_frame > 0:
                        gt_boxes_xyxy = gt_tuples_to_xyxy(gt_for_frame_tuples)
                        target_dict = {
                            'boxes': torch.tensor(gt_boxes_xyxy, dtype=torch.float32),
                            # Assume GT is also for the configured person_class_index
                            'labels': torch.tensor([self.person_class_index] * num_gt_boxes_frame, dtype=torch.long)
                        }
                    # Accumulate total GT count
                    results['total_gt_boxes'] += num_gt_boxes_frame

                # --- Store for aggregate mAP calculation ---
                # Only add if GT was available for the run, otherwise mAP doesn't make sense
                if gt_available:
                    self.all_predictions_for_map.append(pred_dict)
                    self.all_targets_for_map.append(target_dict)

                # --- Logging ---
                if current_frame_count % 100 == 0 or current_frame_count == total_frames_to_process:
                    log_msg = (
                        f"Processed {current_frame_count}/{total_frames_to_process} frames... "
                        f"(Idx {frame_idx}/{len(self.data_loader) - 1}) "
                        f"Cam {cam_id}: {num_detections_frame} dets, {inference_time_ms:.2f}ms"
                    )
                    if gt_available:
                        log_msg += f" | GT Boxes: {num_gt_boxes_frame}"
                    logger.info(log_msg)

            # --- Finalize Performance Stats ---
            results['total_processing_time_sec'] = time.perf_counter() - start_time_total
            results['unique_frame_indices_processed'] = len(processed_indices)
            results['detections_per_camera'] = dict(detections_per_camera)
            results['inference_time_per_camera'] = dict(inference_time_per_camera)
            results['frame_count_per_camera'] = dict(frame_count_per_camera)
            # results['gt_available'] = gt_available # Store if GT was loaded

            self.raw_results = dict(results)  # Store performance results
            self.processed = True
            logger.info("--- Frame Processing Finished (Data collected for mAP) ---")
            # Log sizes for debugging
            logger.info(
                f"Collected {len(self.all_predictions_for_map)} predictions and {len(self.all_targets_for_map)} targets for mAP calculation.")
            return True

        except Exception as e:
            logger.error(f"Error during frame processing loop: {e}", exc_info=True)
            self.processed = False
            # Store partial results if any
            results['total_processing_time_sec'] = time.perf_counter() - start_time_total
            results['unique_frame_indices_processed'] = len(processed_indices)
            results['detections_per_camera'] = dict(detections_per_camera)
            results['inference_time_per_camera'] = dict(inference_time_per_camera)
            results['frame_count_per_camera'] = dict(frame_count_per_camera)
            # results['gt_available'] = gt_available
            self.raw_results = dict(results)  # Store potentially incomplete results
            return False

    def calculate_metrics(self) -> bool:
        """Calculates aggregate performance metrics and mAP."""
        if not self.processed:
            logger.error("Cannot calculate metrics: Frame processing did not complete successfully.")
            return False
        if not self.raw_results:
            logger.error("Cannot calculate metrics: No raw performance results available.")
            return False
        if not self.data_loader:
            logger.error("Cannot calculate metrics: Data loader not available.")
            return False
        # mAP calculation requires ground truth
        if self.ground_truth_data is None:
            logger.warning(
                "Ground truth not available. Skipping mAP calculation, calculating performance metrics only.")
            # Fallback to basic calculation without mAP part? Or let calculate_metrics_with_map handle it.
            # Let's rely on calculate_metrics_with_map to handle the missing GT data gracefully.
            pass

        try:
            # Use the modified function that includes mAP
            self.calculated_metrics = calculate_metrics_with_map(
                self.raw_results,
                self.data_loader.active_camera_ids,
                self.all_predictions_for_map,  # Pass collected data
                self.all_targets_for_map,  # Pass collected data
                self.person_class_index,  # Pass class index
                torch.device('cpu')  # Use CPU for metric calculation typically
            )
            self.metrics_calculated = bool(self.calculated_metrics)
            return self.metrics_calculated
        except Exception as e:
            logger.error(f"Error calculating aggregate metrics (including mAP): {e}", exc_info=True)
            self.metrics_calculated = False
            return False

    # run() method remains largely the same, calling the modified methods
    def run(self) -> Tuple[bool, Dict[str, Any], Optional[List[str]], Optional[int]]:
        """
        Executes the full pipeline: initialization, processing, and metric calculation (now with mAP).
        """
        if not self.initialize_components():
            return False, {}, None, None

        if self.data_loader is None:
            logger.critical("Data loader is None even after successful initialization.")
            return False, {}, None, None

        active_cameras = self.data_loader.active_camera_ids
        num_frame_indices = len(self.data_loader)

        if not self.process_frames():
            # Attempt metrics calculation even on failure, might have partial results
            metrics_ok = self.calculate_metrics()
            logger.warning(
                f"Frame processing failed. Metrics calculation attempted: {'Success' if metrics_ok else 'Failed'}")
            # Return failure, but include any calculated metrics
            return False, self.calculated_metrics, active_cameras, num_frame_indices

        if not self.calculate_metrics():
            return False, {}, active_cameras, num_frame_indices

        # Check if any frames were actually processed (check raw results)
        if self.raw_results.get('frame_counter', 0) == 0:
            logger.warning("Pipeline ran, but zero frames were successfully processed.")
            return False, self.calculated_metrics, active_cameras, num_frame_indices

        return True, self.calculated_metrics, active_cameras, num_frame_indices
