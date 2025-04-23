import logging
import time
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List

import torch

from src.data.loader import FrameDataLoader
from src.evaluation.metrics import load_ground_truth, evaluate_frame_detections, calculate_aggregate_metrics, \
    GroundTruthData
from src.tracking.strategies import get_strategy, DetectionTrackingStrategy

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Encapsulates the core logic for a detection experiment."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initializes the pipeline with configuration and target device.
        """
        self.config = config
        self.device = device
        self.data_loader: Optional[FrameDataLoader] = None
        self.detection_strategy: Optional[DetectionTrackingStrategy] = None
        self.ground_truth_data: Optional[GroundTruthData] = None
        self.raw_results: Dict[str, Any] = {}
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
            self.detection_strategy = get_strategy(model_config, self.device)
            logger.info(
                f"Using detection strategy: {self.detection_strategy.__class__.__name__} on device: {self.device}")

            # Load Ground Truth Data (optional)
            try:
                self.ground_truth_data = load_ground_truth(
                    self.data_loader.scene_path,
                    self.data_loader.active_camera_ids,
                    self.data_loader.image_filenames
                )
            except Exception as e:
                logger.error(f"Failed to load ground truth data: {e}", exc_info=True)
                # Continue without GT if loading fails, evaluation will be skipped
                self.ground_truth_data = None

            self.initialized = True
            logger.info("Pipeline components initialized successfully.")
            return True

        except (FileNotFoundError, ValueError, RuntimeError, ImportError, Exception) as e:
            logger.critical(f"Failed to initialize pipeline components: {e}", exc_info=True)
            self.initialized = False
            return False

    def process_frames(self) -> bool:
        """Processes frames using the initialized components and collects raw results."""
        if not self.initialized or not self.data_loader or not self.detection_strategy:
            logger.error("Cannot process frames: Pipeline components not initialized.")
            return False

        logger.info("Starting frame processing loop...")
        results = defaultdict(float)
        results['frame_counter'] = 0
        results['total_tp'] = 0
        results['total_fp'] = 0
        results['total_fn'] = 0
        results['total_gt_boxes'] = 0
        processed_indices = set()
        detections_per_camera = defaultdict(int)
        inference_time_per_camera = defaultdict(float)
        frame_count_per_camera = defaultdict(int)
        gt_exists_for_run = self.ground_truth_data is not None
        iou_threshold = self.config.get("evaluation", {}).get("iou_threshold", 0.5)

        start_time_total = time.perf_counter()
        total_frames_to_process = len(self.data_loader) * len(self.data_loader.active_camera_ids)

        try:
            for frame_idx, cam_id, filename, frame_bgr in self.data_loader:
                if frame_bgr is None:
                    logger.debug(f"Skipping frame {frame_idx} for camera {cam_id} due to load error.")
                    continue  # Skip if frame loading failed

                current_frame_count = int(results['frame_counter']) + 1 # Use next value for modulo check

                results['frame_counter'] = current_frame_count
                frame_count_per_camera[cam_id] += 1
                processed_indices.add(frame_idx)

                # --- Detection ---
                start_time_inference = time.perf_counter()
                # Ensure detection strategy is not None (type hint satisfaction)
                if self.detection_strategy is None:
                    raise RuntimeError("Detection strategy became None during processing.")
                boxes_xywh, _, confidences = self.detection_strategy.process_frame(frame_bgr)
                inference_time_ms = (time.perf_counter() - start_time_inference) * 1000

                num_detections_frame = len(boxes_xywh)
                results['total_inference_time_ms'] += inference_time_ms
                results['total_person_detections'] += num_detections_frame
                detections_per_camera[cam_id] += num_detections_frame
                inference_time_per_camera[cam_id] += inference_time_ms

                # --- Evaluation (if GT available) ---
                frame_tp, frame_fp, frame_fn = 0, 0, 0
                num_gt_boxes_frame = 0
                if gt_exists_for_run and self.ground_truth_data is not None:
                    gt_for_frame = self.ground_truth_data.get((frame_idx, cam_id), [])
                    # Extract only x, y, w, h for evaluation function
                    gt_boxes_xywh = [[x, y, w, h] for _, x, y, w, h in gt_for_frame]
                    num_gt_boxes_frame = len(gt_boxes_xywh)
                    results['total_gt_boxes'] += num_gt_boxes_frame

                    if num_detections_frame > 0 or num_gt_boxes_frame > 0:
                        frame_tp, frame_fp, frame_fn = evaluate_frame_detections(
                            boxes_xywh, gt_boxes_xywh, iou_threshold
                        )
                        results['total_tp'] += frame_tp
                        results['total_fp'] += frame_fp
                        results['total_fn'] += frame_fn

                if current_frame_count % 100 == 0 or current_frame_count == total_frames_to_process:
                    log_msg = (
                        f"Processed {current_frame_count}/{total_frames_to_process} frames... "
                        f"(Idx {frame_idx}/{len(self.data_loader) - 1}) "
                        f"Cam {cam_id}: {num_detections_frame} dets, {inference_time_ms:.2f}ms"
                    )
                    if gt_exists_for_run:
                        log_msg += f" | Eval: TP={frame_tp}, FP={frame_fp}, FN={frame_fn} (GT={num_gt_boxes_frame})"
                    logger.info(log_msg)

            results['total_processing_time_sec'] = time.perf_counter() - start_time_total
            results['unique_frame_indices_processed'] = len(processed_indices)
            results['detections_per_camera'] = dict(detections_per_camera)
            results['inference_time_per_camera'] = dict(inference_time_per_camera)
            results['frame_count_per_camera'] = dict(frame_count_per_camera)
            results['gt_available'] = gt_exists_for_run

            self.raw_results = dict(results)
            self.processed = True
            logger.info("--- Frame Processing Finished ---")
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
            results['gt_available'] = gt_exists_for_run # Keep consistent
            self.raw_results = dict(results) # Store potentially incomplete results
            return False


    def calculate_metrics(self) -> bool:
        """Calculates aggregate metrics based on the raw processing results."""
        if not self.processed:
            logger.error("Cannot calculate metrics: Frame processing did not complete successfully.")
            return False
        if not self.raw_results:
             logger.error("Cannot calculate metrics: No raw results available.")
             return False
        if not self.data_loader:
            logger.error("Cannot calculate metrics: Data loader not available.")
            return False

        try:
            self.calculated_metrics = calculate_aggregate_metrics(
                self.raw_results,
                self.data_loader.active_camera_ids
            )
            self.metrics_calculated = bool(self.calculated_metrics) # True if dict is not empty
            return self.metrics_calculated
        except Exception as e:
            logger.error(f"Error calculating aggregate metrics: {e}", exc_info=True)
            self.metrics_calculated = False
            return False


    def run(self) -> Tuple[bool, Dict[str, Any], Optional[List[str]], Optional[int]]:
        """
        Executes the full pipeline: initialization, processing, and metric calculation.
        """
        if not self.initialize_components():
            return False, {}, None, None

        # Ensure data_loader is available after successful initialization
        if self.data_loader is None:
             logger.critical("Data loader is None even after successful initialization. This should not happen.")
             return False, {}, None, None

        active_cameras = self.data_loader.active_camera_ids
        num_frame_indices = len(self.data_loader)

        if not self.process_frames():
            # Even if processing fails, try to calculate metrics on partial results
            # but the overall run is marked as failed.
            metrics_ok = self.calculate_metrics()
            logger.warning(f"Frame processing failed. Metrics calculation attempted: {'Success' if metrics_ok else 'Failed'}")
            return False, self.calculated_metrics, active_cameras, num_frame_indices


        if not self.calculate_metrics():
             return False, {}, active_cameras, num_frame_indices # Metrics calculation failed

        # Check if any frames were actually processed
        if self.raw_results.get('frame_counter', 0) == 0:
            logger.warning("Pipeline ran, but zero frames were successfully processed.")
            return False, self.calculated_metrics, active_cameras, num_frame_indices

        return True, self.calculated_metrics, active_cameras, num_frame_indices