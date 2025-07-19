
import logging
import time
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List

import torch

from src.components.data.loader import FrameDataLoader
# --- Corrected Import ---
from src.components.evaluation.metrics import (
    load_ground_truth,
    calculate_detection_metrics_with_map,
    GroundTruthData,
    xywh_to_xyxy,
    gt_tuples_to_xyxy,
    MAPPred, # Import types if needed internally
    MAPTarget
)
# --- End Correction ---
from src.detection.strategies import get_strategy, DetectionTrackingStrategy

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Encapsulates the core logic for a detection experiment, using mAP."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initializes the pipeline with configuration and target device.
        """
        self.config = config
        self.device = device  # Device for model inference
        self.data_loader: Optional[FrameDataLoader] = None
        self.detection_strategy: Optional[DetectionTrackingStrategy] = None
        self.ground_truth_data: Optional[GroundTruthData] = None
        self.person_class_index: int = config.get("model", {}).get("person_class_id", 1) # Default to 1 (person) for detection

        self.raw_results: Dict[str, Any] = {}
        self.all_predictions_for_map: List[MAPPred] = []
        self.all_targets_for_map: List[MAPTarget] = []

        self.calculated_metrics: Dict[str, Any] = {}
        self.initialized = False
        self.processed = False
        self.metrics_calculated = False

    def initialize_components(self) -> bool:
        """Initializes data loader, detection strategy, and loads ground truth."""
        logger.info("Initializing detection pipeline components...")
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
            self.person_class_index = model_config.get('person_class_id', 1) # Get from config, default 1
            logger.info(f"Using Person Class Index for detection strategy: {self.person_class_index}")
            self.detection_strategy = get_strategy(model_config, self.device)
            logger.info(
                f"Using detection strategy: {self.detection_strategy.__class__.__name__} on device: {self.device}")

            # Load Ground Truth Data (required for mAP)
            try:
                # Pass the determined person_class_index to loader
                self.ground_truth_data, _ = load_ground_truth(
                    self.data_loader.scene_path,
                    self.data_loader.active_camera_ids,
                    self.data_loader.image_filenames,
                    self.person_class_index # Pass the class index used by GT (often 0 or 1)
                )
                if self.ground_truth_data is None:
                    logger.warning("Ground truth could not be loaded. mAP calculation will be skipped.")
                else:
                    logger.info("Ground truth loaded successfully.")

            except Exception as e:
                logger.error(f"Failed to load ground truth data: {e}", exc_info=True)
                self.ground_truth_data = None

            self.initialized = True
            logger.info("Detection pipeline components initialized successfully.")
            return True

        except (FileNotFoundError, ValueError, RuntimeError, ImportError, Exception) as e:
            logger.critical(f"Failed to initialize detection pipeline components: {e}", exc_info=True)
            self.initialized = False
            return False

    def process_frames(self) -> bool:
        """Processes frames, collects predictions and targets for mAP calculation."""
        if not self.initialized or not self.data_loader or not self.detection_strategy:
            logger.error("Cannot process frames: Pipeline components not initialized.")
            return False

        logger.info("Starting frame processing loop (collecting data for mAP)...")
        self.all_predictions_for_map = []
        self.all_targets_for_map = []
        results = defaultdict(float)
        results['frame_counter'] = 0
        results['total_gt_boxes'] = 0
        processed_indices = set()
        detections_per_camera = defaultdict(int)
        inference_time_per_camera = defaultdict(float)
        frame_count_per_camera = defaultdict(int)
        gt_available = self.ground_truth_data is not None

        start_time_total = time.perf_counter()
        total_frames_to_process_approx = len(self.data_loader) * len(self.data_loader.active_camera_ids)
        pbar = tqdm(total=total_frames_to_process_approx, desc="Detecting")

        try:
            for frame_idx, cam_id, filename, frame_bgr in self.data_loader:
                pbar.update(1)
                if frame_bgr is None:
                    logger.debug(f"Skipping frame {frame_idx} for camera {cam_id} due to load error.")
                    continue

                # --- Frame Accounting ---
                current_frame_count = int(results['frame_counter']) + 1
                results['frame_counter'] = current_frame_count
                frame_count_per_camera[cam_id] += 1
                processed_indices.add(frame_idx)

                # --- Detection ---
                start_time_inference = time.perf_counter()
                if self.detection_strategy is None: raise RuntimeError("Detection strategy None")
                boxes_xywh, _, confidences = self.detection_strategy.process_frame(frame_bgr)
                inference_time_ms = (time.perf_counter() - start_time_inference) * 1000

                num_detections_frame = len(boxes_xywh)
                results['total_inference_time_ms'] += inference_time_ms
                results['total_person_detections'] += num_detections_frame
                detections_per_camera[cam_id] += num_detections_frame
                inference_time_per_camera[cam_id] += inference_time_ms

                # --- Prepare Predictions for mAP ---
                pred_boxes_xyxy = xywh_to_xyxy(boxes_xywh)
                pred_dict: MAPPred = {
                    'boxes': torch.tensor(pred_boxes_xyxy, dtype=torch.float32),
                    'scores': torch.tensor(confidences, dtype=torch.float32),
                    # Use the configured person_class_index for predictions
                    'labels': torch.tensor([self.person_class_index] * num_detections_frame, dtype=torch.long)
                }

                # --- Prepare Targets for mAP ---
                target_dict: MAPTarget = {
                    'boxes': torch.empty((0, 4), dtype=torch.float32),
                    'labels': torch.empty(0, dtype=torch.long)
                }
                num_gt_boxes_frame = 0
                if gt_available and self.ground_truth_data is not None:
                    gt_for_frame_tuples = self.ground_truth_data.get((frame_idx, cam_id), [])
                    num_gt_boxes_frame = len(gt_for_frame_tuples)
                    if num_gt_boxes_frame > 0:
                        valid_gt_boxes_xyxy = []
                        for _, cx, cy, w, h in gt_for_frame_tuples:
                            # Basic check (redundant if loader is good, but safe)
                            if w <= 0 or h <= 0: continue
                            x1, y1 = cx - w / 2, cy - h / 2
                            x2, y2 = cx + w / 2, cy + h / 2
                            # Clip GT boxes to image boundaries for safety
                            img_h, img_w = frame_bgr.shape[:2]
                            x1 = max(0.0, x1); y1 = max(0.0, y1)
                            x2 = min(float(img_w), x2); y2 = min(float(img_h), y2)
                            if x2 > x1 and y2 > y1:
                                valid_gt_boxes_xyxy.append([x1, y1, x2, y2])

                        if valid_gt_boxes_xyxy:
                             target_dict = {
                                 'boxes': torch.tensor(valid_gt_boxes_xyxy, dtype=torch.float32),
                                 # Assign the configured person_class_index to GT labels
                                 'labels': torch.tensor([self.person_class_index] * len(valid_gt_boxes_xyxy), dtype=torch.long)
                             }
                        else: # Reset target if all GT boxes were invalid after clipping
                             target_dict = {
                                 'boxes': torch.empty((0, 4), dtype=torch.float32),
                                 'labels': torch.empty(0, dtype=torch.long)
                             }
                    results['total_gt_boxes'] += len(valid_gt_boxes_xyxy) if 'boxes' in target_dict else 0 # Count valid GT

                # --- Store for aggregate mAP calculation ---
                if gt_available:
                    self.all_predictions_for_map.append(pred_dict)
                    self.all_targets_for_map.append(target_dict)

                pbar.set_postfix_str(f"Cam {cam_id}: {num_detections_frame} dets ({inference_time_ms:.1f}ms)")

            # --- Finalize Performance Stats ---
            results['total_processing_time_sec'] = time.perf_counter() - start_time_total
            results['unique_frame_indices_processed'] = len(processed_indices)
            results['detections_per_camera'] = dict(detections_per_camera)
            results['inference_time_per_camera'] = dict(inference_time_per_camera)
            results['frame_count_per_camera'] = dict(frame_count_per_camera)

            self.raw_results = dict(results)
            self.processed = True
            logger.info("--- Frame Processing Finished (Data collected for mAP) ---")
            logger.info(f"Collected {len(self.all_predictions_for_map)} predictions and {len(self.all_targets_for_map)} targets for mAP.")
            pbar.close()
            return True

        except Exception as e:
            pbar.close()
            logger.error(f"Error during frame processing loop: {e}", exc_info=True)
            self.processed = False
            # Store partial results if any
            results['total_processing_time_sec'] = time.perf_counter() - start_time_total
            results['unique_frame_indices_processed'] = len(processed_indices)
            results['detections_per_camera'] = dict(detections_per_camera)
            results['inference_time_per_camera'] = dict(inference_time_per_camera)
            results['frame_count_per_camera'] = dict(frame_count_per_camera)
            self.raw_results = dict(results)
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

        try:
            # Use the correctly named function
            self.calculated_metrics = calculate_detection_metrics_with_map(
                self.raw_results,
                self.data_loader.active_camera_ids,
                self.all_predictions_for_map,
                self.all_targets_for_map,
                self.person_class_index, # Pass the class index used
                torch.device('cpu') # Use CPU for metric calculation
            )
            self.metrics_calculated = bool(self.calculated_metrics)
            return self.metrics_calculated
        except Exception as e:
            logger.error(f"Error calculating aggregate metrics (including mAP): {e}", exc_info=True)
            self.metrics_calculated = False
            return False

    def run(self) -> Tuple[bool, Dict[str, Any], Optional[List[str]], Optional[int]]:
        """
        Executes the full pipeline: initialization, processing, and metric calculation.
        """
        if not self.initialize_components():
            return False, {}, None, None

        if self.data_loader is None:
             logger.critical("Data loader is None even after successful initialization.")
             return False, {}, None, None

        active_cameras = self.data_loader.active_camera_ids
        num_frame_indices = len(self.data_loader) # Number of unique frame filenames

        if not self.process_frames():
            metrics_ok = self.calculate_metrics()
            logger.warning(f"Frame processing failed. Metrics calculation attempted: {'Success' if metrics_ok else 'Failed'}")
            return False, self.calculated_metrics, active_cameras, num_frame_indices

        if not self.calculate_metrics():
            return False, {}, active_cameras, num_frame_indices # Metrics failed after processing

        if self.raw_results.get('frame_counter', 0) == 0:
            logger.warning("Pipeline ran, but zero frames were successfully processed.")
            # Return success=False because no frames processed, but include metrics (likely 0s)
            return False, self.calculated_metrics, active_cameras, num_frame_indices

        return True, self.calculated_metrics, active_cameras, num_frame_indices