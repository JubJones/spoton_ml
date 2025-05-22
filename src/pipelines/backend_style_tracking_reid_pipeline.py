# File: jubjones-spoton_ml/src/pipelines/backend_style_tracking_reid_pipeline.py
"""
Pipeline for running tracking and Re-ID using logic adapted from the SpotOn backend.
Uses ground truth bounding boxes as input to the tracker.
"""
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Set
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
import mlflow # For getting active run_id

# --- Optional: Import pandas and motmetrics ---
try:
    import pandas as pd # type: ignore
    import motmetrics as mm # type: ignore
    REQUESTED_MOT_METRICS = ['mota', 'motp', 'idf1', 'idsw', 'num_matches', 'num_false_positives', 'num_misses', 'num_switches']
    MOTMETRICS_AVAILABLE = True
except ImportError:
    pd = None 
    mm = None 
    REQUESTED_MOT_METRICS = []
    MOTMETRICS_AVAILABLE = False
    logging.warning("`motmetrics` or `pandas` not found. Install them (`pip install motmetrics pandas`) to calculate standard MOT metrics.")


# --- Local Imports ---
try:
    from src.data.loader import FrameDataLoader
    from src.evaluation.metrics import load_ground_truth, GroundTruthData
    from src.utils.reid_device_utils import get_reid_device_specifier_string
    from src.tracking_backend_logic.common_types_adapter import (
        CameraID, TrackID, GlobalID, FeatureVector, TrackKey, BoundingBoxXYXY,
        HandoffTriggerInfo # Only HandoffTriggerInfo needed here from this group
    )
    from src.tracking_backend_logic.botsort_tracker_adapter import BotSortTrackerAdapter
    from src.tracking_backend_logic.camera_tracker_factory_adapter import CameraTrackerFactoryAdapter
    from src.tracking_backend_logic.reid_manager_adapter import ReIDManagerAdapter
    from src.tracking_backend_logic.handoff_logic_adapter import HandoffLogicAdapter # Import the class
except ImportError as e:
    logging.error(f"Error importing pipeline dependencies: {e}", exc_info=True)
    import sys
    _project_root_fallback = Path(__file__).parent.parent.parent
    if str(_project_root_fallback) not in sys.path: sys.path.insert(0, str(_project_root_fallback))
    from src.data.loader import FrameDataLoader
    from src.evaluation.metrics import load_ground_truth, GroundTruthData
    from src.utils.reid_device_utils import get_reid_device_specifier_string
    from src.tracking_backend_logic.common_types_adapter import (
        CameraID, TrackID, GlobalID, FeatureVector, TrackKey, BoundingBoxXYXY,
        HandoffTriggerInfo
    )
    from src.tracking_backend_logic.botsort_tracker_adapter import BotSortTrackerAdapter
    from src.tracking_backend_logic.camera_tracker_factory_adapter import CameraTrackerFactoryAdapter
    from src.tracking_backend_logic.reid_manager_adapter import ReIDManagerAdapter
    from src.tracking_backend_logic.handoff_logic_adapter import HandoffLogicAdapter

logger = logging.getLogger(__name__)

TrackingReidResultSummary = Dict[str, Any]

class BackendStyleTrackingReidPipeline:
    def __init__(self, config: Dict[str, Any], device: torch.device, project_root: Path):
        self.config = config
        self.preferred_device = device
        self.project_root = project_root

        self.data_loader: Optional[FrameDataLoader] = None
        self.ground_truth_data: Optional[GroundTruthData] = None
        self.tracker_factory: Optional[CameraTrackerFactoryAdapter] = None
        self.reid_manager: Optional[ReIDManagerAdapter] = None
        self.handoff_logic_adapter: Optional[HandoffLogicAdapter] = None # Added

        self.person_class_id = config.get("evaluation", {}).get("person_class_id", 0)
        
        self.reid_config_from_yaml = config.get("reid_params", {})
        self.handoff_config_from_yaml = config.get("handoff_config", {})
        
        self.raw_tracker_outputs_with_global_ids: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
        self.summary_metrics: TrackingReidResultSummary = {}
        self.actual_tracker_devices: Dict[CameraID, Any] = {}

        self.initialized = False
        self.processed = False
        self.metrics_calculated = False

        tracker_type_from_cfg = config.get('tracker_params',{}).get('type','botsort_adapter')
        reid_type_from_cfg = self.reid_config_from_yaml.get('model_type','clip_adapter')
        self.run_name_tag = f"TrkAdap_{tracker_type_from_cfg}_ReIDAdap_{reid_type_from_cfg}"

    def initialize_components(self) -> bool:
        logger.info(f"[{self.run_name_tag}] Initializing Backend-Style Tracking+ReID pipeline components...")
        try:
            # 1. Data Loader
            self.data_loader = FrameDataLoader(self.config)
            if not self.data_loader.active_camera_ids:
                raise ValueError("Data loader found 0 active cameras to process.")
            if len(self.data_loader) == 0:
                raise ValueError("Data loader found 0 frame indices/filenames to process.")
            logger.info(
                f"[{self.run_name_tag}] Data loader initialized for {len(self.data_loader.active_camera_ids)} active cameras: "
                f"{self.data_loader.active_camera_ids}. Processing {len(self.data_loader)} unique frame indices."
            )

            # 2. Ground Truth
            self.ground_truth_data, _ = load_ground_truth(
                self.data_loader.scene_path,
                self.data_loader.active_camera_ids,
                self.data_loader.image_filenames,
                self.person_class_id
            )
            if not self.ground_truth_data:
                logger.warning("Ground truth data (gt.txt) could not be loaded or is empty. MOT metrics will be affected.")
            else:
                logger.info(f"[{self.run_name_tag}] Ground truth loaded for {len(self.ground_truth_data)} (frame_idx, cam_id) pairs.")

            # 3. Initialize Handoff Logic Adapter
            logger.info(f"[{self.run_name_tag}] Initializing HandoffLogicAdapter...")
            self.handoff_logic_adapter = HandoffLogicAdapter(
                handoff_config_dict=self.handoff_config_from_yaml,
                project_root=self.project_root
            )
            # _parse_config_dict is called within HandoffLogicAdapter's __init__
            logger.info(f"[{self.run_name_tag}] HandoffLogicAdapter initialized and config parsed.")

            # 4. Tracker Factory Adapter
            tracker_params_cfg = self.config.get("tracker_params", {})
            reid_weights_rel_path = self.reid_config_from_yaml.get("weights_path", "clip_market1501.pt")
            weights_base_dir_str = self.config.get("data", {}).get("weights_base_dir", "weights/reid")
            reid_full_weights_path = (self.project_root / weights_base_dir_str / reid_weights_rel_path).resolve()
            if not reid_full_weights_path.is_file():
                raise FileNotFoundError(f"Re-ID weights file for tracker not found: {reid_full_weights_path}")

            self.tracker_factory = CameraTrackerFactoryAdapter(
                reid_weights_path=reid_full_weights_path,
                device=self.preferred_device,
                tracker_params={
                    "half_precision": tracker_params_cfg.get("half_precision", False),
                    "per_class": tracker_params_cfg.get("per_class", False)
                }
            )
            self.tracker_factory.preload_prototype_tracker() 
            logger.info(f"[{self.run_name_tag}] CameraTrackerFactoryAdapter initialized and prototype preloaded.")

            # 5. Re-ID Manager Adapter
            active_mlflow_run = mlflow.active_run()
            run_id_for_context = active_mlflow_run.info.run_id if active_mlflow_run else "local_run_context_reid"
            self.reid_manager = ReIDManagerAdapter(
                run_id_context=run_id_for_context,
                reid_config=self.reid_config_from_yaml,
                handoff_config=self.handoff_config_from_yaml
            )
            logger.info(f"[{self.run_name_tag}] ReIDManagerAdapter initialized.")

            self.initialized = True
            return True
        except Exception as e:
            logger.critical(f"[{self.run_name_tag}] Failed to initialize pipeline components: {e}", exc_info=True)
            self.initialized = False
            return False

    def _parse_tracker_output_adapter(
        self, camera_id: CameraID, tracker_output_np: np.ndarray
    ) -> List[Dict[str, Any]]:
        parsed_tracks = []
        if tracker_output_np is None or tracker_output_np.size == 0:
            return parsed_tracks
        num_cols = tracker_output_np.shape[1]
        for row_idx, row in enumerate(tracker_output_np):
            try:
                x1, y1, x2, y2 = map(float, row[0:4])
                track_id_val = row[4]
                if not np.isfinite(track_id_val) or track_id_val < 0: continue
                track_id_int = int(track_id_val)
                if x2 <= x1 or y2 <= y1: continue
                confidence = float(row[5]) if num_cols > 5 and np.isfinite(row[5]) else 0.0
                class_id = int(row[6]) if num_cols > 6 and np.isfinite(row[6]) else self.person_class_id
                feature_vector: Optional[FeatureVector] = None
                if num_cols > 7:
                    feature_data = row[7:]
                    if feature_data.size > 0 and np.isfinite(feature_data).all():
                        feature_vector = FeatureVector(feature_data.astype(np.float32))
                parsed_tracks.append({
                    "track_key": TrackKey((camera_id, TrackID(track_id_int))),
                    "bbox_xyxy": BoundingBoxXYXY([x1, y1, x2, y2]),
                    "confidence": confidence, "class_id": class_id, "feature_vector": feature_vector
                })
            except (ValueError, IndexError, TypeError) as e:
                logger.warning(f"Error parsing BotSortAdapter output row {row_idx} for cam {camera_id}: {row}. Error: {e}", exc_info=False)
        return parsed_tracks

    def process_frames(self) -> bool:
        if not self.initialized or not self.data_loader or not self.tracker_factory or \
           not self.reid_manager or self.ground_truth_data is None or not self.handoff_logic_adapter:
            logger.error("Cannot process frames: Pipeline not fully initialized or critical components missing.")
            return False

        logger.info(f"[{self.run_name_tag}] Starting frame processing loop (backend style)...")
        frame_processing_times_ms = []
        total_gt_boxes_fed_to_tracker = 0
        total_tracks_output_from_botsort = 0
        processed_frame_indices_set: Set[int] = set()
        num_cam_frame_instances_processed = 0
        start_time_total_processing = time.perf_counter()
        total_iterations = len(self.data_loader.image_filenames) * len(self.data_loader.active_camera_ids)
        pbar = tqdm(total=total_iterations, desc=f"Tracking ({self.run_name_tag})")

        if self.tracker_factory: self.tracker_factory.reset_all_trackers()

        for frame_idx in range(len(self.data_loader.image_filenames)):
            current_frame_features: Dict[TrackKey, FeatureVector] = {}
            current_frame_active_track_keys: Set[TrackKey] = set()
            current_frame_handoff_triggers: List[HandoffTriggerInfo] = []

            for cam_id in self.data_loader.active_camera_ids:
                filename = self.data_loader.image_filenames[frame_idx]
                cam_dir_path = self.data_loader.camera_image_dirs[cam_id]
                image_path = cam_dir_path / filename
                frame_bgr: Optional[np.ndarray] = None
                if image_path.is_file():
                    try:
                        img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
                        frame_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                        if frame_bgr is None or frame_bgr.size == 0: frame_bgr = None
                    except Exception: frame_bgr = None
                
                pbar.update(1)
                num_cam_frame_instances_processed +=1
                if frame_bgr is None: continue

                tracker_instance = self.tracker_factory.get_tracker(cam_id)
                if cam_id not in self.actual_tracker_devices and hasattr(tracker_instance, 'device'):
                     self.actual_tracker_devices[cam_id] = tracker_instance.device

                frame_proc_start_time = time.perf_counter()
                img_h, img_w = frame_bgr.shape[:2]
                gt_for_frame_cam = self.ground_truth_data.get((frame_idx, cam_id), [])
                detections_for_tracker_input = []
                for _, cx, cy, w_gt, h_gt in gt_for_frame_cam:
                    if w_gt > 0 and h_gt > 0:
                        x1,y1,x2,y2 = cx-w_gt/2,cy-h_gt/2,cx+w_gt/2,cy+h_gt/2
                        x1_c,y1_c,x2_c,y2_c = max(0.0,x1),max(0.0,y1),min(float(img_w),x2),min(float(img_h),y2)
                        if x2_c > x1_c and y2_c > y1_c:
                            detections_for_tracker_input.append([x1_c,y1_c,x2_c,y2_c,1.0,self.person_class_id])
                
                total_gt_boxes_fed_to_tracker += len(detections_for_tracker_input)
                detections_np_input = np.array(detections_for_tracker_input, dtype=np.float32) \
                    if detections_for_tracker_input else np.empty((0,6),dtype=np.float32)

                raw_tracker_output_np = tracker_instance.update(detections_np_input, frame_bgr)
                total_tracks_output_from_botsort += len(raw_tracker_output_np) if raw_tracker_output_np.size > 0 else 0
                
                parsed_tracks_this_cam = self._parse_tracker_output_adapter(cam_id, raw_tracker_output_np)
                for track_info in parsed_tracks_this_cam:
                    current_frame_active_track_keys.add(track_info["track_key"])
                    if track_info["feature_vector"] is not None:
                        current_frame_features[track_info["track_key"]] = track_info["feature_vector"]

                if raw_tracker_output_np.size > 0:
                    assert self.reid_manager and self.data_loader and self.handoff_logic_adapter # Type hints
                    for track_info_for_handoff in parsed_tracks_this_cam:
                        bbox_for_handoff_check = np.array(track_info_for_handoff["bbox_xyxy"])
                        trigger_for_this_track = self.handoff_logic_adapter.check_exit_rules(
                            environment_id=self.data_loader.selected_env, # type: ignore
                            camera_id=cam_id,
                            bbox_xyxy=bbox_for_handoff_check, # Pass single bbox
                            frame_shape=(img_h, img_w)
                        )
                        if trigger_for_this_track:
                            updated_trigger = HandoffTriggerInfo(
                                source_track_key=track_info_for_handoff["track_key"],
                                rule=trigger_for_this_track.rule,
                                source_bbox=trigger_for_this_track.source_bbox
                            )
                            current_frame_handoff_triggers.append(updated_trigger)
                
                for track_info in parsed_tracks_this_cam:
                    self.raw_tracker_outputs_with_global_ids[(frame_idx, str(cam_id))].append({
                        "bbox_xyxy": track_info["bbox_xyxy"], "track_id": track_info["track_key"][1],
                        "global_id": None, "confidence": track_info["confidence"], "class_id": track_info["class_id"]
                    })
                frame_processing_times_ms.append((time.perf_counter() - frame_proc_start_time) * 1000)
            
            processed_frame_indices_set.add(frame_idx)
            if self.reid_manager and (current_frame_features or current_frame_active_track_keys):
                self.reid_manager.associate_features_and_update_state(
                    current_frame_features, current_frame_active_track_keys,
                    {trigger.source_track_key: trigger for trigger in current_frame_handoff_triggers},
                    frame_idx
                )
        pbar.close()
        
        if self.reid_manager:
            for (f_idx, c_id_str), track_list_for_mot in self.raw_tracker_outputs_with_global_ids.items():
                for track_dict_for_mot in track_list_for_mot:
                    local_track_id_mot = track_dict_for_mot["track_id"]
                    track_key_lookup = (CameraID(c_id_str), TrackID(local_track_id_mot))
                    assigned_gid = self.reid_manager.track_to_global_id.get(track_key_lookup)
                    track_dict_for_mot["global_id"] = assigned_gid

        total_processing_duration_sec = time.perf_counter() - start_time_total_processing
        self.summary_metrics['perf_total_cam_frame_instances_processed'] = num_cam_frame_instances_processed
        self.summary_metrics['perf_unique_frame_indices_processed'] = len(processed_frame_indices_set)
        self.summary_metrics['perf_total_processing_time_sec'] = round(total_processing_duration_sec, 2)
        self.summary_metrics['perf_avg_cam_frame_processing_time_ms'] = round(np.mean(frame_processing_times_ms), 2) if frame_processing_times_ms else 0
        self.summary_metrics['perf_overall_fps_cam_frames'] = round(num_cam_frame_instances_processed / total_processing_duration_sec, 2) if total_processing_duration_sec > 0 else 0
        self.summary_metrics['input_total_gt_boxes_fed_to_trackers'] = total_gt_boxes_fed_to_tracker
        self.summary_metrics['output_total_tracks_from_botsort_adapter'] = total_tracks_output_from_botsort
        
        self.processed = True
        logger.info(f"[{self.run_name_tag}] --- Backend-Style Frame Processing Finished ---")
        return True

    def calculate_metrics(self) -> bool:
        if not self.processed:
            logger.error(f"[{self.run_name_tag}] Cannot calculate metrics: Frame processing not completed.")
            return False
        if not self.data_loader or self.ground_truth_data is None:
            logger.error(f"[{self.run_name_tag}] Cannot calculate metrics: Data loader or GT data missing.")
            return False
        
        if not MOTMETRICS_AVAILABLE:
            logger.warning(f"[{self.run_name_tag}] `motmetrics` library not available. Skipping MOT metrics calculation.")
            for metric_name in ['MOTA', 'MOTP', 'IDF1', 'IDSW']: self.summary_metrics[metric_name.upper()] = -1.0 # Ensure keys are uppercase
            self.metrics_calculated = True
            return True

        logger.info(f"[{self.run_name_tag}] Preparing data for motmetrics...")
        acc = mm.MOTAccumulator(auto_id=True)
        active_frame_indices = sorted(list(set(f_idx for f_idx, _ in self.raw_tracker_outputs_with_global_ids.keys()) | \
                                       set(f_idx for f_idx, _ in self.ground_truth_data.keys())))
        if not active_frame_indices:
            logger.warning(f"[{self.run_name_tag}] No frame indices with GT or hypotheses. MOT metrics will be empty.")
            for metric_name in REQUESTED_MOT_METRICS: self.summary_metrics[f"mot_{metric_name.upper()}"] = 0.0 # Uppercase
            self.metrics_calculated = True
            return True

        for frame_idx in tqdm(active_frame_indices, desc=f"Accumulating MOT ({self.run_name_tag})"):
            gt_ids_this_frame: List[str] = []
            gt_boxes_this_frame: List[List[float]] = []
            hyp_ids_this_frame: List[str] = []
            hyp_boxes_this_frame: List[List[float]] = []

            for cam_id_obj in self.data_loader.active_camera_ids:
                cam_id_str = str(cam_id_obj)
                gt_tuples_cam_frame = self.ground_truth_data.get((frame_idx, cam_id_obj), [])
                for obj_id_gt, cx, cy, w_gt, h_gt in gt_tuples_cam_frame:
                    if w_gt > 0 and h_gt > 0:
                        unique_gt_id = f"{cam_id_str}_{obj_id_gt}"
                        gt_ids_this_frame.append(unique_gt_id)
                        gt_boxes_this_frame.append([cx - w_gt/2, cy - h_gt/2, w_gt, h_gt])
                hyp_list_cam_frame = self.raw_tracker_outputs_with_global_ids.get((frame_idx, cam_id_str), [])
                for hyp_dict in hyp_list_cam_frame:
                    gid = hyp_dict["global_id"]
                    if gid is None: continue
                    x1, y1, x2, y2 = hyp_dict["bbox_xyxy"]
                    w_hyp, h_hyp = x2 - x1, y2 - y1
                    if w_hyp > 0 and h_hyp > 0:
                        hyp_ids_this_frame.append(str(gid))
                        hyp_boxes_this_frame.append([x1, y1, w_hyp, h_hyp])
            
            if gt_ids_this_frame or hyp_ids_this_frame:
                distances = mm.distances.iou_matrix(gt_boxes_this_frame, hyp_boxes_this_frame, max_iou=0.5)
                acc.update(gt_ids_this_frame, hyp_ids_this_frame, distances)

        mh = mm.metrics.create()
        summary_df = mh.compute(acc, metrics=REQUESTED_MOT_METRICS, name=self.run_name_tag)
        logger.info(f"[{self.run_name_tag}] MOT Metrics Calculation Complete:\n{summary_df}")
        
        if not summary_df.empty:
            for metric_name_report in summary_df.columns: # Iterate over actual columns in report
                metric_key_mlflow = f"mot_{metric_name_report.upper().replace('%', '_PCT')}"
                value = summary_df[metric_name_report].iloc[0]
                self.summary_metrics[metric_key_mlflow] = round(float(value), 4) if isinstance(value, (float, np.floating, np.integer)) else int(value) # Handle int values too
        else:
            logger.warning(f"[{self.run_name_tag}] MOTMetrics summary DataFrame is empty. Metrics will be -1.")
            for metric_name in REQUESTED_MOT_METRICS: self.summary_metrics[f"mot_{metric_name.upper()}"] = -1.0
        
        self.metrics_calculated = True
        return True

    def run(self) -> Tuple[bool, TrackingReidResultSummary]:
        success = False
        try:
            if not self.initialize_components():
                 logger.error(f"[{self.run_name_tag}] Pipeline initialization failed.")
                 return False, self.summary_metrics
            if not self.process_frames():
                logger.warning(f"[{self.run_name_tag}] Frame processing did not complete successfully.")
                self.calculate_metrics() # Attempt to calculate on partial if any
                return False, self.summary_metrics
            if not self.calculate_metrics():
                 logger.warning(f"[{self.run_name_tag}] Metrics calculation failed after successful processing.")
                 return True, self.summary_metrics # Processing was OK
            success = True
        except Exception as e:
            logger.critical(f"[{self.run_name_tag}] Unexpected error during pipeline execution: {e}", exc_info=True)
            success = False
            if not isinstance(self.summary_metrics, dict): self.summary_metrics = {}
        logger.info(f"[{self.run_name_tag}] Backend-Style Tracking+ReID Pipeline Run completed. Success: {success}")
        return success, self.summary_metrics