import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np
import torch
from tqdm import tqdm

# --- Local Imports Need Correct Path Handling ---
try:
    from src.data.reid_dataset_loader import ReidDatasetLoader, ReidCropInfo
    from src.reid.strategies import ReIDStrategy, get_reid_strategy_from_run_config
    from src.evaluation.reid_metrics import compute_reid_metrics
except ImportError:
    import sys

    if str(Path(__file__).parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.reid_dataset_loader import ReidDatasetLoader, ReidCropInfo
    from src.reid.strategies import ReIDStrategy, get_reid_strategy_from_run_config
    from evaluation.reid_metrics import compute_reid_metrics
# --- End Local Import Handling ---


logger = logging.getLogger(__name__)

ExtractedFeaturesList = List[Tuple[np.ndarray, int, str, int]]  # (feature, global_id, camera_id, frame_index)
ReidEvalCounts = Dict[str, int]


class ReidPipeline:
    """Encapsulates the core logic for a Re-ID Query/Gallery experiment."""

    def __init__(self, config: Dict[str, Any], device: torch.device, project_root: Path):
        """
        Initializes the pipeline with configuration, target device, and project root.
        """
        self.config = config
        self.device = device
        self.project_root = project_root
        self.data_loader: Optional[ReidDatasetLoader] = None
        self.reid_strategy: Optional[ReIDStrategy] = None
        self.all_crops_data: List[ReidCropInfo] = []
        self.extracted_features: ExtractedFeaturesList = []
        self.evaluation_sets_prepared: bool = False
        self.distance_metric: str = "cosine"

        # Prepared data for metric calculation
        self.query_features_np: Optional[np.ndarray] = None
        self.query_gids_np: Optional[np.ndarray] = None
        self.query_cids_list: Optional[List[str]] = None
        self.gallery_features_np: Optional[np.ndarray] = None
        self.gallery_gids_np: Optional[np.ndarray] = None
        self.gallery_cids_list: Optional[List[str]] = None

        self.calculated_metrics: Dict[str, Any] = {}
        self.evaluation_counts: ReidEvalCounts = {}
        self.initialized = False
        self.features_extracted = False
        self.metrics_calculated = False

    def initialize_components(self) -> bool:
        """Initializes data loader and Re-ID strategy."""
        logger.info("Initializing Re-ID pipeline components...")
        model_config = self.config.get("model", {})
        model_type = model_config.get("model_type", "unknown_reid")
        weights_file = Path(model_config.get("weights_path", "")).stem
        model_name_tag = f"{model_type}_{weights_file}" if weights_file else model_type

        try:
            # Initialize Data Loader
            logger.info(f"[{model_name_tag}] Initializing Re-ID dataset loader...")
            self.data_loader = ReidDatasetLoader(self.config)
            self.all_crops_data = self.data_loader.get_data()
            if not self.all_crops_data:
                raise ValueError("Re-ID data loader returned no crop info.")
            logger.info(f"[{model_name_tag}] Data loader initialized. Found {len(self.all_crops_data)} GT crops.")
            self.evaluation_counts["total_gt_crops_found"] = len(self.all_crops_data)
            self.evaluation_counts["unique_person_ids_found"] = len(set(c.instance_id for c in self.all_crops_data))

            # Initialize Re-ID Strategy
            logger.info(f"[{model_name_tag}] Initializing Re-ID strategy...")
            self.reid_strategy = get_reid_strategy_from_run_config(self.config, self.device, self.project_root)
            if self.reid_strategy is None:
                raise RuntimeError("Failed to get ReID strategy.")
            logger.info(f"[{model_name_tag}] Re-ID strategy '{self.reid_strategy.__class__.__name__}' initialized.")

            self.initialized = True
            logger.info("Re-ID Pipeline components initialized successfully.")
            return True

        except (FileNotFoundError, ValueError, RuntimeError, ImportError, Exception) as e:
            logger.critical(f"[{model_name_tag}] Failed to initialize Re-ID pipeline components: {e}", exc_info=True)
            self.initialized = False
            return False

    def extract_features(self) -> bool:
        """Extracts features for all loaded crops."""
        if not self.initialized or self.reid_strategy is None or not self.all_crops_data:
            logger.error("Cannot extract features: Pipeline not initialized or no data/strategy.")
            return False

        model_config = self.config.get("model", {})
        model_type = model_config.get("model_type", "unknown_reid")
        weights_file = Path(model_config.get("weights_path", "")).stem
        model_name_tag = f"{model_type}_{weights_file}" if weights_file else model_type
        logger.info(f"[{model_name_tag}] Starting feature extraction for {len(self.all_crops_data)} crops...")

        self.extracted_features = []
        data_grouped_by_frame: Dict[str, List[ReidCropInfo]] = defaultdict(list)
        for crop_info in self.all_crops_data:
            data_grouped_by_frame[crop_info.frame_path].append(crop_info)

        extraction_start_time = time.time()
        with tqdm(total=len(data_grouped_by_frame), desc=f"Extracting Features ({model_name_tag})") as pbar:
            for frame_path_str, crops_in_frame in data_grouped_by_frame.items():
                pbar.set_postfix({"Frame": Path(frame_path_str).name})
                frame_path = Path(frame_path_str)
                if not frame_path.is_file():
                    logger.warning(f"Frame file not found: {frame_path}");
                    pbar.update(1);
                    continue
                try:
                    frame_bytes = np.fromfile(str(frame_path), dtype=np.uint8)
                    frame_bgr = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
                    if frame_bgr is None or frame_bgr.size == 0:
                        logger.warning(f"Failed decode frame: {frame_path}");
                        pbar.update(1);
                        continue

                    # Prepare bounding boxes for the strategy
                    bboxes_xyxy = np.array([c.bbox_xyxy for c in crops_in_frame], dtype=np.float32)
                    valid_bbox_indices = [i for i, bbox in enumerate(bboxes_xyxy) if
                                          bbox[2] > bbox[0] and bbox[3] > bbox[1]]
                    if not valid_bbox_indices: continue

                    valid_bboxes_xyxy = bboxes_xyxy[valid_bbox_indices]
                    valid_crops_in_frame = [crops_in_frame[i] for i in valid_bbox_indices]

                    # Extract features for valid boxes in the current frame
                    features_dict: Dict[int, np.ndarray] = self.reid_strategy.extract_features(frame_bgr,
                                                                                               valid_bboxes_xyxy)

                    # Store features mapped back to original crop info
                    if features_dict:
                        for valid_idx, crop_info in enumerate(valid_crops_in_frame):
                            feature = features_dict.get(valid_idx)  # Index corresponds to valid_bboxes_xyxy
                            if feature is not None and isinstance(feature,
                                                                  np.ndarray) and feature.size > 0 and crop_info.frame_index is not None:
                                self.extracted_features.append(
                                    (feature, crop_info.instance_id, crop_info.camera_id, crop_info.frame_index)
                                )
                except Exception as frame_proc_err:
                    logger.error(f"Error processing frame {frame_path} for feature extraction: {frame_proc_err}",
                                 exc_info=True)
                finally:
                    pbar.update(1)

        extraction_time = time.time() - extraction_start_time
        logger.info(f"[{model_name_tag}] Feature extraction completed in {extraction_time:.2f}s.")
        logger.info(f"Extracted {len(self.extracted_features)} features.")
        self.evaluation_counts["perf_feature_extraction_time_sec"] = round(extraction_time, 2)
        self.evaluation_counts["perf_total_features_extracted"] = len(self.extracted_features)

        if not self.extracted_features:
            logger.error(f"[{model_name_tag}] No features were extracted.")
            self.features_extracted = False
            return False

        self.features_extracted = True
        return True

    def prepare_evaluation_sets(self) -> bool:
        """Prepares query and gallery sets from extracted features."""
        if not self.features_extracted:
            logger.error("Cannot prepare evaluation sets: Features not extracted.")
            return False

        model_name_tag = self.reid_strategy.model_type if self.reid_strategy else "unknown_reid"
        logger.info(f"[{model_name_tag}] Preparing standard Query/Gallery sets...")

        selected_env = self.config.get('data', {}).get('selected_environment', '')
        env_specific_config = self.config.get("data", {}).get(selected_env, {})
        query_cams = set(env_specific_config.get("query_cameras", []))
        gallery_cams = set(env_specific_config.get("gallery_cameras", []))
        self.distance_metric = self.config.get("evaluation", {}).get("distance_metric", "cosine")

        if not query_cams or not gallery_cams:
            logger.error("Query or Gallery cameras not defined in config for selected environment. Cannot evaluate.")
            return False
        if not query_cams.isdisjoint(gallery_cams):
            logger.warning(f"Query cameras {query_cams} and gallery cameras {gallery_cams} overlap.")

        query_data: List[Tuple[np.ndarray, int, str]] = []  # (feature, global_id, camera_id)
        gallery_data: List[Tuple[np.ndarray, int, str]] = []  # (feature, global_id, camera_id)
        all_gids = sorted(list(set(item[1] for item in self.extracted_features)))

        # --- Query Selection: Pick one instance per person ID from query cameras ---
        features_by_gid = defaultdict(list)
        for feature, gid, cid, fidx in self.extracted_features:
            if cid in query_cams:
                features_by_gid[gid].append((feature, cid))

        selected_query_count = 0
        for gid in all_gids:
            if gid in features_by_gid:
                chosen_query = random.choice(features_by_gid[gid])
                query_data.append((chosen_query[0], gid, chosen_query[1]))
                selected_query_count += 1

        # --- Gallery Selection: All instances from gallery cameras ---
        for feature, gid, cid, fidx in self.extracted_features:
            if cid in gallery_cams:
                gallery_data.append((feature, gid, cid))

        logger.info(
            f"Prepared {len(query_data)} queries (one per ID found in query cams: {selected_query_count}) and {len(gallery_data)} gallery items."
        )
        self.evaluation_counts["eval_query_count"] = len(query_data)
        self.evaluation_counts["eval_gallery_count"] = len(gallery_data)
        self.evaluation_counts["eval_query_cameras"] = len(query_cams)  # Store count for simpler logging
        self.evaluation_counts["eval_gallery_cameras"] = len(gallery_cams)

        if not query_data or not gallery_data:
            logger.error(f"[{model_name_tag}] Query or Gallery set is empty after preparation.")
            return False

        # Store numpy arrays for metric calculation
        self.query_features_np = np.array([item[0] for item in query_data])
        self.query_gids_np = np.array([item[1] for item in query_data])
        self.query_cids_list = [item[2] for item in query_data]
        self.gallery_features_np = np.array([item[0] for item in gallery_data])
        self.gallery_gids_np = np.array([item[1] for item in gallery_data])
        self.gallery_cids_list = [item[2] for item in gallery_data]

        self.evaluation_sets_prepared = True
        return True

    def calculate_metrics(self) -> bool:
        """Calculates Re-ID metrics (mAP and Rank-k)."""
        if not self.evaluation_sets_prepared:
            logger.error("Cannot calculate metrics: Evaluation sets not prepared.")
            return False
        if not all([
            self.query_features_np is not None, self.query_gids_np is not None, self.query_cids_list is not None,
            self.gallery_features_np is not None, self.gallery_gids_np is not None, self.gallery_cids_list is not None
        ]):
            logger.error("Cannot calculate metrics: Missing prepared numpy data for evaluation.")
            return False

        model_name_tag = self.reid_strategy.model_type if self.reid_strategy else "unknown_reid"
        logger.info(f"[{model_name_tag}] Calculating Re-ID metrics using '{self.distance_metric}' distance...")

        try:
            metrics = compute_reid_metrics(
                self.query_features_np, self.query_gids_np, self.query_cids_list,
                self.gallery_features_np, self.gallery_gids_np, self.gallery_cids_list,
                self.distance_metric
            )
            if metrics:
                logger.info(f"[{model_name_tag}] Re-ID Metrics: {metrics}")
                self.calculated_metrics = metrics
                self.metrics_calculated = True
                return True
            else:
                logger.error(f"[{model_name_tag}] Metric calculation function returned empty results.")
                self.metrics_calculated = False
                return False
        except Exception as eval_err:
            logger.error(f"[{model_name_tag}] Failed to compute Re-ID metrics: {eval_err}", exc_info=True)
            self.calculated_metrics = {}
            self.metrics_calculated = False
            return False

    def run(self) -> Tuple[bool, Dict[str, Any], ReidEvalCounts]:
        """
        Executes the full Re-ID pipeline: initialization, feature extraction, evaluation preparation, and metric calculation.
        """
        model_name_tag = self.config.get("model", {}).get("model_type",
                                                          "unknown_reid")  # Use model type as tag early on
        success = False
        try:
            if not self.initialize_components():
                return False, {}, self.evaluation_counts

            model_name_tag = self.reid_strategy.model_type if self.reid_strategy else "unknown_reid"

            if not self.extract_features():
                return False, {}, self.evaluation_counts

            if not self.prepare_evaluation_sets():
                return False, {}, self.evaluation_counts

            if not self.calculate_metrics():
                return False, {}, self.evaluation_counts

            success = True

        except Exception as e:
            logger.critical(f"[{model_name_tag}] Unexpected error during Re-ID pipeline execution: {e}", exc_info=True)
            success = False
            # Ensure some metrics dict exists, even if empty
            if not self.calculated_metrics: self.calculated_metrics = {}

        # Log final state
        logger.info(f"[{model_name_tag}] Re-ID Pipeline Run completed. Success: {success}")
        logger.info(f"[{model_name_tag}] Evaluation Counts: {self.evaluation_counts}")
        logger.info(f"[{model_name_tag}] Calculated Metrics: {self.calculated_metrics}")

        return success, self.calculated_metrics, self.evaluation_counts
