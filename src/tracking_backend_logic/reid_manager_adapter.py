"""
Adapter for ReIDStateManager from SpotOn backend.
Manages Re-ID state and association logic, including handoff influence,
for the MLflow tracking pipeline. This class expects feature vectors to be
provided to it (extracted by the tracker, e.g., BotSort using CLIP).
"""
import logging
import uuid
import math # For sqrt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
from scipy.spatial.distance import cdist

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    logging.warning("FAISS library not found. FAISS-based Re-ID methods will be unavailable. Install with `pip install faiss-cpu` or `faiss-gpu`.")


from .common_types_adapter import (
    CameraID, TrackID, GlobalID, FeatureVector, TrackKey, HandoffTriggerInfo
)

logger = logging.getLogger(__name__)

class ReIDManagerAdapter:
    """
    Adapts backend's ReIDStateManager for MLflow pipeline.
    Manages state for cross-camera Re-Identification based on provided feature vectors,
    supporting multiple similarity/distance methods including FAISS.
    """

    def __init__(self,
                 run_id_context: str,
                 reid_config: Dict[str, Any],
                 handoff_config: Dict[str, Any],
                 similarity_method: str = "cosine" # NEW parameter
                 ):
        self.run_id_context = run_id_context
        self.reid_config = reid_config
        self.handoff_config = handoff_config
        self.similarity_method = similarity_method.lower()

        self.reid_gallery: Dict[GlobalID, FeatureVector] = {}
        self.lost_track_gallery: Dict[GlobalID, Tuple[FeatureVector, int]] = {}
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {}
        self.global_id_last_seen_cam: Dict[GlobalID, CameraID] = {}
        self.global_id_last_seen_frame: Dict[GlobalID, int] = {}
        self.track_last_reid_frame: Dict[TrackKey, int] = {}

        # --- FAISS specific ---
        self.faiss_index: Optional[Any] = None # Stores faiss.Index
        self.faiss_gallery_gids: List[GlobalID] = [] # Maps FAISS index to GlobalID
        self.faiss_index_dirty: bool = True # Flag to rebuild FAISS index

        # Parameters from config
        self.similarity_threshold_cosine: float = float(reid_config.get("similarity_threshold", 0.65))
        # For L2, lower is better. Derive if not provided.
        self.l2_distance_threshold_explicit: Optional[float] = reid_config.get("l2_distance_threshold")
        if self.l2_distance_threshold_explicit is not None:
            self.l2_distance_threshold_explicit = float(self.l2_distance_threshold_explicit)
        
        # Effective threshold used by the current method
        self.effective_threshold: float = self.similarity_threshold_cosine

        if self.similarity_method in ["l2_derived", "l2_explicit", "faiss_l2"]:
            if self.similarity_method == "l2_explicit" and self.l2_distance_threshold_explicit is not None:
                self.effective_threshold = self.l2_distance_threshold_explicit
                logger.info(f"Using EXPLICIT L2 distance threshold: {self.effective_threshold}")
            else: # l2_derived or l2_explicit without explicit threshold
                # Derived L2 threshold: d = sqrt(2*(1-s_cos)) for normalized vectors
                self.effective_threshold = math.sqrt(max(0, 2 * (1 - self.similarity_threshold_cosine)))
                logger.info(f"Using DERIVED L2 distance threshold: {self.effective_threshold:.4f} (from cosine_sim_thresh {self.similarity_threshold_cosine})")
        elif self.similarity_method in ["cosine", "inner_product", "faiss_ip"]:
             self.effective_threshold = self.similarity_threshold_cosine # Higher is better
        else:
            logger.warning(f"Unknown similarity_method '{self.similarity_method}'. Defaulting to cosine similarity threshold logic.")
            self.similarity_method = "cosine" # Fallback
            self.effective_threshold = self.similarity_threshold_cosine

        self.gallery_ema_alpha: float = float(reid_config.get("gallery_ema_alpha", 0.9))
        self.refresh_interval_frames: int = int(reid_config.get("refresh_interval_frames", 10))
        self.lost_track_buffer_frames: int = int(reid_config.get("lost_track_buffer_frames", 200))
        self.main_gallery_prune_interval: int = int(reid_config.get("main_gallery_prune_interval_frames", 500))
        _default_prune_thresh = self.lost_track_buffer_frames * 2
        self.main_gallery_prune_threshold: int = int(reid_config.get("main_gallery_prune_threshold_frames", _default_prune_thresh))
        
        self.min_bbox_overlap_for_handoff: float = float(handoff_config.get("min_bbox_overlap_ratio_in_quadrant", 0.40))
        _overlaps_raw = handoff_config.get("possible_camera_overlaps", [])
        self.possible_camera_overlaps: Set[Tuple[CameraID, CameraID]] = {
            tuple(sorted((CameraID(c1), CameraID(c2)))) for c1, c2 in _overlaps_raw # type: ignore
        }

        logger.info(f"[Run {self.run_id_context}] ReIDManagerAdapter initialized with method: '{self.similarity_method}'")
        logger.info(f"  Effective Threshold: {self.effective_threshold:.4f} (Interpreted as {'min_dist' if 'l2' in self.similarity_method else 'min_sim'})")

        if "faiss" in self.similarity_method and not FAISS_AVAILABLE:
            raise ImportError(f"Similarity method '{self.similarity_method}' requires FAISS, but it's not installed.")


    def _build_faiss_index(self):
        """Builds or rebuilds the FAISS index from the current main Re-ID gallery."""
        if not FAISS_AVAILABLE or faiss is None:
            logger.error("FAISS not available. Cannot build FAISS index.")
            self.faiss_index = None
            return

        self.faiss_gallery_gids = []
        gallery_embeddings_list: List[FeatureVector] = []
        for gid, feat in self.reid_gallery.items():
            if feat.size > 0: # Ensure non-empty features
                gallery_embeddings_list.append(feat)
                self.faiss_gallery_gids.append(gid)

        if not gallery_embeddings_list:
            logger.debug(f"[Run {self.run_id_context}] Main gallery empty. FAISS index will be empty.")
            self.faiss_index = None
            self.faiss_index_dirty = False
            return

        gallery_embeddings_np = np.array(gallery_embeddings_list, dtype=np.float32)
        dimension = gallery_embeddings_np.shape[1]

        if self.similarity_method == "faiss_ip":
            self.faiss_index = faiss.IndexFlatIP(dimension)
        elif self.similarity_method == "faiss_l2":
            self.faiss_index = faiss.IndexFlatL2(dimension)
        else: # Should not happen if constructor logic is correct
            logger.error(f"Unsupported FAISS method: {self.similarity_method}. Cannot build index.")
            self.faiss_index = None
            return
        
        if gallery_embeddings_np.shape[0] > 0: # Only add if there are embeddings
            self.faiss_index.add(gallery_embeddings_np)
        self.faiss_index_dirty = False
        logger.info(f"[Run {self.run_id_context}] FAISS index ({self.similarity_method}) built with {len(self.faiss_gallery_gids)} GIDs.")


    def get_new_global_id(self) -> GlobalID:
        return GlobalID(str(uuid.uuid4()))

    def _normalize_embedding(self, embedding: Optional[FeatureVector]) -> Optional[FeatureVector]:
        if embedding is None: return None
        if not isinstance(embedding, np.ndarray):
            try: embedding_np = np.array(embedding, dtype=np.float32)
            except Exception: logger.warning(f"Invalid type for embedding: {type(embedding)}. Cannot normalize."); return None
        else: embedding_np = embedding.astype(np.float32)
        if embedding_np.size == 0: return FeatureVector(np.array([], dtype=np.float32))
        norm = np.linalg.norm(embedding_np)
        return FeatureVector(embedding_np / norm) if norm > 1e-6 else FeatureVector(embedding_np)

    def _calculate_scores_from_cdist(
        self, query_embeddings: np.ndarray, gallery_embeddings: np.ndarray
    ) -> Optional[np.ndarray]:
        """Calculates scores (similarity or distance) based on self.similarity_method using cdist."""
        if query_embeddings.ndim == 1: query_embeddings = query_embeddings.reshape(1, -1)
        if gallery_embeddings.ndim == 1: gallery_embeddings = gallery_embeddings.reshape(1, -1)
        if query_embeddings.size == 0 or gallery_embeddings.size == 0: return None

        try:
            if self.similarity_method == "cosine":
                # Cosine similarity: 1 is best, 0 is worst (for normalized vectors)
                scores = 1.0 - cdist(query_embeddings, gallery_embeddings, metric='cosine')
            elif self.similarity_method == "l2_derived" or self.similarity_method == "l2_explicit":
                # L2 distance: 0 is best, higher is worse
                scores = cdist(query_embeddings, gallery_embeddings, metric='euclidean')
            elif self.similarity_method == "inner_product":
                # Inner product (dot product): higher is better for normalized vectors
                # cdist doesn't have 'dot'. We can compute manually or use (1 - cosine_dist) * norm_q * norm_g
                # Since we normalize, this becomes equivalent to cosine similarity if norms are 1.
                # For simplicity, let's ensure vectors are normalized and use cosine here.
                # If performance is critical, direct dot product (e.g., query @ gallery.T) is faster.
                scores = query_embeddings @ gallery_embeddings.T # Assumes gallery is (N_gallery, D)
            else:
                logger.error(f"Unsupported cdist-based similarity method: {self.similarity_method}")
                return None
            return np.clip(scores, -1.0 if self.similarity_method == "inner_product" else 0.0, 1.0) # Clip appropriately
        except Exception as e:
            logger.error(f"[Run {self.run_id_context}] cdist-based score calculation failed: {e}", exc_info=True)
            return None


    def _should_attempt_reid_for_track(
        self, track_key: TrackKey, current_frame_idx: int, active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
    ) -> bool:
        is_known = track_key in self.track_to_global_id
        last_attempt_frame = self.track_last_reid_frame.get(track_key, -self.refresh_interval_frames - 1)
        is_due_for_refresh = (current_frame_idx - last_attempt_frame) >= self.refresh_interval_frames
        is_triggering_handoff = track_key in active_triggers_map
        if not is_known: return True
        if is_due_for_refresh: return True
        if is_triggering_handoff: logger.info(f"[Run {self.run_id_context}][{track_key}] Re-ID attempt due to active handoff trigger."); return True
        return False

    def _get_relevant_handoff_cams(self, target_cam_id: CameraID) -> Set[CameraID]:
        relevant_cams = {target_cam_id}
        for c1, c2 in self.possible_camera_overlaps:
            if c1 == target_cam_id: relevant_cams.add(c2)
            elif c2 == target_cam_id: relevant_cams.add(c1)
        return relevant_cams
        
    def _apply_handoff_filter_for_match(
        self, query_track_key: TrackKey, matched_gid: GlobalID, active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
    ) -> bool:
        trigger_info = active_triggers_map.get(query_track_key)
        current_cam_id, _ = query_track_key
        if not trigger_info:
            relevant_cams_for_current = self._get_relevant_handoff_cams(current_cam_id)
            last_seen_cam_for_gid = self.global_id_last_seen_cam.get(matched_gid)
            if last_seen_cam_for_gid is not None and last_seen_cam_for_gid != current_cam_id:
                if last_seen_cam_for_gid not in relevant_cams_for_current: return False
            return True 
        trigger_target_cam_id = trigger_info.rule.target_cam_id
        relevant_cams_for_trigger_target = self._get_relevant_handoff_cams(trigger_target_cam_id)
        last_seen_cam_for_matched_gid = self.global_id_last_seen_cam.get(matched_gid)
        if last_seen_cam_for_matched_gid is not None:
            if last_seen_cam_for_matched_gid not in relevant_cams_for_trigger_target: return False
        return True

    def _update_gallery_with_ema(
        self, gid: GlobalID, new_embedding: FeatureVector, matched_gallery_type: str
    ):
        alpha = self.gallery_ema_alpha
        current_gallery_embedding: Optional[FeatureVector] = None
        new_embedding_norm = self._normalize_embedding(new_embedding)
        if new_embedding_norm is None or new_embedding_norm.size == 0: return

        # Mark FAISS index as dirty since gallery is changing
        self.faiss_index_dirty = True

        if matched_gallery_type == 'lost':
            if gid in self.lost_track_gallery:
                lost_embedding, _ = self.lost_track_gallery.pop(gid)
                current_gallery_embedding = self._normalize_embedding(lost_embedding)
            elif gid in self.reid_gallery:
                current_gallery_embedding = self.reid_gallery.get(gid)
        elif matched_gallery_type in ['main', 'main_2pass']:
            current_gallery_embedding = self.reid_gallery.get(gid)
        
        if current_gallery_embedding is not None and current_gallery_embedding.size > 0:
            updated_embedding = (alpha * current_gallery_embedding + (1.0 - alpha) * new_embedding_norm)
            self.reid_gallery[gid] = self._normalize_embedding(updated_embedding)
        else:
            self.reid_gallery[gid] = new_embedding_norm


    def associate_features_and_update_state(
        self,
        all_tracks_with_features: Dict[TrackKey, FeatureVector],
        active_track_keys_this_frame: Set[TrackKey],
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo],
        current_frame_idx: int
    ):
        query_features: Dict[TrackKey, FeatureVector] = {}
        for tk, feat in all_tracks_with_features.items():
            if self._should_attempt_reid_for_track(tk, current_frame_idx, active_triggers_map):
                norm_feat = self._normalize_embedding(feat)
                if norm_feat is not None and norm_feat.size > 0:
                    query_features[tk] = norm_feat
                    self.track_last_reid_frame[tk] = current_frame_idx
        
        if not query_features:
            self.update_galleries_lifecycle(active_track_keys_this_frame, current_frame_idx)
            return

        query_track_keys_list = list(query_features.keys())
        query_embeddings_np = np.array([query_features[tk] for tk in query_track_keys_list], dtype=np.float32)

        gallery_gids_combined: List[GlobalID] = []
        gallery_embeddings_list_combined: List[FeatureVector] = []
        gallery_types_combined: List[str] = []

        # Add lost gallery first
        for gid, (feat, _) in self.lost_track_gallery.items():
            norm_feat = self._normalize_embedding(feat)
            if norm_feat is not None and norm_feat.size > 0:
                gallery_gids_combined.append(gid)
                gallery_embeddings_list_combined.append(norm_feat)
                gallery_types_combined.append('lost')
        
        # Add main gallery, avoiding duplicates if a GID was already in lost (shouldn't happen if logic is correct)
        main_gallery_gids_already_added = set(gid for gid, _, type_str in zip(gallery_gids_combined, gallery_embeddings_list_combined, gallery_types_combined) if type_str == 'lost')
        for gid, feat in self.reid_gallery.items():
            if gid not in main_gallery_gids_already_added and feat.size > 0:
                gallery_gids_combined.append(gid)
                gallery_embeddings_list_combined.append(feat) # Already normalized if from gallery
                gallery_types_combined.append('main')
        
        gallery_embeddings_np_combined = np.array(gallery_embeddings_list_combined, dtype=np.float32) if gallery_embeddings_list_combined else np.empty((0,0), dtype=np.float32)

        tentative_assignments: Dict[TrackKey, Tuple[Optional[GlobalID], float, str]] = {} 
        
        # --- Perform Search based on self.similarity_method ---
        if gallery_embeddings_np_combined.size > 0 and query_embeddings_np.size > 0:
            if "faiss" in self.similarity_method:
                if self.faiss_index_dirty or self.faiss_index is None:
                    self._build_faiss_index() # Rebuild from self.reid_gallery for FAISS search

                if self.faiss_index and self.faiss_index.ntotal > 0:
                    k_neighbors = 1
                    # FAISS search returns distances (for L2) or scores (for IP)
                    raw_scores_faiss, indices_faiss = self.faiss_index.search(query_embeddings_np, k_neighbors)

                    for i, tk_query in enumerate(query_track_keys_list):
                        if indices_faiss[i, 0] < 0: continue # No neighbor found by FAISS
                        
                        matched_faiss_gallery_idx = indices_faiss[i, 0]
                        score_or_dist = float(raw_scores_faiss[i, 0])
                        
                        matched_gid = self.faiss_gallery_gids[matched_faiss_gallery_idx] # Map FAISS index to GID
                        
                        # Determine if match based on threshold and method
                        match_made = False
                        if self.similarity_method == "faiss_ip" and score_or_dist >= self.effective_threshold: # Higher is better
                            match_made = True
                        elif self.similarity_method == "faiss_l2" and score_or_dist <= self.effective_threshold: # Lower is better
                            match_made = True
                        
                        if match_made:
                            # Check if this matched_gid was from lost or main (FAISS index is only main gallery)
                            matched_type = 'main' # FAISS index currently uses only main gallery
                            if self._apply_handoff_filter_for_match(tk_query, matched_gid, active_triggers_map):
                                tentative_assignments[tk_query] = (matched_gid, score_or_dist, matched_type)
                            else:
                                tentative_assignments[tk_query] = (None, score_or_dist, "filtered_handoff")
                        else:
                            tentative_assignments[tk_query] = (None, score_or_dist, "below_threshold")
            else: # cdist based methods
                # Use the combined gallery for cdist methods
                scores_matrix = self._calculate_scores_from_cdist(query_embeddings_np, gallery_embeddings_np_combined)
                if scores_matrix is not None:
                    if self.similarity_method in ["l2_derived", "l2_explicit"]: # Lower score is better (distance)
                        best_match_indices = np.argmin(scores_matrix, axis=1)
                        min_distances = scores_matrix[np.arange(len(query_track_keys_list)), best_match_indices]
                        threshold_condition = min_distances <= self.effective_threshold
                        actual_scores_for_assignment = min_distances # Store distance
                    else: # Cosine or Inner Product - Higher score is better (similarity)
                        best_match_indices = np.argmax(scores_matrix, axis=1)
                        max_similarity_scores = scores_matrix[np.arange(len(query_track_keys_list)), best_match_indices]
                        threshold_condition = max_similarity_scores >= self.effective_threshold
                        actual_scores_for_assignment = max_similarity_scores # Store similarity

                    for i, tk_query in enumerate(query_track_keys_list):
                        if threshold_condition[i]:
                            matched_gallery_idx = best_match_indices[i]
                            matched_gid = gallery_gids_combined[matched_gallery_idx]
                            matched_type = gallery_types_combined[matched_gallery_idx]
                            score_val = float(actual_scores_for_assignment[i])
                            if self._apply_handoff_filter_for_match(tk_query, matched_gid, active_triggers_map):
                                tentative_assignments[tk_query] = (matched_gid, score_val, matched_type)
                            else:
                                tentative_assignments[tk_query] = (None, score_val, "filtered_handoff")
                        else:
                            tentative_assignments[tk_query] = (None, float(actual_scores_for_assignment[i]), "below_threshold")
        
        # --- Conflict resolution and final assignment logic (remains largely the same) ---
        current_assignments = tentative_assignments.copy()
        for i, tk in enumerate(query_track_keys_list):
            if tk not in current_assignments or current_assignments[tk][0] is None:
                new_gid = self.get_new_global_id()
                # Score interpretation for "new": -1 for similarity, high value for distance
                new_score = -1.0 if 'l2' not in self.similarity_method else float('inf')
                current_assignments[tk] = (new_gid, new_score, "new")
                original_embedding_for_new = query_features[tk]
                if original_embedding_for_new.size > 0:
                    self.reid_gallery[new_gid] = original_embedding_for_new
                    self.faiss_index_dirty = True # Mark for rebuild
                self.global_id_last_seen_frame[new_gid] = current_frame_idx
                self.global_id_last_seen_cam[new_gid] = tk[0]

        assignments_by_cam_gid: Dict[Tuple[CameraID, GlobalID], List[Tuple[TrackKey, float, str]]] = defaultdict(list)
        for tk, (gid, score, type_str) in current_assignments.items():
            if gid is not None:
                assignments_by_cam_gid[(tk[0], gid)].append((tk, score, type_str))

        reverted_keys_for_second_pass: List[TrackKey] = []
        for (cam_id_conflict, gid_conflict), track_score_list_with_type in assignments_by_cam_gid.items():
            if len(track_score_list_with_type) > 1:
                # Sort: higher score for sim, lower score for dist
                sort_reverse = 'l2' not in self.similarity_method
                track_score_list_with_type.sort(key=lambda x: x[1], reverse=sort_reverse)
                winner_tk, _, _ = track_score_list_with_type[0]
                for i_conflict in range(1, len(track_score_list_with_type)):
                    reverted_tk, _, _ = track_score_list_with_type[i_conflict]
                    current_assignments[reverted_tk] = (None, -1.0 if sort_reverse else float('inf'), "reverted_conflict")
                    reverted_keys_for_second_pass.append(reverted_tk)
        
        for tk, (gid, _, type_str) in current_assignments.items():
            if gid is not None and tk not in reverted_keys_for_second_pass :
                self.track_to_global_id[tk] = gid
                self.global_id_last_seen_cam[gid] = tk[0]
                self.global_id_last_seen_frame[gid] = current_frame_idx
                if type_str not in ["new", "reverted_conflict", "filtered_handoff", "below_threshold"]:
                    original_embedding_for_update = query_features[tk]
                    self._update_gallery_with_ema(gid, original_embedding_for_update, type_str)
        
        if reverted_keys_for_second_pass:
            # Simplified second pass: assign new GIDs to reverted tracks for this demo
            # A full second pass would re-evaluate against the updated gallery.
            for tk_reverted in reverted_keys_for_second_pass:
                new_gid_for_reverted = self.get_new_global_id()
                self.track_to_global_id[tk_reverted] = new_gid_for_reverted
                reverted_embedding = query_features[tk_reverted]
                if reverted_embedding.size > 0:
                    self.reid_gallery[new_gid_for_reverted] = reverted_embedding
                    self.faiss_index_dirty = True
                self.global_id_last_seen_cam[new_gid_for_reverted] = tk_reverted[0]
                self.global_id_last_seen_frame[new_gid_for_reverted] = current_frame_idx

        self.update_galleries_lifecycle(active_track_keys_this_frame, current_frame_idx)


    def update_galleries_lifecycle(self, active_track_keys_this_frame: Set[TrackKey], current_frame_idx: int):
        all_previously_known_track_keys = set(self.track_to_global_id.keys())
        disappeared_track_keys = all_previously_known_track_keys - active_track_keys_this_frame

        for tk_disappeared in disappeared_track_keys:
            gid = self.track_to_global_id.pop(tk_disappeared, None)
            self.track_last_reid_frame.pop(tk_disappeared, None)
            if gid:
                feature = self.reid_gallery.get(gid)
                if feature is not None and feature.size > 0:
                    if gid not in self.lost_track_gallery:
                        last_active_frame_for_gid = self.global_id_last_seen_frame.get(gid, current_frame_idx -1)
                        self.lost_track_gallery[gid] = (feature, last_active_frame_for_gid)
                        self.faiss_index_dirty = True # Gallery changed
        
        expired_lost_gids = [
            gid for gid, (_, frame_added_to_lost) in self.lost_track_gallery.items()
            if (current_frame_idx - frame_added_to_lost) > self.lost_track_buffer_frames
        ]
        for gid in expired_lost_gids:
            self.lost_track_gallery.pop(gid, None)
            # No need to mark faiss_index_dirty here as lost gallery is not part of FAISS index.

        if current_frame_idx > 0 and (current_frame_idx % self.main_gallery_prune_interval == 0):
            gids_to_prune_main: List[GlobalID] = []
            prune_cutoff_frame = current_frame_idx - self.main_gallery_prune_threshold
            active_gids_in_current_frame = set(self.track_to_global_id.values())
            for gid_candidate, last_seen_f in list(self.global_id_last_seen_frame.items()):
                if last_seen_f < prune_cutoff_frame and \
                   gid_candidate not in self.lost_track_gallery and \
                   gid_candidate not in active_gids_in_current_frame:
                    gids_to_prune_main.append(gid_candidate)
            
            if gids_to_prune_main:
                for gid_prune in gids_to_prune_main:
                    if self.reid_gallery.pop(gid_prune, None) is not None:
                        self.faiss_index_dirty = True # Gallery changed
                    self.global_id_last_seen_cam.pop(gid_prune, None)
                    self.global_id_last_seen_frame.pop(gid_prune, None)