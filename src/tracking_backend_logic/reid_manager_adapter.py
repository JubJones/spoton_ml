"""
Adapter for ReIDStateManager from SpotOn backend.
Manages Re-ID state and association logic, including handoff influence,
for the MLflow tracking pipeline. This class expects feature vectors to be
provided to it (extracted by the tracker, e.g., BotSort using CLIP).
"""
import logging
import uuid
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any # Keep Any for config

import numpy as np
from scipy.spatial.distance import cdist # For cosine similarity

# Assuming common_types_adapter provides these from your backend's common_types.py
from .common_types_adapter import (
    CameraID, TrackID, GlobalID, FeatureVector, TrackKey,
    HandoffTriggerInfo # ExitRuleModelAdapter, ParsedCameraHandoffConfigs might be used by caller
)

logger = logging.getLogger(__name__)

class ReIDManagerAdapter:
    """
    Adapts backend's ReIDStateManager for MLflow pipeline.
    Manages state for cross-camera Re-Identification based on provided feature vectors.
    """

    def __init__(self, run_id_context: str, reid_config: Dict[str, Any], handoff_config: Dict[str, Any]):
        """
        Initializes the ReIDManagerAdapter.

        Args:
            run_id_context: Identifier for the current MLflow run (used for logging context).
            reid_config: Dictionary containing Re-ID parameters from the YAML config.
            handoff_config: Dictionary containing handoff parameters from the YAML config.
        """
        self.run_id_context = run_id_context # For logging context
        self.reid_config = reid_config
        self.handoff_config = handoff_config

        # Core state variables from backend's ReIDStateManager
        self.reid_gallery: Dict[GlobalID, FeatureVector] = {}
        self.lost_track_gallery: Dict[GlobalID, Tuple[FeatureVector, int]] = {} # GID -> (Feature, frame_idx_added_to_lost)
        self.track_to_global_id: Dict[TrackKey, GlobalID] = {}
        self.global_id_last_seen_cam: Dict[GlobalID, CameraID] = {}
        self.global_id_last_seen_frame: Dict[GlobalID, int] = {} # Stores current_frame_idx
        self.track_last_reid_frame: Dict[TrackKey, int] = {} # Stores current_frame_idx of last ReID attempt

        # Parameters from config
        self.similarity_threshold: float = float(reid_config.get("similarity_threshold", 0.65))
        self.gallery_ema_alpha: float = float(reid_config.get("gallery_ema_alpha", 0.9))
        self.refresh_interval_frames: int = int(reid_config.get("refresh_interval_frames", 10))
        self.lost_track_buffer_frames: int = int(reid_config.get("lost_track_buffer_frames", 200))
        self.main_gallery_prune_interval: int = int(reid_config.get("main_gallery_prune_interval_frames", 500))
        # Ensure PRUNE_THRESHOLD is also int, and derived correctly
        _default_prune_thresh = self.lost_track_buffer_frames * 2
        self.main_gallery_prune_threshold: int = int(reid_config.get("main_gallery_prune_threshold_frames", _default_prune_thresh))
        
        self.min_bbox_overlap_for_handoff: float = float(handoff_config.get("min_bbox_overlap_ratio_in_quadrant", 0.40))
        
        _overlaps_raw = handoff_config.get("possible_camera_overlaps", [])
        self.possible_camera_overlaps: Set[Tuple[CameraID, CameraID]] = {
            tuple(sorted((CameraID(c1), CameraID(c2)))) for c1, c2 in _overlaps_raw # type: ignore
        }

        logger.info(f"[Run {self.run_id_context}] ReIDManagerAdapter initialized.")
        logger.info(f"  Similarity Threshold: {self.similarity_threshold}, EMA Alpha: {self.gallery_ema_alpha}")
        logger.info(f"  Refresh Interval: {self.refresh_interval_frames}, Lost Buffer: {self.lost_track_buffer_frames}")
        logger.info(f"  Main Gallery Prune Interval: {self.main_gallery_prune_interval}, Threshold Age: {self.main_gallery_prune_threshold}")
        logger.info(f"  Handoff Overlap Ratio: {self.min_bbox_overlap_for_handoff}")
        logger.info(f"  Configured Camera Overlaps: {self.possible_camera_overlaps}")

    def get_new_global_id(self) -> GlobalID:
        """Generates a new unique Global ID."""
        return GlobalID(str(uuid.uuid4()))

    def _normalize_embedding(self, embedding: Optional[FeatureVector]) -> Optional[FeatureVector]:
        """Normalizes a feature embedding. Returns None if input is None or invalid."""
        if embedding is None:
            return None
        if not isinstance(embedding, np.ndarray):
            try:
                embedding_np = np.array(embedding, dtype=np.float32)
            except Exception:
                logger.warning(f"Invalid type for embedding: {type(embedding)}. Cannot normalize.")
                return None
        else:
            embedding_np = embedding.astype(np.float32)

        if embedding_np.size == 0:
            return FeatureVector(np.array([], dtype=np.float32))

        norm = np.linalg.norm(embedding_np)
        return FeatureVector(embedding_np / norm) if norm > 1e-6 else FeatureVector(embedding_np)

    def _calculate_similarity_matrix(
        self, query_embeddings: np.ndarray, gallery_embeddings: np.ndarray
    ) -> Optional[np.ndarray]:
        """Calculates cosine similarity matrix between query and gallery embeddings."""
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        if gallery_embeddings.ndim == 1:
            gallery_embeddings = gallery_embeddings.reshape(1, -1)
        
        if query_embeddings.size == 0 or gallery_embeddings.size == 0:
            return None
        try:
            similarity_matrix = 1.0 - cdist(query_embeddings, gallery_embeddings, metric='cosine')
            return np.clip(similarity_matrix, 0.0, 1.0)
        except Exception as e:
            logger.error(f"[Run {self.run_id_context}] Batched similarity calculation failed: {e}", exc_info=True)
            return None

    def _should_attempt_reid_for_track(
        self,
        track_key: TrackKey,
        current_frame_idx: int,
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
    ) -> bool:
        """Determines if Re-ID should be attempted for a given track."""
        is_known = track_key in self.track_to_global_id
        last_attempt_frame = self.track_last_reid_frame.get(track_key, -self.refresh_interval_frames - 1)
        is_due_for_refresh = (current_frame_idx - last_attempt_frame) >= self.refresh_interval_frames
        is_triggering_handoff = track_key in active_triggers_map

        if not is_known: return True
        if is_due_for_refresh: return True
        if is_triggering_handoff:
            logger.info(f"[Run {self.run_id_context}][{track_key}] Re-ID attempt due to active handoff trigger.")
            return True
        return False

    def _get_relevant_handoff_cams(self, target_cam_id: CameraID) -> Set[CameraID]:
        """Gets the target camera and any cameras configured to possibly overlap with it."""
        relevant_cams = {target_cam_id}
        for c1, c2 in self.possible_camera_overlaps:
            if c1 == target_cam_id: relevant_cams.add(c2)
            elif c2 == target_cam_id: relevant_cams.add(c1)
        return relevant_cams
        
    def _apply_handoff_filter_for_match(
        self,
        query_track_key: TrackKey,
        matched_gid: GlobalID,
        active_triggers_map: Dict[TrackKey, HandoffTriggerInfo]
    ) -> bool:
        """Checks if a potential ReID match should be filtered based on handoff context."""
        trigger_info = active_triggers_map.get(query_track_key)
        current_cam_id, _ = query_track_key
        
        if not trigger_info:
            relevant_cams_for_current = self._get_relevant_handoff_cams(current_cam_id)
            last_seen_cam_for_gid = self.global_id_last_seen_cam.get(matched_gid)
            if last_seen_cam_for_gid is not None and last_seen_cam_for_gid != current_cam_id:
                if last_seen_cam_for_gid not in relevant_cams_for_current:
                    return False
            return True 

        trigger_target_cam_id = trigger_info.rule.target_cam_id
        relevant_cams_for_trigger_target = self._get_relevant_handoff_cams(trigger_target_cam_id)
        last_seen_cam_for_matched_gid = self.global_id_last_seen_cam.get(matched_gid)
        if last_seen_cam_for_matched_gid is not None:
            if last_seen_cam_for_matched_gid not in relevant_cams_for_trigger_target:
                return False
        return True

    def _update_gallery_with_ema(
        self,
        gid: GlobalID,
        new_embedding: FeatureVector,
        matched_gallery_type: str
    ):
        alpha = self.gallery_ema_alpha
        current_gallery_embedding: Optional[FeatureVector] = None
        new_embedding_norm = self._normalize_embedding(new_embedding)
        if new_embedding_norm is None or new_embedding_norm.size == 0: return

        if matched_gallery_type == 'lost':
            if gid in self.lost_track_gallery:
                lost_embedding, _ = self.lost_track_gallery.pop(gid) # Remove from lost as it's being merged
                current_gallery_embedding = self._normalize_embedding(lost_embedding)
            elif gid in self.reid_gallery:
                current_gallery_embedding = self.reid_gallery.get(gid)
        elif matched_gallery_type == 'main' or matched_gallery_type == 'main_2pass': # Handle 2nd pass type
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

        gallery_gids: List[GlobalID] = []
        gallery_embeddings_list: List[FeatureVector] = []
        gallery_types: List[str] = []

        for gid, (feat, _) in self.lost_track_gallery.items():
            norm_feat = self._normalize_embedding(feat)
            if norm_feat is not None and norm_feat.size > 0:
                gallery_gids.append(gid); gallery_embeddings_list.append(norm_feat); gallery_types.append('lost')
        
        main_gallery_gids_added = set(gallery_gids) # Track GIDs already added from lost gallery
        for gid, feat in self.reid_gallery.items():
            if gid not in main_gallery_gids_added and feat.size > 0:
                gallery_gids.append(gid); gallery_embeddings_list.append(feat); gallery_types.append('main')
        
        gallery_embeddings_np = np.array(gallery_embeddings_list, dtype=np.float32) if gallery_embeddings_list else np.empty((0, query_embeddings_np.shape[1] if query_embeddings_np.size > 0 else 0), dtype=np.float32)

        tentative_assignments: Dict[TrackKey, Tuple[Optional[GlobalID], float, str]] = {}
        
        if gallery_embeddings_np.size > 0 and query_embeddings_np.size > 0:
            similarity_matrix = self._calculate_similarity_matrix(query_embeddings_np, gallery_embeddings_np)
            if similarity_matrix is not None:
                best_match_indices = np.argmax(similarity_matrix, axis=1)
                max_similarity_scores = similarity_matrix[np.arange(len(query_track_keys_list)), best_match_indices]

                for i, tk_query in enumerate(query_track_keys_list):
                    best_gallery_idx = best_match_indices[i]
                    max_sim = float(max_similarity_scores[i])
                    
                    if max_sim >= self.similarity_threshold:
                        matched_gid = gallery_gids[best_gallery_idx]
                        matched_type = gallery_types[best_gallery_idx]
                        if self._apply_handoff_filter_for_match(tk_query, matched_gid, active_triggers_map):
                            tentative_assignments[tk_query] = (matched_gid, max_sim, matched_type)
                        else: 
                            tentative_assignments[tk_query] = (None, max_sim, "filtered_handoff")
                    else: 
                        tentative_assignments[tk_query] = (None, max_sim, "below_threshold")
        
        current_assignments = tentative_assignments.copy()
        for i, tk in enumerate(query_track_keys_list): # Use index i to get original embedding
            if tk not in current_assignments or current_assignments[tk][0] is None: # If no valid match or filtered
                new_gid = self.get_new_global_id()
                current_assignments[tk] = (new_gid, -1.0, "new") # score -1 for new
                original_embedding_for_new = query_features[tk] # Already normalized
                if original_embedding_for_new.size > 0:
                    self.reid_gallery[new_gid] = original_embedding_for_new
                self.global_id_last_seen_frame[new_gid] = current_frame_idx
                self.global_id_last_seen_cam[new_gid] = tk[0]

        assignments_by_cam_gid: Dict[Tuple[CameraID, GlobalID], List[Tuple[TrackKey, float, str]]] = defaultdict(list)
        for tk, (gid, score, type_str) in current_assignments.items():
            if gid is not None:
                assignments_by_cam_gid[(tk[0], gid)].append((tk, score, type_str))

        reverted_keys_for_second_pass: List[TrackKey] = []
        for (cam_id_conflict, gid_conflict), track_score_list in assignments_by_cam_gid.items():
            if len(track_score_list) > 1:
                track_score_list.sort(key=lambda x: x[1], reverse=True) # Highest score first
                winner_tk, _, _ = track_score_list[0]
                for i_conflict in range(1, len(track_score_list)):
                    reverted_tk, _, _ = track_score_list[i_conflict]
                    current_assignments[reverted_tk] = (None, -1.0, "reverted_conflict")
                    reverted_keys_for_second_pass.append(reverted_tk)
        
        # Finalize assignments and update galleries for non-reverted tracks
        for tk, (gid, _, type_str) in current_assignments.items(): # Score not needed here
            if gid is not None and tk not in reverted_keys_for_second_pass :
                self.track_to_global_id[tk] = gid
                self.global_id_last_seen_cam[gid] = tk[0]
                self.global_id_last_seen_frame[gid] = current_frame_idx # Use GID as key
                
                if type_str not in ["new", "reverted_conflict", "filtered_handoff", "below_threshold"]:
                    original_embedding_for_update = query_features[tk]
                    self._update_gallery_with_ema(gid, original_embedding_for_update, type_str)
        
        if reverted_keys_for_second_pass:
            # Rebuild gallery_gids and gallery_embeddings_np from the current main gallery state
            # as it might have changed due to EMA updates.
            current_main_gallery_gids = list(self.reid_gallery.keys())
            current_main_gallery_embeds_np = np.array(
                 [self.reid_gallery[g] for g in current_main_gallery_gids if self.reid_gallery[g].size > 0], dtype=np.float32
            ) if current_main_gallery_gids else np.empty((0,0))


            for tk_reverted in reverted_keys_for_second_pass:
                reverted_embedding = query_features[tk_reverted] # Already normalized
                assigned_in_second_pass = False

                if current_main_gallery_embeds_np.size > 0 and reverted_embedding.size > 0:
                    sim_matrix_2pass = self._calculate_similarity_matrix(
                        reverted_embedding.reshape(1, -1), current_main_gallery_embeds_np
                    )
                    if sim_matrix_2pass is not None and sim_matrix_2pass.size > 0:
                        best_match_idx_2pass = np.argmax(sim_matrix_2pass[0])
                        max_sim_2pass = float(sim_matrix_2pass[0, best_match_idx_2pass])
                        
                        if max_sim_2pass >= self.similarity_threshold:
                            gid_2pass = current_main_gallery_gids[best_match_idx_2pass]
                            can_assign_gid_2pass = True
                            # Check if gid_2pass is already assigned to the *original winner* of the conflict in the same camera
                            original_conflict_assignments = assignments_by_cam_gid.get((tk_reverted[0], gid_2pass), [])
                            if original_conflict_assignments:
                                original_winner_tk = original_conflict_assignments[0][0]
                                if original_winner_tk != tk_reverted and self.track_to_global_id.get(original_winner_tk) == gid_2pass:
                                    can_assign_gid_2pass = False
                            
                            if can_assign_gid_2pass and self._apply_handoff_filter_for_match(tk_reverted, gid_2pass, active_triggers_map):
                                self.track_to_global_id[tk_reverted] = gid_2pass
                                self.global_id_last_seen_cam[gid_2pass] = tk_reverted[0]
                                self.global_id_last_seen_frame[gid_2pass] = current_frame_idx
                                self._update_gallery_with_ema(gid_2pass, reverted_embedding, "main_2pass")
                                assigned_in_second_pass = True
                
                if not assigned_in_second_pass:
                    new_gid_for_reverted = self.get_new_global_id()
                    self.track_to_global_id[tk_reverted] = new_gid_for_reverted
                    if reverted_embedding.size > 0: # Ensure non-empty before adding
                        self.reid_gallery[new_gid_for_reverted] = reverted_embedding
                    self.global_id_last_seen_cam[new_gid_for_reverted] = tk_reverted[0]
                    self.global_id_last_seen_frame[new_gid_for_reverted] = current_frame_idx

        self.update_galleries_lifecycle(active_track_keys_this_frame, current_frame_idx)

    def update_galleries_lifecycle(self, active_track_keys_this_frame: Set[TrackKey], current_frame_idx: int):
        """Manages movement between main and lost galleries, and gallery pruning."""
        all_previously_known_track_keys = set(self.track_to_global_id.keys())
        disappeared_track_keys = all_previously_known_track_keys - active_track_keys_this_frame

        for tk_disappeared in disappeared_track_keys:
            gid = self.track_to_global_id.pop(tk_disappeared, None)
            self.track_last_reid_frame.pop(tk_disappeared, None)
            if gid:
                feature = self.reid_gallery.get(gid) # Get from main gallery
                if feature is not None and feature.size > 0:
                    if gid not in self.lost_track_gallery:
                        last_active_frame_for_gid = self.global_id_last_seen_frame.get(gid, current_frame_idx -1)
                        self.lost_track_gallery[gid] = (feature, last_active_frame_for_gid)
        
        expired_lost_gids = [
            gid for gid, (_, frame_added_to_lost) in self.lost_track_gallery.items()
            if (current_frame_idx - frame_added_to_lost) > self.lost_track_buffer_frames
        ]
        for gid in expired_lost_gids:
            self.lost_track_gallery.pop(gid, None)

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
                    self.reid_gallery.pop(gid_prune, None)
                    self.global_id_last_seen_cam.pop(gid_prune, None)
                    self.global_id_last_seen_frame.pop(gid_prune, None)