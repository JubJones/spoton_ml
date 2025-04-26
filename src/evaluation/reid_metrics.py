import logging
from typing import List, Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

def _compute_distance_matrix(query_features: np.ndarray, gallery_features: np.ndarray, metric: str = 'cosine') -> np.ndarray:
    """Computes the distance matrix between query and gallery features."""
    if metric == 'cosine':
        query_features = normalize(query_features, norm='l2', axis=1)
        gallery_features = normalize(gallery_features, norm='l2', axis=1)
        similarity_matrix = np.dot(query_features, gallery_features.T)
        distance_matrix = 1.0 - similarity_matrix
    elif metric == 'euclidean':
        q_norm_sq = np.sum(query_features**2, axis=1, keepdims=True)
        g_norm_sq = np.sum(gallery_features**2, axis=1, keepdims=True)
        dot_product = np.dot(query_features, gallery_features.T)
        dist_sq = q_norm_sq + g_norm_sq.T - 2 * dot_product
        dist_sq[dist_sq < 0] = 0
        distance_matrix = np.sqrt(dist_sq)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}. Choose 'cosine' or 'euclidean'.")
    return distance_matrix


def compute_reid_metrics(
    query_features: np.ndarray,
    query_ids: np.ndarray,
    query_cams: List[str],
    gallery_features: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_cams: List[str],
    distance_metric: str = 'cosine'
) -> Dict[str, float]:
    """
    Calculates standard Re-ID evaluation metrics (mAP and CMC Rank-k).
    """
    if not all([
        query_features.ndim == 2, gallery_features.ndim == 2,
        query_features.shape[1] == gallery_features.shape[1],
        query_ids.shape[0] == query_features.shape[0] == len(query_cams),
        gallery_ids.shape[0] == gallery_features.shape[0] == len(gallery_cams),
        query_features.shape[0] > 0, gallery_features.shape[0] > 0
    ]):
        logger.error("Invalid input shapes or empty data for Re-ID metric calculation.")
        return {}

    logger.info(f"Computing Re-ID metrics for {len(query_ids)} queries and {len(gallery_ids)} gallery items using '{distance_metric}' distance.")

    try:
        dist_matrix = _compute_distance_matrix(query_features, gallery_features, distance_metric)
        logger.info(f"Distance matrix computed with shape: {dist_matrix.shape}")
    except Exception as e:
        logger.error(f"Error computing distance matrix: {e}", exc_info=True)
        return {}

    num_queries = dist_matrix.shape[0]
    num_gallery = dist_matrix.shape[1]
    indices = np.argsort(dist_matrix, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis]).astype(np.int32)

    all_aps = []; all_cmc = np.zeros(num_gallery, dtype=np.int32)

    logger.info("Calculating mAP and CMC Ranks...")
    for q_idx in range(num_queries):
        q_id = query_ids[q_idx]; q_cam = query_cams[q_idx]
        order = indices[q_idx]; remove_mask = (gallery_ids[order] == q_id) & (np.array(gallery_cams)[order] == q_cam)
        keep_mask = ~remove_mask; valid_order = order[keep_mask]; valid_matches = matches[q_idx][keep_mask]

        if not np.any(valid_matches): all_aps.append(0.0); continue

        first_match_idx = np.argmax(valid_matches)
        if first_match_idx < num_gallery: all_cmc[first_match_idx:] += 1

        y_true = valid_matches; y_scores = -dist_matrix[q_idx][valid_order]
        if np.sum(y_true) > 0:
             try: ap = average_precision_score(y_true, y_scores); all_aps.append(ap)
             except ValueError as ap_err: logger.warning(f"Query {q_idx} (ID {q_id}): Error calculating AP: {ap_err}. Setting AP=0."); all_aps.append(0.0)
        else: all_aps.append(0.0)

    if not all_aps: mAP = 0.0
    else: mAP = np.mean(all_aps)

    all_cmc = all_cmc.astype(np.float32) / num_queries
    rank1 = all_cmc[0] if len(all_cmc) > 0 else 0.0
    rank5 = all_cmc[4] if len(all_cmc) > 4 else 0.0
    rank10 = all_cmc[9] if len(all_cmc) > 9 else 0.0

    metrics = { 'reid_mAP': round(mAP, 5), 'reid_Rank-1': round(rank1, 5), 'reid_Rank-5': round(rank5, 5), 'reid_Rank-10': round(rank10, 5) }
    logger.info(f"Metrics calculated: {metrics}")
    return metrics
