"""
Adapter for the ReIDManager from the SpotOn backend, for use in the MLflow framework.
This class mirrors the structure and expected behavior of the backend's ReID manager.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.tracking_backend_logic.common_types_adapter import (
    CameraID,
    FeatureVector,
    GlobalID,
    TrackID
)

logger = logging.getLogger(__name__)

class ReIDManagerAdapter:
    """
    Manages Re-ID operations using CLIP model, adapted from the SpotOn backend.
    This class handles feature extraction and matching for person re-identification.
    """
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        half_precision: bool = False,
        similarity_threshold: float = 0.7
    ):
        """
        Initializes the ReIDManagerAdapter.

        Args:
            model_path: Path to the CLIP model weights file.
            device: The torch.device to use for the model.
            half_precision: Whether to use half precision for model operations.
            similarity_threshold: Threshold for considering two features as a match.
        """
        self.model_path = model_path
        self.device = device
        self.half_precision = half_precision
        self.similarity_threshold = similarity_threshold
        self._model = None
        self._preprocessor = None
        self._model_loaded_flag = False

        logger.info(
            f"ReIDManagerAdapter configured. Model: {self.model_path}, "
            f"Device: {self.device}, Half: {self.half_precision}, "
            f"Threshold: {self.similarity_threshold}"
        )

    def load_model(self):
        """
        Loads and initializes the CLIP model for Re-ID.
        This method is synchronous to match typical model loading patterns.
        """
        if self._model_loaded_flag and self._model is not None:
            logger.info("CLIP model (Adapter) already loaded.")
            return

        logger.info(f"Loading CLIP model (Adapter) on device: {self.device}...")

        try:
            import clip
            self._model, self._preprocessor = clip.load(
                "ViT-B/32",
                device=self.device,
                download_root=str(self.model_path.parent)
            )
            if self.half_precision and self.device.type == 'cuda':
                self._model = self._model.half()
            self._model_loaded_flag = True
            logger.info("CLIP model (Adapter) loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading CLIP model (Adapter): {e}", exc_info=True)
            self._model = None
            self._preprocessor = None
            self._model_loaded_flag = False
            raise

    def warmup(self, dummy_image_shape: Tuple[int, int, int] = (640, 480, 3)):
        """Warms up the Re-ID model by performing a dummy feature extraction."""
        if not self._model_loaded_flag or self._model is None:
            logger.warning("CLIP model (Adapter) not loaded. Cannot perform warmup.")
            return
        
        logger.info(f"Warming up ReIDManagerAdapter on device {self.device}...")
        try:
            dummy_image = Image.fromarray(np.uint8(np.random.rand(*dummy_image_shape) * 255))
            _ = self.extract_features(dummy_image)
            logger.info("ReIDManagerAdapter warmup successful.")
        except Exception as e:
            logger.error(f"ReIDManagerAdapter warmup failed: {e}", exc_info=True)

    def extract_features(self, image: Image.Image) -> FeatureVector:
        """
        Extracts features from an image using the CLIP model.

        Args:
            image: PIL Image to extract features from.

        Returns:
            A feature vector representing the image.
        """
        if not self._model_loaded_flag or self._model is None or self._preprocessor is None:
            raise RuntimeError("CLIP model (Adapter) not loaded. Call load_model() first.")

        try:
            # Preprocess image
            image_input = self._preprocessor(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().squeeze()
        except Exception as e:
            logger.error(f"Error extracting features (Adapter): {e}", exc_info=True)
            raise

    def compute_similarity(
        self,
        features1: FeatureVector,
        features2: FeatureVector
    ) -> float:
        """
        Computes cosine similarity between two feature vectors.

        Args:
            features1: First feature vector.
            features2: Second feature vector.

        Returns:
            Cosine similarity score between the features.
        """
        try:
            # Convert to numpy if needed
            if isinstance(features1, torch.Tensor):
                features1 = features1.cpu().numpy()
            if isinstance(features2, torch.Tensor):
                features2 = features2.cpu().numpy()

            # Ensure features are 1D
            features1 = features1.flatten()
            features2 = features2.flatten()

            # Compute cosine similarity
            similarity = np.dot(features1, features2) / (
                np.linalg.norm(features1) * np.linalg.norm(features2)
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity (Adapter): {e}", exc_info=True)
            return 0.0

    def find_matches(
        self,
        query_features: FeatureVector,
        gallery_features: List[FeatureVector],
        gallery_ids: List[Tuple[CameraID, TrackID]]
    ) -> List[Tuple[Tuple[CameraID, TrackID], float]]:
        """
        Finds matches for a query feature vector in a gallery of features.

        Args:
            query_features: Feature vector to find matches for.
            gallery_features: List of gallery feature vectors.
            gallery_ids: List of (camera_id, track_id) tuples corresponding to gallery features.

        Returns:
            List of ((camera_id, track_id), similarity) tuples for matches above threshold.
        """
        if not gallery_features or not gallery_ids:
            return []

        matches = []
        for features, (camera_id, track_id) in zip(gallery_features, gallery_ids):
            similarity = self.compute_similarity(query_features, features)
            if similarity >= self.similarity_threshold:
                matches.append(((camera_id, track_id), similarity))

        # Sort by similarity in descending order
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def assign_global_id(
        self,
        query_features: FeatureVector,
        existing_features: Dict[GlobalID, FeatureVector]
    ) -> Optional[GlobalID]:
        """
        Assigns a global ID to a query feature vector based on similarity to existing features.

        Args:
            query_features: Feature vector to assign global ID to.
            existing_features: Dictionary mapping global IDs to their feature vectors.

        Returns:
            Global ID if a match is found, None otherwise.
        """
        if not existing_features:
            return None

        best_match_id = None
        best_similarity = self.similarity_threshold

        for global_id, features in existing_features.items():
            similarity = self.compute_similarity(query_features, features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = global_id

        return best_match_id 