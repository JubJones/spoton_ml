import abc
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import torch
from PIL import Image

# BoxMOT imports
try:
    from boxmot.appearance.reid_auto_backend import ReidAutoBackend
    from boxmot.appearance.backends.base_backend import BaseModelBackend
except ImportError as e:
    logging.critical(f"Failed to import boxmot components needed for ReID. Is boxmot installed? Error: {e}")
    BaseModelBackend = type(None)
    ReidAutoBackend = None

logger = logging.getLogger(__name__)

def get_reid_device_specifier_string(device: torch.device) -> str:
    if device.type == 'cuda': return str(device.index if device.index is not None else 0)
    if device.type == 'mps': return 'mps'
    return 'cpu'

class ReIDStrategy(abc.ABC):
    @abc.abstractmethod
    def __init__(self, model_config: Dict[str, Any], device: torch.device, project_root: Path):
        self.model_config = model_config
        self.model_type = model_config.get("model_type", "unknown_reid")
        self.device = device
        self.project_root = project_root # Store project root
        self.model: Optional[Any] = None

    @abc.abstractmethod
    def extract_features(self, frame_bgr: np.ndarray, bboxes_xyxy: np.ndarray) -> Dict[int, np.ndarray]:
        pass

    def get_model(self) -> Optional[Any]: return self.model
    def _warmup(self): logger.info(f"Warmup for {self.model_type} handled by BoxMOT backend if applicable.")


class BoxMOTReIDStrategy(ReIDStrategy):
    """Re-Identification strategy using BoxMOT's ReidAutoBackend."""
    def __init__(self, model_config: Dict[str, Any], device: torch.device, project_root: Path):
        super().__init__(model_config, device, project_root)
        if ReidAutoBackend is None: raise ImportError("BoxMOT ReidAutoBackend not available.")

        weights_identifier: Optional[str] = None
        # Get base dir from the *full* run_config passed down via model_config['_run_config']
        run_cfg = model_config.get("_run_config", {})
        data_cfg = run_cfg.get("data", {})
        weights_base_dir_str = data_cfg.get("weights_base_dir", "weights/reid")
        weights_base_dir = self.project_root / weights_base_dir_str

        config_weights_path_str = model_config.get("weights_path")
        if config_weights_path_str:
            potential_path = weights_base_dir / config_weights_path_str
            if potential_path.is_file():
                weights_identifier = str(potential_path.resolve())
                logger.info(f"Using specific weights path from config: {weights_identifier}")
            else:
                logger.warning(f"Weights path specified in config not found: {potential_path}. Will attempt lookup by type: {self.model_type}")
                weights_identifier = self.model_type # Fallback to type name
        else:
            weights_identifier = self.model_type
            logger.info(f"No specific weights path. Using BoxMOT identifier/type: {weights_identifier}")

        if not weights_identifier: raise ValueError("Could not determine weights identifier for ReID model.")

        logger.info(f"Initializing BoxMOT ReID Backend: {self.model_type} on {device}")
        logger.info(f"Using weights identifier: {weights_identifier}")

        try:
            reid_device_specifier = get_reid_device_specifier_string(self.device)
            self.reid_backend_handler = ReidAutoBackend(weights=weights_identifier, device=reid_device_specifier, half=False)
            self.model = self.reid_backend_handler.model
            if self.model is None: raise RuntimeError("BoxMOT ReidAutoBackend did not load a model.")
            self._warmup()
            logger.info(f"BoxMOT ReID strategy '{self.model_type}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load BoxMOT ReID model '{self.model_type}' with identifier '{weights_identifier}': {e}", exc_info=True)
            raise RuntimeError(f"BoxMOT ReID model loading failed for {self.model_type}") from e

    @torch.no_grad()
    def extract_features(self, frame_bgr: np.ndarray, bboxes_xyxy: np.ndarray) -> Dict[int, np.ndarray]:
        features: Dict[int, np.ndarray] = {}
        if self.reid_backend_handler is None: logger.error(f"[{self.model_type}] Backend handler not init."); return features
        if frame_bgr is None or frame_bgr.size == 0: return features
        if bboxes_xyxy is None or bboxes_xyxy.shape[0] == 0: return features
        if bboxes_xyxy.ndim != 2 or bboxes_xyxy.shape[1] != 4: logger.warning(f"[{self.model_type}] Invalid bbox shape: {bboxes_xyxy.shape}"); return features

        try:
            batch_features_np: Optional[np.ndarray] = self.reid_backend_handler.get_features(bboxes_xyxy, frame_bgr)
            if batch_features_np is not None and batch_features_np.shape[0] == bboxes_xyxy.shape[0]:
                for i, feature_vec in enumerate(batch_features_np):
                    if feature_vec is not None and np.isfinite(feature_vec).all() and feature_vec.size > 0: features[i] = feature_vec
            elif batch_features_np is not None: logger.warning(f"[{self.model_type}] Feature count mismatch: Got {batch_features_np.shape[0]}, expected {bboxes_xyxy.shape[0]}.")
        except Exception as e: logger.error(f"[{self.model_type}] Feature extraction call failed: {e}", exc_info=True)
        return features

# Updated factory function to pass project_root and full config context
def get_reid_strategy(model_config: Dict[str, Any], device: torch.device, project_root: Path) -> Optional[ReIDStrategy]:
    try:
        if not model_config.get("model_type"): logger.error("Model config missing 'model_type'."); return None
        strategy = BoxMOTReIDStrategy(model_config, device, project_root)
        return strategy
    except Exception as e: logger.error(f"Failed to create ReID strategy for config {model_config}: {e}"); return None

def get_reid_strategy_from_run_config(run_config: Dict[str, Any], device: torch.device, project_root: Path) -> Optional[ReIDStrategy]:
     model_config = run_config.get("model", {})
     model_config["_run_config"] = run_config # Inject full run_config for context
     return get_reid_strategy(model_config, device, project_root)