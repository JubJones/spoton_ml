"""
Adapter for the BotSortTracker from the SpotOn backend, for use in the MLflow framework.
This class mirrors the structure and expected behavior of the backend's tracker.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple # Added Tuple

import numpy as np
import torch

# Attempt to import BoxMOT's BotSort directly, as in the backend
try:
    from boxmot.trackers.botsort.botsort import BotSort
    BOXMOT_BOTSORT_AVAILABLE = True
except ImportError as e:
    logging.critical(f"Failed to import boxmot.trackers.botsort.BotSort. BotSortTrackerAdapter unavailable. Error: {e}")
    BotSort = None # type: ignore
    BOXMOT_BOTSORT_AVAILABLE = False

from src.utils.reid_device_utils import get_reid_device_specifier_string # Use ML repo's util

logger = logging.getLogger(__name__)

class BotSortTrackerAdapter:
    """
    Adapter for the BotSort tracker, using the BoxMOT library implementation
    similar to the SpotOn backend.
    """
    def __init__(
        self,
        reid_weights_path: Path,
        device: torch.device,
        half_precision: bool = False,
        per_class: bool = False
    ):
        """
        Initializes the BotSortTrackerAdapter.

        Args:
            reid_weights_path: Path to the Re-ID model weights file.
            device: The torch.device to use for the tracker.
            half_precision: Whether to use half precision for tracker operations.
            per_class: Whether the tracker should operate on a per-class basis.
        """
        if not BOXMOT_BOTSORT_AVAILABLE or BotSort is None:
            raise ImportError("BoxMOT BotSort class is required but not available.")

        self.tracker_instance: Optional[BotSort] = None
        self.device = device # Store the torch.device
        self.reid_model_path: Path = reid_weights_path
        self.use_half: bool = half_precision
        self.per_class: bool = per_class
        self._model_loaded_flag: bool = False

        logger.info(
            f"BotSortTrackerAdapter configured. ReID Weights: {self.reid_model_path}, "
            f"Device: {self.device}, Half: {self.use_half}, PerClass: {self.per_class}"
        )

    def load_model(self):
        """
        Loads and initializes the BotSort tracker, including its ReID model.
        This method is synchronous to match typical model loading patterns.
        If BoxMOT's BotSort init is async, this would need adjustment.
        """
        if self._model_loaded_flag and self.tracker_instance is not None:
            logger.info("BotSort tracker (Adapter) already loaded.")
            return

        logger.info(f"Loading BotSort tracker (Adapter) on device: {self.device}...")

        # Convert torch.device to BoxMOT-compatible string ('cpu', '0', 'mps', etc.)
        boxmot_device_str = get_reid_device_specifier_string(self.device)
        effective_half = self.use_half and self.device.type == 'cuda'

        try:
            self.tracker_instance = BotSort(
                reid_weights=self.reid_model_path,
                device=boxmot_device_str,
                half=effective_half,
                per_class=self.per_class
            )
            self._model_loaded_flag = True
            logger.info(
                f"BotSort tracker instance (Adapter) created with ReID model '{self.reid_model_path}'."
            )
        except TypeError as te:
            logger.error(f"TypeError loading BotSort tracker (Adapter). Check constructor arguments: {te}", exc_info=True)
            self.tracker_instance = None
            self._model_loaded_flag = False
            raise
        except Exception as e:
            logger.error(f"Error loading BotSort tracker (Adapter): {e}", exc_info=True)
            self.tracker_instance = None
            self._model_loaded_flag = False
            raise

    def warmup(self, dummy_image_shape: Tuple[int, int, int] = (640, 480, 3)):
        """Warms up the tracker by performing a dummy update."""
        if not self._model_loaded_flag or not self.tracker_instance:
            logger.warning("BotSort tracker (Adapter) not loaded. Cannot perform warmup.")
            return
        
        logger.info(f"Warming up BotSortTrackerAdapter on device {self.device}...")
        try:
            dummy_frame = np.uint8(np.random.rand(*dummy_image_shape) * 255)
            dummy_dets = np.empty((0, 6), dtype=np.float32)
            _ = self.update(dummy_dets, dummy_frame)
            logger.info("BotSortTrackerAdapter warmup successful.")
        except Exception as e:
            logger.error(f"BotSortTrackerAdapter warmup failed: {e}", exc_info=True)

    def update(self, detections: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
        """
        Updates tracks with new detections for the current frame.

        Args:
            detections: A NumPy array of detections in [x1, y1, x2, y2, conf, cls_id] format.
            image_bgr: The current frame as a NumPy array (BGR).

        Returns:
            A NumPy array representing tracked objects, typically BoxMOT format.
        """
        if not self._model_loaded_flag or self.tracker_instance is None:
            raise RuntimeError("BotSort tracker (Adapter) not loaded. Call load_model() first.")

        detections_for_update = detections
        if detections.ndim != 2 or detections.shape[1] != 6:
            if detections.size == 0:
                 detections_for_update = np.empty((0, 6), dtype=np.float32)
            else:
                logger.error(
                    f"Invalid detections shape: {detections.shape}. Expected (N, 6). Tracker might fail."
                )
                # Allow tracker to handle, or return empty based on strictness
                # For now, pass through, BoxMOT might handle or error.
        
        if image_bgr is None or image_bgr.size == 0:
            logger.error("Invalid image provided to tracker update (Adapter).")
            return np.empty((0, 8)) # Default BoxMOT output columns if features are included

        try:
            # BoxMOT's update is synchronous
            tracked_output_np = self.tracker_instance.update(detections_for_update, image_bgr)

            # Validate output structure (basic check)
            if tracked_output_np is None or not isinstance(tracked_output_np, np.ndarray):
                return np.empty((0, 8))
            if tracked_output_np.size == 0:
                return np.empty((0, 8))
            if tracked_output_np.ndim != 2 or tracked_output_np.shape[1] < 5: # x1,y1,x2,y2,id
                 logger.warning(
                     f"Tracker output has unexpected shape: {tracked_output_np.shape}. "
                     "Expected at least 5 columns for track data."
                 )
                 # Return what we got, downstream parsing will handle it or fail
            return tracked_output_np
        except Exception as e:
            logger.error(f"Error during BotSort tracker update (Adapter): {e}", exc_info=True)
            return np.empty((0, 8)) # Default to empty with expected cols for ReID if features exist

    def reset(self):
        """Resets the tracker's state."""
        if self.tracker_instance and hasattr(self.tracker_instance, 'reset'):
            try:
                self.tracker_instance.reset()
                logger.info("BotSort tracker (Adapter) state reset.")
            except Exception as e:
                logger.error(f"Error resetting BotSort tracker (Adapter): {e}")
        elif self.tracker_instance:
            logger.warning(
                "BotSort tracker instance (Adapter) does not have a 'reset' method. Re-initializing for reset."
            )
            self._model_loaded_flag = False
            self.tracker_instance = None
            self.load_model()
        else:
            logger.warning("BotSort tracker instance (Adapter) not available to reset.") 