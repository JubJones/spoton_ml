"""
Adapter for the CameraTrackerFactory from the SpotOn backend, for use in the MLflow framework.
This class mirrors the structure and expected behavior of the backend's factory.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import torch

from src.tracking_backend_logic.botsort_tracker_adapter import BotSortTrackerAdapter
from src.tracking_backend_logic.common_types_adapter import CameraID

logger = logging.getLogger(__name__)

class CameraTrackerFactoryAdapter:
    """
    Factory class for creating and managing camera-specific trackers.
    This adapter mirrors the structure of the SpotOn backend's CameraTrackerFactory.
    """
    def __init__(
        self,
        reid_weights_path: Path,
        device: torch.device,
        half_precision: bool = False,
        per_class: bool = False
    ):
        """
        Initializes the CameraTrackerFactoryAdapter.

        Args:
            reid_weights_path: Path to the Re-ID model weights file.
            device: The torch.device to use for the trackers.
            half_precision: Whether to use half precision for tracker operations.
            per_class: Whether the trackers should operate on a per-class basis.
        """
        self.reid_weights_path = reid_weights_path
        self.device = device
        self.half_precision = half_precision
        self.per_class = per_class
        self._trackers: Dict[CameraID, BotSortTrackerAdapter] = {}

        logger.info(
            f"CameraTrackerFactoryAdapter initialized with ReID weights: {self.reid_weights_path}, "
            f"Device: {self.device}, Half: {self.half_precision}, PerClass: {self.per_class}"
        )

    def get_tracker(self, camera_id: CameraID) -> BotSortTrackerAdapter:
        """
        Gets or creates a tracker for the specified camera.

        Args:
            camera_id: The ID of the camera to get a tracker for.

        Returns:
            A BotSortTrackerAdapter instance for the specified camera.
        """
        if camera_id not in self._trackers:
            logger.info(f"Creating new BotSortTrackerAdapter for camera {camera_id}")
            self._trackers[camera_id] = BotSortTrackerAdapter(
                reid_weights_path=self.reid_weights_path,
                device=self.device,
                half_precision=self.half_precision,
                per_class=self.per_class
            )
            self._trackers[camera_id].load_model()
        return self._trackers[camera_id]

    def reset_tracker(self, camera_id: CameraID) -> None:
        """
        Resets the tracker for the specified camera.

        Args:
            camera_id: The ID of the camera whose tracker should be reset.
        """
        if camera_id in self._trackers:
            logger.info(f"Resetting tracker for camera {camera_id}")
            self._trackers[camera_id].reset()
        else:
            logger.warning(f"No tracker found for camera {camera_id} to reset")

    def reset_all_trackers(self) -> None:
        """Resets all camera trackers."""
        logger.info("Resetting all camera trackers")
        for camera_id in self._trackers:
            self.reset_tracker(camera_id)

    def remove_tracker(self, camera_id: CameraID) -> None:
        """
        Removes the tracker for the specified camera.

        Args:
            camera_id: The ID of the camera whose tracker should be removed.
        """
        if camera_id in self._trackers:
            logger.info(f"Removing tracker for camera {camera_id}")
            del self._trackers[camera_id]
        else:
            logger.warning(f"No tracker found for camera {camera_id} to remove")

    def remove_all_trackers(self) -> None:
        """Removes all camera trackers."""
        logger.info("Removing all camera trackers")
        self._trackers.clear()

    def get_all_trackers(self) -> Dict[CameraID, BotSortTrackerAdapter]:
        """
        Gets all camera trackers.

        Returns:
            A dictionary mapping camera IDs to their respective trackers.
        """
        return self._trackers.copy() 