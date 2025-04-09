import abc
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torchvision
from PIL import Image
import cv2

# Conditional imports for models - improves portability if some aren't installed
try:
    from ultralytics import YOLO, RTDETR
except ImportError:
    YOLO, RTDETR = None, None
    logging.warning("Ultralytics library not found. YOLO and RTDETR strategies unavailable.")

try:
    from rfdetr import RFDETRLarge
except ImportError:
    RFDETRLarge = None
    logging.warning("rfdetr library not found. RFDETR strategy unavailable.")

try:
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
except ImportError:
    FasterRCNN_ResNet50_FPN_Weights = None
    logging.warning("torchvision FasterRCNN weights not found.")


logger = logging.getLogger(__name__)

# Define type alias for detection results
DetectionResult = Tuple[List[List[float]], List[int], List[float]] # boxes_xywh, track_ids, confidences

class DetectionTrackingStrategy(abc.ABC):
    """Abstract base class for object detection and tracking strategies."""

    @abc.abstractmethod
    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        """
        Load the specific detection/tracking model.

        Args:
            model_path: Path to the model weights file.
            device: The torch.device to run inference on.
            config: Dictionary containing model-specific configurations
                    (e.g., confidence_threshold, person_class_id).
        """
        self.model_path = model_path
        self.device = device
        self.config = config
        self.model = None # To be initialized by subclasses
        self.person_class_id = config.get('person_class_id', 0) # Default to 0 (COCO person)
        self.confidence_threshold = config.get('confidence_threshold', 0.5) # Default threshold
        self.placeholder_track_id = -1 # Used when tracking is not available/enabled

    @abc.abstractmethod
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Process a single frame to detect objects (specifically persons).
        Tracking logic (assigning consistent IDs) is handled by the main Tracker class,
        although some strategies might return preliminary track IDs.

        Args:
            frame: The input frame in BGR format (from OpenCV).

        Returns:
            A tuple containing:
            - List of bounding boxes ([center_x, center_y, width, height]).
            - List of track IDs (integer for tracked objects, -1 for detections without ID).
            - List of confidence scores for each detection/track.
        """
        pass

    def _warmup(self):
        """Optional warmup routine for the model."""
        if self.model:
            try:
                logger.info(f"Warming up {self.__class__.__name__} model...")
                # Use a small dummy frame
                dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)
                _ = self.process_frame(dummy_frame)
                logger.info(f"{self.__class__.__name__} warmup complete.")
            except Exception as e:
                logger.warning(f"Warmup failed for {self.__class__.__name__}: {e}")


class YoloStrategy(DetectionTrackingStrategy):
    """Detection and tracking using YOLO models (v8, v9, etc.) via Ultralytics."""

    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        super().__init__(model_path, device, config)
        if YOLO is None:
            raise ImportError("Ultralytics YOLO is required for YoloStrategy but not found.")
        logger.info(f"Initializing YOLO strategy with model: {model_path} on device: {device}")
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self._warmup()
            logger.info(f"YOLO model loaded successfully onto {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{model_path}' onto {self.device}: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        boxes_xywh, track_ids, confidences = [], [], []
        try:
            # Note: Ultralytics tracker needs state between frames.
            # If used for pure detection per frame, use model.predict()
            # If used for tracking, ensure persist=True is managed externally by tracker if needed.
            # For now, assume predict for simplicity in this refactor.
            results = self.model.predict(
                frame,
                classes=[self.person_class_id],
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )

            if results and results[0].boxes is not None:
                res_boxes = results[0].boxes
                if res_boxes.xywh is not None:
                    boxes_xywh = res_boxes.xywh.cpu().numpy().tolist()
                if res_boxes.conf is not None:
                    confidences = res_boxes.conf.cpu().numpy().tolist()

                # YOLO predict doesn't provide track IDs, assign placeholder
                track_ids = [self.placeholder_track_id] * len(boxes_xywh)

                # Ensure all lists have the same length
                min_len = min(len(boxes_xywh), len(track_ids), len(confidences))
                boxes_xywh = boxes_xywh[:min_len]
                track_ids = track_ids[:min_len]
                confidences = confidences[:min_len]

        except Exception as e:
            logger.error(f"Error during YOLO processing: {e}", exc_info=True)
            return [], [], [] # Return empty lists on error

        return boxes_xywh, track_ids, confidences


class RTDetrStrategy(DetectionTrackingStrategy):
    """Detection using RT-DETR models via Ultralytics."""

    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        super().__init__(model_path, device, config)
        if RTDETR is None:
            raise ImportError("Ultralytics RTDETR is required for RTDetrStrategy but not found.")
        logger.info(f"Initializing RT-DETR strategy with model: {model_path} on device: {device}")
        try:
            self.model = RTDETR(model_path)
            self.model.to(self.device)
            self._warmup()
            logger.info(f"RT-DETR model loaded successfully onto {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load RT-DETR model '{model_path}' onto {self.device}: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        boxes_xywh, track_ids, confidences = [], [], []
        try:
            # RTDETR predict only gives detections
            results = self.model.predict(
                frame,
                classes=[self.person_class_id],
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )

            if results and results[0].boxes is not None:
                res_boxes = results[0].boxes
                if res_boxes.xywh is not None:
                    boxes_xywh = res_boxes.xywh.cpu().numpy().tolist()
                if res_boxes.conf is not None:
                    confidences = res_boxes.conf.cpu().numpy().tolist()

                # RTDETR predict doesn't provide track IDs
                track_ids = [self.placeholder_track_id] * len(boxes_xywh)

                min_len = min(len(boxes_xywh), len(track_ids), len(confidences))
                boxes_xywh = boxes_xywh[:min_len]
                track_ids = track_ids[:min_len]
                confidences = confidences[:min_len]

        except Exception as e:
            logger.error(f"Error during RT-DETR processing: {e}", exc_info=True)
            return [], [], []

        return boxes_xywh, track_ids, confidences


class FasterRCNNStrategy(DetectionTrackingStrategy):
    """Detection using Faster R-CNN (ResNet50 FPN) from TorchVision."""

    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        # model_path is ignored for default torchvision weights, but kept for consistency
        super().__init__(model_path, device, config)
        if FasterRCNN_ResNet50_FPN_Weights is None:
             raise ImportError("torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights not found.")

        logger.info(f"Initializing Faster R-CNN strategy (TorchVision default weights) on device: {device}")
        try:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            self.transforms = weights.transforms()
            logger.info("Faster R-CNN model loaded successfully.")
            self._warmup()
        except Exception as e:
            logger.error(f"Failed to load Faster R-CNN model: {e}")
            raise

    @torch.no_grad() # Disable gradient calculations for inference
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        boxes_xywh, track_ids, confidences = [], [], []
        try:
            # Preprocessing: BGR to RGB PIL Image -> Tensor -> Batch
            img_rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = self.transforms(img_rgb_pil)
            input_batch = [input_tensor.to(self.device)]

            predictions = self.model(input_batch)

            # Postprocessing
            pred_boxes_xyxy = predictions[0]["boxes"].cpu().numpy()
            pred_labels = predictions[0]["labels"].cpu().numpy()
            pred_scores = predictions[0]["scores"].cpu().numpy()

            for box_xyxy, label, score in zip(pred_boxes_xyxy, pred_labels, pred_scores):
                # COCO label for person is 1 in torchvision models
                is_person = (label == 1) # FasterRCNN uses COCO label 1 for person
                if is_person and score >= self.confidence_threshold:
                    x1, y1, x2, y2 = box_xyxy
                    width = x2 - x1
                    height = y2 - y1
                    if width > 0 and height > 0:
                        center_x = x1 + width / 2
                        center_y = y1 + height / 2
                        boxes_xywh.append([center_x, center_y, width, height])
                        track_ids.append(self.placeholder_track_id) # FasterRCNN provides no tracking
                        confidences.append(float(score))

        except Exception as e:
            logger.error(f"Error during Faster R-CNN processing step: {e}", exc_info=True)
            return [], [], []

        return boxes_xywh, track_ids, confidences


class RfDetrStrategy(DetectionTrackingStrategy):
    """Detection using RF-DETR model from the rfdetr library."""

    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
         # model_path is ignored for RFDETRLarge, but kept for consistency
        super().__init__(model_path, device, config)
        if RFDETRLarge is None:
            raise ImportError("RFDETRLarge is required for RfDetrStrategy but not found.")

        # RFDETRLarge might not explicitly take torch.device, but string ('cpu', 'cuda')
        device_str = 'cuda' if device.type == 'cuda' else 'cpu'
        logger.info(f"Initializing RF-DETR strategy (RFDETRLarge) targeting device string: '{device_str}'")

        try:
            # Assuming RFDETRLarge determines device internally or via its own mechanisms
            # Pass device string if constructor accepts it, otherwise it might use global settings
            # Check RFDETR documentation for precise device handling
            self.model = RFDETRLarge() # device=device_str - Adjust if constructor API changes

            # Check if model ended up on the correct device (might need adjustment based on rfdetr lib)
            # Example check (may need refinement):
            # model_device = next(self.model.parameters()).device
            # if model_device.type != device.type:
            #    logger.warning(f"RFDETRLarge model initialized on {model_device}, expected {device}. Check library behavior.")
            #    # Attempt to move if possible/necessary
            #    # self.model.to(device) # Might not work depending on library design

            self.person_label_index = 1 # Assuming COCO index 1 for person
            logger.info(f"RF-DETR strategy initialized.")
            self._warmup()

        except Exception as e:
            logger.error(f"Failed during RFDETRLarge initialization/warmup: {e}")
            raise

    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        boxes_xywh, track_ids, confidences = [], [], []
        if not self.model:
            logger.error("RF-DETR model not initialized.")
            return [], [], []
        try:
            # Preprocessing: BGR to RGB PIL Image
            img_rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # RFDETR predict returns a Detections object (check library for exact structure)
            detections = self.model.predict(img_rgb_pil, threshold=self.confidence_threshold)

            if detections and hasattr(detections, 'xyxy') and hasattr(detections, 'class_id') and hasattr(detections, 'confidence'):
                # Adapt based on the actual attributes of the 'detections' object returned by rfdetr
                pred_boxes_xyxy = detections.xyxy
                pred_labels = detections.class_id
                pred_scores = detections.confidence

                for box_xyxy, label, score in zip(pred_boxes_xyxy, pred_labels, pred_scores):
                    # Assuming COCO index 1 for person in rfdetr model
                    is_person = (label == self.person_label_index)
                    if is_person and score >= self.confidence_threshold: # Check threshold again just in case predict didn't filter perfectly
                        x1, y1, x2, y2 = box_xyxy
                        width = x2 - x1
                        height = y2 - y1
                        if width > 0 and height > 0:
                            center_x = x1 + width / 2
                            center_y = y1 + height / 2
                            boxes_xywh.append([center_x, center_y, width, height])
                            track_ids.append(self.placeholder_track_id) # RFDETR likely provides no tracking IDs
                            confidences.append(float(score))
            else:
                 logger.warning("RF-DETR predict returned None or unexpected Detections object structure.")

        except Exception as e:
            logger.error(f"Error during RF-DETR processing step: {e}", exc_info=True)
            return [], [], []

        return boxes_xywh, track_ids, confidences

# --- Factory function to get the strategy instance ---
def get_strategy(model_config: Dict[str, Any], device: torch.device) -> DetectionTrackingStrategy:
    """
    Factory function to instantiate the correct detection strategy based on config.

    Args:
        model_config: Dictionary containing model configuration ('type', 'weights_path', etc.).
        device: The torch.device to use.

    Returns:
        An instance of a DetectionTrackingStrategy subclass.

    Raises:
        ValueError: If the model type in the config is unsupported.
        ImportError: If the required library for the model type is not installed.
    """
    model_type = model_config.get("type", "").lower()
    model_path = model_config.get("weights_path", "")

    if model_type == "yolo":
        return YoloStrategy(model_path, device, model_config)
    elif model_type == "rtdetr":
        return RTDetrStrategy(model_path, device, model_config)
    elif model_type == "fasterrcnn":
        return FasterRCNNStrategy(model_path, device, model_config)
    elif model_type == "rfdetr":
         # Note: RFDETR strategy might have specific dependencies or initialization nuances
        return RfDetrStrategy(model_path, device, model_config)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Must be one of "
                         f"'yolo', 'rtdetr', 'fasterrcnn', 'rfdetr'.")