import abc
import logging
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

# Conditional imports for models - improves portability if some aren't installed
try:
    from ultralytics import YOLO, RTDETR
except ImportError:
    YOLO, RTDETR = None, None
    logging.warning("Ultralytics library not found. YOLO and RTDETR strategies unavailable.")

try:
    # Assuming RFDETRLarge is the class name, adjust if necessary
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

DetectionResult = Tuple[List[List[float]], List[int], List[float]]  # boxes_xywh, track_ids, confidences


class DetectionTrackingStrategy(abc.ABC):
    """Abstract base class for object detection strategies."""

    @abc.abstractmethod
    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        """
        Load the specific detection model.
        """
        self.model_path = model_path
        self.device = device
        self.config = config
        self.model: Optional[Any] = None
        self.person_class_id = config.get('person_class_id', 0)  # Default COCO index for person
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.placeholder_track_id = -1

    @abc.abstractmethod
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Process a single frame to detect persons.
        """
        pass

    def get_model(self) -> Optional[Any]:
        """Returns the loaded model object."""
        return self.model

    def get_sample_input_tensor(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """
        Preprocesses a sample frame to get the expected input tensor format.
        Needs to be implemented specifically by subclasses if signature/input example
        is desired for model logging. Returns None if not implemented or fails.
        """
        logger.warning(f"get_sample_input_tensor not implemented for {self.__class__.__name__}")
        return None

    def _warmup(self):
        """Optional warmup routine for the model."""
        if self.model:
            try:
                logger.info(f"Warming up {self.__class__.__name__} model...")
                # Use a small, valid-looking frame size for warmup
                warmup_h, warmup_w = 64, 64
                dummy_frame = np.zeros((warmup_h, warmup_w, 3), dtype=np.uint8)

                # Try getting a sample tensor first if available
                sample_tensor = self.get_sample_input_tensor(dummy_frame)
                if sample_tensor is not None and isinstance(self.model, torch.nn.Module):
                    input_batch = [sample_tensor.to(self.device)]
                    with torch.no_grad():
                        _ = self.model(input_batch)
                elif hasattr(self.model, 'predict'): # For ultralytics/rfdetr style
                     _ = self.process_frame(dummy_frame) # Fallback to process_frame
                else:
                     logger.warning(f"Cannot determine appropriate warmup method for {self.__class__.__name__}")

                logger.info(f"{self.__class__.__name__} warmup complete.")
            except Exception as e:
                logger.warning(f"Warmup failed for {self.__class__.__name__}: {e}", exc_info=True)


class YoloStrategy(DetectionTrackingStrategy):
    """Detection using YOLO models via Ultralytics."""

    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        super().__init__(model_path, device, config)
        if YOLO is None:
            raise ImportError("Ultralytics YOLO is required for YoloStrategy but not found.")
        logger.info(f"Initializing YOLO strategy with model: {model_path} on device: {device}")
        try:
            self.model = YOLO(model_path)
            # Note: Ultralytics models handle device internally in predict
            # self.model.to(self.device) # Let predict handle it
            self._warmup()
            logger.info(f"YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{model_path}': {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        boxes_xywh, track_ids, confidences = [], [], []
        try:
            results = self.model.predict(
                frame,
                classes=[self.person_class_id],
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )

            if results and results[0].boxes is not None:
                res_boxes = results[0].boxes
                # Ensure tensors exist before accessing cpu/numpy
                if hasattr(res_boxes, 'xywh') and res_boxes.xywh is not None:
                    boxes_xywh = res_boxes.xywh.cpu().numpy().tolist()
                if hasattr(res_boxes, 'conf') and res_boxes.conf is not None:
                    confidences = res_boxes.conf.cpu().numpy().tolist()
                else: # Handle cases where confidence might be missing
                    confidences = [self.confidence_threshold] * len(boxes_xywh)

                # YOLO predict doesn't provide track IDs, assign placeholder
                track_ids = [self.placeholder_track_id] * len(boxes_xywh)

                # Ensure all lists have the same length after potential filtering
                min_len = min(len(boxes_xywh), len(track_ids), len(confidences))
                boxes_xywh = boxes_xywh[:min_len]
                track_ids = track_ids[:min_len]
                confidences = confidences[:min_len]

        except Exception as e:
            logger.error(f"Error during YOLO processing: {e}", exc_info=True)
            return [], [], []  # Return empty lists on error

        return boxes_xywh, track_ids, confidences

    def get_sample_input_tensor(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0  # Normalize to [0, 1]
            if img.ndimension() == 3:
                img = img.permute(2, 0, 1) # HWC to CHW
            img = img.unsqueeze(0) # Add batch dimension
            return img # Return NCHW tensor
        except Exception as e:
            logger.error(f"Failed to create sample input tensor for YOLO: {e}")
            return None

class RTDetrStrategy(DetectionTrackingStrategy):
    """Detection using RT-DETR models via Ultralytics."""

    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        super().__init__(model_path, device, config)
        if RTDETR is None:
            raise ImportError("Ultralytics RTDETR is required for RTDetrStrategy but not found.")
        logger.info(f"Initializing RT-DETR strategy with model: {model_path} on device: {device}")
        try:
            self.model = RTDETR(model_path)
            # Note: Ultralytics models handle device internally in predict
            # self.model.to(self.device) # Let predict handle it
            self._warmup()
            logger.info(f"RT-DETR model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load RT-DETR model '{model_path}': {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        boxes_xywh, track_ids, confidences = [], [], []
        try:
            results = self.model.predict(
                frame,
                classes=[self.person_class_id],
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )

            if results and results[0].boxes is not None:
                res_boxes = results[0].boxes
                if hasattr(res_boxes, 'xywh') and res_boxes.xywh is not None:
                    boxes_xywh = res_boxes.xywh.cpu().numpy().tolist()
                if hasattr(res_boxes, 'conf') and res_boxes.conf is not None:
                    confidences = res_boxes.conf.cpu().numpy().tolist()
                else:
                    confidences = [self.confidence_threshold] * len(boxes_xywh)

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


    def get_sample_input_tensor(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.permute(2, 0, 1) # HWC to CHW
            img = img.unsqueeze(0) # Add batch dimension
            return img # Return NCHW tensor
        except Exception as e:
            logger.error(f"Failed to create sample input tensor for RTDETR: {e}")
            return None

class FasterRCNNStrategy(DetectionTrackingStrategy):
    """Detection using Faster R-CNN (ResNet50 FPN) from TorchVision."""

    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        super().__init__(model_path, device, config)
        if FasterRCNN_ResNet50_FPN_Weights is None:
            raise ImportError("torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights not found.")

        logger.info(f"Initializing Faster R-CNN strategy (TorchVision default weights) on device: {device}")
        try:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            self.transforms = weights.transforms() # Store transforms
            logger.info("Faster R-CNN model loaded successfully.")
            self._warmup()
        except Exception as e:
            logger.error(f"Failed to load Faster R-CNN model: {e}")
            raise

    @torch.no_grad()  # Disable gradient calculations for inference
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        boxes_xywh, track_ids, confidences = [], [], []
        if self.model is None or self.transforms is None:
             logger.error("FasterRCNN model or transforms not initialized.")
             return [], [], []
        try:
            # Preprocessing: BGR to RGB PIL Image -> Tensor -> Batch
            img_rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = self.transforms(img_rgb_pil)
            input_batch = [input_tensor.to(self.device)]

            predictions = self.model(input_batch)

            # Postprocessing
            # Check predictions format, it's a list containing a dict
            if predictions and isinstance(predictions, list) and isinstance(predictions[0], dict):
                pred_dict = predictions[0]
                pred_boxes_xyxy = pred_dict["boxes"].cpu().numpy()
                pred_labels = pred_dict["labels"].cpu().numpy()
                pred_scores = pred_dict["scores"].cpu().numpy()

                for box_xyxy, label, score in zip(pred_boxes_xyxy, pred_labels, pred_scores):
                    # Check person_class_id (comes from config, defaults to 1 for torchvision F-RCNN)
                    is_person = (label == self.person_class_id)
                    if is_person and score >= self.confidence_threshold:
                        x1, y1, x2, y2 = box_xyxy
                        width = x2 - x1
                        height = y2 - y1
                        if width > 0 and height > 0:
                            center_x = x1 + width / 2
                            center_y = y1 + height / 2
                            boxes_xywh.append([center_x, center_y, width, height])
                            track_ids.append(self.placeholder_track_id)  # FasterRCNN provides no tracking
                            confidences.append(float(score))
            else:
                logger.warning(f"Unexpected prediction format from FasterRCNN: {type(predictions)}")


        except Exception as e:
            logger.error(f"Error during Faster R-CNN processing step: {e}", exc_info=True)
            return [], [], []

        return boxes_xywh, track_ids, confidences

    def get_sample_input_tensor(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        if self.transforms is None: return None
        try:
            img_rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = self.transforms(img_rgb_pil)
            return input_tensor
        except Exception as e:
             logger.error(f"Failed to create sample input tensor for FasterRCNN: {e}")
             return None


class RfDetrStrategy(DetectionTrackingStrategy):
    """Detection using RF-DETR model from the rfdetr library."""

    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        # model_path might be ignored if RFDETRLarge loads fixed weights, but kept for consistency
        super().__init__(model_path, device, config)
        if RFDETRLarge is None:
            raise ImportError("RFDETRLarge is required for RfDetrStrategy but not found.")

        # RFDETRLarge might not explicitly take torch.device, but string ('cpu', 'cuda')
        logger.info(f"Initializing RF-DETR strategy (RFDETRLarge) targeting device: {device}")

        try:
            self.model = RFDETRLarge()
            if hasattr(self.model, 'to') and isinstance(self.model, torch.nn.Module): # Check if it's a torch module
                 self.model.to(self.device)
                 self.model.eval()

            # Assuming COCO index 1 for person here. Adjust if needed.
            self.person_label_index = self.person_class_id

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

            detections = self.model.predict(img_rgb_pil, threshold=self.confidence_threshold)

            # Check the structure of the 'detections' object based on rfdetr library docs
            if detections and hasattr(detections, 'xyxy') and hasattr(detections, 'class_id') and hasattr(detections, 'confidence'):
                pred_boxes_xyxy = detections.xyxy
                pred_labels = detections.class_id
                pred_scores = detections.confidence

                # Handle potential tensor outputs
                if isinstance(pred_boxes_xyxy, torch.Tensor): pred_boxes_xyxy = pred_boxes_xyxy.cpu().numpy()
                if isinstance(pred_labels, torch.Tensor): pred_labels = pred_labels.cpu().numpy()
                if isinstance(pred_scores, torch.Tensor): pred_scores = pred_scores.cpu().numpy()


                for box_xyxy, label, score in zip(pred_boxes_xyxy, pred_labels, pred_scores):
                    is_person = (label == self.person_label_index)
                    if is_person and score >= self.confidence_threshold:
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
                logger.warning(f"RF-DETR predict returned None or unexpected object structure: {type(detections)}")

        except Exception as e:
            logger.error(f"Error during RF-DETR processing step: {e}", exc_info=True)
            return [], [], []

        return boxes_xywh, track_ids, confidences

    def get_sample_input_tensor(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        try:
            # Fallback simple tensor:
            img_t = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
            return img_t # Return CHW tensor (batch dim added later)

        except Exception as e:
             logger.error(f"Failed to create sample input tensor for RFDETR: {e}")
             return None


def get_strategy(model_config: Dict[str, Any], device: torch.device) -> DetectionTrackingStrategy:
    """
    Factory function to instantiate the correct detection strategy based on config.
    """
    model_type = model_config.get("type", "").lower()
    model_path = model_config.get("weights_path", "") # May not be used by all strategies

    if model_type == "yolo":
        return YoloStrategy(model_path, device, model_config)
    elif model_type == "rtdetr":
        return RTDetrStrategy(model_path, device, model_config)
    elif model_type == "fasterrcnn":
        return FasterRCNNStrategy(model_path, device, model_config)
    elif model_type == "rfdetr":
        return RfDetrStrategy(model_path, device, model_config)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Must be one of "
                         f"'yolo', 'rtdetr', 'fasterrcnn', 'rfdetr'.")