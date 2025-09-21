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
        """Load the specific detection model."""
        self.model_path = model_path
        self.device = device
        self.config = config
        self.model: Optional[Any] = None
        self.person_class_id = config.get('person_class_id', 0)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.placeholder_track_id = -1

    @abc.abstractmethod
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Process a single frame to detect persons."""
        pass

    def get_model(self) -> Optional[Any]:
        """Returns the loaded model object."""
        return self.model

    def get_sample_input_tensor(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocesses a sample frame to get the expected input tensor (CHW)."""
        logger.warning(f"get_sample_input_tensor not specifically implemented for {self.__class__.__name__}. Using generic fallback.")
        try:
            # Generic fallback: Convert BGR numpy to RGB CHW tensor normalized to [0,1]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1) # HWC to CHW
            img_t = img_t.float() / 255.0
            return img_t
        except Exception as e:
            logger.error(f"Generic get_sample_input_tensor failed: {e}")
            return None

    def _warmup(self):
        """Optional warmup routine for the model."""
        if not self.model:
            return
            
        model_name = self.__class__.__name__
        logger.info(f"Warming up {model_name} model...")
        
        try:
            dummy_frame_np = np.zeros((64, 64, 3), dtype=np.uint8)
            
            # Strategy-specific warmup approaches
            warmup_success = False
            
            # Ultralytics models (YOLO, RT-DETR)
            if isinstance(self, (YoloStrategy, RTDetrStrategy)) and hasattr(self.model, 'predict'):
                self.model.predict(dummy_frame_np, device=self.device, verbose=False)
                logger.info(f"{model_name} warmup complete (Ultralytics predict)")
                warmup_success = True
                
            # RF-DETR models
            elif isinstance(self, RfDetrStrategy) and hasattr(self.model, 'predict'):
                dummy_pil = Image.fromarray(dummy_frame_np)
                self.model.predict(dummy_pil)
                logger.info(f"{model_name} warmup complete (RFDETR predict)")
                warmup_success = True
                
            # PyTorch models (FasterRCNN, etc.)
            elif isinstance(self.model, torch.nn.Module):
                sample_tensor = self.get_sample_input_tensor(dummy_frame_np)
                if sample_tensor is not None and sample_tensor.dim() == 3:
                    input_list = [sample_tensor.to(self.device)]
                    with torch.no_grad():
                        self.model(input_list)
                    logger.info(f"{model_name} warmup complete (PyTorch tensor input)")
                    warmup_success = True
                else:
                    logger.warning(f"Could not get valid sample tensor for {model_name} warmup")
            
            # Fallback to process_frame if other methods failed
            if not warmup_success:
                logger.warning(f"Standard warmup failed for {model_name}, trying process_frame fallback")
                try:
                    self.process_frame(dummy_frame_np)
                    logger.info(f"{model_name} warmup complete (process_frame fallback)")
                except Exception as pf_err:
                    logger.warning(f"Warmup fallback process_frame failed: {pf_err}")
                    
        except Exception as e:
            logger.warning(f"Warmup failed for {model_name}: {e}", exc_info=True)
    
    def _process_ultralytics_results(self, results, model_name: str) -> DetectionResult:
        """Common processing logic for Ultralytics models (YOLO, RT-DETR)."""
        boxes_xywh, track_ids, confidences = [], [], []
        
        if results and results[0].boxes:
            res = results[0].boxes
            if hasattr(res, 'xywh') and res.xywh is not None:
                boxes_xywh = res.xywh.cpu().numpy().tolist()
            if hasattr(res, 'conf') and res.conf is not None:
                confidences = res.conf.cpu().numpy().tolist()
            else:
                confidences = [self.confidence_threshold] * len(boxes_xywh)
            track_ids = [self.placeholder_track_id] * len(boxes_xywh)
            
            # Ensure all lists have the same length
            min_len = min(len(boxes_xywh), len(track_ids), len(confidences))
            boxes_xywh = boxes_xywh[:min_len]
            track_ids = track_ids[:min_len]
            confidences = confidences[:min_len]
        
        return boxes_xywh, track_ids, confidences


class YoloStrategy(DetectionTrackingStrategy):
    """Detection using YOLO models via Ultralytics."""
    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        super().__init__(model_path, device, config)
        if YOLO is None: raise ImportError("Ultralytics YOLO required.")
        logger.info(f"Initializing YOLO: {model_path} on {device}")
        try: self.model = YOLO(model_path); self._warmup(); logger.info("YOLO loaded.")
        except Exception as e: logger.error(f"YOLO load failed: {e}"); raise

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        try:
            results = self.model.predict(frame, classes=[self.person_class_id], 
                                       conf=self.confidence_threshold, device=self.device, verbose=False)
            return self._process_ultralytics_results(results, "YOLO")
        except Exception as e:
            logger.error(f"YOLO process error: {e}", exc_info=True)
            return [], [], []

    # get_sample_input_tensor uses generic fallback which is suitable


class RTDetrStrategy(DetectionTrackingStrategy):
    """Detection using RT-DETR models via Ultralytics."""
    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        super().__init__(model_path, device, config)
        if RTDETR is None: raise ImportError("Ultralytics RTDETR required.")
        logger.info(f"Initializing RT-DETR: {model_path} on {device}")
        try: self.model = RTDETR(model_path); self._warmup(); logger.info("RT-DETR loaded.")
        except Exception as e: logger.error(f"RT-DETR load failed: {e}"); raise

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        try:
            results = self.model.predict(frame, classes=[self.person_class_id], 
                                       conf=self.confidence_threshold, device=self.device, verbose=False)
            return self._process_ultralytics_results(results, "RT-DETR")
        except Exception as e:
            logger.error(f"RT-DETR process error: {e}", exc_info=True)
            return [], [], []

    # get_sample_input_tensor uses generic fallback


class FasterRCNNStrategy(DetectionTrackingStrategy):
    """Detection using Faster R-CNN (ResNet50 FPN) from TorchVision."""
    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        super().__init__(model_path, device, config)
        if FasterRCNN_ResNet50_FPN_Weights is None: raise ImportError("torchvision FasterRCNN weights required.")
        logger.info(f"Initializing Faster R-CNN (TorchVision) on {device}")
        try:
            weights_enum_name = config.get("weights_path", "DEFAULT") # Allow specifying weights like V2
            try: # Try to get specific weights
                 weights = getattr(FasterRCNN_ResNet50_FPN_Weights, weights_enum_name)
                 logger.info(f"Using FasterRCNN weights: {weights_enum_name}")
            except AttributeError:
                 logger.warning(f"Weights '{weights_enum_name}' not found, using DEFAULT.")
                 weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            self.model.to(self.device); self.model.eval()
            self.transforms = weights.transforms() # Store transforms
            self._warmup(); logger.info("Faster R-CNN loaded.")
        except Exception as e: logger.error(f"Faster R-CNN load failed: {e}"); raise

    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        boxes_xywh, track_ids, confidences = [], [], []
        if self.model is None or self.transforms is None: logger.error("F-RCNN model/transforms not init."); return [], [], []
        try:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = self.transforms(img_pil) # Gets CHW tensor
            input_batch = [input_tensor.to(self.device)] # List of 3D tensors
            predictions = self.model(input_batch)

            if predictions and isinstance(predictions, list) and isinstance(predictions[0], dict):
                p = predictions[0]
                boxes, labels, scores = p["boxes"].cpu().numpy(), p["labels"].cpu().numpy(), p["scores"].cpu().numpy()
                for box, label, score in zip(boxes, labels, scores):
                    if label == self.person_class_id and score >= self.confidence_threshold:
                        x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
                        if w > 0 and h > 0:
                           cx, cy = x1 + w / 2, y1 + h / 2
                           boxes_xywh.append([cx, cy, w, h]); track_ids.append(self.placeholder_track_id); confidences.append(float(score))
            else: logger.warning(f"Unexpected F-RCNN prediction format: {type(predictions)}")
        except Exception as e: logger.error(f"Faster R-CNN process error: {e}", exc_info=True); return [], [], []
        return boxes_xywh, track_ids, confidences

    def get_sample_input_tensor(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Specific implementation for FasterRCNN using its transforms."""
        if self.transforms is None: logger.error("FasterRCNN transforms not available."); return None
        try:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = self.transforms(img_pil) # Should return CHW
            return input_tensor
        except Exception as e: logger.error(f"FasterRCNN get_sample_input_tensor failed: {e}"); return None


class RfDetrStrategy(DetectionTrackingStrategy):
    """Detection using RF-DETR model from the rfdetr library."""
    def __init__(self, model_path: str, device: torch.device, config: Dict[str, Any]):
        super().__init__(model_path, device, config)
        if RFDETRLarge is None: raise ImportError("RFDETRLarge required.")
        logger.info(f"Initializing RF-DETR on {device}")
        try:
            self.model = RFDETRLarge() # Does it take device? Check docs. Assume internal handling or CPU default.
            # If it's a torch module and needs device:
            # if hasattr(self.model, 'to') and isinstance(self.model, torch.nn.Module):
            #      self.model.to(self.device); self.model.eval()
            self.person_label_index = self.person_class_id # Use configured ID
            self._warmup(); logger.info("RF-DETR initialized.")
        except Exception as e: logger.error(f"RF-DETR init failed: {e}"); raise

    @torch.no_grad() # Add if predict uses torch internally
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        boxes_xywh, track_ids, confidences = [], [], []
        if not self.model: logger.error("RF-DETR model not init."); return [], [], []
        try:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = self.model.predict(img_pil, threshold=self.confidence_threshold) # Assuming predict takes PIL

            if detections and hasattr(detections, 'xyxy') and hasattr(detections, 'class_id') and hasattr(detections, 'confidence'):
                boxes, labels, scores = detections.xyxy, detections.class_id, detections.confidence
                if isinstance(boxes, torch.Tensor): boxes = boxes.cpu().numpy()
                if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
                if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()
                for box, label, score in zip(boxes, labels, scores):
                    if label == self.person_label_index and score >= self.confidence_threshold:
                        x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
                        if w > 0 and h > 0:
                            cx, cy = x1 + w / 2, y1 + h / 2
                            boxes_xywh.append([cx, cy, w, h]); track_ids.append(self.placeholder_track_id); confidences.append(float(score))
            else: logger.warning(f"Unexpected RF-DETR detection obj: {type(detections)}")
        except Exception as e: logger.error(f"RF-DETR process error: {e}", exc_info=True); return [], [], []
        return boxes_xywh, track_ids, confidences

    def get_sample_input_tensor(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """RFDETR predict likely takes PIL, so tensor might not be primary input type."""
        logger.warning(f"get_sample_input_tensor for RfDetrStrategy returning generic tensor. Predict uses PIL.")
        # Return generic tensor for potential PyFunc signature, but note primary input is PIL
        return super().get_sample_input_tensor(frame)


def get_strategy(model_config: Dict[str, Any], device: torch.device) -> DetectionTrackingStrategy:
    """Factory function to instantiate the correct detection strategy."""
    model_type = model_config.get("type", "").lower()
    model_path = model_config.get("weights_path", "") # May be name, path, or ignored
    strategy_map = { "yolo": YoloStrategy, "rtdetr": RTDetrStrategy, "fasterrcnn": FasterRCNNStrategy, "rfdetr": RfDetrStrategy }
    strategy_class = strategy_map.get(model_type)
    if strategy_class: return strategy_class(model_path, device, model_config)
    else: raise ValueError(f"Unsupported model_type: {model_type}. Choose from {list(strategy_map.keys())}")
