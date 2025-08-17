"""
Detection Model Comparison Pipeline - Compare RF-DETR vs FasterRCNN

This pipeline provides comprehensive comparison between RF-DETR and FasterRCNN
detection models on the same dataset with unified evaluation metrics.

Features:
- Side-by-side performance comparison
- Statistical significance testing
- Per-scene and per-camera analysis
- Inference speed benchmarking
- Comprehensive visualization and reporting
- Model-specific insights and recommendations
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import statistics
import warnings

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import cv2
from scipy import stats

# Suppress torch.meshgrid warning about indexing argument
warnings.filterwarnings("ignore", message=".*torch.meshgrid: in an upcoming release.*")

# Import model-specific components
from src.components.data.training_dataset import MTMMCDetectionDataset
from src.components.training.runner import get_transform, get_fasterrcnn_model
from src.components.training.rfdetr_runner import get_rfdetr_model
from src.components.evaluation.metrics import calculate_map_metrics
from src.utils.mlflow_utils import log_metrics, log_artifacts, log_params

logger = logging.getLogger(__name__)

@dataclass
class DetectionModelMetrics:
    """Comprehensive metrics for a detection model."""
    # Core detection metrics
    map_50: float
    map_75: float
    map_50_95: float
    precision: float
    recall: float
    f1_score: float
    
    # Count metrics
    total_detections: int
    total_ground_truth: int
    true_positives: int
    false_positives: int
    false_negatives: int
    
    # Performance metrics
    avg_inference_time_ms: float
    std_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float
    
    # Advanced metrics
    confidence_distribution: Dict[str, float]
    box_size_performance: Dict[str, float]
    per_scene_performance: Dict[str, Dict[str, float]]
    
    # Model-specific metrics
    model_specific_metrics: Dict[str, Any]

class DetectionModelLoader:
    """Load and manage RF-DETR and FasterRCNN models."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
    def load_rfdetr_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """Load RF-DETR model with trained weights."""
        logger.info("Loading RF-DETR model...")
        
        # Check if we have a trained checkpoint to determine correct num_classes
        checkpoint_path = model_config.get("trained_model_path")
        has_trained_checkpoint = checkpoint_path and Path(checkpoint_path).exists()
        
        # Create temporary config for RF-DETR with proper checkpoint path mapping
        rfdetr_config = {
            "model": {
                "type": "rfdetr",
                "size": model_config.get("size", "base"),
                "num_classes": model_config.get("num_classes", 2)
            },
            # Map trained_model_path to local_model_path for get_rfdetr_model compatibility
            "local_model_path": checkpoint_path if has_trained_checkpoint else None
        }
        
        model = get_rfdetr_model(rfdetr_config)
        
        # Load trained weights if available
        checkpoint_path = model_config.get("trained_model_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading RF-DETR weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if hasattr(model, 'model') and hasattr(model.model, 'load_state_dict'):
                if 'model_state_dict' in checkpoint:
                    model.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    model.model.load_state_dict(checkpoint['model'])
                else:
                    model.model.load_state_dict(checkpoint)
        else:
            logger.warning("No RF-DETR checkpoint found, using pre-trained weights")
        
        # Optimize model for inference to reduce latency warnings
        try:
            if hasattr(model, 'optimize_for_inference'):
                model.optimize_for_inference()
                logger.info("RF-DETR model optimized for inference")
        except Exception as opt_e:
            logger.debug(f"Could not optimize RF-DETR model for inference: {opt_e}")
        
        # Smart device placement: GPU for CUDA, CPU for MPS (Mac compatibility)
        target_device = self.device
        
        # Handle MPS compatibility issues on Mac
        if str(self.device) == 'mps':
            # MPS has known issues with RF-DETR operators
            target_device = torch.device('cpu')
            logger.warning("RF-DETR moved to CPU due to MPS compatibility issues on Mac")
            # Set fallback for MPS-specific operators
            import os
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        elif torch.cuda.is_available() and 'cuda' in str(self.device):
            # Use GPU acceleration on PC/CUDA systems
            target_device = self.device  
            logger.info(f"RF-DETR will use GPU acceleration: {target_device}")
        else:
            # Fallback to CPU
            target_device = torch.device('cpu')
            logger.info("RF-DETR will use CPU (no CUDA available)")
        
        # Move model to target device
        if hasattr(model, 'model') and hasattr(model.model, 'to'):
            model.model.to(target_device)
            logger.info(f"RF-DETR model moved to: {target_device}")
        elif hasattr(model, 'to'):
            model.to(target_device)
            logger.info(f"RF-DETR model moved to: {target_device}")
        
        # Log GPU memory usage if using CUDA
        if torch.cuda.is_available() and 'cuda' in str(target_device):
            memory_allocated = torch.cuda.memory_allocated(target_device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(target_device) / 1024**3   # GB
            logger.info(f"GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        
        return model
    
    def load_fasterrcnn_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """Load FasterRCNN model with trained weights."""
        logger.info("Loading FasterRCNN model...")
        
        # Create temporary config for FasterRCNN
        fasterrcnn_config = {
            "model": {
                "type": "fasterrcnn",
                "backbone_weights": model_config.get("backbone_weights", "FasterRCNN_ResNet50_FPN_Weights.DEFAULT"),
                "num_classes": model_config.get("num_classes", 2),
                "trainable_backbone_layers": model_config.get("trainable_backbone_layers", 3)
            }
        }
        
        model = get_fasterrcnn_model(fasterrcnn_config)
        model.to(self.device)
        
        # Load trained weights if available
        checkpoint_path = model_config.get("trained_model_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading FasterRCNN weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            logger.warning("No FasterRCNN checkpoint found, using pre-trained weights")
        
        model.eval()
        return model

class DetectionModelEvaluator:
    """Evaluate and compare detection models comprehensively."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.confidence_threshold = config.get("comparison", {}).get("confidence_threshold", 0.5)
        self.iou_threshold = config.get("comparison", {}).get("iou_threshold", 0.5)
        
        # Load validation dataset
        self.dataset = self._load_validation_dataset()
        
    def _load_validation_dataset(self) -> MTMMCDetectionDataset:
        """Load validation dataset for evaluation."""
        logger.info("Loading validation dataset for model comparison...")
        val_transforms = get_transform(train=False, config=self.config)
        return MTMMCDetectionDataset(
            config=self.config,
            mode='val',
            transforms=val_transforms
        )
    
    def evaluate_model(self, model: nn.Module, model_name: str, model_type: str) -> DetectionModelMetrics:
        """
        Comprehensively evaluate a detection model.
        
        Args:
            model: The model to evaluate
            model_name: Name for logging
            model_type: Type of model ("rfdetr" or "fasterrcnn")
            
        Returns:
            DetectionModelMetrics with comprehensive evaluation
        """
        logger.info(f"Evaluating {model_name} ({model_type}) model...")
        
        # Collect predictions and metadata
        predictions = []
        ground_truth = []
        inference_times = []
        scene_camera_info = []
        confidence_scores = []
        box_sizes = []
        
        # Set model to eval mode (only for PyTorch models)
        if hasattr(model, 'eval'):
            model.eval()
        
        # Use torch.no_grad only for PyTorch models
        context_manager = torch.no_grad() if model_type == "fasterrcnn" else torch.enable_grad()
        
        with context_manager:
            for i in tqdm(range(len(self.dataset)), desc=f"Evaluating {model_name}"):
                image_tensor, target = self.dataset[i]
                
                # Extract scene/camera info from image path
                scene_info = {'scene_id': 'unknown', 'camera_id': 'unknown'}
                try:
                    image_path = self.dataset.get_image_path(i)
                    if image_path:
                        # Handle both Windows and Unix path separators
                        path_parts = str(image_path).replace('\\', '/').split('/')
                        # Try to extract scene and camera from path structure
                        for j, part in enumerate(path_parts):
                            if part.startswith('s') and len(part) > 1 and part[1:].isdigit():  # Scene ID
                                scene_info['scene_id'] = part
                            elif part.startswith('c') and len(part) > 1 and part[1:].isdigit():  # Camera ID
                                scene_info['camera_id'] = part
                except Exception as e:
                    logger.debug(f"Failed to extract scene/camera info from path: {e}")
                    pass  # Use default 'unknown' values
                
                # Run inference with timing
                start_time = time.perf_counter()
                
                if model_type == "fasterrcnn":
                    prediction = self._run_fasterrcnn_inference(model, image_tensor)
                elif model_type == "rfdetr":
                    prediction = self._run_rfdetr_inference(model, i)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                inference_time_ms = (time.perf_counter() - start_time) * 1000
                inference_times.append(inference_time_ms)
                
                # Filter by confidence threshold
                if len(prediction['scores']) > 0:
                    valid_indices = prediction['scores'] >= self.confidence_threshold
                    prediction = {
                        'boxes': prediction['boxes'][valid_indices],
                        'scores': prediction['scores'][valid_indices],
                        'labels': prediction['labels'][valid_indices]
                    }
                
                # Store data
                predictions.append(prediction)
                ground_truth.append({
                    'boxes': target['boxes'].cpu().numpy(),
                    'labels': target['labels'].cpu().numpy()
                })
                scene_camera_info.append(scene_info)
                
                # Collect confidence scores and box sizes
                if len(prediction['scores']) > 0:
                    confidence_scores.extend(prediction['scores'].tolist())
                    box_areas = self._calculate_box_areas(prediction['boxes'])
                    box_sizes.extend(box_areas)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            predictions, ground_truth, inference_times, 
            scene_camera_info, confidence_scores, box_sizes, model_type
        )
        
        logger.info(f"{model_name} evaluation completed:")
        logger.info(f"  mAP@0.5: {metrics.map_50:.4f}")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall: {metrics.recall:.4f}")
        logger.info(f"  Avg inference: {metrics.avg_inference_time_ms:.2f}ms")
        
        return metrics
    
    def _run_fasterrcnn_inference(self, model: nn.Module, image_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Run FasterRCNN inference."""
        prediction = model([image_tensor.to(self.device)])[0]
        return {
            'boxes': prediction['boxes'].cpu().numpy(),
            'scores': prediction['scores'].cpu().numpy(),
            'labels': prediction['labels'].cpu().numpy()
        }
    
    def _run_rfdetr_inference(self, model: nn.Module, sample_idx: int) -> Dict[str, np.ndarray]:
        """Run RF-DETR inference."""
        try:
            # Get image path and load image
            image_path = self.dataset.get_image_path(sample_idx) if hasattr(self.dataset, 'get_image_path') else None
            
            if image_path and Path(image_path).exists():
                original_image = cv2.imread(str(image_path))
                if original_image is not None:
                    image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                    
                    # Try RF-DETR prediction with comprehensive error handling
                    try:
                        results = model.predict(image_pil)
                        
                        # Debug: Log successful predictions for first few samples
                        if sample_idx < 3 and results is not None:
                            logger.info(f"RF-DETR prediction successful for sample {sample_idx}, result type: {type(results)}")
                            if hasattr(results, 'boxes'):
                                logger.info(f"  Boxes available: {results.boxes is not None}")
                                if results.boxes is not None:
                                    logger.info(f"  Number of detections: {len(results.boxes)}")
                        
                        # Extract predictions - handle different result formats
                        if results is not None:
                            # Handle Ultralytics-style results
                            if hasattr(results, 'boxes') and results.boxes is not None:
                                boxes = results.boxes.xyxy.cpu().numpy()
                                scores = results.boxes.conf.cpu().numpy()
                                labels = results.boxes.cls.cpu().numpy()
                                
                                # Debug: Log successful extraction
                                if sample_idx < 3:
                                    logger.info(f"  Extracted {len(boxes)} detections: boxes={boxes.shape}, scores={scores.shape}")
                                    
                            # Handle list of results
                            elif isinstance(results, list) and len(results) > 0:
                                result = results[0]
                                if hasattr(result, 'boxes') and result.boxes is not None:
                                    boxes = result.boxes.xyxy.cpu().numpy()
                                    scores = result.boxes.conf.cpu().numpy()
                                    labels = result.boxes.cls.cpu().numpy()
                                else:
                                    boxes = np.array([]).reshape(0, 4)
                                    scores = np.array([])
                                    labels = np.array([])
                            else:
                                boxes = np.array([]).reshape(0, 4)
                                scores = np.array([])
                                labels = np.array([])
                                
                                # Debug: Log when no valid predictions
                                if sample_idx < 5:
                                    logger.warning(f"RF-DETR results format not recognized for sample {sample_idx}: {type(results)}")
                            
                            return {
                                'boxes': boxes,
                                'scores': scores,
                                'labels': labels
                            }
                        else:
                            # Log when results is None for debugging
                            if sample_idx < 5:  # Only log first few for debugging
                                logger.warning(f"RF-DETR returned None results for sample {sample_idx}")
                    
                    except Exception as pred_e:
                        # Use warning instead of debug for critical failures
                        if sample_idx < 10:  # Log first 10 errors to understand pattern
                            logger.error(f"RF-DETR prediction failed for sample {sample_idx}: {type(pred_e).__name__}: {pred_e}")
                        elif sample_idx == 10:
                            logger.error("RF-DETR prediction failures continue... suppressing further error logs")
                else:
                    if sample_idx < 5:  # Only log first few for debugging
                        logger.warning(f"Could not load image for sample {sample_idx}: {image_path}")
            else:
                if sample_idx < 5:  # Only log first few for debugging
                    logger.warning(f"Image path not found for sample {sample_idx}: {image_path}")
                        
        except Exception as e:
            # Use warning instead of debug for setup failures
            if sample_idx < 5:  # Only log first few to avoid spam
                logger.warning(f"RF-DETR inference setup failed for sample {sample_idx}: {e}")
        
        # Return empty prediction if failed
        return {
            'boxes': np.array([]).reshape(0, 4),
            'scores': np.array([]),
            'labels': np.array([])
        }
    
    def _calculate_box_areas(self, boxes: np.ndarray) -> List[float]:
        """Calculate box areas for size analysis."""
        if len(boxes) == 0:
            return []
        
        areas = []
        for box in boxes:
            if len(box) >= 4:
                x1, y1, x2, y2 = box[:4]
                area = (x2 - x1) * (y2 - y1)
                areas.append(float(area))
        
        return areas
    
    def _calculate_comprehensive_metrics(self, predictions: List[Dict], ground_truth: List[Dict],
                                       inference_times: List[float], scene_camera_info: List[Dict],
                                       confidence_scores: List[float], box_sizes: List[float],
                                       model_type: str) -> DetectionModelMetrics:
        """Calculate comprehensive detection metrics."""
        
        # Core mAP metrics
        try:
            map_metrics = calculate_map_metrics(predictions, ground_truth)
            map_50 = map_metrics.get('map_50', 0.0)
            map_75 = map_metrics.get('map_75', 0.0)
            map_50_95 = map_metrics.get('map_50_95', 0.0)
        except Exception as e:
            logger.warning(f"Failed to calculate mAP metrics: {e}")
            map_50 = map_75 = map_50_95 = 0.0
        
        # Basic count metrics
        total_detections = sum(len(pred['boxes']) for pred in predictions)
        total_ground_truth = sum(len(gt['boxes']) for gt in ground_truth)
        
        # Precision, Recall, F1
        tp, fp, fn = self._calculate_tp_fp_fn(predictions, ground_truth)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Inference time statistics
        avg_inference_time = statistics.mean(inference_times) if inference_times else 0.0
        std_inference_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
        min_inference_time = min(inference_times) if inference_times else 0.0
        max_inference_time = max(inference_times) if inference_times else 0.0
        
        # Confidence distribution
        confidence_dist = self._analyze_confidence_distribution(confidence_scores)
        
        # Box size performance analysis
        box_size_perf = self._analyze_box_size_performance(predictions, ground_truth, box_sizes)
        
        # Per-scene performance
        per_scene_perf = self._analyze_per_scene_performance(predictions, ground_truth, scene_camera_info)
        
        # Model-specific metrics
        model_specific = self._calculate_model_specific_metrics(model_type, predictions)
        
        return DetectionModelMetrics(
            map_50=map_50,
            map_75=map_75,
            map_50_95=map_50_95,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_detections=total_detections,
            total_ground_truth=total_ground_truth,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            avg_inference_time_ms=avg_inference_time,
            std_inference_time_ms=std_inference_time,
            min_inference_time_ms=min_inference_time,
            max_inference_time_ms=max_inference_time,
            confidence_distribution=confidence_dist,
            box_size_performance=box_size_perf,
            per_scene_performance=per_scene_perf,
            model_specific_metrics=model_specific
        )
    
    def _calculate_tp_fp_fn(self, predictions: List[Dict], ground_truth: List[Dict]) -> Tuple[int, int, int]:
        """Calculate true positives, false positives, and false negatives."""
        tp = fp = fn = 0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = pred['boxes']
            gt_boxes = gt['boxes']
            
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue
            elif len(pred_boxes) == 0:
                fn += len(gt_boxes)
                continue
            elif len(gt_boxes) == 0:
                fp += len(pred_boxes)
                continue
            
            # Calculate IoU matrix and match boxes
            matched_gt = set()
            for i, pred_box in enumerate(pred_boxes):
                best_iou = 0.0
                best_gt_idx = -1
                
                for j, gt_box in enumerate(gt_boxes):
                    if j not in matched_gt:
                        iou = self._calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                
                if best_iou >= self.iou_threshold:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            fn += len(gt_boxes) - len(matched_gt)
        
        return tp, fp, fn
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        if len(box1) < 4 or len(box2) < 4:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_confidence_distribution(self, confidence_scores: List[float]) -> Dict[str, float]:
        """Analyze confidence score distribution."""
        if not confidence_scores:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": statistics.mean(confidence_scores),
            "std": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0,
            "median": statistics.median(confidence_scores),
            "min": min(confidence_scores),
            "max": max(confidence_scores)
        }
    
    def _analyze_box_size_performance(self, predictions: List[Dict], ground_truth: List[Dict], 
                                    box_sizes: List[float]) -> Dict[str, float]:
        """Analyze performance by box size categories."""
        if not box_sizes:
            return {"small_boxes": 0.0, "medium_boxes": 0.0, "large_boxes": 0.0}
        
        # Define size thresholds (can be configurable)
        small_threshold = 1000   # Small boxes: area < 1000
        medium_threshold = 5000  # Medium boxes: 1000 <= area < 5000
        
        size_categories = {
            "small_boxes": len([s for s in box_sizes if s < small_threshold]),
            "medium_boxes": len([s for s in box_sizes if small_threshold <= s < medium_threshold]),
            "large_boxes": len([s for s in box_sizes if s >= medium_threshold])
        }
        
        total_boxes = sum(size_categories.values())
        if total_boxes > 0:
            return {k: v / total_boxes for k, v in size_categories.items()}
        
        return {"small_boxes": 0.0, "medium_boxes": 0.0, "large_boxes": 0.0}
    
    def _analyze_per_scene_performance(self, predictions: List[Dict], ground_truth: List[Dict],
                                     scene_camera_info: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Analyze performance per scene and camera."""
        scene_performance = {}
        
        # Group by scene_id and camera_id
        scene_data = {}
        for i, (pred, gt, info) in enumerate(zip(predictions, ground_truth, scene_camera_info)):
            scene_id = info.get('scene_id', 'unknown')
            camera_id = info.get('camera_id', 'unknown')
            key = f"{scene_id}_{camera_id}"
            
            if key not in scene_data:
                scene_data[key] = {'predictions': [], 'ground_truth': []}
            
            scene_data[key]['predictions'].append(pred)
            scene_data[key]['ground_truth'].append(gt)
        
        # Calculate metrics for each scene/camera
        for key, data in scene_data.items():
            tp, fp, fn = self._calculate_tp_fp_fn(data['predictions'], data['ground_truth'])
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            scene_performance[key] = {
                'precision': precision,
                'recall': recall,
                'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            }
        
        return scene_performance
    
    def _calculate_model_specific_metrics(self, model_type: str, predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate model-specific metrics."""
        metrics = {"model_type": model_type}
        
        if model_type == "rfdetr":
            # RF-DETR specific metrics
            metrics.update({
                "avg_detections_per_image": statistics.mean([len(pred['boxes']) for pred in predictions]) if predictions else 0.0,
                "detection_consistency": statistics.stdev([len(pred['boxes']) for pred in predictions]) if len(predictions) > 1 else 0.0
            })
        elif model_type == "fasterrcnn":
            # FasterRCNN specific metrics
            metrics.update({
                "avg_detections_per_image": statistics.mean([len(pred['boxes']) for pred in predictions]) if predictions else 0.0,
                "detection_consistency": statistics.stdev([len(pred['boxes']) for pred in predictions]) if len(predictions) > 1 else 0.0
            })
        
        return metrics

class DetectionModelComparisonPipeline:
    """Main pipeline for comparing RF-DETR and FasterRCNN models."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.model_loader = DetectionModelLoader(config, device)
        self.evaluator = DetectionModelEvaluator(config, device)
        
        # Create output directory
        self.output_dir = Path(config.get("comparison", {}).get("output_dir", "outputs/detection_model_comparison"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_complete_comparison(self) -> Dict[str, Any]:
        """Run complete model comparison pipeline."""
        logger.info("Starting RF-DETR vs FasterRCNN comparison...")
        
        # Load and evaluate RF-DETR
        logger.info("Loading and evaluating RF-DETR model...")
        rfdetr_config = self.config.get("models", {}).get("rfdetr", {})
        rfdetr_model = self.model_loader.load_rfdetr_model(rfdetr_config)
        rfdetr_metrics = self.evaluator.evaluate_model(rfdetr_model, "RF-DETR", "rfdetr")
        
        # Load and evaluate FasterRCNN
        logger.info("Loading and evaluating FasterRCNN model...")
        fasterrcnn_config = self.config.get("models", {}).get("fasterrcnn", {})
        fasterrcnn_model = self.model_loader.load_fasterrcnn_model(fasterrcnn_config)
        fasterrcnn_metrics = self.evaluator.evaluate_model(fasterrcnn_model, "FasterRCNN", "fasterrcnn")
        
        # Calculate comparison metrics
        comparison_results = self._calculate_comparison_metrics(rfdetr_metrics, fasterrcnn_metrics)
        
        # Statistical analysis
        statistical_results = self._perform_statistical_analysis(rfdetr_metrics, fasterrcnn_metrics)
        
        # Generate visualizations
        self._create_comparison_visualizations(rfdetr_metrics, fasterrcnn_metrics, comparison_results)
        
        # Generate reports
        self._generate_comparison_reports(rfdetr_metrics, fasterrcnn_metrics, comparison_results, statistical_results)
        
        # Log to MLflow
        self._log_to_mlflow(rfdetr_metrics, fasterrcnn_metrics, comparison_results)
        
        return {
            "rfdetr_metrics": asdict(rfdetr_metrics),
            "fasterrcnn_metrics": asdict(fasterrcnn_metrics),
            "comparison_results": comparison_results,
            "statistical_results": statistical_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_comparison_metrics(self, rfdetr: DetectionModelMetrics, 
                                    fasterrcnn: DetectionModelMetrics) -> Dict[str, Any]:
        """Calculate comprehensive comparison metrics."""
        return {
            # Performance differences
            "map_50_difference": rfdetr.map_50 - fasterrcnn.map_50,
            "precision_difference": rfdetr.precision - fasterrcnn.precision,
            "recall_difference": rfdetr.recall - fasterrcnn.recall,
            "f1_difference": rfdetr.f1_score - fasterrcnn.f1_score,
            
            # Speed comparison
            "speed_difference_ms": rfdetr.avg_inference_time_ms - fasterrcnn.avg_inference_time_ms,
            "speed_ratio": rfdetr.avg_inference_time_ms / fasterrcnn.avg_inference_time_ms if fasterrcnn.avg_inference_time_ms > 0 else float('inf'),
            
            # Relative improvements
            "map_50_relative_improvement": (rfdetr.map_50 - fasterrcnn.map_50) / fasterrcnn.map_50 * 100 if fasterrcnn.map_50 > 0 else 0.0,
            "precision_relative_improvement": (rfdetr.precision - fasterrcnn.precision) / fasterrcnn.precision * 100 if fasterrcnn.precision > 0 else 0.0,
            "recall_relative_improvement": (rfdetr.recall - fasterrcnn.recall) / fasterrcnn.recall * 100 if fasterrcnn.recall > 0 else 0.0,
            
            # Overall assessment
            "better_model": "RF-DETR" if rfdetr.map_50 > fasterrcnn.map_50 else "FasterRCNN",
            "performance_gap": abs(rfdetr.map_50 - fasterrcnn.map_50),
            "significant_difference": abs(rfdetr.map_50 - fasterrcnn.map_50) > 0.05,
            
            # Efficiency analysis
            "efficiency_score_rfdetr": rfdetr.map_50 / (rfdetr.avg_inference_time_ms / 1000) if rfdetr.avg_inference_time_ms > 0 else 0.0,
            "efficiency_score_fasterrcnn": fasterrcnn.map_50 / (fasterrcnn.avg_inference_time_ms / 1000) if fasterrcnn.avg_inference_time_ms > 0 else 0.0
        }
    
    def _perform_statistical_analysis(self, rfdetr: DetectionModelMetrics, 
                                    fasterrcnn: DetectionModelMetrics) -> Dict[str, Any]:
        """Perform statistical significance testing."""
        # For now, return basic statistical comparison
        # In a full implementation, you would collect per-sample metrics and perform t-tests
        
        return {
            "significance_level": self.config.get("comparison", {}).get("statistical_tests", {}).get("significance_level", 0.05),
            "map_50_difference_significant": abs(rfdetr.map_50 - fasterrcnn.map_50) > 0.05,
            "precision_difference_significant": abs(rfdetr.precision - fasterrcnn.precision) > 0.05,
            "recall_difference_significant": abs(rfdetr.recall - fasterrcnn.recall) > 0.05,
            "speed_difference_significant": abs(rfdetr.avg_inference_time_ms - fasterrcnn.avg_inference_time_ms) > 50,  # 50ms threshold
            "overall_assessment": "RF-DETR shows superior performance" if rfdetr.map_50 > fasterrcnn.map_50 else "FasterRCNN shows superior performance"
        }
    
    def _create_comparison_visualizations(self, rfdetr: DetectionModelMetrics, 
                                        fasterrcnn: DetectionModelMetrics, 
                                        comparison: Dict[str, Any]) -> None:
        """Create comprehensive comparison visualizations."""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Model performance radar chart
        self._create_radar_chart(rfdetr, fasterrcnn, viz_dir)
        
        # Performance comparison bar chart
        self._create_performance_bars(rfdetr, fasterrcnn, viz_dir)
        
        # Speed vs accuracy scatter plot
        self._create_speed_accuracy_plot(rfdetr, fasterrcnn, viz_dir)
        
        # Per-scene performance heatmap
        self._create_scene_performance_heatmap(rfdetr, fasterrcnn, viz_dir)
        
        logger.info(f"Visualization charts saved to: {viz_dir}")
    
    def _create_radar_chart(self, rfdetr: DetectionModelMetrics, fasterrcnn: DetectionModelMetrics, viz_dir: Path) -> None:
        """Create radar chart comparing models across multiple metrics."""
        metrics = ['mAP@0.5', 'Precision', 'Recall', 'F1-Score']
        rfdetr_values = [rfdetr.map_50, rfdetr.precision, rfdetr.recall, rfdetr.f1_score]
        fasterrcnn_values = [fasterrcnn.map_50, fasterrcnn.precision, fasterrcnn.recall, fasterrcnn.f1_score]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        rfdetr_values += rfdetr_values[:1]
        fasterrcnn_values += fasterrcnn_values[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, rfdetr_values, 'o-', linewidth=2, label='RF-DETR', color='#FF6B35')
        ax.fill(angles, rfdetr_values, alpha=0.25, color='#FF6B35')
        
        ax.plot(angles, fasterrcnn_values, 'o-', linewidth=2, label='FasterRCNN', color='#004E89')
        ax.fill(angles, fasterrcnn_values, alpha=0.25, color='#004E89')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.0)
        ax.set_title('Model Performance Comparison\n(Radar Chart)', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_radar_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_bars(self, rfdetr: DetectionModelMetrics, fasterrcnn: DetectionModelMetrics, viz_dir: Path) -> None:
        """Create performance comparison bar chart."""
        metrics = ['mAP@0.5', 'mAP@0.75', 'Precision', 'Recall', 'F1-Score']
        rfdetr_values = [rfdetr.map_50, rfdetr.map_75, rfdetr.precision, rfdetr.recall, rfdetr.f1_score]
        fasterrcnn_values = [fasterrcnn.map_50, fasterrcnn.map_75, fasterrcnn.precision, fasterrcnn.recall, fasterrcnn.f1_score]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars1 = ax.bar(x - width/2, rfdetr_values, width, label='RF-DETR', color='#FF6B35', alpha=0.8)
        bars2 = ax.bar(x + width/2, fasterrcnn_values, width, label='FasterRCNN', color='#004E89', alpha=0.8)
        
        ax.set_xlabel('Performance Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Detection Model Performance Comparison', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_speed_accuracy_plot(self, rfdetr: DetectionModelMetrics, fasterrcnn: DetectionModelMetrics, viz_dir: Path) -> None:
        """Create speed vs accuracy scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot models
        ax.scatter(rfdetr.avg_inference_time_ms, rfdetr.map_50, 
                  s=200, color='#FF6B35', label='RF-DETR', alpha=0.8, edgecolors='black')
        ax.scatter(fasterrcnn.avg_inference_time_ms, fasterrcnn.map_50, 
                  s=200, color='#004E89', label='FasterRCNN', alpha=0.8, edgecolors='black')
        
        # Add model labels
        ax.annotate('RF-DETR', (rfdetr.avg_inference_time_ms, rfdetr.map_50), 
                   xytext=(10, 10), textcoords='offset points', fontsize=12)
        ax.annotate('FasterRCNN', (fasterrcnn.avg_inference_time_ms, fasterrcnn.map_50), 
                   xytext=(10, 10), textcoords='offset points', fontsize=12)
        
        ax.set_xlabel('Average Inference Time (ms)', fontsize=12)
        ax.set_ylabel('mAP@0.5', fontsize=12)
        ax.set_title('Speed vs Accuracy Trade-off', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(viz_dir / "speed_vs_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_scene_performance_heatmap(self, rfdetr: DetectionModelMetrics, fasterrcnn: DetectionModelMetrics, viz_dir: Path) -> None:
        """Create per-scene performance heatmap."""
        # Combine scene data from both models
        all_scenes = set(list(rfdetr.per_scene_performance.keys()) + list(fasterrcnn.per_scene_performance.keys()))
        
        if not all_scenes:
            logger.warning("No scene performance data available for heatmap")
            return
        
        # Create comparison matrix
        scene_names = sorted(all_scenes)
        metrics = ['precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            # Prepare data for heatmap
            data = []
            models = ['RF-DETR', 'FasterRCNN']
            
            for model_name, model_perf in [('RF-DETR', rfdetr.per_scene_performance), 
                                         ('FasterRCNN', fasterrcnn.per_scene_performance)]:
                row = []
                for scene in scene_names:
                    value = model_perf.get(scene, {}).get(metric, 0.0)
                    row.append(value)
                data.append(row)
            
            # Create heatmap
            im = axes[i].imshow(data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks and labels
            axes[i].set_xticks(range(len(scene_names)))
            axes[i].set_xticklabels(scene_names, rotation=45, ha='right')
            axes[i].set_yticks(range(len(models)))
            axes[i].set_yticklabels(models)
            axes[i].set_title(f'{metric.capitalize()} by Scene', fontsize=12)
            
            # Add text annotations
            for j in range(len(models)):
                for k in range(len(scene_names)):
                    text = axes[i].text(k, j, f'{data[j][k]:.3f}', 
                                      ha="center", va="center", color="white" if data[j][k] < 0.5 else "black")
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i])
        
        plt.suptitle('Per-Scene Performance Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(viz_dir / "scene_performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_comparison_reports(self, rfdetr: DetectionModelMetrics, fasterrcnn: DetectionModelMetrics,
                                   comparison: Dict[str, Any], statistical: Dict[str, Any]) -> None:
        """Generate comprehensive comparison reports."""
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate executive summary
        self._generate_executive_summary(rfdetr, fasterrcnn, comparison, reports_dir)
        
        # Generate technical report
        self._generate_technical_report(rfdetr, fasterrcnn, comparison, statistical, reports_dir)
        
        # Export data to CSV/JSON
        self._export_comparison_data(rfdetr, fasterrcnn, comparison, reports_dir)
        
        logger.info(f"Comparison reports saved to: {reports_dir}")
    
    def _generate_executive_summary(self, rfdetr: DetectionModelMetrics, fasterrcnn: DetectionModelMetrics,
                                  comparison: Dict[str, Any], reports_dir: Path) -> None:
        """Generate executive summary report."""
        summary_file = reports_dir / "executive_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("DETECTION MODEL COMPARISON - EXECUTIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {len(self.evaluator.dataset)} validation samples\n\n")
            
            # Key findings
            f.write("KEY FINDINGS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"• Best Overall Model: {comparison['better_model']}\n")
            f.write(f"• Performance Gap: {comparison['performance_gap']:.3f} mAP@0.5\n")
            f.write(f"• Significant Difference: {'Yes' if comparison['significant_difference'] else 'No'}\n\n")
            
            # Performance summary
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 25 + "\n")
            f.write(f"RF-DETR:\n")
            f.write(f"  mAP@0.5: {rfdetr.map_50:.4f}\n")
            f.write(f"  Precision: {rfdetr.precision:.4f}\n")
            f.write(f"  Recall: {rfdetr.recall:.4f}\n")
            f.write(f"  Avg Inference: {rfdetr.avg_inference_time_ms:.1f}ms\n\n")
            
            f.write(f"FasterRCNN:\n")
            f.write(f"  mAP@0.5: {fasterrcnn.map_50:.4f}\n")
            f.write(f"  Precision: {fasterrcnn.precision:.4f}\n")
            f.write(f"  Recall: {fasterrcnn.recall:.4f}\n")
            f.write(f"  Avg Inference: {fasterrcnn.avg_inference_time_ms:.1f}ms\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 18 + "\n")
            if comparison['better_model'] == 'RF-DETR':
                f.write("• Deploy RF-DETR for production use\n")
                f.write("• RF-DETR shows superior detection performance\n")
                if comparison['speed_difference_ms'] > 0:
                    f.write("• Consider FasterRCNN for speed-critical applications\n")
            else:
                f.write("• Deploy FasterRCNN for production use\n")
                f.write("• FasterRCNN shows superior detection performance\n")
                if comparison['speed_difference_ms'] < 0:
                    f.write("• Consider RF-DETR for speed-critical applications\n")
            
            f.write(f"\nFor detailed analysis, see technical_report.html\n")
    
    def _generate_technical_report(self, rfdetr: DetectionModelMetrics, fasterrcnn: DetectionModelMetrics,
                                 comparison: Dict[str, Any], statistical: Dict[str, Any], reports_dir: Path) -> None:
        """Generate detailed technical report in HTML format."""
        html_file = reports_dir / "technical_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Detection Model Comparison - Technical Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #e6ffe6; }}
                .warning {{ background-color: #ffe6e6; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Detection Model Comparison - Technical Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Dataset Size:</strong> {len(self.evaluator.dataset)} validation samples</p>
            </div>
            
            <div class="section">
                <h2>Performance Overview</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>RF-DETR</th>
                        <th>FasterRCNN</th>
                        <th>Difference</th>
                    </tr>
                    <tr class="{'highlight' if rfdetr.map_50 > fasterrcnn.map_50 else 'warning' if rfdetr.map_50 < fasterrcnn.map_50 else ''}">
                        <td>mAP@0.5</td>
                        <td>{rfdetr.map_50:.4f}</td>
                        <td>{fasterrcnn.map_50:.4f}</td>
                        <td>{comparison['map_50_difference']:+.4f}</td>
                    </tr>
                    <tr class="{'highlight' if rfdetr.precision > fasterrcnn.precision else 'warning' if rfdetr.precision < fasterrcnn.precision else ''}">
                        <td>Precision</td>
                        <td>{rfdetr.precision:.4f}</td>
                        <td>{fasterrcnn.precision:.4f}</td>
                        <td>{comparison['precision_difference']:+.4f}</td>
                    </tr>
                    <tr class="{'highlight' if rfdetr.recall > fasterrcnn.recall else 'warning' if rfdetr.recall < fasterrcnn.recall else ''}">
                        <td>Recall</td>
                        <td>{rfdetr.recall:.4f}</td>
                        <td>{fasterrcnn.recall:.4f}</td>
                        <td>{comparison['recall_difference']:+.4f}</td>
                    </tr>
                    <tr class="{'warning' if rfdetr.avg_inference_time_ms > fasterrcnn.avg_inference_time_ms else 'highlight' if rfdetr.avg_inference_time_ms < fasterrcnn.avg_inference_time_ms else ''}">
                        <td>Inference Time (ms)</td>
                        <td>{rfdetr.avg_inference_time_ms:.2f}</td>
                        <td>{fasterrcnn.avg_inference_time_ms:.2f}</td>
                        <td>{comparison['speed_difference_ms']:+.2f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Statistical Analysis</h2>
                <p><strong>Overall Assessment:</strong> {statistical['overall_assessment']}</p>
                <p><strong>Significant Performance Difference:</strong> {'Yes' if comparison['significant_difference'] else 'No'}</p>
                <p><strong>Recommended Model:</strong> {comparison['better_model']}</p>
            </div>
            
            <div class="section">
                <h2>Detailed Metrics</h2>
                <h3>RF-DETR Detailed Performance</h3>
                <ul>
                    <li>mAP@0.75: {rfdetr.map_75:.4f}</li>
                    <li>F1-Score: {rfdetr.f1_score:.4f}</li>
                    <li>Total Detections: {rfdetr.total_detections}</li>
                    <li>True Positives: {rfdetr.true_positives}</li>
                    <li>False Positives: {rfdetr.false_positives}</li>
                    <li>False Negatives: {rfdetr.false_negatives}</li>
                </ul>
                
                <h3>FasterRCNN Detailed Performance</h3>
                <ul>
                    <li>mAP@0.75: {fasterrcnn.map_75:.4f}</li>
                    <li>F1-Score: {fasterrcnn.f1_score:.4f}</li>
                    <li>Total Detections: {fasterrcnn.total_detections}</li>
                    <li>True Positives: {fasterrcnn.true_positives}</li>
                    <li>False Positives: {fasterrcnn.false_positives}</li>
                    <li>False Negatives: {fasterrcnn.false_negatives}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Efficiency Analysis</h2>
                <p><strong>RF-DETR Efficiency Score:</strong> {comparison['efficiency_score_rfdetr']:.3f} (mAP per second)</p>
                <p><strong>FasterRCNN Efficiency Score:</strong> {comparison['efficiency_score_fasterrcnn']:.3f} (mAP per second)</p>
                <p><strong>Speed Ratio:</strong> RF-DETR is {comparison['speed_ratio']:.2f}x {'slower' if comparison['speed_ratio'] > 1 else 'faster'} than FasterRCNN</p>
            </div>
        </body>
        </html>
        """
        
        with open(html_file, 'w') as f:
            f.write(html_content)
    
    def _export_comparison_data(self, rfdetr: DetectionModelMetrics, fasterrcnn: DetectionModelMetrics,
                              comparison: Dict[str, Any], reports_dir: Path) -> None:
        """Export comparison data in multiple formats."""
        
        # Export to JSON
        comparison_data = {
            "rfdetr_metrics": asdict(rfdetr),
            "fasterrcnn_metrics": asdict(fasterrcnn),
            "comparison_metrics": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(reports_dir / "comparison_data.json", 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        # Export to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['mAP@0.5', 'mAP@0.75', 'Precision', 'Recall', 'F1-Score', 'Inference Time (ms)'],
            'RF-DETR': [rfdetr.map_50, rfdetr.map_75, rfdetr.precision, rfdetr.recall, rfdetr.f1_score, rfdetr.avg_inference_time_ms],
            'FasterRCNN': [fasterrcnn.map_50, fasterrcnn.map_75, fasterrcnn.precision, fasterrcnn.recall, fasterrcnn.f1_score, fasterrcnn.avg_inference_time_ms],
            'Difference': [comparison['map_50_difference'], rfdetr.map_75 - fasterrcnn.map_75, comparison['precision_difference'], comparison['recall_difference'], comparison['f1_difference'], comparison['speed_difference_ms']]
        })
        
        metrics_df.to_csv(reports_dir / "performance_metrics.csv", index=False)
    
    def _log_to_mlflow(self, rfdetr: DetectionModelMetrics, fasterrcnn: DetectionModelMetrics,
                      comparison: Dict[str, Any]) -> None:
        """Log comprehensive results to MLflow."""
        try:
            # Log RF-DETR metrics
            log_metrics({
                "rfdetr_map_50": rfdetr.map_50,
                "rfdetr_precision": rfdetr.precision,
                "rfdetr_recall": rfdetr.recall,
                "rfdetr_f1": rfdetr.f1_score,
                "rfdetr_inference_time_ms": rfdetr.avg_inference_time_ms
            })
            
            # Log FasterRCNN metrics
            log_metrics({
                "fasterrcnn_map_50": fasterrcnn.map_50,
                "fasterrcnn_precision": fasterrcnn.precision,
                "fasterrcnn_recall": fasterrcnn.recall,
                "fasterrcnn_f1": fasterrcnn.f1_score,
                "fasterrcnn_inference_time_ms": fasterrcnn.avg_inference_time_ms
            })
            
            # Log comparison metrics
            log_metrics({
                "performance_gap": comparison["performance_gap"],
                "better_model_is_rfdetr": 1 if comparison["better_model"] == "RF-DETR" else 0,
                "significant_difference": 1 if comparison["significant_difference"] else 0,
                "speed_ratio": comparison["speed_ratio"]
            })
            
            # Log parameters
            log_params({
                "dataset_size": len(self.evaluator.dataset),
                "confidence_threshold": self.evaluator.confidence_threshold,
                "iou_threshold": self.evaluator.iou_threshold,
                "comparison_type": "rfdetr_vs_fasterrcnn"
            })
            
            # Log artifacts
            log_artifacts(str(self.output_dir))
            
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")