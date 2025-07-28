"""
Model Comparison Pipeline - Compare Initial vs Trained Detection Models

This pipeline provides a unified interface for comparing detection models
(FasterRCNN or RF-DETR) between their initial pre-trained weights and 
trained weights.

Features:
- Unified model loading for both FasterRCNN and RF-DETR
- Comprehensive performance metrics calculation
- Detailed comparison reports and visualizations
- MLflow integration for experiment tracking
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import cv2

# Import model-specific components
from src.components.data.training_dataset import MTMMCDetectionDataset
from src.components.training.runner import get_transform, get_fasterrcnn_model
from src.components.training.rfdetr_runner import get_rfdetr_model
from src.components.evaluation.metrics import calculate_map_metrics
from src.utils.mlflow_utils import log_metrics, log_artifacts, log_params

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a single model."""
    map_50: float
    map_75: float
    map_50_95: float
    precision: float
    recall: float
    f1_score: float
    total_detections: int
    total_ground_truth: int
    true_positives: int
    false_positives: int
    false_negatives: int
    inference_time_ms: float

class UnifiedModelLoader:
    """Unified model loader for both FasterRCNN and RF-DETR."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.model_type = config.get("model", {}).get("type", "").lower()
        
    def load_initial_model(self) -> nn.Module:
        """Load model with initial (pre-trained) weights."""
        logger.info(f"Loading initial {self.model_type} model...")
        
        if self.model_type == "fasterrcnn":
            model = get_fasterrcnn_model(self.config)
            model.to(self.device)
            model.eval()
            return model
            
        elif self.model_type == "rfdetr":
            model = get_rfdetr_model(self.config)
            # RF-DETR handles device internally
            return model
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def load_trained_model(self, checkpoint_path: str) -> nn.Module:
        """Load model with trained weights from checkpoint."""
        logger.info(f"Loading trained {self.model_type} model from: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if self.model_type == "fasterrcnn":
            model = get_fasterrcnn_model(self.config)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            return model
            
        elif self.model_type == "rfdetr":
            model = get_rfdetr_model(self.config)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different RF-DETR checkpoint formats
            if hasattr(model, 'model') and hasattr(model.model, 'load_state_dict'):
                if 'model_state_dict' in checkpoint:
                    model.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    model.model.load_state_dict(checkpoint['model'])
                else:
                    model.model.load_state_dict(checkpoint)
            
            return model
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

class ModelEvaluator:
    """Evaluates model performance on validation dataset."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.model_type = config.get("model", {}).get("type", "").lower()
        
        # Load validation dataset
        self.dataset = self._load_validation_dataset()
        
    def _load_validation_dataset(self) -> MTMMCDetectionDataset:
        """Load validation dataset for evaluation."""
        logger.info("Loading validation dataset...")
        val_transforms = get_transform(train=False, config=self.config)
        return MTMMCDetectionDataset(
            config=self.config,
            mode='val',
            transforms=val_transforms
        )
    
    def evaluate_model(self, model: nn.Module, model_name: str) -> ModelPerformanceMetrics:
        """
        Evaluate a model on the validation dataset.
        
        Args:
            model: The model to evaluate
            model_name: Name for logging purposes
            
        Returns:
            ModelPerformanceMetrics object with all metrics
        """
        logger.info(f"Evaluating {model_name} model...")
        
        # Collect predictions and ground truth
        predictions = []
        ground_truth = []
        inference_times = []
        
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(self.dataset)), desc=f"Evaluating {model_name}"):
                # Get sample
                image_tensor, target = self.dataset[i]
                
                # Prepare input
                start_time = time.time()
                
                if self.model_type == "fasterrcnn":
                    # FasterRCNN inference
                    prediction = model([image_tensor.to(self.device)])[0]
                    pred_boxes = prediction['boxes'].cpu().numpy()
                    pred_scores = prediction['scores'].cpu().numpy()
                    pred_labels = prediction['labels'].cpu().numpy()
                    
                elif self.model_type == "rfdetr":
                    # RF-DETR inference
                    # Convert tensor to PIL image
                    image_path = self.dataset.get_image_path(i)
                    if image_path:
                        original_image = cv2.imread(str(image_path))
                        if original_image is not None:
                            image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                            results = model.predict(image_pil)
                            
                            # Extract predictions
                            if results and hasattr(results, 'boxes') and results.boxes is not None:
                                pred_boxes = results.boxes.xyxy.cpu().numpy()
                                pred_scores = results.boxes.conf.cpu().numpy()
                                pred_labels = results.boxes.cls.cpu().numpy()
                            else:
                                pred_boxes = np.array([]).reshape(0, 4)
                                pred_scores = np.array([])
                                pred_labels = np.array([])
                        else:
                            pred_boxes = np.array([]).reshape(0, 4)
                            pred_scores = np.array([])
                            pred_labels = np.array([])
                    else:
                        pred_boxes = np.array([]).reshape(0, 4)
                        pred_scores = np.array([])
                        pred_labels = np.array([])
                
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                inference_times.append(inference_time)
                
                # Filter by confidence threshold
                conf_threshold = self.config.get("evaluation", {}).get("confidence_threshold", 0.5)
                if len(pred_scores) > 0:
                    valid_indices = pred_scores >= conf_threshold
                    pred_boxes = pred_boxes[valid_indices]
                    pred_scores = pred_scores[valid_indices]
                    pred_labels = pred_labels[valid_indices]
                
                # Store predictions
                predictions.append({
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'labels': pred_labels
                })
                
                # Store ground truth
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                ground_truth.append({
                    'boxes': gt_boxes,
                    'labels': gt_labels
                })
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, ground_truth, inference_times)
        
        logger.info(f"{model_name} evaluation completed:")
        logger.info(f"  mAP@0.5: {metrics.map_50:.4f}")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall: {metrics.recall:.4f}")
        logger.info(f"  Avg inference time: {metrics.inference_time_ms:.2f}ms")
        
        return metrics
    
    def _calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict], 
                          inference_times: List[float]) -> ModelPerformanceMetrics:
        """Calculate performance metrics from predictions and ground truth."""
        
        # Calculate mAP using torchmetrics
        try:
            map_metrics = calculate_map_metrics(predictions, ground_truth)
            map_50 = map_metrics.get('map_50', 0.0)
            map_75 = map_metrics.get('map_75', 0.0)
            map_50_95 = map_metrics.get('map_50_95', 0.0)
        except Exception as e:
            logger.warning(f"Failed to calculate mAP metrics: {e}")
            map_50 = map_75 = map_50_95 = 0.0
        
        # Calculate basic metrics
        total_detections = sum(len(pred['boxes']) for pred in predictions)
        total_ground_truth = sum(len(gt['boxes']) for gt in ground_truth)
        
        # Calculate precision, recall, F1 using IoU matching
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        iou_threshold = self.config.get("evaluation", {}).get("iou_threshold", 0.5)
        
        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = pred['boxes']
            gt_boxes = gt['boxes']
            
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue
            elif len(pred_boxes) == 0:
                false_negatives += len(gt_boxes)
                continue
            elif len(gt_boxes) == 0:
                false_positives += len(pred_boxes)
                continue
            
            # Calculate IoU between all predictions and ground truth
            ious = self._calculate_iou_matrix(pred_boxes, gt_boxes)
            
            # Match predictions to ground truth
            matched_gt = set()
            for i, pred_box in enumerate(pred_boxes):
                best_iou = 0.0
                best_gt_idx = -1
                
                for j, gt_box in enumerate(gt_boxes):
                    if j not in matched_gt and ious[i, j] > best_iou:
                        best_iou = ious[i, j]
                        best_gt_idx = j
                
                if best_iou >= iou_threshold:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1
            
            # Unmatched ground truth are false negatives
            false_negatives += len(gt_boxes) - len(matched_gt)
        
        # Calculate derived metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        
        return ModelPerformanceMetrics(
            map_50=map_50,
            map_75=map_75,
            map_50_95=map_50_95,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_detections=total_detections,
            total_ground_truth=total_ground_truth,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            inference_time_ms=avg_inference_time
        )
    
    def _calculate_iou_matrix(self, pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU matrix between predicted and ground truth boxes."""
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return np.zeros((len(pred_boxes), len(gt_boxes)))
        
        # Calculate IoU for each pair
        ious = np.zeros((len(pred_boxes), len(gt_boxes)))
        
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                ious[i, j] = self._calculate_iou(pred_box, gt_box)
        
        return ious
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        # Convert to [x1, y1, x2, y2] format if needed
        if len(box1) == 4 and len(box2) == 4:
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
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
        
        return 0.0

class ModelComparisonPipeline:
    """Main pipeline for comparing initial vs trained models."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.model_loader = UnifiedModelLoader(config, device)
        self.evaluator = ModelEvaluator(config, device)
        
        # Create output directory
        self.output_dir = Path(config.get("comparison", {}).get("output_dir", "outputs/model_comparison"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_complete_comparison(self) -> Dict[str, Any]:
        """Run complete model comparison pipeline."""
        logger.info("Starting complete model comparison...")
        
        # Load models
        initial_model = self.model_loader.load_initial_model()
        
        trained_model_path = self.config.get("trained_model_path")
        if trained_model_path and Path(trained_model_path).exists():
            trained_model = self.model_loader.load_trained_model(trained_model_path)
        else:
            logger.warning("No trained model found, using initial model for both evaluations")
            trained_model = initial_model
        
        # Evaluate models
        initial_metrics = self.evaluator.evaluate_model(initial_model, "Initial")
        trained_metrics = self.evaluator.evaluate_model(trained_model, "Trained")
        
        # Calculate improvement metrics
        improvement_metrics = self._calculate_improvements(initial_metrics, trained_metrics)
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(initial_metrics, trained_metrics, improvement_metrics)
        
        # Create visualizations
        self._create_comparison_visualizations(initial_metrics, trained_metrics, improvement_metrics)
        
        # Log to MLflow
        self._log_to_mlflow(initial_metrics, trained_metrics, improvement_metrics)
        
        return {
            "initial_metrics": self._metrics_to_dict(initial_metrics),
            "trained_metrics": self._metrics_to_dict(trained_metrics),
            "improvement_metrics": improvement_metrics,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_improvements(self, initial: ModelPerformanceMetrics, 
                              trained: ModelPerformanceMetrics) -> Dict[str, float]:
        """Calculate improvement metrics."""
        return {
            "map_50_improvement": trained.map_50 - initial.map_50,
            "map_75_improvement": trained.map_75 - initial.map_75,
            "map_50_95_improvement": trained.map_50_95 - initial.map_50_95,
            "precision_improvement": trained.precision - initial.precision,
            "recall_improvement": trained.recall - initial.recall,
            "f1_improvement": trained.f1_score - initial.f1_score,
            "speed_improvement": initial.inference_time_ms - trained.inference_time_ms,  # Negative means slower
            "relative_map_improvement": ((trained.map_50 - initial.map_50) / initial.map_50 * 100) if initial.map_50 > 0 else 0.0
        }
    
    def _generate_comparison_summary(self, initial: ModelPerformanceMetrics, 
                                   trained: ModelPerformanceMetrics, 
                                   improvements: Dict[str, float]) -> Dict[str, Any]:
        """Generate comparison summary."""
        return {
            "model_type": self.config.get("model", {}).get("type", "unknown"),
            "dataset_size": len(self.evaluator.dataset),
            "best_performing_model": "trained" if trained.map_50 > initial.map_50 else "initial",
            "significant_improvement": improvements["map_50_improvement"] > 0.05,
            "speed_comparison": "faster" if improvements["speed_improvement"] > 0 else "slower",
            "overall_assessment": self._assess_overall_performance(improvements)
        }
    
    def _assess_overall_performance(self, improvements: Dict[str, float]) -> str:
        """Assess overall performance improvement."""
        map_improvement = improvements["map_50_improvement"]
        
        if map_improvement > 0.1:
            return "Significant improvement achieved"
        elif map_improvement > 0.05:
            return "Moderate improvement achieved"
        elif map_improvement > 0.01:
            return "Minor improvement achieved"
        elif map_improvement > -0.01:
            return "No significant change"
        else:
            return "Performance degradation detected"
    
    def _create_comparison_visualizations(self, initial: ModelPerformanceMetrics, 
                                        trained: ModelPerformanceMetrics, 
                                        improvements: Dict[str, float]) -> None:
        """Create comparison visualizations."""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics comparison bar chart
        plt.figure(figsize=(12, 8))
        
        metrics_names = ['mAP@0.5', 'mAP@0.75', 'Precision', 'Recall', 'F1-Score']
        initial_values = [initial.map_50, initial.map_75, initial.precision, initial.recall, initial.f1_score]
        trained_values = [trained.map_50, trained.map_75, trained.precision, trained.recall, trained.f1_score]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        plt.bar(x - width/2, initial_values, width, label='Initial Model', alpha=0.8)
        plt.bar(x + width/2, trained_values, width, label='Trained Model', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'Model Performance Comparison ({self.config.get("model", {}).get("type", "Unknown")})')
        plt.xticks(x, metrics_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Improvement visualization
        plt.figure(figsize=(10, 6))
        
        improvement_names = ['mAP@0.5', 'mAP@0.75', 'Precision', 'Recall', 'F1-Score']
        improvement_values = [
            improvements["map_50_improvement"],
            improvements["map_75_improvement"],
            improvements["precision_improvement"],
            improvements["recall_improvement"],
            improvements["f1_improvement"]
        ]
        
        colors = ['green' if x > 0 else 'red' for x in improvement_values]
        
        plt.bar(improvement_names, improvement_values, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Metrics')
        plt.ylabel('Improvement (Trained - Initial)')
        plt.title('Performance Improvement by Metric')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "improvement_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to: {viz_dir}")
    
    def _log_to_mlflow(self, initial: ModelPerformanceMetrics, 
                      trained: ModelPerformanceMetrics, 
                      improvements: Dict[str, float]) -> None:
        """Log results to MLflow."""
        try:
            # Log initial model metrics
            log_metrics({
                "initial_map_50": initial.map_50,
                "initial_map_75": initial.map_75,
                "initial_precision": initial.precision,
                "initial_recall": initial.recall,
                "initial_f1": initial.f1_score,
                "initial_inference_time_ms": initial.inference_time_ms
            })
            
            # Log trained model metrics
            log_metrics({
                "trained_map_50": trained.map_50,
                "trained_map_75": trained.map_75,
                "trained_precision": trained.precision,
                "trained_recall": trained.recall,
                "trained_f1": trained.f1_score,
                "trained_inference_time_ms": trained.inference_time_ms
            })
            
            # Log improvement metrics
            log_metrics(improvements)
            
            # Log parameters
            log_params({
                "model_type": self.config.get("model", {}).get("type", "unknown"),
                "dataset_size": len(self.evaluator.dataset),
                "confidence_threshold": self.config.get("evaluation", {}).get("confidence_threshold", 0.5),
                "iou_threshold": self.config.get("evaluation", {}).get("iou_threshold", 0.5)
            })
            
            # Log artifacts
            log_artifacts(str(self.output_dir))
            
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    def _metrics_to_dict(self, metrics: ModelPerformanceMetrics) -> Dict[str, float]:
        """Convert ModelPerformanceMetrics to dictionary."""
        return {
            "map_50": metrics.map_50,
            "map_75": metrics.map_75,
            "map_50_95": metrics.map_50_95,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "total_detections": metrics.total_detections,
            "total_ground_truth": metrics.total_ground_truth,
            "true_positives": metrics.true_positives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "inference_time_ms": metrics.inference_time_ms
        }