"""
Cross-Model Detection Comparison System for Phase 1.2.

This module implements comprehensive comparison analysis across multiple detection models:
1. Model Performance Matrix generation
2. Ensemble Analysis Opportunities
3. Side-by-side failure comparison visualizations
4. Model-specific failure pattern identification
5. Model recommendation matrix by scenario type
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import cv2
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

from src.components.detection.strategies import get_strategy, DetectionTrackingStrategy
from src.components.analysis.enhanced_detection_analysis import (
    SceneContext, DetectionFailure, SceneAnalyzer, EnhancedDetectionAnalyzer
)

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Single model prediction result."""
    model_name: str
    boxes_xywh: List[List[float]]  # [center_x, center_y, width, height]
    confidences: List[float]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class MultiModelFrameResult:
    """Results from multiple models on a single frame."""
    frame_idx: int
    scene_id: str
    camera_id: str
    ground_truth_boxes: List[List[float]]  # xyxy format
    ground_truth_ids: List[int]
    model_predictions: Dict[str, ModelPrediction]
    scene_context: SceneContext
    timestamp: Optional[str] = None


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a single model."""
    model_name: str
    total_frames: int
    total_gt_objects: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    average_confidence: float
    average_processing_time: float
    success_rate: float  # Percentage of frames processed without error
    

@dataclass
class ModelComparisonMatrix:
    """Comprehensive comparison across all models."""
    models_compared: List[str]
    individual_metrics: Dict[str, ModelPerformanceMetrics]
    pairwise_agreement: Dict[Tuple[str, str], float]  # IoU agreement between model pairs
    scenario_performance: Dict[str, Dict[str, float]]  # Performance by scenario type
    ensemble_opportunities: Dict[str, List[str]]  # Scenarios where models complement each other
    model_recommendations: Dict[str, str]  # Best model per scenario


@dataclass 
class EnsembleOpportunity:
    """Identified ensemble opportunity between models."""
    scenario_type: str
    primary_model: str
    supporting_models: List[str]
    potential_improvement: float  # Estimated performance gain
    complementarity_score: float  # How well models complement each other
    recommended_strategy: str  # e.g., 'voting', 'sequential', 'confidence_based'


class MultiModelDetectionRunner:
    """Runs multiple detection models on the same input for comparison."""
    
    def __init__(self, model_configs: Dict[str, Dict[str, Any]], device: torch.device):
        self.device = device
        self.models: Dict[str, DetectionTrackingStrategy] = {}
        self.model_configs = model_configs
        
        # Load all configured models
        self._load_models()
        
    def _load_models(self):
        """Load all detection models for comparison."""
        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Loading model for comparison: {model_name}")
                strategy = get_strategy(config, self.device)
                self.models[model_name] = strategy
                logger.info(f"Successfully loaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                # Continue loading other models
    
    def process_frame_all_models(self, frame: np.ndarray, frame_idx: int, 
                               scene_id: str, camera_id: str,
                               ground_truth: Optional[Dict[str, torch.Tensor]] = None,
                               scene_context: Optional[SceneContext] = None,
                               timestamp: Optional[str] = None) -> MultiModelFrameResult:
        """Process single frame with all loaded models."""
        model_predictions = {}
        
        for model_name, model_strategy in self.models.items():
            start_time = datetime.now()
            
            try:
                boxes_xywh, track_ids, confidences = model_strategy.process_frame(frame)
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                prediction = ModelPrediction(
                    model_name=model_name,
                    boxes_xywh=boxes_xywh,
                    confidences=confidences,
                    processing_time=processing_time,
                    success=True
                )
                
            except Exception as e:
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                prediction = ModelPrediction(
                    model_name=model_name,
                    boxes_xywh=[],
                    confidences=[],
                    processing_time=processing_time,
                    success=False,
                    error_message=str(e)
                )
                logger.error(f"Model {model_name} failed on frame {frame_idx}: {e}")
            
            model_predictions[model_name] = prediction
        
        # Extract ground truth data
        gt_boxes = []
        gt_ids = []
        if ground_truth:
            if "boxes" in ground_truth and len(ground_truth["boxes"]) > 0:
                # Convert from xyxy torch tensor to list
                gt_boxes = ground_truth["boxes"].cpu().numpy().tolist()
            if "labels" in ground_truth and len(ground_truth["labels"]) > 0:
                gt_ids = ground_truth["labels"].cpu().numpy().tolist()
        
        return MultiModelFrameResult(
            frame_idx=frame_idx,
            scene_id=scene_id,
            camera_id=camera_id,
            ground_truth_boxes=gt_boxes,
            ground_truth_ids=gt_ids,
            model_predictions=model_predictions,
            scene_context=scene_context,
            timestamp=timestamp
        )


class ModelComparisonAnalyzer:
    """Analyzes and compares performance across multiple models."""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.results: List[MultiModelFrameResult] = []
        
    def add_frame_result(self, result: MultiModelFrameResult):
        """Add a frame result for analysis."""
        self.results.append(result)
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes."""
        # Convert xywh to xyxy if needed
        if len(box1) == 4 and len(box2) == 4:
            # Assume box1 might be xywh (from predictions) and box2 is xyxy (from GT)
            if all(isinstance(x, (int, float)) for x in box1):
                # Convert xywh to xyxy for box1 if it looks like center format
                if box1[2] < box1[0] or box1[3] < box1[1]:  # Likely center format
                    cx, cy, w, h = box1
                    box1 = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_model_metrics(self, model_name: str) -> ModelPerformanceMetrics:
        """Calculate comprehensive metrics for a single model."""
        if not self.results:
            return ModelPerformanceMetrics(
                model_name=model_name,
                total_frames=0, total_gt_objects=0, true_positives=0,
                false_positives=0, false_negatives=0, precision=0.0,
                recall=0.0, f1_score=0.0, average_confidence=0.0,
                average_processing_time=0.0, success_rate=0.0
            )
        
        total_frames = len(self.results)
        total_gt_objects = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_confidence = 0.0
        confidence_count = 0
        total_processing_time = 0.0
        successful_frames = 0
        
        for result in self.results:
            if model_name not in result.model_predictions:
                continue
                
            prediction = result.model_predictions[model_name]
            total_processing_time += prediction.processing_time
            
            if prediction.success:
                successful_frames += 1
                
                # Count ground truth objects
                num_gt = len(result.ground_truth_boxes)
                total_gt_objects += num_gt
                
                # Track confidence scores
                if prediction.confidences:
                    total_confidence += sum(prediction.confidences)
                    confidence_count += len(prediction.confidences)
                
                # Calculate TP, FP, FN
                gt_matched = [False] * num_gt
                pred_matched = [False] * len(prediction.boxes_xywh)
                
                # Match predictions to ground truth
                for pred_idx, pred_box in enumerate(prediction.boxes_xywh):
                    best_iou = 0.0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_box in enumerate(result.ground_truth_boxes):
                        if gt_matched[gt_idx]:
                            continue
                        
                        iou = self.calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                        true_positives += 1
                        gt_matched[best_gt_idx] = True
                        pred_matched[pred_idx] = True
                
                # Count unmatched predictions as false positives
                false_positives += sum(1 for matched in pred_matched if not matched)
                
                # Count unmatched ground truth as false negatives
                false_negatives += sum(1 for matched in gt_matched if not matched)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        average_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0
        average_processing_time = total_processing_time / total_frames if total_frames > 0 else 0.0
        success_rate = successful_frames / total_frames if total_frames > 0 else 0.0
        
        return ModelPerformanceMetrics(
            model_name=model_name,
            total_frames=total_frames,
            total_gt_objects=total_gt_objects,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            average_confidence=average_confidence,
            average_processing_time=average_processing_time,
            success_rate=success_rate
        )
    
    def calculate_pairwise_agreement(self, model1: str, model2: str) -> float:
        """Calculate agreement between two models based on prediction overlap."""
        if not self.results:
            return 0.0
        
        total_agreement = 0.0
        valid_comparisons = 0
        
        for result in self.results:
            if (model1 not in result.model_predictions or 
                model2 not in result.model_predictions):
                continue
            
            pred1 = result.model_predictions[model1]
            pred2 = result.model_predictions[model2]
            
            if not (pred1.success and pred2.success):
                continue
            
            # Calculate IoU-based agreement
            if not pred1.boxes_xywh and not pred2.boxes_xywh:
                # Both models detected nothing - perfect agreement
                total_agreement += 1.0
            elif not pred1.boxes_xywh or not pred2.boxes_xywh:
                # One detected something, other didn't - no agreement
                total_agreement += 0.0
            else:
                # Calculate best matches between predictions
                matches = 0
                total_possible = max(len(pred1.boxes_xywh), len(pred2.boxes_xywh))
                
                for box1 in pred1.boxes_xywh:
                    best_iou = 0.0
                    for box2 in pred2.boxes_xywh:
                        iou = self.calculate_iou(box1, box2)
                        best_iou = max(best_iou, iou)
                    
                    if best_iou >= self.iou_threshold:
                        matches += 1
                
                frame_agreement = matches / total_possible if total_possible > 0 else 0.0
                total_agreement += frame_agreement
            
            valid_comparisons += 1
        
        return total_agreement / valid_comparisons if valid_comparisons > 0 else 0.0
    
    def analyze_scenario_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze model performance by different scenario types."""
        scenario_results = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            if not result.scene_context:
                continue
            
            # Group by different scenario dimensions
            scenarios = {
                f"lighting_{result.scene_context.lighting_condition}": result.scene_context.lighting_condition,
                f"density_{result.scene_context.crowd_density}": result.scene_context.crowd_density,
                f"occlusion_{result.scene_context.occlusion_level}": result.scene_context.occlusion_level,
                f"quality_{self._get_quality_category(result.scene_context.frame_quality)}": result.scene_context.frame_quality
            }
            
            for scenario_key, scenario_value in scenarios.items():
                for model_name, prediction in result.model_predictions.items():
                    if not prediction.success:
                        continue
                    
                    # Calculate F1 score for this frame (simplified)
                    frame_tp = 0
                    frame_fp = len(prediction.boxes_xywh)
                    frame_fn = len(result.ground_truth_boxes)
                    
                    # Basic matching (could be improved)
                    for pred_box in prediction.boxes_xywh:
                        for gt_box in result.ground_truth_boxes:
                            if self.calculate_iou(pred_box, gt_box) >= self.iou_threshold:
                                frame_tp += 1
                                frame_fp -= 1
                                frame_fn -= 1
                                break
                    
                    precision = frame_tp / (frame_tp + frame_fp) if (frame_tp + frame_fp) > 0 else 0.0
                    recall = frame_tp / (frame_tp + frame_fn) if (frame_tp + frame_fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    scenario_results[scenario_key][model_name].append(f1)
        
        # Average the results
        scenario_performance = {}
        for scenario, model_scores in scenario_results.items():
            scenario_performance[scenario] = {}
            for model, scores in model_scores.items():
                scenario_performance[scenario][model] = np.mean(scores) if scores else 0.0
        
        return scenario_performance
    
    def _get_quality_category(self, quality_score: float) -> str:
        """Convert quality score to category."""
        if quality_score >= 0.7:
            return 'high'
        elif quality_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def identify_ensemble_opportunities(self) -> List[EnsembleOpportunity]:
        """Identify opportunities for ensemble methods."""
        opportunities = []
        scenario_performance = self.analyze_scenario_performance()
        
        for scenario, model_scores in scenario_performance.items():
            if len(model_scores) < 2:
                continue
            
            # Sort models by performance
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            best_model = sorted_models[0][0]
            best_score = sorted_models[0][1]
            
            # Find complementary models
            complementary_models = []
            for model_name, score in sorted_models[1:]:
                if score >= 0.6 * best_score:  # Within 60% of best
                    complementary_models.append(model_name)
            
            if complementary_models:
                # Calculate potential improvement (simplified)
                potential_improvement = min(0.15, (1.0 - best_score) * 0.5)
                
                # Calculate complementarity score
                complementarity_score = len(complementary_models) / len(model_scores)
                
                opportunity = EnsembleOpportunity(
                    scenario_type=scenario,
                    primary_model=best_model,
                    supporting_models=complementary_models,
                    potential_improvement=potential_improvement,
                    complementarity_score=complementarity_score,
                    recommended_strategy='confidence_based' if len(complementary_models) <= 2 else 'voting'
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def generate_model_recommendations(self) -> Dict[str, str]:
        """Generate model recommendations by scenario."""
        scenario_performance = self.analyze_scenario_performance()
        recommendations = {}
        
        for scenario, model_scores in scenario_performance.items():
            if model_scores:
                best_model = max(model_scores.items(), key=lambda x: x[1])[0]
                recommendations[scenario] = best_model
        
        return recommendations
    
    def generate_comparison_matrix(self) -> ModelComparisonMatrix:
        """Generate comprehensive comparison matrix."""
        if not self.results:
            return ModelComparisonMatrix(
                models_compared=[],
                individual_metrics={},
                pairwise_agreement={},
                scenario_performance={},
                ensemble_opportunities={},
                model_recommendations={}
            )
        
        # Get all model names
        all_models = set()
        for result in self.results:
            all_models.update(result.model_predictions.keys())
        all_models = list(all_models)
        
        # Calculate individual metrics
        individual_metrics = {}
        for model in all_models:
            individual_metrics[model] = self.calculate_model_metrics(model)
        
        # Calculate pairwise agreement
        pairwise_agreement = {}
        for i, model1 in enumerate(all_models):
            for j, model2 in enumerate(all_models):
                if i < j:  # Avoid duplicate pairs
                    agreement = self.calculate_pairwise_agreement(model1, model2)
                    pairwise_agreement[(model1, model2)] = agreement
        
        # Analyze scenario performance
        scenario_performance = self.analyze_scenario_performance()
        
        # Identify ensemble opportunities
        opportunities = self.identify_ensemble_opportunities()
        ensemble_opportunities = {}
        for opp in opportunities:
            ensemble_opportunities[opp.scenario_type] = opp.supporting_models
        
        # Generate recommendations
        model_recommendations = self.generate_model_recommendations()
        
        return ModelComparisonMatrix(
            models_compared=all_models,
            individual_metrics=individual_metrics,
            pairwise_agreement=pairwise_agreement,
            scenario_performance=scenario_performance,
            ensemble_opportunities=ensemble_opportunities,
            model_recommendations=model_recommendations
        )


class ComparisonVisualizationGenerator:
    """Generates visualizations for cross-model comparison."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_performance_comparison_chart(self, comparison_matrix: ModelComparisonMatrix) -> Path:
        """Generate performance comparison chart."""
        models = comparison_matrix.models_compared
        if not models:
            return None
        
        # Prepare data
        metrics_data = {
            'Model': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'Processing Time (s)': []
        }
        
        for model in models:
            if model in comparison_matrix.individual_metrics:
                metrics = comparison_matrix.individual_metrics[model]
                metrics_data['Model'].append(model)
                metrics_data['Precision'].append(metrics.precision)
                metrics_data['Recall'].append(metrics.recall)
                metrics_data['F1-Score'].append(metrics.f1_score)
                metrics_data['Processing Time (s)'].append(metrics.average_processing_time)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Precision comparison
        axes[0, 0].bar(metrics_data['Model'], metrics_data['Precision'], color='skyblue')
        axes[0, 0].set_title('Precision by Model')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        axes[0, 1].bar(metrics_data['Model'], metrics_data['Recall'], color='lightcoral')
        axes[0, 1].set_title('Recall by Model')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        axes[1, 0].bar(metrics_data['Model'], metrics_data['F1-Score'], color='lightgreen')
        axes[1, 0].set_title('F1-Score by Model')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Processing time comparison
        axes[1, 1].bar(metrics_data['Model'], metrics_data['Processing Time (s)'], color='orange')
        axes[1, 1].set_title('Processing Time by Model')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        chart_path = self.output_dir / 'model_performance_comparison.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def generate_agreement_heatmap(self, comparison_matrix: ModelComparisonMatrix) -> Path:
        """Generate pairwise agreement heatmap."""
        models = comparison_matrix.models_compared
        if len(models) < 2:
            return None
        
        # Create agreement matrix
        agreement_matrix = np.zeros((len(models), len(models)))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i == j:
                    agreement_matrix[i, j] = 1.0  # Perfect self-agreement
                else:
                    # Find agreement in pairwise_agreement dict
                    key1 = (model1, model2)
                    key2 = (model2, model1)
                    
                    if key1 in comparison_matrix.pairwise_agreement:
                        agreement_matrix[i, j] = comparison_matrix.pairwise_agreement[key1]
                    elif key2 in comparison_matrix.pairwise_agreement:
                        agreement_matrix[i, j] = comparison_matrix.pairwise_agreement[key2]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(agreement_matrix, 
                   xticklabels=models, 
                   yticklabels=models,
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Agreement Score'})
        
        plt.title('Model Agreement Matrix (IoU-based)')
        plt.xlabel('Model')
        plt.ylabel('Model')
        plt.tight_layout()
        
        heatmap_path = self.output_dir / 'model_agreement_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return heatmap_path
    
    def generate_scenario_performance_chart(self, comparison_matrix: ModelComparisonMatrix) -> Path:
        """Generate scenario-based performance chart."""
        scenario_perf = comparison_matrix.scenario_performance
        if not scenario_perf:
            return None
        
        # Prepare data for plotting
        scenarios = list(scenario_perf.keys())
        models = comparison_matrix.models_compared
        
        # Create a dataframe for easier plotting
        data = []
        for scenario in scenarios:
            for model in models:
                if model in scenario_perf[scenario]:
                    data.append({
                        'Scenario': scenario,
                        'Model': model,
                        'F1-Score': scenario_perf[scenario][model]
                    })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Pivot data for grouped bar chart
        pivot_df = df.pivot(index='Scenario', columns='Model', values='F1-Score')
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('Model Performance by Scenario')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('F1-Score')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        scenario_chart_path = self.output_dir / 'scenario_performance_comparison.png'
        plt.savefig(scenario_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return scenario_chart_path
    
    def generate_side_by_side_failure_visualization(self, frame_result: MultiModelFrameResult,
                                                  original_image: np.ndarray) -> Path:
        """Generate side-by-side visualization of model predictions on a failure case."""
        models = list(frame_result.model_predictions.keys())
        num_models = len(models)
        
        if num_models == 0:
            return None
        
        # Create figure with subplots
        cols = min(3, num_models)
        rows = (num_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        
        for idx, model_name in enumerate(models):
            if rows > 1:
                ax = axes[idx // cols, idx % cols]
            else:
                ax = axes[idx] if num_models > 1 else axes
            
            # Copy image
            img_vis = original_image.copy()
            
            # Draw ground truth boxes in green
            for gt_box in frame_result.ground_truth_boxes:
                x1, y1, x2, y2 = map(int, gt_box)
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img_vis, 'GT', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw model predictions in red
            prediction = frame_result.model_predictions[model_name]
            if prediction.success:
                for i, pred_box in enumerate(prediction.boxes_xywh):
                    # Convert xywh to xyxy
                    cx, cy, w, h = pred_box
                    x1, y1, x2, y2 = int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)
                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    conf = prediction.confidences[i] if i < len(prediction.confidences) else 0.0
                    cv2.putText(img_vis, f'{conf:.2f}', (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img_rgb)
            ax.set_title(f'{model_name}\n{"Success" if prediction.success else "Failed"}')
            ax.axis('off')
        
        # Hide unused subplots
        if rows > 1:
            for idx in range(num_models, rows * cols):
                axes[idx // cols, idx % cols].axis('off')
        
        plt.suptitle(f'Model Comparison - Frame {frame_result.frame_idx}')
        plt.tight_layout()
        
        viz_path = self.output_dir / f'comparison_frame_{frame_result.frame_idx}_{frame_result.camera_id}.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path


class CrossModelComparisonSystem:
    """Main system for cross-model detection comparison."""
    
    def __init__(self, model_configs: Dict[str, Dict[str, Any]], 
                 device: torch.device, output_dir: Path):
        self.model_configs = model_configs
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_runner = MultiModelDetectionRunner(model_configs, device)
        self.comparison_analyzer = ModelComparisonAnalyzer()
        self.visualizer = ComparisonVisualizationGenerator(output_dir / 'visualizations')
        self.scene_analyzer = SceneAnalyzer() if SceneAnalyzer else None
        
    def analyze_dataset(self, dataset, max_frames: Optional[int] = None) -> ModelComparisonMatrix:
        """Analyze entire dataset with all models."""
        logger.info("Starting cross-model comparison analysis...")
        
        processed_frames = 0
        for i in range(len(dataset)):
            if max_frames and processed_frames >= max_frames:
                break
            
            try:
                # Get sample
                image_tensor, target = dataset[i]
                sample_info = dataset.get_sample_info(i)
                
                if not sample_info:
                    continue
                
                # Convert tensor to numpy image
                if isinstance(image_tensor, torch.Tensor):
                    # Convert from CHW to HWC and denormalize
                    image_np = image_tensor.permute(1, 2, 0).numpy()
                    if image_np.max() <= 1.0:  # If normalized
                        image_np = (image_np * 255).astype(np.uint8)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    continue
                
                # Analyze scene context if available
                scene_context = None
                if hasattr(self, 'scene_analyzer') and self.scene_analyzer:
                    gt_boxes = target.get("boxes", torch.empty(0, 4))
                    gt_ids = target.get("labels", torch.empty(0))
                    scene_context = self.scene_analyzer.analyze_scene_context(
                        image_bgr, gt_boxes, len(gt_ids)
                    )
                
                # Process frame with all models
                frame_result = self.model_runner.process_frame_all_models(
                    image_bgr,
                    frame_idx=i,
                    scene_id=sample_info['scene_id'],
                    camera_id=sample_info['camera_id'],
                    ground_truth=target,
                    scene_context=scene_context
                )
                
                # Add to analyzer
                self.comparison_analyzer.add_frame_result(frame_result)
                processed_frames += 1
                
                if processed_frames % 50 == 0:
                    logger.info(f"Processed {processed_frames} frames for cross-model comparison")
                
            except Exception as e:
                logger.error(f"Error processing frame {i}: {e}")
                continue
        
        logger.info(f"Cross-model analysis complete. Processed {processed_frames} frames.")
        return self.comparison_analyzer.generate_comparison_matrix()
    
    def generate_comprehensive_report(self, comparison_matrix: ModelComparisonMatrix) -> Dict[str, Path]:
        """Generate comprehensive cross-model comparison report."""
        logger.info("Generating cross-model comparison report...")
        
        generated_files = {}
        
        # Generate visualizations
        perf_chart = self.visualizer.generate_performance_comparison_chart(comparison_matrix)
        if perf_chart:
            generated_files['performance_chart'] = perf_chart
        
        agreement_heatmap = self.visualizer.generate_agreement_heatmap(comparison_matrix)
        if agreement_heatmap:
            generated_files['agreement_heatmap'] = agreement_heatmap
        
        scenario_chart = self.visualizer.generate_scenario_performance_chart(comparison_matrix)
        if scenario_chart:
            generated_files['scenario_chart'] = scenario_chart
        
        # Generate detailed report
        report_data = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'models_compared': comparison_matrix.models_compared,
                'analysis_type': 'Cross-Model Detection Comparison'
            },
            'model_metrics': {
                model: asdict(metrics) 
                for model, metrics in comparison_matrix.individual_metrics.items()
            },
            'pairwise_agreement': {
                f"{pair[0]}_vs_{pair[1]}": agreement 
                for pair, agreement in comparison_matrix.pairwise_agreement.items()
            },
            'scenario_performance': comparison_matrix.scenario_performance,
            'ensemble_opportunities': comparison_matrix.ensemble_opportunities,
            'model_recommendations': comparison_matrix.model_recommendations,
            'key_findings': self._extract_key_findings(comparison_matrix)
        }
        
        report_path = self.output_dir / 'cross_model_comparison_report.json'
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        generated_files['detailed_report'] = report_path
        
        # Generate summary
        summary_path = self._generate_executive_summary(comparison_matrix, generated_files)
        generated_files['executive_summary'] = summary_path
        
        logger.info(f"Cross-model comparison report complete. Generated {len(generated_files)} files.")
        return generated_files
    
    def _extract_key_findings(self, comparison_matrix: ModelComparisonMatrix) -> Dict[str, Any]:
        """Extract key findings from comparison analysis."""
        findings = {}
        
        if comparison_matrix.individual_metrics:
            # Best performing model overall
            best_f1_model = max(comparison_matrix.individual_metrics.items(), 
                              key=lambda x: x[1].f1_score)
            findings['best_overall_model'] = {
                'model': best_f1_model[0],
                'f1_score': best_f1_model[1].f1_score
            }
            
            # Fastest model
            fastest_model = min(comparison_matrix.individual_metrics.items(),
                              key=lambda x: x[1].average_processing_time)
            findings['fastest_model'] = {
                'model': fastest_model[0],
                'processing_time': fastest_model[1].average_processing_time
            }
            
            # Most reliable model
            most_reliable = max(comparison_matrix.individual_metrics.items(),
                              key=lambda x: x[1].success_rate)
            findings['most_reliable_model'] = {
                'model': most_reliable[0],
                'success_rate': most_reliable[1].success_rate
            }
        
        # Model agreement insights
        if comparison_matrix.pairwise_agreement:
            avg_agreement = np.mean(list(comparison_matrix.pairwise_agreement.values()))
            findings['average_model_agreement'] = avg_agreement
            
            highest_agreement = max(comparison_matrix.pairwise_agreement.items(), 
                                  key=lambda x: x[1])
            findings['highest_agreement_pair'] = {
                'models': highest_agreement[0],
                'agreement': highest_agreement[1]
            }
        
        # Ensemble opportunities count
        findings['ensemble_opportunities_count'] = len(comparison_matrix.ensemble_opportunities)
        
        return findings
    
    def _generate_executive_summary(self, comparison_matrix: ModelComparisonMatrix, 
                                  generated_files: Dict[str, Path]) -> Path:
        """Generate executive summary report."""
        summary_data = {
            'executive_summary': {
                'analysis_type': 'Cross-Model Detection Comparison - Phase 1.2',
                'timestamp': datetime.now().isoformat(),
                'models_analyzed': len(comparison_matrix.models_compared),
                'key_insights': self._extract_key_findings(comparison_matrix),
                'recommendations': self._generate_recommendations(comparison_matrix)
            },
            'generated_files': {key: str(path) for key, path in generated_files.items()}
        }
        
        summary_path = self.output_dir / 'executive_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        return summary_path
    
    def _generate_recommendations(self, comparison_matrix: ModelComparisonMatrix) -> List[str]:
        """Generate actionable recommendations from comparison analysis."""
        recommendations = []
        
        if comparison_matrix.individual_metrics:
            # Performance recommendations
            best_model = max(comparison_matrix.individual_metrics.items(), 
                           key=lambda x: x[1].f1_score)[0]
            recommendations.append(f"Use {best_model} for best overall detection performance")
            
            fastest_model = min(comparison_matrix.individual_metrics.items(),
                              key=lambda x: x[1].average_processing_time)[0]
            recommendations.append(f"Use {fastest_model} for real-time applications requiring speed")
        
        # Ensemble recommendations
        if comparison_matrix.ensemble_opportunities:
            recommendations.append(f"Consider ensemble methods for {len(comparison_matrix.ensemble_opportunities)} identified scenarios")
        
        # Scenario-specific recommendations
        for scenario, best_model in comparison_matrix.model_recommendations.items():
            recommendations.append(f"Use {best_model} for {scenario} scenarios")
        
        return recommendations