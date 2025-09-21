"""
Surveillance-Optimized Detector Module
Specialized detector configurations for surveillance person detection scenarios
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """Configuration for surveillance detector optimizations."""
    
    # Detection thresholds
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 300
    
    # Person-specific thresholds
    person_confidence_threshold: float = 0.3  # Lower threshold for persons
    person_nms_threshold: float = 0.5
    
    # Multi-scale detection
    scale_aware_nms: bool = True
    small_object_boost: float = 1.2
    large_object_penalty: float = 0.9
    
    # Crowd handling
    crowd_nms_enabled: bool = True
    crowd_density_threshold: int = 15  # Persons per image
    crowd_nms_threshold: float = 0.6   # Higher for crowded scenes
    
    # Temporal consistency (for video)
    temporal_consistency: bool = False
    tracking_enabled: bool = False
    
    # Performance optimization
    batch_nms: bool = True
    optimize_for_speed: bool = False


class ScaleAwareNMS:
    """Scale-aware Non-Maximum Suppression for better person detection."""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
    
    def __call__(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply scale-aware NMS.
        
        Args:
            boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
            scores: Confidence scores [N]
            classes: Class predictions [N]
            image_size: (height, width)
            
        Returns:
            Filtered boxes, scores, classes
        """
        
        if len(boxes) == 0:
            return boxes, scores, classes
        
        # Calculate box areas for scale classification
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        image_area = image_size[0] * image_size[1]
        relative_areas = areas / image_area
        
        # Classify boxes by scale
        small_mask = relative_areas < 0.001   # Very small objects
        medium_mask = (relative_areas >= 0.001) & (relative_areas < 0.01)  # Medium objects
        large_mask = relative_areas >= 0.01   # Large objects
        
        keep_indices = []
        
        # Process each scale separately
        for mask, scale_name in [(small_mask, 'small'), (medium_mask, 'medium'), (large_mask, 'large')]:
            if not mask.any():
                continue
                
            scale_boxes = boxes[mask]
            scale_scores = scores[mask]
            scale_classes = classes[mask]
            
            # Adjust NMS threshold based on scale
            if scale_name == 'small':
                nms_threshold = self.config.nms_threshold * 0.8  # More permissive for small objects
                score_boost = self.config.small_object_boost
            elif scale_name == 'large':
                nms_threshold = self.config.nms_threshold * 1.2  # More strict for large objects
                score_boost = self.config.large_object_penalty
            else:
                nms_threshold = self.config.nms_threshold
                score_boost = 1.0
            
            # Boost scores for small objects, penalize large objects
            adjusted_scores = scale_scores * score_boost
            
            # Apply NMS within this scale
            scale_keep = self._nms_single_scale(
                scale_boxes, adjusted_scores, scale_classes, nms_threshold
            )
            
            # Map back to original indices
            original_indices = torch.where(mask)[0]
            keep_indices.extend(original_indices[scale_keep].tolist())
        
        if not keep_indices:
            return torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.long)
        
        keep_indices = torch.tensor(keep_indices, device=boxes.device)
        
        return boxes[keep_indices], scores[keep_indices], classes[keep_indices]
    
    def _nms_single_scale(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """Apply NMS within a single scale."""
        
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Sort by score
        sorted_indices = torch.argsort(scores, descending=True)
        
        keep = []
        while len(sorted_indices) > 0:
            # Take the box with highest score
            current_idx = sorted_indices[0]
            keep.append(current_idx.item())
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current_idx].unsqueeze(0)
            remaining_boxes = boxes[sorted_indices[1:]]
            
            ious = self._calculate_iou(current_box, remaining_boxes)
            
            # Keep boxes with IoU less than threshold
            keep_mask = ious.squeeze() < threshold
            sorted_indices = sorted_indices[1:][keep_mask]
        
        return torch.tensor(keep, device=boxes.device, dtype=torch.long)
    
    def _calculate_iou(self, box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between one box and multiple boxes."""
        
        # Calculate intersection
        x1 = torch.max(box1[:, 0], boxes2[:, 0])
        y1 = torch.max(box1[:, 1], boxes2[:, 1])
        x2 = torch.min(box1[:, 2], boxes2[:, 2])
        y2 = torch.min(box1[:, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-8)


class CrowdAwareDetector:
    """Crowd-aware detection post-processing for dense surveillance scenes."""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.scale_nms = ScaleAwareNMS(config)
    
    def __call__(
        self,
        predictions: Dict[str, torch.Tensor],
        image_sizes: List[Tuple[int, int]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Process model predictions with crowd-aware post-processing.
        
        Args:
            predictions: Model predictions containing logits and boxes
            image_sizes: List of (height, width) for each image
            
        Returns:
            List of detection results per image
        """
        
        batch_size = len(image_sizes)
        results = []
        
        # Extract predictions
        if 'pred_logits' in predictions and 'pred_boxes' in predictions:
            logits = predictions['pred_logits']  # [B, N, num_classes]
            boxes = predictions['pred_boxes']     # [B, N, 4]
            
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            for i in range(batch_size):
                image_results = self._process_single_image(
                    probs[i], boxes[i], image_sizes[i]
                )
                results.append(image_results)
        
        return results
    
    def _process_single_image(
        self,
        probs: torch.Tensor,
        boxes: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """Process predictions for a single image."""
        
        # Convert normalized boxes to absolute coordinates
        height, width = image_size
        boxes = boxes * torch.tensor([width, height, width, height], device=boxes.device)
        
        # Convert from center format to corner format if needed
        if boxes.size(-1) == 4:  # Assuming (cx, cy, w, h) format
            x_center, y_center, w, h = boxes.unbind(-1)
            boxes = torch.stack([
                x_center - w/2, y_center - h/2,  # x1, y1
                x_center + w/2, y_center + h/2   # x2, y2
            ], dim=-1)
        
        # Get class predictions (excluding background class 0)
        class_probs = probs[:, 1:]  # [N, num_classes-1]
        max_probs, class_indices = class_probs.max(dim=-1)
        class_indices = class_indices + 1  # Add 1 to account for background class
        
        # Filter by confidence threshold
        person_class = 1  # Assuming person is class 1
        person_mask = class_indices == person_class
        
        # Different thresholds for person vs other classes
        confidence_mask = torch.zeros_like(max_probs, dtype=torch.bool)
        confidence_mask[person_mask] = max_probs[person_mask] >= self.config.person_confidence_threshold
        confidence_mask[~person_mask] = max_probs[~person_mask] >= self.config.confidence_threshold
        
        # Apply confidence filtering
        valid_boxes = boxes[confidence_mask]
        valid_scores = max_probs[confidence_mask]
        valid_classes = class_indices[confidence_mask]
        
        if len(valid_boxes) == 0:
            return {
                'boxes': torch.empty(0, 4, device=boxes.device),
                'scores': torch.empty(0, device=boxes.device),
                'labels': torch.empty(0, dtype=torch.long, device=boxes.device)
            }
        
        # Determine if this is a crowded scene
        person_count = (valid_classes == person_class).sum().item()
        is_crowded = person_count > self.config.crowd_density_threshold
        
        # Apply appropriate NMS strategy
        if self.config.scale_aware_nms:
            final_boxes, final_scores, final_classes = self.scale_nms(
                valid_boxes, valid_scores, valid_classes, image_size
            )
        else:
            # Standard NMS
            nms_threshold = (self.config.crowd_nms_threshold if is_crowded 
                           else self.config.nms_threshold)
            keep_indices = self._standard_nms(valid_boxes, valid_scores, nms_threshold)
            final_boxes = valid_boxes[keep_indices]
            final_scores = valid_scores[keep_indices]
            final_classes = valid_classes[keep_indices]
        
        # Limit number of detections
        if len(final_boxes) > self.config.max_detections:
            # Keep top detections by score
            top_indices = torch.argsort(final_scores, descending=True)[:self.config.max_detections]
            final_boxes = final_boxes[top_indices]
            final_scores = final_scores[top_indices]
            final_classes = final_classes[top_indices]
        
        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_classes,
            'is_crowded': is_crowded,
            'person_count': person_count
        }
    
    def _standard_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """Standard non-maximum suppression."""
        
        from torchvision.ops import nms
        return nms(boxes, scores, threshold)


class SurveillanceMetrics:
    """Comprehensive metrics for surveillance person detection evaluation."""
    
    def __init__(self, iou_thresholds: List[float] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.ground_truths = []
        self.image_info = []
    
    def update(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        ground_truths: List[Dict[str, torch.Tensor]],
        image_ids: Optional[List[int]] = None
    ):
        """Update metrics with new predictions and ground truths."""
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            image_id = image_ids[i] if image_ids else len(self.predictions)
            
            self.predictions.append({
                'image_id': image_id,
                'boxes': pred['boxes'].cpu(),
                'scores': pred['scores'].cpu(),
                'labels': pred['labels'].cpu()
            })
            
            self.ground_truths.append({
                'image_id': image_id,
                'boxes': gt['boxes'].cpu(),
                'labels': gt['labels'].cpu(),
                'iscrowd': gt.get('iscrowd', torch.zeros(len(gt['boxes']), dtype=torch.bool))
            })
            
            # Store additional image information
            self.image_info.append({
                'image_id': image_id,
                'is_crowded': pred.get('is_crowded', False),
                'person_count_pred': pred.get('person_count', 0),
                'person_count_gt': (gt['labels'] == 1).sum().item()
            })
    
    def compute(self) -> Dict[str, float]:
        """Compute comprehensive surveillance metrics."""
        
        if not self.predictions:
            return {}
        
        metrics = {}
        
        # Standard COCO metrics
        coco_metrics = self._compute_coco_metrics()
        metrics.update(coco_metrics)
        
        # Surveillance-specific metrics
        surveillance_metrics = self._compute_surveillance_metrics()
        metrics.update(surveillance_metrics)
        
        # Scene-specific analysis
        scene_metrics = self._compute_scene_metrics()
        metrics.update(scene_metrics)
        
        return metrics
    
    def _compute_coco_metrics(self) -> Dict[str, float]:
        """Compute standard COCO detection metrics."""
        
        # This would integrate with pycocotools for official mAP computation
        # For now, implementing simplified version
        
        all_ious = []
        all_scores = []
        all_matches = []
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            if len(pred['boxes']) == 0 or len(gt['boxes']) == 0:
                continue
            
            # Calculate IoU matrix
            ious = self._calculate_iou_matrix(pred['boxes'], gt['boxes'])
            
            # For each IoU threshold, determine matches
            for iou_thresh in self.iou_thresholds:
                matches, scores = self._match_predictions(
                    ious, pred['scores'], iou_thresh
                )
                
                all_ious.extend(ious.max(dim=1)[0].tolist())
                all_scores.extend(scores.tolist())
                all_matches.extend(matches)
        
        # Compute AP for different IoU thresholds
        metrics = {}
        for iou_thresh in self.iou_thresholds:
            ap = self._compute_average_precision(all_matches, all_scores, iou_thresh)
            metrics[f'AP_{int(iou_thresh*100)}'] = ap
        
        # Overall mAP
        metrics['mAP'] = np.mean([metrics[f'AP_{int(t*100)}'] for t in self.iou_thresholds])
        
        return metrics
    
    def _compute_surveillance_metrics(self) -> Dict[str, float]:
        """Compute surveillance-specific metrics."""
        
        metrics = {}
        
        # Person detection accuracy
        person_predictions = []
        person_ground_truths = []
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            # Filter for person class (assuming class 1)
            person_mask_pred = pred['labels'] == 1
            person_mask_gt = gt['labels'] == 1
            
            if person_mask_pred.any():
                person_predictions.append({
                    'boxes': pred['boxes'][person_mask_pred],
                    'scores': pred['scores'][person_mask_pred]
                })
            else:
                person_predictions.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0)})
            
            if person_mask_gt.any():
                person_ground_truths.append({
                    'boxes': gt['boxes'][person_mask_gt]
                })
            else:
                person_ground_truths.append({'boxes': torch.empty(0, 4)})
        
        # Person-specific mAP
        if person_predictions and person_ground_truths:
            person_metrics = self._compute_person_ap(person_predictions, person_ground_truths)
            metrics.update(person_metrics)
        
        # Crowd scene performance
        crowd_metrics = self._compute_crowd_metrics()
        metrics.update(crowd_metrics)
        
        return metrics
    
    def _compute_scene_metrics(self) -> Dict[str, float]:
        """Compute scene-specific metrics."""
        
        # Group by scene characteristics
        normal_scenes = []
        crowded_scenes = []
        
        for i, info in enumerate(self.image_info):
            if info['is_crowded']:
                crowded_scenes.append(i)
            else:
                normal_scenes.append(i)
        
        metrics = {}
        
        # Performance on normal vs crowded scenes
        if normal_scenes:
            normal_ap = self._compute_scene_specific_ap(normal_scenes)
            metrics['normal_scene_mAP'] = normal_ap
        
        if crowded_scenes:
            crowded_ap = self._compute_scene_specific_ap(crowded_scenes)
            metrics['crowded_scene_mAP'] = crowded_ap
        
        # Performance degradation in crowds
        if 'normal_scene_mAP' in metrics and 'crowded_scene_mAP' in metrics:
            degradation = metrics['normal_scene_mAP'] - metrics['crowded_scene_mAP']
            metrics['crowd_degradation'] = degradation
        
        return metrics
    
    def _compute_person_ap(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        ground_truths: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Compute person-specific AP metrics."""
        
        # Simplified person AP computation
        all_tps = []
        all_fps = []
        all_scores = []
        total_gt = sum(len(gt['boxes']) for gt in ground_truths)
        
        for pred, gt in zip(predictions, ground_truths):
            if len(pred['boxes']) == 0:
                continue
            
            if len(gt['boxes']) == 0:
                all_fps.extend([1] * len(pred['boxes']))
                all_tps.extend([0] * len(pred['boxes']))
            else:
                ious = self._calculate_iou_matrix(pred['boxes'], gt['boxes'])
                matches = ious.max(dim=1)[0] > 0.5  # IoU threshold 0.5
                
                all_tps.extend(matches.int().tolist())
                all_fps.extend((~matches).int().tolist())
            
            all_scores.extend(pred['scores'].tolist())
        
        if not all_scores:
            return {'person_AP': 0.0, 'person_recall': 0.0, 'person_precision': 0.0}
        
        # Sort by score
        indices = np.argsort(all_scores)[::-1]
        tps = np.array(all_tps)[indices]
        fps = np.array(all_fps)[indices]
        
        # Compute cumulative precision and recall
        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(fps)
        
        recalls = tp_cumsum / max(total_gt, 1)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Compute AP using trapezoidal rule
        ap = np.trapz(precisions, recalls) if len(recalls) > 1 else 0.0
        
        return {
            'person_AP': float(ap),
            'person_recall': float(recalls[-1]) if len(recalls) > 0 else 0.0,
            'person_precision': float(precisions[-1]) if len(precisions) > 0 else 0.0
        }
    
    def _compute_crowd_metrics(self) -> Dict[str, float]:
        """Compute crowd-specific metrics."""
        
        crowd_accuracy = []
        
        for info in self.image_info:
            gt_count = info['person_count_gt']
            pred_count = info['person_count_pred']
            
            if gt_count == 0 and pred_count == 0:
                accuracy = 1.0
            elif gt_count == 0:
                accuracy = 0.0
            else:
                accuracy = 1.0 - abs(gt_count - pred_count) / gt_count
            
            crowd_accuracy.append(max(0.0, accuracy))
        
        return {
            'crowd_counting_accuracy': np.mean(crowd_accuracy) if crowd_accuracy else 0.0,
            'avg_crowd_detection_error': np.mean([
                abs(info['person_count_gt'] - info['person_count_pred'])
                for info in self.image_info
            ])
        }
    
    def _compute_scene_specific_ap(self, scene_indices: List[int]) -> float:
        """Compute AP for specific scene indices."""
        
        scene_predictions = [self.predictions[i] for i in scene_indices]
        scene_ground_truths = [self.ground_truths[i] for i in scene_indices]
        
        # Simplified AP computation for scene subset
        all_matches = []
        all_scores = []
        
        for pred, gt in zip(scene_predictions, scene_ground_truths):
            if len(pred['boxes']) == 0 or len(gt['boxes']) == 0:
                continue
            
            ious = self._calculate_iou_matrix(pred['boxes'], gt['boxes'])
            matches = ious.max(dim=1)[0] > 0.5
            
            all_matches.extend(matches.tolist())
            all_scores.extend(pred['scores'].tolist())
        
        if not all_scores:
            return 0.0
        
        # Simple AP approximation
        sorted_indices = np.argsort(all_scores)[::-1]
        sorted_matches = np.array(all_matches)[sorted_indices]
        
        precision_sum = 0.0
        true_positive_count = 0
        
        for i, match in enumerate(sorted_matches):
            if match:
                true_positive_count += 1
                precision_sum += true_positive_count / (i + 1)
        
        return precision_sum / max(true_positive_count, 1)
    
    def _calculate_iou_matrix(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """Calculate IoU matrix between two sets of boxes."""
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # Left-top corner
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # Right-bottom corner
        
        wh = (rb - lt).clamp(min=0)
        intersection = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - intersection
        iou = intersection / (union + 1e-8)
        
        return iou
    
    def _match_predictions(
        self,
        ious: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float
    ) -> Tuple[List[bool], torch.Tensor]:
        """Match predictions to ground truth based on IoU threshold."""
        
        matches = []
        used_gt = set()
        
        # Sort predictions by score
        sorted_indices = torch.argsort(scores, descending=True)
        
        for pred_idx in sorted_indices:
            pred_ious = ious[pred_idx]
            
            # Find best matching ground truth
            best_iou, best_gt_idx = pred_ious.max(dim=0)
            
            if best_iou >= iou_threshold and best_gt_idx.item() not in used_gt:
                matches.append(True)
                used_gt.add(best_gt_idx.item())
            else:
                matches.append(False)
        
        return matches, scores[sorted_indices]
    
    def _compute_average_precision(
        self,
        matches: List[bool],
        scores: List[float],
        iou_threshold: float
    ) -> float:
        """Compute average precision for given IoU threshold."""
        
        if not matches:
            return 0.0
        
        # Sort by score
        sorted_pairs = sorted(zip(scores, matches), key=lambda x: x[0], reverse=True)
        sorted_matches = [match for _, match in sorted_pairs]
        
        # Compute precision at each recall level
        tp_sum = 0
        precisions = []
        
        for i, match in enumerate(sorted_matches):
            if match:
                tp_sum += 1
            
            precision = tp_sum / (i + 1)
            precisions.append(precision)
        
        # Compute AP using average precision
        if tp_sum == 0:
            return 0.0
        
        return np.mean(precisions)


def create_surveillance_detector(
    confidence_threshold: float = 0.5,
    person_confidence_threshold: float = 0.3,
    scale_aware_nms: bool = True,
    crowd_nms_enabled: bool = True,
    **kwargs
) -> CrowdAwareDetector:
    """
    Create surveillance-optimized detector.
    
    Args:
        confidence_threshold: General confidence threshold
        person_confidence_threshold: Lower threshold for person class
        scale_aware_nms: Enable scale-aware NMS
        crowd_nms_enabled: Enable crowd-aware NMS
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured surveillance detector
    """
    
    config = DetectorConfig(
        confidence_threshold=confidence_threshold,
        person_confidence_threshold=person_confidence_threshold,
        scale_aware_nms=scale_aware_nms,
        crowd_nms_enabled=crowd_nms_enabled,
        **kwargs
    )
    
    return CrowdAwareDetector(config)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Surveillance Detector")
    
    # Create detector
    detector = create_surveillance_detector(
        confidence_threshold=0.5,
        person_confidence_threshold=0.3,
        scale_aware_nms=True
    )
    
    # Mock predictions
    batch_size = 2
    num_queries = 100
    num_classes = 91
    
    mock_predictions = {
        'pred_logits': torch.randn(batch_size, num_queries, num_classes),
        'pred_boxes': torch.rand(batch_size, num_queries, 4)  # Normalized coordinates
    }
    
    # Mock image sizes
    image_sizes = [(480, 640), (512, 768)]
    
    # Test detection processing
    results = detector(mock_predictions, image_sizes)
    
    print(f"✅ Detection processing completed")
    print(f"  Batch size: {len(results)}")
    for i, result in enumerate(results):
        print(f"  Image {i}: {len(result['boxes'])} detections, crowded: {result.get('is_crowded', False)}")
    
    # Test metrics computation
    metrics = SurveillanceMetrics()
    
    # Mock ground truth
    mock_gt = [
        {
            'boxes': torch.tensor([[100, 100, 200, 300], [300, 150, 400, 350]], dtype=torch.float),
            'labels': torch.tensor([1, 1], dtype=torch.long)  # Person class
        },
        {
            'boxes': torch.tensor([[50, 50, 150, 250]], dtype=torch.float),
            'labels': torch.tensor([1], dtype=torch.long)
        }
    ]
    
    # Update metrics
    metrics.update(results, mock_gt, image_ids=[0, 1])
    
    # Compute metrics
    computed_metrics = metrics.compute()
    
    print(f"✅ Metrics computation completed")
    for metric_name, value in computed_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    print("✅ Surveillance Detector testing completed")