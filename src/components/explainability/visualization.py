import logging
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

# OpenCV Colormap Mapping
CV2_COLORMAPS = {name: getattr(cv2, name) for name in dir(cv2) if name.startswith('COLORMAP_')}


def overlay_heatmap(
    image_np: np.ndarray,
    heatmap_np: np.ndarray,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Overlays a heatmap onto an image."""
    # (Implementation remains the same)
    if image_np.dtype != np.uint8:
        logger.warning(f"Input image dtype is {image_np.dtype}, converting to uint8.")
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    if heatmap_np.ndim != 2:
        if heatmap_np.ndim == 3 and heatmap_np.shape[0] == 1:
             heatmap_np = heatmap_np.squeeze(0)
        elif heatmap_np.ndim == 3 and heatmap_np.shape[-1] == 1:
             heatmap_np = heatmap_np.squeeze(-1)
        else:
             raise ValueError(f"Heatmap must be 2D, but got shape {heatmap_np.shape}")

    try:
        min_val, max_val = heatmap_np.min(), heatmap_np.max()
        if max_val > min_val:
            heatmap_normalized = (heatmap_np - min_val) / (max_val - min_val) * 255
        else:
            heatmap_normalized = np.zeros_like(heatmap_np)

        heatmap_uint8 = heatmap_normalized.astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap_uint8, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay
    except Exception as e:
        logger.error(f"Error overlaying heatmap: {e}")
        return image_np


def visualize_explanation(
    original_image_rgb: np.ndarray,
    attribution_map: torch.Tensor,
    output_path: Path,
    # box_to_highlight: Optional[List[float]] = None, # <<< Removed
    boxes_to_draw: Optional[List[List[float]]] = None, # <<< Renamed parameter
    score: Optional[float] = None, # Score of the detection being explained
    label: Optional[int] = None, # Label of the detection being explained
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET,
    figsize: Tuple[int, int] = (10, 10)
):
    """
    Visualizes the Grad-CAM explanation by overlaying the heatmap and drawing
    all provided bounding boxes with a consistent style.

    Args:
        original_image_rgb: The original image (NumPy array, RGB, uint8).
        attribution_map: The raw attribution heatmap from Captum (PyTorch Tensor, CHW or HW).
                         This heatmap corresponds to the prediction indicated by score/label.
        output_path: Path to save the visualization.
        boxes_to_draw: List of coordinates [[x1, y1, x2, y2], ...] for ALL detections
                       (above initial threshold) to be drawn on the image.
        score: Confidence score of the specific detection being explained by the heatmap.
        label: Class label of the specific detection being explained by the heatmap.
        alpha: Transparency for the heatmap overlay.
        colormap: OpenCV colormap *constant* (e.g., cv2.COLORMAP_JET).
        figsize: Figure size for the plot.
    """
    fig = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if attribution_map.dim() == 3:
            processed_heatmap_np = attribution_map.sum(dim=0).cpu().numpy()
        elif attribution_map.dim() == 2:
            processed_heatmap_np = attribution_map.cpu().numpy()
        else:
            raise ValueError(f"Unsupported attribution map dimension: {attribution_map.dim()}")

        if not np.all(np.isfinite(processed_heatmap_np)):
             logger.warning(f"Non-finite values found in heatmap for {output_path.name}. Clamping to 0.")
             processed_heatmap_np = np.nan_to_num(processed_heatmap_np, nan=0.0, posinf=0.0, neginf=0.0)

        overlay_img = overlay_heatmap(
            original_image_rgb.copy(),
            processed_heatmap_np,
            alpha=alpha,
            colormap=colormap
        )

        fig = plt.figure(figsize=figsize)
        plt.imshow(overlay_img)
        plt.axis('off')

        # Simplified title, indicates which prediction the heatmap belongs to
        title = "Explanation Heatmap"
        if label is not None and score is not None:
             title += f" (Explaining Class: {label}, Score: {score:.3f} Detection)"
        plt.title(title, fontsize=10)

        # --- Draw ALL detected boxes with the same style ---
        if boxes_to_draw:
            for box in boxes_to_draw:
                 x1, y1, x2, y2 = box
                 rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      fill=False, edgecolor='lime', linewidth=1.5, linestyle='-') # Consistent style: lime, solid, thinner
                 plt.gca().add_patch(rect)
        # ----------------------------------------------------

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        logger.info(f"Saved explanation visualization to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to create visualization for {output_path.name}: {e}", exc_info=True)
        if fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)


def save_analysis_visualization(
    image: np.ndarray,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    score: float,
    output_dir: Path,
    base_filename: str,
    config: dict,
):
    """
    Draws ground truth and prediction boxes on an image and saves it.

    Args:
        image: The original image in RGB format (numpy array).
        pred_boxes: Predicted bounding boxes (Tensor).
        pred_scores: Scores for predicted boxes (Tensor).
        gt_boxes: Ground truth bounding boxes (Tensor).
        score: The overall performance score for the frame.
        output_dir: The directory to save the image.
        base_filename: The original name of the image file.
        config: The analysis configuration dictionary for viz parameters.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        vis_config = config.get("analysis", {}).get("visualization", {})
        
        # Convert RGB to BGR for OpenCV
        img_to_draw = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get visualization params from config
        gt_color = vis_config.get("colors", {}).get("ground_truth", (0, 255, 0))
        pred_color = vis_config.get("colors", {}).get("prediction", (0, 0, 255))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = vis_config.get("font_scale", 0.6)
        thickness = vis_config.get("thickness", 2)

        # Draw Ground Truth boxes
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), gt_color, thickness)
            cv2.putText(img_to_draw, "GT", (x1, y1 - 10), font, font_scale, gt_color, thickness)

        # Draw Prediction boxes
        for box, scr in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), pred_color, thickness)
            label = f"Pred: {scr:.2f}"
            cv2.putText(img_to_draw, label, (x1, y2 + 20), font, font_scale, pred_color, thickness)
        
        # Add overall score to the image
        score_text = f"Frame Score: {score:.3f}"
        cv2.putText(img_to_draw, score_text, (20, 40), font, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img_to_draw, score_text, (20, 40), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        # Save the image
        new_filename = f"{Path(base_filename).stem}_score_{score:.2f}.png"
        output_path = output_dir / new_filename
        cv2.imwrite(str(output_path), img_to_draw)

    except Exception as e:
        logger.error(f"Failed to save visualization for {base_filename}: {e}", exc_info=True)