import logging
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

# --- Add OpenCV Colormap Mapping ---
CV2_COLORMAPS = {name: getattr(cv2, name) for name in dir(cv2) if name.startswith('COLORMAP_')}
# --- End Colormap Mapping ---


def overlay_heatmap(
    image_np: np.ndarray,
    heatmap_np: np.ndarray,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET # Keep default int here
) -> np.ndarray:
    """
    Overlays a heatmap onto an image.

    Args:
        image_np: Original image as NumPy array (RGB, 0-255, uint8).
        heatmap_np: Heatmap as NumPy array (should be 2D).
        alpha: Transparency of the heatmap overlay.
        colormap: OpenCV colormap constant (e.g., cv2.COLORMAP_JET).

    Returns:
        NumPy array of the image with the heatmap overlaid.
    """
    if image_np.dtype != np.uint8:
        logger.warning(f"Input image dtype is {image_np.dtype}, converting to uint8.")
        image_np = image_np.astype(np.uint8)
    if heatmap_np.ndim != 2:
        if heatmap_np.ndim == 3 and heatmap_np.shape[0] == 1:
             heatmap_np = heatmap_np.squeeze(0)
        elif heatmap_np.ndim == 3 and heatmap_np.shape[-1] == 1:
             heatmap_np = heatmap_np.squeeze(-1)
        else:
             raise ValueError(f"Heatmap must be 2D, but got shape {heatmap_np.shape}")

    try:
        heatmap_normalized = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap_resized = cv2.resize(heatmap_normalized, (image_np.shape[1], image_np.shape[0]))
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
    box_to_highlight: Optional[List[float]] = None,
    score: Optional[float] = None,
    label: Optional[int] = None,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET, # Expect integer colormap here
    figsize: Tuple[int, int] = (10, 10)
):
    """
    Visualizes the Grad-CAM explanation by overlaying the heatmap and optionally
    highlighting the target bounding box.

    Args:
        original_image_rgb: The original image (NumPy array, RGB, uint8).
        attribution_map: The raw attribution heatmap from Captum (PyTorch Tensor, CHW or HW).
        output_path: Path to save the visualization.
        box_to_highlight: Coordinates [x1, y1, x2, y2] of the detection box to draw.
        score: Confidence score of the detection to display.
        label: Class label of the detection to display.
        alpha: Transparency for the heatmap overlay.
        colormap: OpenCV colormap *constant* (e.g., cv2.COLORMAP_JET).
        figsize: Figure size for the plot.
    """
    # (Function body remains unchanged from previous response)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if attribution_map.dim() == 3:
            processed_heatmap_np = attribution_map.sum(dim=0).cpu().numpy()
        elif attribution_map.dim() == 2:
            processed_heatmap_np = attribution_map.cpu().numpy()
        else:
            raise ValueError(f"Unsupported attribution map dimension: {attribution_map.dim()}")

        overlay_img = overlay_heatmap(
            original_image_rgb, processed_heatmap_np, alpha=alpha, colormap=colormap
        )

        plt.figure(figsize=figsize)
        plt.imshow(overlay_img)
        plt.axis('off')

        title = "Explanation Heatmap"
        if label is not None and score is not None:
             title += f" (Class: {label}, Score: {score:.2f})"
        plt.title(title)

        if box_to_highlight:
            x1, y1, x2, y2 = box_to_highlight
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor='lime', linewidth=2.5)
            plt.gca().add_patch(rect)

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        logger.info(f"Saved explanation visualization to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to create visualization for {output_path.name}: {e}", exc_info=True)
        if 'plt' in locals() and plt.gcf().number > 0:
            plt.close()
