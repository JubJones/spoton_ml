# FILE: src/explainability/faster_rcnn_explainer.py
import logging
from typing import Optional

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN

try:
    from captum.attr import LayerGradCam
    CAPTUM_AVAILABLE = True
except ImportError:
    LayerGradCam = None
    CAPTUM_AVAILABLE = False
    logging.warning("Captum library not found. Explainability features unavailable. Install with `pip install captum`")

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ['gradcam'] # Add more if implemented, e.g., 'integratedgradients'

def get_target_layer(model: FasterRCNN, layer_name_str: str) -> Optional[nn.Module]:
    """
    Retrieves a layer module from the model using a dot-separated name string.

    Args:
        model: The FasterRCNN model instance.
        layer_name_str: Dot-separated path to the layer (e.g., "backbone.body.layer4").

    Returns:
        The torch.nn.Module if found, otherwise None.
    """
    layers = layer_name_str.split('.')
    current_module = model
    try:
        for layer in layers:
            current_module = getattr(current_module, layer)
        logger.info(f"Successfully retrieved target layer: {layer_name_str}")
        return current_module
    except AttributeError:
        logger.error(f"Could not find layer '{layer_name_str}' in the model.")
        return None

def explain_detection_gradcam(
    model: FasterRCNN,
    input_tensor: torch.Tensor, # This is the original 3D tensor [C, H, W]
    target_layer: nn.Module,
    target_index: int,
    target_class_index: int,
    device: torch.device
) -> Optional[torch.Tensor]:
    """
    Generates a Grad-CAM explanation for a specific detection.

    Args:
        model: The FasterRCNN model (must be on the correct device).
        input_tensor: The preprocessed input image tensor ([C, H, W], must be on the correct device).
        target_layer: The convolutional layer to compute Grad-CAM for.
        target_index: The index of the detection box in the model's output list
                      (from the single-image inference result) to explain.
        target_class_index: The class index (e.g., 1 for 'person') to explain.
        device: The device the model and tensor are on.

    Returns:
        The Grad-CAM attribution heatmap tensor (on CPU), or None if an error occurs.
    """
    if not CAPTUM_AVAILABLE or LayerGradCam is None:
        logger.error("Captum not available, cannot generate Grad-CAM.")
        return None

    if not isinstance(target_layer, nn.Module):
        logger.error("Invalid target_layer provided.")
        return None

    if input_tensor.dim() != 3:
         logger.error(f"Input tensor for Grad-CAM must be 3D [C, H, W], but got shape {input_tensor.shape}")
         return None

    model.eval() # Ensure model is in eval mode

    # --- Define a Forward Function for Captum ---
    def detection_forward_func(batched_inputs: torch.Tensor) -> torch.Tensor:
        """
        Wrapper function for Captum. Runs model inference and extracts the
        classification score for the target detection box and class.
        'batched_inputs' here is expected to have a batch dimension [B, C, H, W] (added by Captum).
        Returns a 1D tensor containing the target score(s).
        """
        with torch.enable_grad(): # Ensure gradients are enabled for captum
            if batched_inputs.dim() == 4 and batched_inputs.shape[0] == 1:
                model_input_list = [batched_inputs.squeeze(0)] # Create list with the single 3D tensor
            elif batched_inputs.dim() == 3:
                model_input_list = [batched_inputs]
            else:
                logger.error(f"Unexpected input dimension ({batched_inputs.dim()}) in detection_forward_func. Expected 4.")
                # Return a dummy 1-element tensor on error
                return torch.tensor([0.0], device=batched_inputs.device)

            outputs = model(model_input_list)

        if outputs and isinstance(outputs, list) and isinstance(outputs[0], dict):
            pred = outputs[0]
            if 'scores' in pred and target_index < len(pred['scores']):
                target_score = pred['scores'][target_index]
                # --- MODIFICATION: Ensure output is at least 1D ---
                if target_score.dim() == 0:
                    return target_score.unsqueeze(0) # Return as [1] tensor
                else:
                    # Should not happen for a single score, but handle defensively
                    return target_score
                # --- END MODIFICATION ---
            else:
                score_len = len(pred.get('scores',[]))
                logger.warning(f"Target index {target_index} out of bounds for scores (len={score_len}) or 'scores' key missing. Returning zero.")
                # Return a dummy 1-element tensor on error
                return torch.tensor([0.0], device=batched_inputs.device)
        else:
             logger.error(f"Unexpected model output format during captum forward: {type(outputs)}. Returning zero.")
             # Return a dummy 1-element tensor on error
             return torch.tensor([0.0], device=batched_inputs.device)


    # --- Initialize and Run Grad-CAM ---
    try:
        layer_gc = LayerGradCam(detection_forward_func, target_layer)

        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)

        # Calculate attributions
        # Target=None should work now because the forward function returns a tensor
        # that (implicitly, by selection within the function) represents the target score.
        # Captum's default behavior for target=None with a single-output tensor from forward_func
        # is usually to attribute to that single output.
        attributions = layer_gc.attribute(
            input_batch,
            target=None, # Keep target=None for now
            relu_attributions=True
        )

        heatmap = attributions.squeeze(0).cpu().detach()
        logger.info(f"Grad-CAM attribution generated for target_index {target_index}, class {target_class_index}.")
        return heatmap

    except IndexError as ie:
         # Catch the specific error if it persists
         logger.error(f"IndexError during Grad-CAM (likely 0-dim tensor issue persisted): {ie}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {e}", exc_info=True)
        return None