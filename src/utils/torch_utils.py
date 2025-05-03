import torch
from typing import List, Tuple, Dict, Any, Optional
import logging

def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function for object detection tasks using PyTorch models.
    Handles batches where images might be Tensors or T.Image containers.
    """
    images = []
    targets = []
    for item in batch:
        # Ensure image is a plain tensor if it's a T.Image container
        image_data = item[0]
        if not isinstance(image_data, torch.Tensor):
            # Assuming item[0] might be T.Image or similar if transforms are used
            # Convert it back to a tensor if necessary (depends on exact transform output)
            # This might require specific handling based on the transform library (e.g., torchvision v2)
            # For safety, let's assume transforms output tensors directly if using standard Compose
            # Or if using T.Image, models might handle it, but explicitly converting is safer.
             if hasattr(image_data, 'to_tensor'): # Hypothetical method
                 images.append(image_data.to_tensor())
             elif isinstance(image_data, torch.Tensor): # Should already be tensor
                 images.append(image_data)
             else:
                 # Fallback or raise error if type is unexpected
                 # For now, just append assuming it's usable tensor-like
                 images.append(image_data)
                 # logger.warning(f"Unexpected image type in collate_fn: {type(image_data)}")
        else:
            images.append(image_data)

        targets.append(item[1])

    # Images are typically expected as a list by detection models like FasterRCNN
    # Targets must remain as a list of dictionaries.
    return images, targets

def get_optimizer(name: str, params: List[torch.Tensor], lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Factory function to create an optimizer."""
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        # SGD often needs momentum specified
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        logging.warning(f"Unsupported optimizer '{name}'. Defaulting to AdamW.")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def get_lr_scheduler(name: str, optimizer: torch.optim.Optimizer, **kwargs) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Factory function to create a learning rate scheduler."""
    name = name.lower()
    if name == "steplr":
        step_size = kwargs.get('step_size', 5)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "cosineannealinglr":
        T_max = kwargs.get('T_max', 50) # Number of iterations for the first restart
        eta_min = kwargs.get('eta_min', 0) # Minimum learning rate
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    # Add other schedulers like MultiStepLR, ReduceLROnPlateau etc. if needed
    else:
        logging.warning(f"Unsupported LR scheduler '{name}'. No scheduler will be used.")
        return None