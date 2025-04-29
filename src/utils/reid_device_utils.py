import logging
import torch

logger = logging.getLogger(__name__)

def get_reid_device_specifier_string(device: torch.device) -> str:
    """
    Converts a torch.device object into a string specifier suitable for BoxMOT backends.
    """
    if not isinstance(device, torch.device):
        logger.warning(f"Input is not a torch.device object ({type(device)}). Attempting fallback.")
        # Try to create a device object from the input if it's a string
        try:
            device = torch.device(str(device))
        except Exception:
             logger.error(f"Could not convert input '{device}' to torch.device. Defaulting to 'cpu'.")
             return 'cpu'

    device_type = device.type
    logger.debug(f"Converting torch.device '{device}' to BoxMOT specifier string.")

    if device_type == 'cuda':
        # BoxMOT expects the device index as a string, or just '0' if index is None (default device)
        device_index = device.index if device.index is not None else 0
        specifier = str(device_index)
        logger.debug(f"CUDA device index {device_index} -> specifier '{specifier}'")
        return specifier
    elif device_type == 'mps':
        specifier = 'mps'
        logger.debug(f"MPS device -> specifier '{specifier}'")
        return specifier
    elif device_type == 'cpu':
        specifier = 'cpu'
        logger.debug(f"CPU device -> specifier '{specifier}'")
        return specifier
    else:
        logger.warning(f"Unsupported torch.device type: '{device_type}'. Defaulting to 'cpu'.")
        return 'cpu'
