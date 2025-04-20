import logging

import torch

logger = logging.getLogger(__name__)


def get_selected_device(requested_device: str = "auto") -> torch.device:
    """
    Gets the torch.device based on the requested setting and availability.
    Prioritizes CUDA > MPS > CPU for "auto".
    """
    req_device_lower = requested_device.lower()
    logger.info(f"--- Determining Device (Requested: '{requested_device}') ---")

    if req_device_lower.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                # Allow specific device index like "cuda:0", "cuda:1"
                device = torch.device(req_device_lower)
                device_name = torch.cuda.get_device_name(device)
                logger.info(f"Selected device: {device} ({device_name})")
                # Test tensor creation
                _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
                return device
            except Exception as e:
                logger.warning(
                    f"Requested CUDA device '{requested_device}' not valid, "
                    f"available, or test failed ({e}). Falling back to CPU."
                )
                return torch.device("cpu")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    elif req_device_lower == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                device = torch.device("mps")
                # Test tensor creation on MPS
                _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
                logger.info(f"Selected device: {device}")
                return device
            except Exception as e:
                logger.warning(f"MPS available but test failed ({e}). Falling back to CPU.")
                return torch.device("cpu")
        else:
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    elif req_device_lower == "cpu":
        logger.info("Selected device: cpu")
        return torch.device("cpu")

    elif req_device_lower == "auto":
        logger.info("Attempting auto-detection: CUDA > MPS > CPU")
        # 1. Try CUDA
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda")  # Default CUDA device
                device_name = torch.cuda.get_device_name(device)
                # Test tensor creation
                _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
                logger.info(f"Auto-selected device: {device} ({device_name})")
                return device
            except Exception as e:
                logger.warning(f"CUDA available but failed to initialize/test ({e}). Checking MPS.")
        else:
            logger.info("CUDA not available.")

        # 2. Try MPS (if CUDA not available or failed)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                device = torch.device("mps")
                # Test tensor creation on MPS
                _ = torch.tensor([1.0], device=device) + torch.tensor([1.0], device=device)
                logger.info(f"Auto-selected device: {device}")
                return device
            except Exception as e:
                logger.warning(f"MPS available but test failed ({e}). Falling back to CPU.")
        else:
            logger.info("MPS not available.")

        # 3. Fallback to CPU
        device = torch.device("cpu")
        logger.info(f"Auto-selected device: {device}")
        return device

    else:
        logger.warning(
            f"Unknown device requested: '{requested_device}'. Falling back to CPU."
        )
        return torch.device("cpu")
