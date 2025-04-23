import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Sets the seed for random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set seed {seed} for random, numpy, torch, and CUDA.")
    else:
        logger.info(f"Set seed {seed} for random, numpy, and torch (CUDA not available).")