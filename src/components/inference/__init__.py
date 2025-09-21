# This file makes the 'inference' directory a Python package.

from .detector import infer_single_image, load_trained_fasterrcnn

__all__ = ["infer_single_image", "load_trained_fasterrcnn"]