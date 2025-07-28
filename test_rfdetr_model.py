#!/usr/bin/env python3
"""
Quick test script to verify RF-DETR model loading works.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.components.training.rfdetr_runner import get_rfdetr_model
    import torch
    
    # Test configuration
    config = {
        "model": {
            "type": "rfdetr",
            "size": "base",
            "num_classes": 2
        }
    }
    
    print("Testing RF-DETR model loading...")
    
    # Try to load the model
    model = get_rfdetr_model(config)
    print(f"✓ Successfully loaded RF-DETR model: {type(model)}")
    
    # Test inference on a dummy image
    from PIL import Image
    import numpy as np
    
    dummy_image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    print("Testing model inference...")
    
    results = model.predict(dummy_image)
    print(f"✓ Model inference successful: {type(results)}")
    
    print("✓ RF-DETR model test passed!")
    
except Exception as e:
    print(f"✗ RF-DETR model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)