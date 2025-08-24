"""
RF-DETR Inference Debugging Script

This script helps diagnose RF-DETR inference issues by testing:
1. Model loading and initialization
2. Checkpoint loading 
3. Inference on sample images
4. Output format analysis
5. Class mapping verification

Usage:
    python src/debug_rfdetr_inference.py
"""

import logging
import sys
import traceback
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logging
from src.components.training.rfdetr_runner import get_rfdetr_model

# Setup logging
logger = logging.getLogger(__name__)
setup_logging("debug_rfdetr", PROJECT_ROOT / "logs")

def test_rfdetr_model_loading():
    """Test RF-DETR model loading with and without checkpoint."""
    logger.info("=== Testing RF-DETR Model Loading ===")
    
    config_path = "configs/rfdetr_detection_analysis_config.yaml"
    config = load_config(config_path)
    
    if not config:
        logger.error(f"Failed to load config from {config_path}")
        return False
    
    try:
        # Test 1: Load model architecture
        logger.info("Test 1: Loading RF-DETR model architecture...")
        model = get_rfdetr_model(config)
        logger.info(f"✅ Model loaded successfully: {type(model)}")
        logger.info(f"   Model has attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}...")
        
        # Test 2: Check model structure
        logger.info("Test 2: Analyzing model structure...")
        if hasattr(model, 'model'):
            logger.info(f"   model.model type: {type(model.model)}")
            if hasattr(model.model, 'model'):
                logger.info(f"   model.model.model type: {type(model.model.model)}")
        
        # Test 3: Check class configuration
        logger.info("Test 3: Checking class configuration...")
        if hasattr(model, 'class_names'):
            logger.info(f"   model.class_names: {model.class_names}")
        if hasattr(model.model, 'class_names'):
            logger.info(f"   model.model.class_names: {model.model.class_names}")
        if hasattr(model.model, 'num_classes'):
            logger.info(f"   model.model.num_classes: {model.model.num_classes}")
            
        # Test 4: Check predict method
        logger.info("Test 4: Checking predict method...")
        if hasattr(model, 'predict'):
            logger.info(f"   ✅ model.predict method exists")
        else:
            logger.error(f"   ❌ model.predict method missing")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_checkpoint_loading():
    """Test checkpoint loading specifically."""
    logger.info("=== Testing RF-DETR Checkpoint Loading ===")
    
    config_path = "configs/rfdetr_detection_analysis_config.yaml"
    config = load_config(config_path)
    checkpoint_path = config.get("local_model_path")
    
    if not checkpoint_path or not Path(checkpoint_path).exists():
        logger.warning(f"Checkpoint not found at: {checkpoint_path}")
        logger.info("Testing with pre-trained model instead...")
        return True
    
    try:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        logger.info(f"Checkpoint type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                logger.info(f"Model state type: {type(model_state)}")
                if isinstance(model_state, dict):
                    sample_keys = list(model_state.keys())[:5]
                    logger.info(f"Sample model state keys: {sample_keys}")
                    
            if 'args' in checkpoint:
                args = checkpoint['args']
                logger.info(f"Args type: {type(args)}")
                if hasattr(args, '__dict__'):
                    logger.info(f"Args attributes: {list(vars(args).keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Checkpoint loading failed: {e}")
        logger.error(traceback.format_exc())
        return False

def create_test_image():
    """Create a simple test image."""
    # Create a 640x640 RGB test image with some patterns
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Add some rectangles to simulate objects
    cv2.rectangle(img, (100, 100), (200, 300), (255, 100, 100), -1)  # Person-like rectangle
    cv2.rectangle(img, (400, 200), (500, 400), (100, 255, 100), -1)  # Another rectangle
    
    # Convert to PIL Image
    return Image.fromarray(img)

def test_rfdetr_inference():
    """Test RF-DETR inference on a sample image."""
    logger.info("=== Testing RF-DETR Inference ===")
    
    config_path = "configs/rfdetr_detection_analysis_config.yaml"
    config = load_config(config_path)
    
    try:
        # Load model
        logger.info("Loading RF-DETR model...")
        model = get_rfdetr_model(config)
        
        # Create test image
        logger.info("Creating test image...")
        test_image = create_test_image()
        logger.info(f"Test image size: {test_image.size}, mode: {test_image.mode}")
        
        # Test different confidence thresholds
        confidence_thresholds = [0.1, 0.3, 0.5, 0.7]
        
        for threshold in confidence_thresholds:
            logger.info(f"Testing inference with confidence threshold: {threshold}")
            
            try:
                # Run inference
                results = model.predict(test_image, threshold=threshold)
                
                logger.info(f"  Results type: {type(results)}")
                logger.info(f"  Results: {results}")
                
                if results is not None:
                    if hasattr(results, '__len__'):
                        logger.info(f"  Number of detections: {len(results)}")
                    
                    if hasattr(results, 'xyxy'):
                        logger.info(f"  Boxes shape: {results.xyxy.shape}")
                        logger.info(f"  Sample boxes: {results.xyxy[:3] if len(results.xyxy) > 0 else 'No boxes'}")
                    
                    if hasattr(results, 'confidence'):
                        logger.info(f"  Confidence shape: {results.confidence.shape}")
                        logger.info(f"  Sample confidences: {results.confidence[:3] if len(results.confidence) > 0 else 'No confidences'}")
                        
                    if hasattr(results, 'class_id'):
                        logger.info(f"  Class ID shape: {results.class_id.shape}")
                        logger.info(f"  Sample class IDs: {results.class_id[:3] if len(results.class_id) > 0 else 'No class IDs'}")
                        logger.info(f"  Unique classes detected: {np.unique(results.class_id) if len(results.class_id) > 0 else 'None'}")
                else:
                    logger.warning(f"  No results returned for threshold {threshold}")
                    
            except Exception as inference_error:
                logger.error(f"  ❌ Inference failed at threshold {threshold}: {inference_error}")
                logger.error(f"  Error details: {traceback.format_exc()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_class_mapping():
    """Test class mapping and verification."""
    logger.info("=== Testing RF-DETR Class Mapping ===")
    
    config_path = "configs/rfdetr_detection_analysis_config.yaml"
    config = load_config(config_path)
    
    try:
        model = get_rfdetr_model(config)
        
        # Check class names property
        logger.info("Checking class names...")
        if hasattr(model, 'class_names'):
            class_names = model.class_names
            logger.info(f"model.class_names: {class_names}")
            logger.info(f"class_names type: {type(class_names)}")
            
            if isinstance(class_names, dict):
                logger.info(f"Class mapping: {class_names}")
                person_classes = [k for k, v in class_names.items() if 'person' in v.lower()]
                logger.info(f"Person class IDs: {person_classes}")
            elif isinstance(class_names, list):
                logger.info(f"Class list: {class_names}")
                person_indices = [i for i, name in enumerate(class_names) if 'person' in name.lower()]
                logger.info(f"Person class indices: {person_indices}")
        
        # Check model configuration
        logger.info("Checking model configuration...")
        if hasattr(model, 'model_config'):
            logger.info(f"model_config: {model.model_config}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Class mapping test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all RF-DETR diagnostic tests."""
    logger.info("🔍 Starting RF-DETR Diagnostic Tests")
    
    tests = [
        ("Model Loading", test_rfdetr_model_loading),
        ("Checkpoint Loading", test_checkpoint_loading),
        ("Inference", test_rfdetr_inference),
        ("Class Mapping", test_class_mapping)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{test_name} Test: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name} Test: ❌ FAILED with exception: {e}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! RF-DETR should work correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
        logger.info("Common issues:")
        logger.info("1. Checkpoint format mismatch")
        logger.info("2. Class configuration problems")
        logger.info("3. Model architecture incompatibility")
        logger.info("4. Missing dependencies")

if __name__ == "__main__":
    main()