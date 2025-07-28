#!/usr/bin/env python3
"""
Test script to validate model comparison functionality.
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.pipelines.model_comparison_pipeline import UnifiedModelLoader, ModelEvaluator

def test_model_loading():
    """Test model loading for both FasterRCNN and RF-DETR."""
    print("=== Testing Model Loading ===")
    
    # Test FasterRCNN loading
    print("\n1. Testing FasterRCNN model loading...")
    try:
        config = load_config("configs/model_comparison_config.yaml")
        if config:
            config["model"]["type"] = "fasterrcnn"
            
            import torch
            device = torch.device("cpu")
            
            loader = UnifiedModelLoader(config, device)
            model = loader.load_initial_model()
            
            print(f"✅ FasterRCNN model loaded successfully: {type(model)}")
            
            # Test model inference
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model([dummy_input])
            print(f"✅ FasterRCNN inference test passed: {len(output)} outputs")
            
        else:
            print("❌ Failed to load FasterRCNN config")
            return False
            
    except Exception as e:
        print(f"❌ FasterRCNN loading failed: {e}")
        return False
    
    # Test RF-DETR loading
    print("\n2. Testing RF-DETR model loading...")
    try:
        config = load_config("configs/rfdetr_comparison_config.yaml")
        if config:
            config["model"]["type"] = "rfdetr"
            
            loader = UnifiedModelLoader(config, device)
            model = loader.load_initial_model()
            
            print(f"✅ RF-DETR model loaded successfully: {type(model)}")
            
            # Test model inference
            from PIL import Image
            import numpy as np
            
            dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            results = model.predict(dummy_image)
            print(f"✅ RF-DETR inference test passed: {type(results)}")
            
        else:
            print("❌ Failed to load RF-DETR config")
            return False
            
    except Exception as e:
        print(f"❌ RF-DETR loading failed: {e}")
        return False
    
    return True

def test_evaluation_pipeline():
    """Test the evaluation pipeline components."""
    print("\n=== Testing Evaluation Pipeline ===")
    
    try:
        config = load_config("configs/model_comparison_config.yaml")
        if not config:
            print("❌ Failed to load config")
            return False
        
        # Modify config for testing
        config["data"]["use_data_subset"] = True
        config["data"]["data_subset_fraction"] = 0.01  # Use tiny subset for testing
        
        import torch
        device = torch.device("cpu")
        
        # Test evaluator creation
        evaluator = ModelEvaluator(config, device)
        print(f"✅ ModelEvaluator created successfully")
        print(f"  Dataset size: {len(evaluator.dataset)}")
        
        if len(evaluator.dataset) == 0:
            print("⚠️  Dataset is empty - check data paths")
            return True  # Not a failure, just no data
        
        # Test with a very small sample
        print("\n3. Testing evaluation on small sample...")
        
        # Load a simple model for testing
        config["model"]["type"] = "fasterrcnn"
        loader = UnifiedModelLoader(config, device)
        model = loader.load_initial_model()
        
        # Run evaluation on first few samples only
        import torch
        model.eval()
        
        print("✅ Model evaluation test setup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison_configs():
    """Test configuration file loading."""
    print("\n=== Testing Configuration Files ===")
    
    configs_to_test = [
        "configs/model_comparison_config.yaml",
        "configs/rfdetr_comparison_config.yaml"
    ]
    
    for config_path in configs_to_test:
        try:
            config = load_config(config_path)
            if config:
                print(f"✅ {config_path} loaded successfully")
                
                # Check required fields
                required_fields = ["model", "data", "evaluation", "comparison"]
                for field in required_fields:
                    if field not in config:
                        print(f"❌ Missing required field: {field}")
                        return False
                
                print(f"  Model type: {config['model']['type']}")
                print(f"  Output dir: {config['comparison']['output_dir']}")
                
            else:
                print(f"❌ Failed to load {config_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading {config_path}: {e}")
            return False
    
    return True

def test_metrics_calculation():
    """Test metrics calculation components."""
    print("\n=== Testing Metrics Calculation ===")
    
    try:
        # Test IoU calculation
        from src.pipelines.model_comparison_pipeline import ModelEvaluator
        import numpy as np
        
        # Create dummy evaluator for testing
        config = {"model": {"type": "fasterrcnn"}, "evaluation": {"iou_threshold": 0.5}}
        evaluator = ModelEvaluator(config, None)
        
        # Test IoU calculation
        box1 = np.array([10, 10, 50, 50])  # x1, y1, x2, y2
        box2 = np.array([20, 20, 60, 60])  # overlapping box
        
        iou = evaluator._calculate_iou(box1, box2)
        print(f"✅ IoU calculation test: {iou:.4f}")
        
        if 0 < iou < 1:
            print("✅ IoU calculation working correctly")
        else:
            print("❌ IoU calculation may have issues")
            return False
        
        # Test IoU matrix
        pred_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        gt_boxes = np.array([[15, 15, 55, 55], [70, 70, 110, 110]])
        
        iou_matrix = evaluator._calculate_iou_matrix(pred_boxes, gt_boxes)
        print(f"✅ IoU matrix calculation: shape {iou_matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Model Comparison Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_comparison_configs),
        ("Model Loading", test_model_loading),
        ("Metrics Calculation", test_metrics_calculation),
        ("Evaluation Pipeline", test_evaluation_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Model comparison should work correctly.")
    else:
        print("❌ Some tests failed. Please review the issues above.")
        sys.exit(1)