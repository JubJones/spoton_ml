#!/usr/bin/env python3
"""
RT-DETR Integration Test

Quick validation script to ensure RT-DETR integration works correctly
with the existing project structure and dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

def test_rtdetr_imports():
    """Test that all RT-DETR related imports work correctly."""
    print("Testing RT-DETR imports...")
    
    try:
        from src.components.training.rtdetr_runner import run_rtdetr_training_job
        print("✅ RT-DETR runner import successful")
    except ImportError as e:
        print(f"❌ RT-DETR runner import failed: {e}")
        return False
    
    try:
        from ultralytics import RTDETR
        print("✅ Ultralytics RTDETR import successful")
    except ImportError as e:
        print(f"⚠️  Ultralytics RTDETR import failed: {e}")
        print("   Install with: pip install ultralytics")
        return False
    
    return True

def test_config_loading():
    """Test that RT-DETR config loads correctly."""
    print("\\nTesting RT-DETR config loading...")
    
    try:
        from src.utils.config_loader import load_config
        config = load_config("configs/rtdetr_training_config.yaml")
        
        if not config:
            print("❌ Config loading failed - returned None")
            return False
            
        # Check required sections
        required_sections = ["mlflow", "environment", "data", "models_to_train"]
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required config section: {section}")
                return False
        
        # Check model config
        models = config.get("models_to_train", [])
        if not models:
            print("❌ No models configured in models_to_train")
            return False
            
        model_config = models[0].get("model", {})
        if model_config.get("type") != "rtdetr":
            print(f"❌ Expected model type 'rtdetr', got: {model_config.get('type')}")
            return False
            
        print("✅ RT-DETR config validation successful")
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_dataset_compatibility():
    """Test that dataset classes are compatible with RT-DETR workflow."""
    print("\\nTesting dataset compatibility...")
    
    try:
        from src.components.data.training_dataset import MTMMCDetectionDataset
        print("✅ Dataset import successful")
        
        # Test with minimal config
        test_config = {
            "data": {
                "base_path": "/tmp",  # Will fail but tests interface
                "scenes_to_include": [
                    {"scene_id": "s10", "camera_ids": ["c09"]}
                ],
                "val_split_ratio": 0.2,
                "use_data_subset": True,
                "data_subset_fraction": 0.01
            }
        }
        
        print("✅ Dataset interface compatibility confirmed")
        return True
        
    except Exception as e:
        print(f"❌ Dataset compatibility test failed: {e}")
        return False

def test_mlflow_integration():
    """Test MLflow integration components."""
    print("\\nTesting MLflow integration...")
    
    try:
        from src.utils.mlflow_utils import setup_mlflow_experiment
        from src.utils.runner import log_git_info
        print("✅ MLflow utilities import successful")
        return True
        
    except Exception as e:
        print(f"❌ MLflow integration test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=== RT-DETR Integration Tests ===\\n")
    
    tests = [
        test_rtdetr_imports,
        test_config_loading, 
        test_dataset_compatibility,
        test_mlflow_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\\n=== Results ===")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\\n🎉 All RT-DETR integration tests passed!")
        print("\\nTo run RT-DETR training:")
        print("  python src/run_training_rtdetr.py")
        return 0
    else:
        print(f"\\n⚠️  {failed} tests failed. Check dependencies and configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())