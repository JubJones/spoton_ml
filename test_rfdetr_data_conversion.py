#!/usr/bin/env python3
"""
Test script to validate RF-DETR data conversion and identify potential issues.
"""

import sys
import json
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.components.data.training_dataset import MTMMCDetectionDataset
from src.components.training.runner import get_transform
from src.components.training.rfdetr_runner import convert_mtmmc_to_coco_format

def test_data_conversion():
    """Test the RF-DETR data conversion process."""
    print("=== Testing RF-DETR Data Conversion ===")
    
    # Load config
    config = load_config("configs/rfdetr_training_config.yaml")
    if not config:
        print("❌ Failed to load config")
        return False
    
    # Create a small dataset for testing
    print("\n1. Creating test dataset...")
    transforms = get_transform(train=True, config=config)
    dataset = MTMMCDetectionDataset(config=config, mode='train', transforms=transforms)
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return False
    
    # Test first few samples
    print("\n2. Testing sample data...")
    for i in range(min(3, len(dataset))):
        sample = dataset.samples_split[i]
        image_path, annotations = sample
        
        print(f"\nSample {i}:")
        print(f"  Image: {image_path}")
        print(f"  Annotations: {len(annotations)} objects")
        
        # Check annotation format
        for j, (obj_id, cx, cy, w, h) in enumerate(annotations):
            print(f"    Object {j}: id={obj_id}, center=({cx:.1f}, {cy:.1f}), size=({w:.1f}, {h:.1f})")
            
            # Validate coordinates
            if cx < 0 or cy < 0 or w <= 0 or h <= 0:
                print(f"    ❌ Invalid coordinates detected!")
                return False
    
    # Test COCO conversion
    print("\n3. Testing COCO conversion...")
    try:
        coco_path = convert_mtmmc_to_coco_format(dataset, "test_output", "test_split")
        print(f"✅ COCO conversion successful: {coco_path}")
        
        # Validate COCO file
        coco_file = Path(coco_path) / "test_split" / "_annotations.coco.json"
        if coco_file.exists():
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            print(f"\n4. Validating COCO data...")
            print(f"  Images: {len(coco_data['images'])}")
            print(f"  Annotations: {len(coco_data['annotations'])}")
            print(f"  Categories: {len(coco_data['categories'])}")
            
            # Check categories
            for cat in coco_data['categories']:
                print(f"  Category: id={cat['id']}, name='{cat['name']}'")
                if cat['id'] != 0:
                    print(f"    ❌ Category ID should be 0, got {cat['id']}")
                    return False
            
            # Check annotations
            for i, ann in enumerate(coco_data['annotations'][:5]):
                print(f"  Annotation {i}: category_id={ann['category_id']}, bbox={ann['bbox']}")
                
                # Validate class index
                if ann['category_id'] != 0:
                    print(f"    ❌ Category ID should be 0, got {ann['category_id']}")
                    return False
                
                # Validate bbox
                bbox = ann['bbox']
                if len(bbox) != 4 or any(v < 0 for v in bbox):
                    print(f"    ❌ Invalid bbox: {bbox}")
                    return False
                
                # Validate bbox size
                if bbox[2] <= 0 or bbox[3] <= 0:
                    print(f"    ❌ Invalid bbox size: width={bbox[2]}, height={bbox[3]}")
                    return False
            
            print("✅ COCO data validation passed!")
            return True
            
        else:
            print(f"❌ COCO file not found: {coco_file}")
            return False
            
    except Exception as e:
        print(f"❌ COCO conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tensor_creation():
    """Test tensor creation with the converted data."""
    print("\n=== Testing Tensor Creation ===")
    
    # Create dummy data similar to what RF-DETR would process
    try:
        # Test class indices
        class_indices = torch.tensor([0, 0, 0], dtype=torch.int64)  # 0-based indices
        print(f"Class indices: {class_indices}")
        
        # Test bounding boxes
        bbox_data = torch.tensor([
            [100, 100, 50, 80],   # [x_min, y_min, width, height]
            [200, 150, 60, 90],
            [300, 200, 70, 100]
        ], dtype=torch.float32)
        print(f"Bounding boxes: {bbox_data}")
        
        # Test conversion to center format (what RF-DETR expects)
        # Convert from [x_min, y_min, width, height] to [center_x, center_y, width, height]
        center_x = bbox_data[:, 0] + bbox_data[:, 2] / 2
        center_y = bbox_data[:, 1] + bbox_data[:, 3] / 2
        center_format = torch.stack([center_x, center_y, bbox_data[:, 2], bbox_data[:, 3]], dim=1)
        print(f"Center format: {center_format}")
        
        # Test normalization (what RF-DETR does internally)
        img_width, img_height = 640, 480
        normalized_boxes = center_format.clone()
        normalized_boxes[:, 0] /= img_width   # normalize center_x
        normalized_boxes[:, 1] /= img_height  # normalize center_y
        normalized_boxes[:, 2] /= img_width   # normalize width
        normalized_boxes[:, 3] /= img_height  # normalize height
        print(f"Normalized boxes: {normalized_boxes}")
        
        # Check if normalized coordinates are in valid range [0, 1]
        if torch.any(normalized_boxes < 0) or torch.any(normalized_boxes > 1):
            print("❌ Normalized coordinates out of [0, 1] range!")
            return False
        
        print("✅ Tensor creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tensor creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("RF-DETR Data Conversion Test")
    print("=" * 50)
    
    success1 = test_data_conversion()
    success2 = test_tensor_creation()
    
    if success1 and success2:
        print("\n✅ All tests passed! RF-DETR data conversion should work correctly.")
    else:
        print("\n❌ Some tests failed. Please review the issues above.")
        sys.exit(1)