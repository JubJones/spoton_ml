#!/usr/bin/env python3
"""
Debug script to inspect RF-DETR training data format and identify issues.
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.components.data.training_dataset import MTMMCDetectionDataset
from src.components.training.runner import get_transform

def inspect_dataset_sample(dataset, idx=0):
    """Inspect a single dataset sample."""
    print(f"\n=== Inspecting Dataset Sample {idx} ===")
    
    # Get sample info
    info = dataset.get_sample_info(idx)
    print(f"Sample info: {info}")
    
    # Get raw sample data
    sample = dataset.samples_split[idx]
    image_path, annotations = sample
    
    print(f"Image path: {image_path}")
    print(f"Annotations: {annotations}")
    
    # Load image to check dimensions
    image = Image.open(image_path)
    width, height = image.size
    print(f"Image dimensions: {width}x{height}")
    
    # Analyze annotations
    print(f"Number of annotations: {len(annotations)}")
    for i, (obj_id, cx, cy, w, h) in enumerate(annotations):
        print(f"  Annotation {i}: obj_id={obj_id}, center=({cx:.1f}, {cy:.1f}), size=({w:.1f}, {h:.1f})")
        
        # Check if coordinates are reasonable
        if cx < 0 or cx > width or cy < 0 or cy > height:
            print(f"    ❌ CENTER OUT OF BOUNDS: center=({cx:.1f}, {cy:.1f}), image_size=({width}, {height})")
        
        if w <= 0 or h <= 0:
            print(f"    ❌ INVALID SIZE: w={w:.1f}, h={h:.1f}")
            
        # Check bounding box bounds
        x_min = cx - w/2
        y_min = cy - h/2
        x_max = cx + w/2
        y_max = cy + h/2
        
        if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
            print(f"    ⚠️  BBOX OUT OF BOUNDS: bbox=({x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f})")
    
    return image_path, annotations, (width, height)

def inspect_ground_truth_file(gt_file_path):
    """Inspect raw ground truth file."""
    print(f"\n=== Inspecting Ground Truth File: {gt_file_path} ===")
    
    if not gt_file_path.exists():
        print("❌ Ground truth file not found!")
        return
    
    lines = []
    with open(gt_file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 10:  # Show first 10 lines
                lines.append(line.strip())
            else:
                break
    
    print(f"First {len(lines)} lines:")
    for i, line in enumerate(lines):
        print(f"  {i+1}: {line}")
    
    # Parse and analyze first few entries
    print("\nParsed entries:")
    for i, line in enumerate(lines[:5]):
        parts = line.split(",")
        if len(parts) >= 6:
            try:
                frame_idx = int(parts[0])
                obj_id = int(parts[1])
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                
                center_x = bb_left + bb_width / 2
                center_y = bb_top + bb_height / 2
                
                print(f"  Entry {i+1}: frame={frame_idx}, obj_id={obj_id}")
                print(f"    Raw bbox: left={bb_left:.1f}, top={bb_top:.1f}, w={bb_width:.1f}, h={bb_height:.1f}")
                print(f"    Center: ({center_x:.1f}, {center_y:.1f})")
                
            except ValueError as e:
                print(f"  Entry {i+1}: Parse error: {e}")

def main():
    """Main debug function."""
    print("=== RF-DETR Data Debug Script ===")
    
    # Load config
    config = load_config("configs/rfdetr_training_config.yaml")
    if not config:
        print("❌ Failed to load config")
        return
    
    # Create dataset
    print("\n=== Creating Dataset ===")
    transforms = get_transform(train=True, config=config)
    dataset = MTMMCDetectionDataset(config=config, mode='train', transforms=transforms)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Scenes: {config['data']['scenes_to_include']}")
    
    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return
    
    # Inspect first few samples
    for i in range(min(3, len(dataset))):
        image_path, annotations, image_size = inspect_dataset_sample(dataset, i)
        
        # Also inspect the raw ground truth file
        if i == 0:
            # Get scene and camera info
            scene_info = dataset.get_sample_info(i)
            if scene_info:
                base_path = Path(config['data']['base_path'])
                scene_path = base_path / "train" / scene_info['scene_id']
                gt_file = scene_path / scene_info['camera_id'] / "gt" / "gt.txt"
                inspect_ground_truth_file(gt_file)
    
    # Test RF-DETR COCO conversion
    print("\n=== Testing RF-DETR COCO Conversion ===")
    try:
        from src.components.training.rfdetr_runner import convert_mtmmc_to_coco_format
        
        # Convert a small subset
        print("Converting dataset to COCO format...")
        coco_path = convert_mtmmc_to_coco_format(dataset, "debug_output", "debug_split")
        print(f"COCO conversion successful: {coco_path}")
        
        # Inspect the generated COCO file
        coco_file = Path(coco_path) / "debug_split" / "_annotations.coco.json"
        if coco_file.exists():
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            print(f"COCO data summary:")
            print(f"  Images: {len(coco_data['images'])}")
            print(f"  Annotations: {len(coco_data['annotations'])}")
            print(f"  Categories: {len(coco_data['categories'])}")
            
            # Show first few annotations
            for i, ann in enumerate(coco_data['annotations'][:5]):
                print(f"  Annotation {i+1}: {ann}")
                
    except Exception as e:
        print(f"❌ COCO conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()