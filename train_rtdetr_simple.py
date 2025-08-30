#!/usr/bin/env python3
"""
Simple RT-DETR Training Script

A minimal script to train RT-DETR on MTMMC dataset using Ultralytics.
Based on the training syntax from RTDETR_TRAINING.md

Usage:
    python train_rtdetr_simple.py
"""

import os
import json
from pathlib import Path
from ultralytics import RTDETR

# ===== Configuration =====
# Update this to point to your MTMMC dataset
BASE_PATH = "D:/MTMMC"  # Change this to your dataset path

# Training settings
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = "0"  # GPU device (use "cpu" for CPU training)

# Dataset settings - using factory environment, scene s10
SELECTED_ENVIRONMENT = "factory"  # "factory" or "campus"
SCENE_ID = "s10"
CAMERA_IDS = ["c09", "c12", "c13", "c16"]  # Factory scene cameras
MAX_FRAMES = 500  # Max frames per camera (-1 for all)

# Output directory
OUTPUT_DIR = Path("rtdetr_training_output")
OUTPUT_DIR.mkdir(exist_ok=True)

def create_coco_format_data():
    """Convert MTMMC data to COCO format for RT-DETR training"""
    print("Creating COCO format dataset...")
    
    base_path = Path(BASE_PATH)
    train_path = base_path / "train" / SCENE_ID
    
    if not train_path.exists():
        print(f"Error: Dataset path not found: {train_path}")
        print("Please update BASE_PATH in the script")
        exit(1)
    
    # Create output directories
    images_dir = OUTPUT_DIR / "images"
    labels_dir = OUTPUT_DIR / "labels"  
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    annotations = []
    image_id = 0
    
    print(f"Processing scene {SCENE_ID} from {SELECTED_ENVIRONMENT} environment...")
    
    for camera_id in CAMERA_IDS:
        camera_path = train_path / camera_id
        rgb_path = camera_path / "rgb"
        gt_path = camera_path / "gt" / "gt.txt"
        
        if not rgb_path.exists() or not gt_path.exists():
            print(f"Warning: Skipping camera {camera_id} - missing data")
            continue
            
        print(f"Processing camera {camera_id}...")
        
        # Read ground truth
        gt_data = {}
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    frame_id = int(parts[0])
                    obj_id = int(parts[1])
                    x, y, w, h = map(float, parts[2:6])
                    
                    if frame_id not in gt_data:
                        gt_data[frame_id] = []
                    gt_data[frame_id].append((obj_id, x, y, w, h))
        
        # Process images
        image_files = sorted([f for f in rgb_path.glob("*.jpg")])
        if MAX_FRAMES > 0:
            image_files = image_files[:MAX_FRAMES]
            
        for img_file in image_files:
            frame_id = int(img_file.stem)
            
            # Copy image to output directory
            new_img_name = f"{SCENE_ID}_{camera_id}_{frame_id:06d}.jpg"
            new_img_path = images_dir / new_img_name
            
            # Copy image file
            import shutil
            shutil.copy2(img_file, new_img_path)
            
            # Get image dimensions
            from PIL import Image
            with Image.open(img_file) as img:
                img_width, img_height = img.size
            
            # Create YOLO format labels if annotations exist for this frame
            if frame_id in gt_data:
                label_file = labels_dir / f"{SCENE_ID}_{camera_id}_{frame_id:06d}.txt"
                with open(label_file, 'w') as f:
                    for obj_id, x, y, w, h in gt_data[frame_id]:
                        # Convert to YOLO format (normalized center coordinates)
                        # x,y are top-left coordinates, need to convert to center
                        center_x = (x + w/2) / img_width
                        center_y = (y + h/2) / img_height
                        norm_w = w / img_width
                        norm_h = h / img_height
                        
                        # Clip coordinates to valid range [0, 1]
                        center_x = max(0.0, min(1.0, center_x))
                        center_y = max(0.0, min(1.0, center_y))
                        norm_w = max(0.0, min(1.0, norm_w))
                        norm_h = max(0.0, min(1.0, norm_h))
                        
                        # Skip invalid boxes
                        if norm_w <= 0 or norm_h <= 0:
                            continue
                            
                        # Class 0 for person
                        f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
            image_id += 1
            
            if image_id % 100 == 0:
                print(f"Processed {image_id} images...")

    print(f"Dataset creation complete! Processed {image_id} images.")
    return images_dir, labels_dir

def create_dataset_yaml():
    """Create dataset YAML file for RT-DETR"""
    yaml_content = f"""
# MTMMC Person Detection Dataset
path: {OUTPUT_DIR.absolute()}
train: images
val: images  # Using same for validation (you can split if needed)

# Classes
nc: 1  # number of classes
names: ['person']  # class names
"""
    
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    
    print(f"Created dataset YAML: {yaml_path}")
    return yaml_path

def main():
    """Main training function"""
    print("=" * 50)
    print("Simple RT-DETR Training Script")
    print("=" * 50)
    
    # Create dataset
    images_dir, labels_dir = create_coco_format_data()
    dataset_yaml = create_dataset_yaml()
    
    # Initialize RT-DETR model
    print("\nInitializing RT-DETR model...")
    model = RTDETR("rtdetr-l.pt")  # Load pretrained RT-DETR Large model
    
    # Start training
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    
    # Disable MLflow to avoid Windows path issues
    os.environ['MLFLOW_TRACKING_URI'] = ''
    
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            project=str(OUTPUT_DIR),
            name="rtdetr_experiment",
            save=True,
            plots=True,
            val=True,
            cache=False,  # Set to True if you have enough RAM
            workers=4,
            patience=50,  # Early stopping patience
            lr0=0.001,  # Initial learning rate
            weight_decay=0.0005,
            warmup_epochs=3,
        )
        
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print(f"Results saved in: {OUTPUT_DIR}/rtdetr_experiment")
        print("=" * 50)
        
        # Print some basic results
        if hasattr(results, 'maps'):
            print(f"Final mAP@0.5: {results.maps[0]:.4f}")
            print(f"Final mAP@0.5:0.95: {results.maps[1]:.4f}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Make sure you have:")
        print("1. Correct dataset path")
        print("2. Ultralytics installed: pip install ultralytics")
        print("3. Sufficient GPU memory or use CPU")
        raise

if __name__ == "__main__":
    main()