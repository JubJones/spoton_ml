"""
RF-DETR Training Runner
This file contains the runner function for RF-DETR training.
It adapts the MTMMC dataset to RF-DETR's expected COCO format.
"""

import logging
import traceback
import time
import json
import shutil
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import torch
import mlflow

from src.components.data.training_dataset import MTMMCDetectionDataset

logger = logging.getLogger(__name__)


def _log_params_recursive(params: Dict[str, Any], prefix: str = "") -> None:
    """
    Log parameters recursively to MLflow, flattening nested dictionaries.
    """
    for key, value in params.items():
        param_name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _log_params_recursive(value, param_name)
        else:
            # Convert to string and truncate if too long
            str_value = str(value)
            if len(str_value) > 250:
                str_value = str_value[:247] + "..."
            mlflow.log_param(param_name, str_value)


def _log_git_info() -> None:
    """
    Log git information to MLflow.
    """
    try:
        # Get git commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.STDOUT
        ).decode().strip()
        mlflow.set_tag("git.commit", commit_hash)
        
        # Get git branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.STDOUT
        ).decode().strip()
        mlflow.set_tag("git.branch", branch)
        
        # Get git status
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.STDOUT
        ).decode().strip()
        mlflow.set_tag("git.is_dirty", "true" if status else "false")
        
    except subprocess.CalledProcessError:
        logger.warning("Failed to get git information")
    except Exception as e:
        logger.warning(f"Error getting git info: {e}")


def get_rfdetr_model(config: Dict[str, Any]):
    """Loads an RF-DETR model based on configuration."""
    model_config = config["model"]
    model_size = model_config.get("size", "base").lower()
    
    # Import RF-DETR classes
    from rfdetr import RFDETRBase, RFDETRLarge, RFDETRSmall, RFDETRMedium, RFDETRNano
    
    # Map model size to class
    model_classes = {
        "base": RFDETRBase,
        "large": RFDETRLarge,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "nano": RFDETRNano
    }
    
    if model_size not in model_classes:
        raise ValueError(f"Invalid RF-DETR model size: {model_size}. Available: {list(model_classes.keys())}")
    
    model_class = model_classes[model_size]
    
    # Get model configuration parameters
    model_kwargs = {}
    if "num_classes" in model_config:
        model_kwargs["num_classes"] = model_config["num_classes"]
    
    logger.info(f"Loading RF-DETR {model_size} model with config: {model_kwargs}")
    
    # Create model instance
    model = model_class(**model_kwargs)
    
    return model


def validate_coco_format(coco_data: Dict[str, Any], split_name: str) -> bool:
    """
    Validate that the COCO format data structure is correct.
    
    Args:
        coco_data: The COCO format dictionary
        split_name: Name of the split for logging context
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["info", "images", "annotations", "categories"]
    
    for field in required_fields:
        if field not in coco_data:
            logger.error(f"COCO format validation failed for {split_name}: Missing required field '{field}'")
            return False
    
    # Validate info field
    info_required = ["description", "version", "year", "contributor", "date_created"]
    for field in info_required:
        if field not in coco_data["info"]:
            logger.warning(f"COCO format info field missing recommended field '{field}' for {split_name}")
    
    # Validate categories
    if not coco_data["categories"]:
        logger.error(f"COCO format validation failed for {split_name}: Categories cannot be empty")
        return False
    
    # Validate that each category has required fields
    for cat in coco_data["categories"]:
        if "id" not in cat or "name" not in cat:
            logger.error(f"COCO format validation failed for {split_name}: Category missing id or name")
            return False
    
    # Validate images
    for img in coco_data["images"]:
        img_required = ["id", "file_name", "width", "height"]
        for field in img_required:
            if field not in img:
                logger.error(f"COCO format validation failed for {split_name}: Image missing required field '{field}'")
                return False
    
    # Validate annotations
    for ann in coco_data["annotations"]:
        ann_required = ["id", "image_id", "category_id", "bbox", "area"]
        for field in ann_required:
            if field not in ann:
                logger.error(f"COCO format validation failed for {split_name}: Annotation missing required field '{field}'")
                return False
        
        # Validate bbox format
        if not isinstance(ann["bbox"], list) or len(ann["bbox"]) != 4:
            logger.error(f"COCO format validation failed for {split_name}: Invalid bbox format")
            return False
    
    logger.info(f"COCO format validation passed for {split_name}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    return True


def convert_mtmmc_to_coco_format(dataset: MTMMCDetectionDataset, output_dir: Path, split_name: str, debug_mode: bool = False):
    """
    Convert MTMMC dataset to COCO format required by RF-DETR.
    
    Args:
        dataset: MTMMCDetectionDataset instance
        output_dir: Directory to save COCO format data
        split_name: "train" or "val"
        debug_mode: Enable debug logging for box processing
    """
    # Create output directories
    images_dir = output_dir / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize COCO format structure with required fields
    # CRITICAL: RF-DETR expects 0-based class indices
    # The 'info' field is required by pycocotools
    current_time = datetime.now().strftime("%Y-%m-%d")
    coco_data = {
        "info": {
            "description": f"MTMMC Dataset converted to COCO format for RF-DETR training - {split_name} split",
            "url": "https://github.com/ergyun/MTMMC",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "MTMMC-to-COCO converter",
            "date_created": current_time
        },
        "licenses": [
            {
                "id": 1,
                "name": "Custom License",
                "url": "https://github.com/ergyun/MTMMC"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 0,  # 0-based indexing for RF-DETR
                "name": "person",
                "supercategory": "person"
            }
        ]
    }
    
    annotation_id = 1
    total_original_annotations = 0
    total_skipped_annotations = 0
    
    logger.info(f"Converting {len(dataset)} samples to COCO format for {split_name}")
    
    for idx in range(len(dataset)):
        # Get sample without transforms for annotation conversion
        sample = dataset.samples_split[idx]
        image_path, annotations = sample
        
        # Load image to get dimensions
        image = Image.open(image_path)
        width, height = image.size
        
        # Copy image to output directory
        image_filename = f"{split_name}_{idx:06d}.jpg"
        output_image_path = images_dir / image_filename
        
        # Convert to RGB if needed and save
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(output_image_path)
        
        # Add image info with license reference
        image_info = {
            "id": idx,
            "file_name": image_filename,
            "width": width,
            "height": height,
            "license": 1,  # Reference to license in licenses array
            "flickr_url": "",
            "coco_url": "",
            "date_captured": f"{current_time} 12:00:00"
        }
        coco_data["images"].append(image_info)
        
        # Convert annotations: MTMMC format → COCO format with comprehensive validation
        image_annotations = len(annotations)
        image_skipped = 0
        total_original_annotations += image_annotations
        
        for obj_id, cx, cy, w, h in annotations:
            # MTMMC annotations: (obj_id, center_x, center_y, box_width, box_height) in PIXEL coordinates
            # RF-DETR expects: COCO format [x_min, y_min, width, height] in PIXEL coordinates
            #                  But the model internally expects NORMALIZED coordinates [0,1]
            
            # Debug logging for problematic boxes
            if debug_mode:
                logger.info(f"Processing annotation: obj_id={obj_id}, center=({cx:.1f}, {cy:.1f}), size=({w:.1f}, {h:.1f}), image_size=({width}, {height})")
            
            # Validate input coordinates are reasonable (should be pixel coordinates)
            if w <= 0 or h <= 0:
                logger.warning(f"Skipping invalid box: negative/zero size=({w:.1f}, {h:.1f})")
                image_skipped += 1
                continue
            
            # Calculate box boundaries
            x_min = cx - w / 2
            y_min = cy - h / 2
            x_max = cx + w / 2
            y_max = cy + h / 2
            
            # Debug original boundaries
            if debug_mode:
                logger.info(f"Original boundaries: x_min={x_min:.1f}, y_min={y_min:.1f}, x_max={x_max:.1f}, y_max={y_max:.1f}")
            
            # Check if the box has any overlap with the image (more lenient validation)
            if x_max <= 0 or x_min >= width or y_max <= 0 or y_min >= height:
                logger.warning(f"Skipping invalid box: no overlap with image. center=({cx:.1f}, {cy:.1f}), size=({w:.1f}, {h:.1f}), image_size=({width}, {height})")
                image_skipped += 1
                continue
            
            # Clamp to image bounds - FIX: Use proper x_max/y_max calculation
            x_min_clamped = max(0, x_min)
            y_min_clamped = max(0, y_min)
            x_max_clamped = min(width, x_max)  # Fixed: was using cx + w / 2
            y_max_clamped = min(height, y_max)  # Fixed: was using cy + h / 2
            
            # Recalculate width/height after clamping
            box_width = x_max_clamped - x_min_clamped
            box_height = y_max_clamped - y_min_clamped
            
            # Debug clamped boundaries
            if debug_mode:
                logger.info(f"Clamped boundaries: x_min={x_min_clamped:.1f}, y_min={y_min_clamped:.1f}, x_max={x_max_clamped:.1f}, y_max={y_max_clamped:.1f}")
                logger.info(f"Final box: width={box_width:.1f}, height={box_height:.1f}")
            
            # Skip invalid boxes after clamping - add minimum size threshold
            MIN_BOX_SIZE = 1.0  # Minimum box size in pixels
            if box_width < MIN_BOX_SIZE or box_height < MIN_BOX_SIZE:
                logger.warning(f"Skipping invalid box after clipping: bbox=({x_min_clamped:.1f}, {y_min_clamped:.1f}, {box_width:.1f}, {box_height:.1f}), too small (min={MIN_BOX_SIZE})")
                image_skipped += 1
                continue
            
            # Create COCO annotation (RF-DETR will handle normalization internally)
            # CRITICAL: RF-DETR expects 0-based class indices, not 1-based COCO indices
            annotation = {
                "id": annotation_id,
                "image_id": idx,
                "category_id": 0,  # person class (0-based indexing for RF-DETR)
                "bbox": [x_min_clamped, y_min_clamped, box_width, box_height],  # COCO format: [x_min, y_min, width, height]
                "area": box_width * box_height,
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1
    
        # Log summary for this image
        image_kept = image_annotations - image_skipped
        total_skipped_annotations += image_skipped
        if image_skipped > 0:
            logger.info(f"Image {idx}: {image_kept}/{image_annotations} annotations kept, {image_skipped} skipped")
    
    # Validate COCO format before saving
    if not validate_coco_format(coco_data, split_name):
        # Log debug information for troubleshooting
        logger.error(f"COCO format validation failed for {split_name}")
        logger.error(f"Dataset structure keys: {list(coco_data.keys())}")
        if "info" in coco_data:
            logger.error(f"Info structure: {coco_data['info']}")
        raise ValueError(f"Generated COCO format is invalid for {split_name} split")
    
    # Save COCO annotations
    annotations_file = output_dir / split_name / "_annotations.coco.json"
    with open(annotations_file, "w") as f:
        json.dump(coco_data, f, indent=2)
    
    total_kept = len(coco_data['annotations'])
    logger.info(f"Saved {len(coco_data['images'])} images and {total_kept} annotations to {annotations_file}")
    
    # Calculate overall skip rate
    if total_original_annotations > 0:
        skip_rate = (total_skipped_annotations / total_original_annotations) * 100
        logger.info(f"Overall annotation processing: {total_kept} kept, {total_skipped_annotations} skipped ({skip_rate:.1f}% skip rate)")
    
    return str(output_dir)


def run_single_rfdetr_training_job(
    run_config: Dict[str, Any], device: torch.device, project_root: Path
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Executes a single RF-DETR training job based on the provided configuration.
    """
    active_run = mlflow.active_run()
    if not active_run:
        logger.critical("run_single_rfdetr_training_job called without an active MLflow run!")
        return "FAILED", None
    
    run_id = active_run.info.run_id
    job_status = "FAILED"
    final_metrics = None
    
    model_config = run_config["model"]
    training_config = run_config["training"]
    data_config = run_config["data"]
    model_type = model_config.get("type", "rfdetr").lower()
    run_name_tag = model_config.get("name_tag", f"rfdetr_training_{run_id[:8]}")
    
    logger.info(f"--- Starting RF-DETR Training Job: {run_name_tag} (Run ID: {run_id}) ---")
    logger.info(f"Model Type: {model_type}, Device: {device}")
    
    temp_dataset_dir = None
    
    try:
        # Log parameters to MLflow
        logger.info("Logging parameters to MLflow...")
        _log_params_recursive(run_config)
        mlflow.log_param("environment.actual_device", str(device))
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("engine_type", "rfdetr")
        _log_git_info()
        
        # Create temporary directory for COCO format dataset
        temp_dataset_dir = Path(tempfile.mkdtemp(prefix="rfdetr_dataset_"))
        logger.info(f"Created temporary dataset directory: {temp_dataset_dir}")
        
        # Create datasets
        logger.info("Creating MTMMC datasets...")
        dataset_train = MTMMCDetectionDataset(
            run_config,
            mode="train",
            transforms=None  # No transforms for data conversion
        )
        dataset_val = MTMMCDetectionDataset(
            run_config,
            mode="val", 
            transforms=None
        )
        
        if len(dataset_train) == 0:
            raise ValueError("Training dataset is empty!")
        
        logger.info(f"Train samples: {len(dataset_train)}, Val samples: {len(dataset_val)}")
        mlflow.log_param("dataset.num_train_samples", len(dataset_train))
        mlflow.log_param("dataset.num_val_samples", len(dataset_val))
        
        # Convert to COCO format
        logger.info("Converting datasets to COCO format...")
        # Check if debug mode is enabled
        debug_mode = run_config.get("debug", {}).get("enable_box_debug", False)
        dataset_dir = convert_mtmmc_to_coco_format(dataset_train, temp_dataset_dir, "train", debug_mode)
        if len(dataset_val) > 0:
            convert_mtmmc_to_coco_format(dataset_val, temp_dataset_dir, "valid", debug_mode)
            # RF-DETR expects a test split, use validation data for test
            convert_mtmmc_to_coco_format(dataset_val, temp_dataset_dir, "test", debug_mode)
        
        # Create RF-DETR model
        logger.info("Creating RF-DETR model...")
        model = get_rfdetr_model(run_config)
        
        # Prepare training configuration
        train_config = {
            "dataset_dir": str(temp_dataset_dir),
            "output_dir": str(project_root / "checkpoints" / run_id),
            "epochs": training_config.get("epochs", 100),
            "batch_size": training_config.get("batch_size", 4),
            "lr": training_config.get("learning_rate", 1e-4),
            "lr_encoder": training_config.get("lr_encoder", 1.5e-4),
            "num_workers": data_config.get("num_workers", 2),
            "weight_decay": training_config.get("weight_decay", 1e-4),
            "early_stopping": training_config.get("early_stopping", False),
            "early_stopping_patience": training_config.get("early_stopping_patience", 10),
            "checkpoint_interval": training_config.get("checkpoint_interval", 10)
        }
        
        # Add optional parameters
        optional_params = [
            "grad_accum_steps", "warmup_epochs", "lr_drop", "ema_decay", 
            "ema_tau", "lr_vit_layer_decay", "lr_component_decay", "drop_path",
            "multi_scale", "expanded_scales", "early_stopping_min_delta", "run_test"
        ]
        
        for param in optional_params:
            if param in training_config:
                train_config[param] = training_config[param]
        
        logger.info(f"Training configuration: {train_config}")
        
        # Create output directory
        output_dir = Path(train_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log dataset directory structure for debugging
        logger.info(f"Dataset directory structure:")
        dataset_path = Path(train_config["dataset_dir"])
        for split in ["train", "valid", "test"]:
            split_path = dataset_path / split
            if split_path.exists():
                logger.info(f"  {split}/: {len(list(split_path.glob('*.jpg')))} images")
                ann_file = split_path / "_annotations.coco.json"
                if ann_file.exists():
                    logger.info(f"  {split}/_annotations.coco.json: exists")
                    # Log a sample of the COCO format for debugging
                    try:
                        with open(ann_file, 'r') as f:
                            coco_sample = json.load(f)
                            logger.info(f"  {split} COCO format keys: {list(coco_sample.keys())}")
                            logger.info(f"  {split} has {len(coco_sample.get('images', []))} images, {len(coco_sample.get('annotations', []))} annotations")
                            if 'info' in coco_sample:
                                logger.info(f"  {split} info field: {coco_sample['info']}")
                    except Exception as e:
                        logger.warning(f"  Could not read {split}/_annotations.coco.json: {e}")
                else:
                    logger.info(f"  {split}/_annotations.coco.json: missing")
            else:
                logger.info(f"  {split}/: directory missing")
        
        # Start training
        logger.info("Starting RF-DETR training...")
        start_time = time.time()
        
        try:
            # Train the model
            model.train(**train_config)
            
            training_duration = time.time() - start_time
            logger.info(f"Training completed in {training_duration:.2f} seconds")
            
            # Log training artifacts
            if output_dir.exists():
                logger.info("Logging training artifacts...")
                for artifact_file in output_dir.glob("*.pth"):
                    mlflow.log_artifact(str(artifact_file), artifact_path="checkpoints")
                
                # Log training logs if they exist
                for log_file in output_dir.glob("*.log"):
                    mlflow.log_artifact(str(log_file), artifact_path="logs")
            
            # Set final metrics (RF-DETR training doesn't return metrics directly)
            final_metrics = {
                "training_duration": training_duration,
                "epochs_completed": train_config["epochs"]
            }
            
            mlflow.log_metric("training_duration", training_duration)
            mlflow.log_metric("epochs_completed", train_config["epochs"])
            
            job_status = "FINISHED"
            
        except Exception as train_error:
            logger.error(f"RF-DETR training failed: {train_error}", exc_info=True)
            raise
            
    except KeyboardInterrupt:
        logger.warning(f"[{run_name_tag}] Training job interrupted.")
        job_status = "KILLED"
        if mlflow.active_run():
            mlflow.set_tag("run_outcome", "Killed by user")
        raise
        
    except Exception as e:
        logger.critical(f"[{run_name_tag}] Uncaught error: {e}", exc_info=True)
        job_status = "FAILED"
        if mlflow.active_run():
            mlflow.set_tag("run_outcome", "Crashed")
            try:
                error_log_content = (
                    f"Error: {type(e).__name__}\n{e}\n{traceback.format_exc()}"
                )
                mlflow.log_text(error_log_content, "error_log.txt")
            except Exception as log_err:
                logger.error(f"Failed to log error details: {log_err}")
                
    finally:
        # Cleanup temporary dataset directory
        if temp_dataset_dir and temp_dataset_dir.exists():
            try:
                shutil.rmtree(temp_dataset_dir)
                logger.info(f"Cleaned up temporary dataset directory: {temp_dataset_dir}")
            except Exception as cleanup_err:
                logger.warning(f"Could not cleanup temp dataset dir {temp_dataset_dir}: {cleanup_err}")
        
        logger.info(f"--- Finished RF-DETR Training Job: {run_name_tag} (Status: {job_status}) ---")
    
    return job_status, final_metrics