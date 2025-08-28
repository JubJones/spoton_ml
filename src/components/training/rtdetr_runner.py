"""
RT-DETR Training Runner using Ultralytics

This module provides RT-DETR training functionality using the Ultralytics framework,
with MLflow integration for comprehensive metrics logging compatible with the existing
FasterRCNN training workflow.
"""

import logging
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import mlflow
import shutil
import os
import json

try:
    from ultralytics import RTDETR
    from ultralytics.utils.callbacks import mlflow as ultralytics_mlflow
except ImportError:
    RTDETR = None
    ultralytics_mlflow = None

from src.components.data.training_dataset import MTMMCDetectionDataset
from src.utils.runner import log_git_info

logger = logging.getLogger(__name__)

def log_params_recursive(config_dict: Dict[str, Any], prefix: str = "") -> None:
    """
    Recursively logs configuration parameters to MLflow, handling nested dictionaries.
    """
    for key, value in config_dict.items():
        param_name = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            log_params_recursive(value, param_name)
        elif isinstance(value, (list, tuple)):
            # Log list/tuple as string representation
            mlflow.log_param(param_name, str(value))
        else:
            # Log primitive values directly
            mlflow.log_param(param_name, value)

def create_ultralytics_dataset_config(
    train_dataset: MTMMCDetectionDataset,
    val_dataset: MTMMCDetectionDataset,
    project_root: Path
) -> Path:
    """
    Create an Ultralytics-compatible dataset configuration file in YOLO format.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        project_root: Project root directory
        
    Returns:
        Path to the created dataset config file
    """
    
    # Create temporary directory for dataset conversion
    temp_dir = project_root / "temp_ultralytics_data"
    temp_dir.mkdir(exist_ok=True)
    
    train_images_dir = temp_dir / "images" / "train"
    val_images_dir = temp_dir / "images" / "val"
    train_labels_dir = temp_dir / "labels" / "train"
    val_labels_dir = temp_dir / "labels" / "val"
    
    # Create directories
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    def convert_dataset_to_yolo(dataset: MTMMCDetectionDataset, images_dir: Path, labels_dir: Path):
        """Convert dataset to YOLO format"""
        logger.info(f"Converting {len(dataset)} samples to YOLO format...")
        
        for idx in range(len(dataset)):
            # Get the image path and annotations directly from samples_split
            image_path, annotations = dataset.samples_split[idx]
            
            # Copy image
            image_filename = f"{dataset.mode}_{idx:06d}.jpg"
            dest_image_path = images_dir / image_filename
            shutil.copy2(image_path, dest_image_path)
            
            # Create YOLO format label file
            label_filename = f"{dataset.mode}_{idx:06d}.txt"
            label_path = labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                for ann in annotations.persons:
                    # Convert to YOLO format: class_id center_x center_y width height (normalized)
                    # Get image dimensions
                    try:
                        from PIL import Image
                        with Image.open(image_path) as img:
                            img_width, img_height = img.size
                    except:
                        # Fallback to standard resolution if image can't be opened
                        img_width, img_height = 1920, 1080
                    
                    # Convert bbox from (x, y, w, h) to normalized center format
                    x, y, w, h = ann.bbox
                    center_x = (x + w/2) / img_width
                    center_y = (y + h/2) / img_height
                    norm_width = w / img_width
                    norm_height = h / img_height
                    
                    # Class 0 for person (Ultralytics uses 0-based indexing)
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\\n")
    
    # Convert datasets
    convert_dataset_to_yolo(train_dataset, train_images_dir, train_labels_dir)
    convert_dataset_to_yolo(val_dataset, val_images_dir, val_labels_dir)
    
    # Create dataset config file
    dataset_config = {
        'path': str(temp_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val', 
        'names': {0: 'person'},
        'nc': 1  # number of classes
    }
    
    config_path = temp_dir / "dataset.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    logger.info(f"Created Ultralytics dataset config at: {config_path}")
    return config_path

def run_rtdetr_training_job(
    run_config: Dict[str, Any], 
    device: torch.device, 
    project_root: Path
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Execute RT-DETR training using Ultralytics with comprehensive MLflow logging.
    
    Args:
        run_config: Training configuration dictionary
        device: Training device
        project_root: Project root directory
        
    Returns:
        Tuple of (status, final_metrics)
    """
    
    if RTDETR is None:
        logger.error("Ultralytics not installed. Please install with: pip install ultralytics")
        return "FAILED", None
    
    active_run = mlflow.active_run()
    if not active_run:
        logger.critical("run_rtdetr_training_job called without an active MLflow run!")
        return "FAILED", None
    
    run_id = active_run.info.run_id
    job_status = "FAILED"  
    final_metrics = None
    
    model_config = run_config["model"]
    training_config = run_config["training"]
    data_config = run_config["data"]
    env_config = run_config["environment"]
    
    model_type = model_config.get("type", "rtdetr").lower()
    model_size = model_config.get("model_size", "rtdetr-l.pt")
    run_name_tag = model_config.get("name_tag", f"rtdetr_training_{run_id[:8]}")
    
    logger.info(f"--- Starting RT-DETR Training Job: {run_name_tag} (Run ID: {run_id}) ---")
    logger.info(f"Model Size: {model_size}, Device: {device}")
    
    try:
        # Log parameters to MLflow
        logger.info("Logging parameters to MLflow...")
        log_params_recursive(run_config)
        mlflow.log_param("environment.actual_device", str(device))
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("engine_type", "ultralytics")
        log_git_info()
        
        # Create datasets
        logger.info("Creating Datasets...")
        train_dataset = MTMMCDetectionDataset(
            run_config,
            mode="train",
            transforms=None  # Ultralytics handles augmentations internally
        )
        
        val_dataset = MTMMCDetectionDataset(
            run_config, 
            mode="val",
            transforms=None
        )
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        mlflow.log_param("dataset.train_size", len(train_dataset))
        mlflow.log_param("dataset.val_size", len(val_dataset))
        
        # Create Ultralytics dataset config
        logger.info("Converting dataset to Ultralytics format...")
        dataset_config_path = create_ultralytics_dataset_config(
            train_dataset, val_dataset, project_root
        )
        
        # Initialize RT-DETR model
        logger.info(f"Loading RT-DETR model: {model_size}")
        model = RTDETR(model_size)
        
        # Log model info
        model.info()
        mlflow.log_param("model.parameters", model.model[-1].nc if hasattr(model.model[-1], 'nc') else 'unknown')
        
        # Configure training parameters
        train_args = {
            'data': str(dataset_config_path),
            'epochs': training_config.get('epochs', 100),
            'batch': training_config.get('batch_size', 4),
            'imgsz': training_config.get('imgsz', 640),
            'lr0': training_config.get('learning_rate', 0.001),
            'device': str(device) if device != torch.device('cpu') else 'cpu',
            'project': str(project_root / training_config.get('project', 'checkpoints')),
            'name': training_config.get('name', 'rtdetr_run'),
            'exist_ok': training_config.get('exist_ok', True),
            'pretrained': training_config.get('pretrained', True),
            'optimizer': training_config.get('optimizer', 'AdamW'),
            'verbose': training_config.get('verbose', True),
            'seed': env_config.get('seed', 42),
            'deterministic': training_config.get('deterministic', True),
            'single_cls': training_config.get('single_cls', True),
            'patience': training_config.get('patience', 50),
            'save_period': training_config.get('save_period', 10),
            'cache': training_config.get('cache', False),
            'workers': training_config.get('workers', 2),
            'rect': training_config.get('rect', False),
            'cos_lr': training_config.get('cos_lr', False),
            'close_mosaic': training_config.get('close_mosaic', 10),
            'resume': training_config.get('resume', False),
            'amp': training_config.get('amp', True),
            'fraction': training_config.get('fraction', 1.0),
            'profile': training_config.get('profile', False),
            'freeze': training_config.get('freeze', None),
            'lr0': training_config.get('lr0', 0.01),
            'lrf': training_config.get('lrf', 0.01),
            'momentum': training_config.get('momentum', 0.937),
            'weight_decay': training_config.get('weight_decay', 0.0005),
            'warmup_epochs': training_config.get('warmup_epochs', 3.0),
            'warmup_momentum': training_config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': training_config.get('warmup_bias_lr', 0.1),
            'box': training_config.get('box', 7.5),
            'cls': training_config.get('cls', 0.5),
            'dfl': training_config.get('dfl', 1.5),
            'label_smoothing': training_config.get('label_smoothing', 0.0),
            'nbs': training_config.get('nbs', 64),
            'hsv_h': training_config.get('hsv_h', 0.015),
            'hsv_s': training_config.get('hsv_s', 0.7),
            'hsv_v': training_config.get('hsv_v', 0.4),
            'degrees': training_config.get('degrees', 0.0),
            'translate': training_config.get('translate', 0.1),
            'scale': training_config.get('scale', 0.5),
            'shear': training_config.get('shear', 0.0),
            'perspective': training_config.get('perspective', 0.0),
            'flipud': training_config.get('flipud', 0.0),
            'fliplr': training_config.get('fliplr', 0.5),
            'mosaic': training_config.get('mosaic', 1.0),
            'mixup': training_config.get('mixup', 0.0),
            'copy_paste': training_config.get('copy_paste', 0.0),
            'val': training_config.get('val', True),
            'split': training_config.get('split', 'val'),
            'save_json': training_config.get('save_json', True),
            'save_hybrid': training_config.get('save_hybrid', False),
            'conf': training_config.get('conf', None),
            'iou': training_config.get('iou', 0.7),
            'max_det': training_config.get('max_det', 300),
            'half': training_config.get('half', False),
            'dnn': training_config.get('dnn', False),
            'plots': training_config.get('plots', True),
        }
        
        # Log training arguments
        for key, value in train_args.items():
            mlflow.log_param(f"ultralytics.{key}", value)
        
        # Enable MLflow callback in Ultralytics
        try:
            # Add MLflow integration callback
            from ultralytics.utils.callbacks import add_integration_callbacks
            add_integration_callbacks(model)
        except Exception as e:
            logger.warning(f"Could not enable Ultralytics MLflow integration: {e}")
        
        # Start training
        logger.info("Starting RT-DETR training...")
        results = model.train(**train_args)
        
        # Extract and log metrics
        if results:
            try:
                # Get training metrics from results
                if hasattr(results, 'results_dict'):
                    metrics = results.results_dict
                elif isinstance(results, dict):
                    metrics = results
                else:
                    # Try to extract from model trainer
                    metrics = {}
                    if hasattr(model, 'trainer') and hasattr(model.trainer, 'metrics'):
                        metrics = model.trainer.metrics
                
                # Log key metrics that match FasterRCNN logging format
                if metrics:
                    final_metrics = {}
                    
                    # Map Ultralytics metrics to our format
                    metric_mapping = {
                        'metrics/mAP50-95(B)': 'val_map',
                        'metrics/mAP50(B)': 'val_map_50', 
                        'metrics/mAP75(B)': 'val_map_75',
                        'metrics/precision(B)': 'val_precision',
                        'metrics/recall(B)': 'val_recall',
                        'train/box_loss': 'train_box_loss',
                        'train/cls_loss': 'train_cls_loss',
                        'train/dfl_loss': 'train_dfl_loss',
                        'val/box_loss': 'val_box_loss',
                        'val/cls_loss': 'val_cls_loss',
                        'val/dfl_loss': 'val_dfl_loss',
                        'lr/pg0': 'learning_rate',
                        'lr/pg1': 'learning_rate_pg1',
                        'lr/pg2': 'learning_rate_pg2'
                    }
                    
                    for ultralytics_key, our_key in metric_mapping.items():
                        if ultralytics_key in metrics:
                            value = metrics[ultralytics_key]
                            mlflow.log_metric(our_key, float(value))
                            final_metrics[our_key] = float(value)
                            logger.info(f"Logged metric {our_key}: {value}")
                    
                    # Also log any additional metrics found
                    for key, value in metrics.items():
                        if key not in metric_mapping and isinstance(value, (int, float)):
                            safe_key = key.replace('/', '_').replace('(', '_').replace(')', '')
                            mlflow.log_metric(f"ultralytics_{safe_key}", float(value))
                
                # Log final model path
                if hasattr(model, 'trainer') and hasattr(model.trainer, 'best'):
                    best_model_path = model.trainer.best
                    if best_model_path and Path(best_model_path).exists():
                        mlflow.log_artifact(str(best_model_path), artifact_path="models")
                        logger.info(f"Logged best model: {best_model_path}")
                
                job_status = "FINISHED"
                logger.info("RT-DETR training completed successfully!")
                
            except Exception as e:
                logger.error(f"Error extracting training metrics: {e}")
                job_status = "FINISHED"  # Still consider successful if training completed
                
        else:
            logger.warning("No results returned from training")
            job_status = "FINISHED"
        
    except Exception as e:
        logger.error(f"RT-DETR training failed: {e}", exc_info=True)
        job_status = "FAILED"
        final_metrics = None
        
    finally:
        # Cleanup temporary dataset files
        temp_dir = project_root / "temp_ultralytics_data"
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary dataset files")
            except Exception as e:
                logger.warning(f"Could not cleanup temporary files: {e}")
    
    logger.info(f"RT-DETR training job completed with status: {job_status}")
    return job_status, final_metrics