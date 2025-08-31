"""
RT-DETR Training Runner with MLflow Integration

This script runs RT-DETR model training using the Ultralytics framework,
following the same structure as the FasterRCNN training runner.

Usage:
    python src/run_training_rtdetr.py

The script will:
1. Load configuration from configs/rtdetr_training_config.yaml
2. Setup MLflow experiment tracking with comprehensive metrics logging
3. Train RT-DETR model using Ultralytics with person detection focus
4. Log all training metrics including mAP, precision, recall, F1-score
"""

import logging
import sys
import time
import warnings
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"PROJECT_ROOT added to sys.path: {PROJECT_ROOT}")

# --- Local Imports ---
try:
    from src.utils.config_loader import load_config
    from src.utils.reproducibility import set_seed
    from src.utils.logging_utils import setup_logging
    from src.utils.mlflow_utils import setup_mlflow_experiment
    from src.utils.device_utils import get_selected_device
    from src.utils.runner import log_params_recursive, log_git_info
except ImportError as e:
    print(f"Error importing local modules in run_training_rtdetr.py: {e}")
    print("Please ensure all modules exist and PYTHONPATH is set correctly.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = setup_logging(log_prefix="train_rtdetr", log_dir=PROJECT_ROOT / "logs")
logger = logging.getLogger(__name__)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")
warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")

# Import Ultralytics after logging setup
from ultralytics import RTDETR
import ultralytics.utils

# ===== Configuration Constants =====
# These will be overridden by config or can be used as defaults
DEFAULT_BASE_PATH = "D:/MTMMC"  # Default dataset path
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 4
DEFAULT_IMAGE_SIZE = 320
DEFAULT_ENVIRONMENT = "factory"
DEFAULT_SCENE_ID = "s10"
DEFAULT_CAMERA_IDS = ["c09"]
DEFAULT_MAX_FRAMES = 50

# GPU Detection and Device Selection
def get_best_device():
    """Detect and return the best available device"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"🔥 CUDA available! Found {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            return "0"  # Use first GPU
        else:
            logger.warning("⚠️  CUDA not available, falling back to CPU")
            return "cpu"
    except ImportError:
        logger.warning("⚠️  PyTorch not available for device detection, using default")
        return "0"


def create_coco_format_data(run_config: Dict[str, Any], output_dir: Path):
    """Convert MTMMC data to COCO format for RT-DETR training"""
    logger.info("Creating COCO format dataset...")
    
    data_config = run_config.get("data", {})
    base_path = Path(data_config.get("base_path", DEFAULT_BASE_PATH))
    environment = data_config.get("environment", DEFAULT_ENVIRONMENT)
    scene_id = data_config.get("scene_id", DEFAULT_SCENE_ID)
    camera_ids = data_config.get("camera_ids", DEFAULT_CAMERA_IDS)
    max_frames = data_config.get("max_frames", DEFAULT_MAX_FRAMES)
    
    train_path = base_path / "train" / scene_id
    
    if not train_path.exists():
        error_msg = f"Dataset path not found: {train_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Create output directories
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"  
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    image_id = 0
    
    logger.info(f"Processing scene {scene_id} from {environment} environment...")
    
    for camera_id in camera_ids:
        camera_path = train_path / camera_id
        rgb_path = camera_path / "rgb"
        gt_path = camera_path / "gt" / "gt.txt"
        
        if not rgb_path.exists() or not gt_path.exists():
            logger.warning(f"Skipping camera {camera_id} - missing data")
            continue
            
        logger.info(f"Processing camera {camera_id}...")
        
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
        if max_frames > 0:
            image_files = image_files[:max_frames]
            
        for img_file in image_files:
            frame_id = int(img_file.stem)
            
            # Copy image to output directory
            new_img_name = f"{scene_id}_{camera_id}_{frame_id:06d}.jpg"
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
                label_file = labels_dir / f"{scene_id}_{camera_id}_{frame_id:06d}.txt"
                with open(label_file, 'w') as f:
                    for obj_id, x, y, w, h in gt_data[frame_id]:
                        # Convert to YOLO format (normalized center coordinates)
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
                logger.info(f"Processed {image_id} images...")

    logger.info(f"Dataset creation complete! Processed {image_id} images.")
    return images_dir, labels_dir


def create_dataset_yaml(output_dir: Path):
    """Create dataset YAML file for RT-DETR"""
    yaml_content = f"""# MTMMC Person Detection Dataset
path: {output_dir.absolute()}
train: images
val: images  # Using same for validation (you can split if needed)

# Classes
nc: 1  # number of classes
names: ['person']  # class names
"""
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    
    logger.info(f"Created dataset YAML: {yaml_path}")
    return yaml_path


def run_rtdetr_training_job(run_config: Dict[str, Any], device: str, project_root: Path) -> tuple[str, Optional[Dict[str, Any]]]:
    """Execute RT-DETR training within MLflow context"""
    active_run = mlflow.active_run()
    if not active_run:
        logger.critical("run_rtdetr_training_job called without an active MLflow run!")
        return "FAILED", None
    
    run_id = active_run.info.run_id
    job_status = "FAILED"
    final_metrics = {"status": "initialized"}
    
    logger.info(f"--- Starting RT-DETR Training Job (Run ID: {run_id}) ---")
    
    # Extract training parameters
    training_config = run_config.get("training", {})
    epochs = training_config.get("epochs", DEFAULT_EPOCHS)
    batch_size = training_config.get("batch_size", DEFAULT_BATCH_SIZE)
    image_size = training_config.get("image_size", DEFAULT_IMAGE_SIZE)
    
    # Output directory
    output_dir = project_root / "rtdetr_training_output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Log parameters
        logger.info("Logging parameters to MLflow...")
        log_params_recursive(run_config)
        mlflow.log_param("environment.actual_device", str(device))
        mlflow.log_param("training.epochs", epochs)
        mlflow.log_param("training.batch_size", batch_size)
        mlflow.log_param("training.image_size", image_size)
        mlflow.set_tag("model_type", "rtdetr")
        mlflow.set_tag("engine_type", "ultralytics")
        log_git_info()
        
        # Create dataset
        logger.info("Creating COCO format dataset...")
        images_dir, labels_dir = create_coco_format_data(run_config, output_dir)
        dataset_yaml = create_dataset_yaml(output_dir)
        
        # Log dataset info
        mlflow.log_param("dataset.images_created", len(list(images_dir.glob("*.jpg"))))
        mlflow.log_param("dataset.labels_created", len(list(labels_dir.glob("*.txt"))))
        
        # Initialize RT-DETR model
        logger.info("Initializing RT-DETR model...")
        model = RTDETR("rtdetr-l.pt")  # Load pretrained RT-DETR Large model
        
        # Configure Ultralytics settings to disable built-in tracking
        try:
            ultralytics.utils.SETTINGS['mlflow'] = False
            ultralytics.utils.SETTINGS['comet'] = False
            ultralytics.utils.SETTINGS['wandb'] = False
            logger.info("Disabled Ultralytics built-in tracking")
        except Exception as e:
            logger.warning(f"Could not configure Ultralytics settings: {e}")
        
        # Start training
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Device: {device}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Image size: {image_size}")
        
        # Extract all training parameters from config for comprehensive augmentation support
        training_params = {
            # Core training parameters
            'data': str(dataset_yaml),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': training_config.get('imgsz', image_size),
            'device': device,
            'project': str(output_dir),
            'name': training_config.get('name', 'rtdetr_experiment'),
            'save': training_config.get('save', True),
            'plots': training_config.get('plots', True),
            'val': training_config.get('val', True),
            'cache': training_config.get('cache', False),
            'workers': training_config.get('workers', 4),
            'patience': training_config.get('patience', 50),
            'amp': training_config.get('amp', True),
            'half': training_config.get('half', False),
            
            # Learning rate and optimization
            'lr0': training_config.get('lr0', 0.001),
            'lrf': training_config.get('lrf', 0.01),
            'momentum': training_config.get('momentum', 0.937),
            'weight_decay': training_config.get('weight_decay', 0.0005),
            'warmup_epochs': training_config.get('warmup_epochs', 3.0),
            'warmup_momentum': training_config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': training_config.get('warmup_bias_lr', 0.1),
            
            # Loss function parameters
            'box': training_config.get('box', 7.5),
            'cls': training_config.get('cls', 0.5),
            'dfl': training_config.get('dfl', 1.5),
            'label_smoothing': training_config.get('label_smoothing', 0.0),
            
            # Color space augmentations
            'hsv_h': training_config.get('hsv_h', 0.015),
            'hsv_s': training_config.get('hsv_s', 0.7),
            'hsv_v': training_config.get('hsv_v', 0.4),
            
            # Geometric augmentations
            'degrees': training_config.get('degrees', 0.0),
            'translate': training_config.get('translate', 0.1),
            'scale': training_config.get('scale', 0.5),
            'shear': training_config.get('shear', 0.0),
            'perspective': training_config.get('perspective', 0.0),
            
            # Flip augmentations
            'flipud': training_config.get('flipud', 0.0),
            'fliplr': training_config.get('fliplr', 0.5),
            
            # Advanced composition augmentations
            'mosaic': training_config.get('mosaic', 1.0),
            'mixup': training_config.get('mixup', 0.0),
            'copy_paste': training_config.get('copy_paste', 0.0),
            
            # Advanced augmentation parameters
            'cutmix': training_config.get('cutmix', 0.0),
            'erasing': training_config.get('erasing', 0.0),
            'auto_augment': training_config.get('auto_augment', None),
            
            # Training strategies
            'multi_scale': training_config.get('multi_scale', False),
            'rect': training_config.get('rect', False),
            'cos_lr': training_config.get('cos_lr', False),
            'close_mosaic': training_config.get('close_mosaic', 10),
            
            # Other training parameters
            'optimizer': training_config.get('optimizer', 'AdamW'),
            'verbose': training_config.get('verbose', True),
            'seed': training_config.get('seed', 42),
            'deterministic': training_config.get('deterministic', True),
            'single_cls': training_config.get('single_cls', True),
            'resume': training_config.get('resume', False),
            'exist_ok': training_config.get('exist_ok', True),
            'pretrained': training_config.get('pretrained', True),
            'fraction': training_config.get('fraction', 1.0),
            'profile': training_config.get('profile', False),
            'freeze': training_config.get('freeze', None),
        }
        
        # Remove None values to avoid issues
        training_params = {k: v for k, v in training_params.items() if v is not None}
        
        logger.info("Enhanced training with complex augmentations:")
        logger.info(f"  Color augmentations: HSV-H={training_params.get('hsv_h')}, HSV-S={training_params.get('hsv_s')}, HSV-V={training_params.get('hsv_v')}")
        logger.info(f"  Geometric augmentations: Rotation={training_params.get('degrees')}°, Scale={training_params.get('scale')}, Shear={training_params.get('shear')}°")
        logger.info(f"  Composition augmentations: Mosaic={training_params.get('mosaic')}, Mixup={training_params.get('mixup')}, CutMix={training_params.get('cutmix')}")
        logger.info(f"  Advanced features: Multi-scale={training_params.get('multi_scale')}, Auto-augment={training_params.get('auto_augment')}")
        
        # Add training callback to log metrics per epoch (like FasterRCNN)
        class MLflowCallback:
            """Custom callback to log training metrics to MLflow like FasterRCNN training"""
            def __init__(self, run_id):
                self.run_id = run_id
                self.best_map = -1.0
                self.best_epoch = -1
                self.epoch_times = []
                
            def on_train_epoch_start(self, trainer):
                """Track epoch start time"""
                import time
                self.epoch_start_time = time.time()
                logger.info(f"Starting epoch {trainer.epoch if hasattr(trainer, 'epoch') and trainer.epoch is not None else 'N/A'}")
                
            def on_train_epoch_end(self, trainer):
                """Log epoch metrics similar to PyTorch FasterRCNN training"""
                if trainer.epoch is not None:
                    # Log epoch duration (similar to FasterRCNN epoch timing)
                    if hasattr(self, 'epoch_start_time'):
                        epoch_duration = time.time() - self.epoch_start_time
                        self.epoch_times.append(epoch_duration)
                        mlflow.log_metric("epoch_duration_seconds", epoch_duration, step=trainer.epoch)
                        mlflow.log_metric("avg_epoch_duration_seconds", sum(self.epoch_times) / len(self.epoch_times), step=trainer.epoch)
                    
                    # Log basic training metrics
                    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                        # Log training losses (similar to avg_train_loss, avg_train_comp_losses)
                        total_loss = float(trainer.loss_items.mean()) if hasattr(trainer.loss_items, 'mean') else float(sum(trainer.loss_items) / len(trainer.loss_items))
                        mlflow.log_metric("epoch_train_loss_avg", total_loss, step=trainer.epoch)
                        
                        # Log component losses if available (mimicking FasterRCNN component losses)
                        if len(trainer.loss_items) >= 3:  # Typical RT-DETR losses: box, cls, dfl
                            mlflow.log_metric("epoch_train_loss_box", float(trainer.loss_items[0]), step=trainer.epoch)
                            mlflow.log_metric("epoch_train_loss_cls", float(trainer.loss_items[1]), step=trainer.epoch)
                            mlflow.log_metric("epoch_train_loss_dfl", float(trainer.loss_items[2]), step=trainer.epoch)
                            
                            # Additional RT-DETR specific losses if available
                            if len(trainer.loss_items) >= 4:
                                mlflow.log_metric("epoch_train_loss_additional", float(trainer.loss_items[3]), step=trainer.epoch)
                    
                    # Log learning rate (similar to FasterRCNN)
                    if hasattr(trainer.optimizer, 'param_groups') and trainer.optimizer.param_groups:
                        current_lr = trainer.optimizer.param_groups[0]['lr']
                        mlflow.log_metric("learning_rate", current_lr, step=trainer.epoch)
                        
                        # Log momentum if available (similar to FasterRCNN optimizer tracking)
                        if 'momentum' in trainer.optimizer.param_groups[0]:
                            momentum = trainer.optimizer.param_groups[0]['momentum']
                            mlflow.log_metric("momentum", momentum, step=trainer.epoch)
                        elif 'betas' in trainer.optimizer.param_groups[0]:  # For Adam-like optimizers
                            beta1 = trainer.optimizer.param_groups[0]['betas'][0]
                            mlflow.log_metric("beta1", beta1, step=trainer.epoch)
                    
                    # Log GPU memory usage if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                            mlflow.log_metric("gpu_memory_allocated_gb", memory_allocated, step=trainer.epoch)
                            mlflow.log_metric("gpu_memory_reserved_gb", memory_reserved, step=trainer.epoch)
                    except Exception:
                        pass  # Ignore if CUDA not available
            
            def on_val_end(self, validator):
                """Log validation metrics similar to FasterRCNN eval_metrics"""
                if validator.metrics and hasattr(validator, 'trainer') and validator.trainer.epoch is not None:
                    metrics = validator.metrics.results_dict
                    epoch = validator.trainer.epoch
                    
                    # Log validation metrics (similar to eval_metrics in FasterRCNN)
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and not str(key).startswith('_'):
                            # Map RT-DETR metric names to consistent naming (like FasterRCNN eval_metrics)
                            if 'mAP50-95' in key:
                                mlflow.log_metric("eval_map_50_95", float(value), step=epoch)
                            elif 'mAP50' in key and 'mAP50-95' not in key:
                                mlflow.log_metric("eval_map_50", float(value), step=epoch)
                            elif 'precision' in key.lower():
                                mlflow.log_metric("eval_precision", float(value), step=epoch)
                            elif 'recall' in key.lower():
                                mlflow.log_metric("eval_recall", float(value), step=epoch)
                            elif 'f1' in key.lower():
                                mlflow.log_metric("eval_f1_score", float(value), step=epoch)
                            else:
                                # Generic metric mapping
                                metric_name = key.lower().replace('(b)', '').replace('metrics/', 'eval_').replace(' ', '_')
                                mlflow.log_metric(f"epoch_{metric_name}", float(value), step=epoch)
                    
                    # Track best model (similar to best_metric_value in FasterRCNN)
                    current_map = metrics.get('metrics/mAP50-95(B)', -1.0)
                    if current_map > self.best_map:
                        self.best_map = current_map
                        self.best_epoch = epoch
                        mlflow.set_tag("best_map_50_95", f"{self.best_map:.4f}")
                        mlflow.set_tag("best_epoch", str(self.best_epoch))
                        logger.info(f"New best mAP@0.5:0.95: {self.best_map:.4f} at epoch {self.best_epoch}")
                        
                        # Log additional best metrics (similar to FasterRCNN best model tracking)
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)) and 'precision' in key.lower():
                                mlflow.set_tag("best_precision", f"{float(value):.4f}")
                            elif isinstance(value, (int, float)) and 'recall' in key.lower():
                                mlflow.set_tag("best_recall", f"{float(value):.4f}")
                    
                    # Log validation loss if available (similar to avg_val_loss in FasterRCNN)
                    if hasattr(validator, 'loss') and validator.loss is not None:
                        mlflow.log_metric("epoch_val_loss_avg", float(validator.loss), step=epoch)
        
        # Setup MLflow callback
        callback = MLflowCallback(run_id)
        
        # Add callback to training parameters
        training_params['callbacks'] = [callback]
        
        logger.info("Starting RT-DETR training with MLflow logging (similar to FasterRCNN)...")
        start_time_training = time.time()
        
        results = model.train(**training_params)
        
        training_duration = time.time() - start_time_training
        logger.info(f"--- RT-DETR Training Finished in {training_duration:.2f} seconds ---")
        mlflow.log_metric("training_duration_seconds", training_duration)
        
        # Log final training results (enhanced like FasterRCNN)
        if hasattr(results, 'maps') and len(results.maps) > 0:
            if len(results.maps) >= 2:
                mlflow.log_metric("final_map_50", results.maps[0])
                mlflow.log_metric("final_map_50_95", results.maps[1])
                logger.info(f"Final mAP@0.5: {results.maps[0]:.4f}")
                logger.info(f"Final mAP@0.5:0.95: {results.maps[1]:.4f}")
                final_metrics = {"map_50": results.maps[0], "map_50_95": results.maps[1]}
            else:
                # Only one metric available (likely mAP@0.5:0.95)
                mlflow.log_metric("final_map_50_95", results.maps[0])
                logger.info(f"Final mAP@0.5:0.95: {results.maps[0]:.4f}")
                final_metrics = {"map_50_95": results.maps[0]}
        
        # Enhanced checkpoint and artifact logging (similar to FasterRCNN)
        experiment_dir = output_dir / "rtdetr_experiment"
        checkpoint_dir = project_root / "checkpoints" / "rtdetr" / run_id
        
        # Log best model parameters
        if callback.best_epoch >= 0:
            mlflow.log_param("best_model_epoch", callback.best_epoch)
            mlflow.log_param("best_model_map_50_95", f"{callback.best_map:.4f}")
        
        if experiment_dir.exists():
            logger.info("Logging comprehensive RT-DETR training artifacts...")
            
            # Create checkpoint directory structure (like FasterRCNN)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Log model checkpoints with structured paths
            weights_dir = experiment_dir / "weights"
            if weights_dir.exists():
                best_pt = weights_dir / "best.pt"
                last_pt = weights_dir / "last.pt"
                
                # Log best model checkpoint (similar to best_model_path in FasterRCNN)
                if best_pt.exists():
                    logger.info(f"Logging best model checkpoint: {best_pt.name}")
                    mlflow.log_artifact(str(best_pt), artifact_path="checkpoints")
                    
                    # Copy to structured checkpoint dir (like FasterRCNN)
                    best_checkpoint_path = checkpoint_dir / f"ckpt_best_map_50_95.pt"
                    import shutil
                    shutil.copy2(best_pt, best_checkpoint_path)
                
                # Log latest model checkpoint (similar to latest_path in FasterRCNN)
                if last_pt.exists():
                    mlflow.log_artifact(str(last_pt), artifact_path="checkpoints/latest")
                    
                    # Copy to structured checkpoint dir
                    latest_checkpoint_path = checkpoint_dir / f"ckpt_latest.pt"
                    shutil.copy2(last_pt, latest_checkpoint_path)
            
            # Log training plots and results (organized like FasterRCNN)
            for artifact_file in experiment_dir.rglob("*"):
                if artifact_file.is_file():
                    
                    # Organize artifacts by type (similar to FasterRCNN structure)
                    if artifact_file.suffix == '.pt':
                        artifact_path = "checkpoints"
                    elif artifact_file.suffix in ['.png', '.jpg']:
                        artifact_path = "plots"
                    elif artifact_file.suffix in ['.yaml', '.yml']:
                        artifact_path = "config"
                    elif artifact_file.suffix == '.csv':
                        artifact_path = "metrics"
                    else:
                        artifact_path = "training_results"
                    
                    try:
                        mlflow.log_artifact(str(artifact_file), artifact_path=artifact_path)
                    except Exception as e:
                        logger.warning(f"Could not log artifact {artifact_file}: {e}")
            
            logger.info("All RT-DETR training artifacts logged successfully!")
        
        job_status = "FINISHED"
        logger.info("RT-DETR training completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        job_status = "KILLED"
        raise
    except Exception as e:
        logger.critical(f"Training failed: {e}", exc_info=True)
        mlflow.set_tag("error", str(e))
        job_status = "FAILED"
        raise
    
    return job_status, final_metrics


# --- Main Execution ---
def main():
    """
    Runs a single RT-DETR training job based on the configuration file.
    """
    logger.info("--- Starting RT-DETR Training Run ---")
    config_path_str = "configs/rtdetr_training_config.yaml"
    final_status = "FAILED"
    run_id = None

    # 1. Load Configuration
    config = load_config(config_path_str)
    if not config:
        logger.critical(
            f"Failed to load configuration from {config_path_str}. Exiting."
        )
        sys.exit(1)

    # --- Extract the single training job config ---
    models_to_train = config.get("models_to_train")
    if (
        not models_to_train
        or not isinstance(models_to_train, list)
        or len(models_to_train) == 0
    ):
        logger.critical(
            f"Config {config_path_str} must contain a list 'models_to_train' with at least one entry."
        )
        sys.exit(1)
    if len(models_to_train) > 1:
        logger.warning(
            f"Config contains {len(models_to_train)} entries in 'models_to_train'. Using only the first one."
        )

    job_config_entry = models_to_train[0]
    if "model" not in job_config_entry or "training" not in job_config_entry:
        logger.critical(
            "The first entry in 'models_to_train' must contain 'model' and 'training' keys."
        )
        sys.exit(1)

    # --- Construct the config needed by run_rtdetr_training_job ---
    single_run_config = {
        "environment": config.get("environment", {}),
        "data": config.get("data", {}),
        "mlflow": config.get(
            "mlflow", {}
        ),  # Pass mlflow config for experiment name etc.
        "model": job_config_entry["model"],
        "training": job_config_entry["training"],
        # Add run name if specified at top level, otherwise runner uses model tag
        "run_name": config.get(
            "run_name", job_config_entry["model"].get("name_tag", "rtdetr_run")
        ),
    }

    # 2. Setup MLflow Experiment
    experiment_id = setup_mlflow_experiment(
        config, default_experiment_name="Default RT-DETR Training"
    )
    if not experiment_id:
        logger.critical("MLflow experiment setup failed. Exiting.")
        sys.exit(1)

    # 3. Set Seed & Determine Device
    seed = config.get("environment", {}).get("seed", int(time.time()))
    set_seed(seed)
    logger.info(f"Global random seed set to: {seed}")
    base_device_preference = config.get("environment", {}).get("device", "auto")
    resolved_device = get_selected_device(base_device_preference)
    if resolved_device is None:
        resolved_device = get_best_device()
    logger.info(f"Resolved base device: {resolved_device}")

    try:
        # 4. Start ONE MLflow Run
        run_name = single_run_config["run_name"]
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            logger.info(f"--- MLflow Run Started ---")
            logger.info(f"Run Name: {run_name}")
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Experiment ID: {experiment_id}")
            logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

            # Log the original config file path as an artifact
            orig_conf_path = PROJECT_ROOT / config_path_str
            if orig_conf_path.is_file():
                mlflow.log_artifact(str(orig_conf_path), artifact_path="config")

            # 5. Execute the single training job using the RT-DETR runner function
            run_status, final_metrics = run_rtdetr_training_job(
                run_config=single_run_config,
                device=str(resolved_device),
                project_root=PROJECT_ROOT,
            )
            final_status = run_status

    except KeyboardInterrupt:
        logger.warning("Training run interrupted by user (KeyboardInterrupt).")
        final_status = "KILLED"
        if run_id:
            try:
                MlflowClient().set_terminated(run_id, status="KILLED")
            except Exception as term_err:
                logger.warning(
                    f"Could not terminate run {run_id} after KILLED: {term_err}"
                )
    except Exception as e:
        logger.critical(f"An uncaught error occurred: {e}", exc_info=True)
        final_status = "FAILED"
        if run_id:
            try:
                client = MlflowClient()
                client.set_tag(run_id, "run_outcome", "Crashed - Outer")
                client.set_terminated(run_id, status="FAILED")
            except Exception as term_err:
                logger.warning(
                    f"Could not terminate run {run_id} after CRASHED: {term_err}"
                )
    finally:
        logger.info(f"--- Finalizing Run (Final Status: {final_status}) ---")
        # Log the main script log file to the run if it exists
        if run_id and log_file.exists():
            try:
                for handler in logging.getLogger().handlers:
                    handler.flush()
                client = MlflowClient()
                client.log_artifact(run_id, str(log_file), artifact_path="logs")
                logger.info(
                    f"Main training log file '{log_file.name}' logged as artifact to run {run_id}."
                )
            except Exception as log_artifact_err:
                logger.warning(
                    f"Could not log main training log file artifact '{log_file}': {log_artifact_err}"
                )

        # Ensure run termination status is set correctly
        active_run = mlflow.active_run()
        if active_run and active_run.info.run_id == run_id:
            logger.info(
                f"Ensuring MLflow run {run_id} is terminated with status {final_status}."
            )
            mlflow.end_run(status=final_status)
        elif run_id:
            try:
                logger.warning(
                    f"Attempting to terminate run {run_id} outside active context with status {final_status}."
                )
                MlflowClient().set_terminated(run_id, status=final_status)
            except Exception as term_err:
                logger.error(f"Failed to terminate run {run_id} forcefully: {term_err}")

    logger.info(f"--- RT-DETR Training Run Completed (Status: {final_status}) ---")
    exit_code = 0 if final_status == "FINISHED" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()