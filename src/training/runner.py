# ================================================
# FILE: src/training/runner.py (MODIFIED)
# ================================================
import logging
import traceback
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch

from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights # Import Enum directly
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# --- MODIFICATION: Import tv_tensors if needed, otherwise just T ---
# from torchvision import tv_tensors
from torchvision.transforms import v2 as T
# -----------------------------------------------------------------

import mlflow

from src.utils.torch_utils import collate_fn, get_optimizer, get_lr_scheduler
from src.data.training_dataset import MTMMCDetectionDataset
from src.training.pytorch_engine import train_one_epoch, evaluate as evaluate_pytorch
from src.core.runner import log_params_recursive, log_git_info  # Reuse logging helpers

logger = logging.getLogger(__name__)


# --- Model Loading Functions ---
def get_fasterrcnn_model(config: Dict[str, Any]) -> FasterRCNN:
    """Loads a pre-trained Faster R-CNN model and modifies the head."""
    model_config = config['model']
    num_classes = model_config["num_classes"]  # Should be Person(1) + Background(0) = 2
    weights_str = model_config.get("backbone_weights", "DEFAULT")
    trainable_layers = model_config.get("trainable_backbone_layers", 3)

    logger.info(f"Loading FasterRCNN model with weights identifier: {weights_str}")

    weight_key = weights_str
    enum_class_name = "FasterRCNN_ResNet50_FPN_Weights"
    if enum_class_name in weights_str:
         parts = weights_str.split('.')
         if len(parts) > 1: weight_key = parts[-1]
         else: weight_key = weights_str

    logger.info(f"Attempting to access weights enum member: {weight_key}")

    try:
        weights_enum_member = getattr(FasterRCNN_ResNet50_FPN_Weights, weight_key)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights_enum_member,
            trainable_backbone_layers=trainable_layers
        )
    except AttributeError:
        logger.error(f"Invalid backbone_weights key: '{weight_key}'. Check torchvision documentation for available FasterRCNN_ResNet50_FPN_Weights members.")
        raise
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        raise

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    logger.info(f"Model loaded. Output classes: {num_classes}. Trainable backbone layers: {trainable_layers}")
    return model


# --- Transforms ---
def get_transform(train: bool, config: Dict[str, Any]) -> T.Compose:
    """Gets the appropriate transforms for training or validation."""
    transforms = []
    # --- MODIFICATION: Convert PIL/np.ndarray to T.Image happens in Dataset __getitem__ ---
    # transforms.append(T.ToImage()) # Moved to Dataset
    # ----------------------------------------------------------------------------------
    if train:
        # These transforms work directly on T.Image and tv_tensors.BoundingBoxes
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        # Add more augmentations here if desired (e.g., T.ColorJitter)
    transforms.append(T.ToDtype(torch.float32, scale=True))  # Scales image to [0.0, 1.0]
    # Consider adding normalization if the backbone expects it
    # transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    # --- MODIFICATION: Remove SanitizeBoundingBoxes ---
    # transforms.append(T.SanitizeBoundingBoxes()) # REMOVED - Caused error with empty boxes
    # -------------------------------------------------
    return T.Compose(transforms)


# --- Main Runner Function for a Single Training Job ---
def run_single_training_job(
        run_config: Dict[str, Any],  # Contains combined config for this specific job
        device: torch.device,
        project_root: Path
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Executes a single training job based on the provided configuration.
    Handles model selection, engine dispatching, and MLflow logging within a nested run context.
    """
    active_run = mlflow.active_run()
    if not active_run:
        logger.critical("run_single_training_job called without an active MLflow run!")
        return "FAILED", None
    run_id = active_run.info.run_id
    job_status = "FAILED"
    final_metrics = None

    model_config = run_config['model']
    training_config = run_config['training']
    data_config = run_config['data']
    env_config = run_config['environment']
    model_type = model_config.get('type', 'unknown').lower()
    engine_type = training_config.get('engine', 'pytorch').lower()
    run_name_tag = model_config.get("name_tag", f"{model_type}_training_{run_id[:8]}")

    logger.info(f"--- Starting Single Training Job: {run_name_tag} (Run ID: {run_id}) ---")
    logger.info(f"Model Type: {model_type}, Engine: {engine_type}, Device: {device}")

    run_artifact_dir = project_root / "mlruns_temp_artifacts" / run_id
    run_artifact_dir.mkdir(parents=True, exist_ok=True)

    pytorch_person_class_id = 1

    try:
        logger.info("Logging parameters to MLflow...")
        log_params_recursive(run_config)
        mlflow.log_param("environment.actual_device", str(device))
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("engine_type", engine_type)
        log_git_info()

        logger.info("Creating Datasets and DataLoaders...")
        try:
            # Pass the configuration to get_transform
            dataset_train = MTMMCDetectionDataset(run_config, mode="train",
                                                  transforms=get_transform(train=True, config=run_config))
            dataset_val = MTMMCDetectionDataset(run_config, mode="val",
                                                transforms=get_transform(train=False, config=run_config))

            if len(dataset_train) == 0: raise ValueError("Training dataset is empty!")
            if len(dataset_val) == 0: logger.warning("Validation dataset is empty!")

            mlflow.log_param("dataset.num_train_samples", len(dataset_train))
            mlflow.log_param("dataset.num_val_samples", len(dataset_val))

            num_workers = data_config.get("num_workers", 0)
            batch_size = training_config.get("batch_size", 2)

            data_loader_train = DataLoader(
                dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True if device.type == 'cuda' else False,
                persistent_workers=True if num_workers > 0 else False,
                drop_last=True
            )
            data_loader_val = None
            if len(dataset_val) > 0:
                data_loader_val = DataLoader(
                    dataset_val, batch_size=max(1, batch_size // 1), shuffle=False, num_workers=num_workers,
                    collate_fn=collate_fn, pin_memory=True if device.type == 'cuda' else False,
                    persistent_workers=True if num_workers > 0 else False
                )
            logger.info("Datasets and DataLoaders created.")
        except Exception as e:
            logger.critical(f"Failed to create datasets/dataloaders: {e}", exc_info=True)
            raise

        if engine_type == "pytorch":
            logger.info("Executing with PyTorch training engine...")

            logger.info("Initializing PyTorch model...")
            if model_type != 'fasterrcnn':
                raise ValueError(f"PyTorch engine currently only supports 'fasterrcnn', got '{model_type}'")
            model = get_fasterrcnn_model(run_config)
            model.to(device)

            logger.info("Initializing optimizer and scheduler...")
            lr = training_config.get("learning_rate", 0.001)
            weight_decay = training_config.get("weight_decay", 0.005)
            opt_name = training_config.get("optimizer", "AdamW")
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = get_optimizer(opt_name, params, lr, weight_decay)

            scheduler_name = training_config.get("lr_scheduler", "StepLR").lower()
            scheduler_params = {
                "step_size": training_config.get("lr_scheduler_step_size", 5),
                "gamma": training_config.get("lr_scheduler_gamma", 0.1),
                "T_max": training_config.get("epochs", 10)
            }
            lr_scheduler = get_lr_scheduler(scheduler_name, optimizer, **scheduler_params)

            scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
            logger.info(f"AMP Scaler Enabled: {scaler is not None}")

            checkpoint_dir_str = training_config.get("checkpoint_dir", "checkpoints")
            checkpoint_dir = project_root / checkpoint_dir_str / run_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoints will be saved in: {checkpoint_dir}")

            num_epochs = training_config.get("epochs", 10)
            save_best_metric_name = training_config.get("save_best_metric", "val_map_50").replace("val_", "eval_")
            higher_is_better = "map" in save_best_metric_name
            best_metric_value = -float('inf') if higher_is_better else float('inf')
            logger.info(
                f"Tracking best model based on '{save_best_metric_name}' ({'higher' if higher_is_better else 'lower'} is better).")
            best_model_path = None
            latest_path = None

            logger.info(f"--- Starting PyTorch Training Loop ({num_epochs} Epochs) ---")
            start_time_training = time.time()

            for epoch in range(num_epochs):
                epoch_start_time = time.time()

                avg_train_loss, avg_train_comp_losses = train_one_epoch(
                    model, optimizer, data_loader_train, device, epoch, scaler=scaler
                )
                mlflow.log_metric("epoch_train_loss_avg", avg_train_loss, step=epoch)
                for k, v in avg_train_comp_losses.items():
                    mlflow.log_metric(f"epoch_train_loss_{k}", v, step=epoch)

                avg_val_loss = float('nan')
                eval_metrics = {}
                if data_loader_val:
                    avg_val_loss, eval_metrics = evaluate_pytorch(
                        model, data_loader_val, device, epoch, person_class_id=pytorch_person_class_id
                    )
                    mlflow.log_metric("epoch_val_loss_avg", avg_val_loss, step=epoch)
                    for k, v in eval_metrics.items():
                        mlflow.log_metric(k, v, step=epoch)
                else:
                    logger.warning(f"Epoch {epoch}: No validation data loader. Skipping evaluation.")

                current_lr = optimizer.param_groups[0]['lr']
                mlflow.log_metric("learning_rate", current_lr, step=epoch)
                if lr_scheduler:
                    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        lr_scheduler.step(avg_val_loss if data_loader_val else avg_train_loss)
                    else:
                        lr_scheduler.step()

                epoch_duration = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds. LR: {current_lr:.6f}")

                current_metric_value = eval_metrics.get(save_best_metric_name, None)
                metric_improved = False
                if current_metric_value is not None and not np.isnan(current_metric_value): # Check for NaN
                    if higher_is_better and current_metric_value > best_metric_value:
                        metric_improved = True
                        best_metric_value = current_metric_value
                    elif not higher_is_better and current_metric_value < best_metric_value:
                        metric_improved = True
                        best_metric_value = current_metric_value

                latest_path = checkpoint_dir / f"ckpt_epoch_{epoch}_latest.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                    'val_loss': avg_val_loss,
                    'val_metrics': eval_metrics
                }, latest_path)

                if metric_improved:
                    best_model_path = checkpoint_dir / f"ckpt_best_{save_best_metric_name}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                    }, best_model_path)
                    logger.info(
                        f"Saved new best model checkpoint ({save_best_metric_name}={best_metric_value:.4f}): {best_model_path.name}")
                    mlflow.log_param("best_model_epoch", epoch)
                    mlflow.set_tag(f"best_{save_best_metric_name}", f"{best_metric_value:.4f}")
                elif current_metric_value is None and epoch == 0:
                    best_model_path = checkpoint_dir / f"ckpt_best_epoch_0.pth"
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, best_model_path)
                    logger.info(f"Saved first epoch checkpoint as best (no validation): {best_model_path.name}")

            training_duration = time.time() - start_time_training
            logger.info(f"--- PyTorch Training Finished in {training_duration:.2f} seconds ---")

            if best_model_path and best_model_path.exists():
                logger.info(f"Logging best model checkpoint artifact: {best_model_path.name}")
                mlflow.log_artifact(str(best_model_path), artifact_path="checkpoints")
            if latest_path and latest_path.exists():
                mlflow.log_artifact(str(latest_path), artifact_path="checkpoints/latest")

            final_metrics = eval_metrics if data_loader_val else {"train_loss": avg_train_loss}
            job_status = "FINISHED"

        elif engine_type == "ultralytics":
            logger.error(f"Attempted to use Ultralytics engine, but it's not configured or supported in this setup.")
            job_status = "FAILED"

        else:
            logger.error(f"Unknown engine type: '{engine_type}' specified in training config.")
            job_status = "FAILED"

    except KeyboardInterrupt:
        logger.warning(f"[{run_name_tag}] Training job interrupted by user.")
        job_status = "KILLED"
        if mlflow.active_run(): mlflow.set_tag("run_outcome", "Killed by user")
        raise

    except Exception as e:
        logger.critical(f"[{run_name_tag}] An uncaught error occurred during the training job: {e}", exc_info=True)
        job_status = "FAILED"
        if mlflow.active_run():
            mlflow.set_tag("run_outcome", "Crashed")
            try:
                error_log_content = f"Error Type: {type(e).__name__}\nError Message: {e}\n\nTraceback:\n{traceback.format_exc()}"
                mlflow.log_text(error_log_content, "error_log.txt")
            except Exception as log_err:
                logger.error(f"Failed to log training error details to MLflow: {log_err}")

    finally:
        logger.info(f"--- Finished Single Training Job: {run_name_tag} (Attempted Status: {job_status}) ---")
        try:
            import shutil
            # Also need to import numpy for the NaN check in the loop
            import numpy as np
            if run_artifact_dir.exists():
                shutil.rmtree(run_artifact_dir)
                logger.debug(f"Cleaned up temporary artifact directory: {run_artifact_dir}")
        except Exception as cleanup_err:
            logger.warning(f"Could not cleanup temporary artifact directory {run_artifact_dir}: {cleanup_err}")

    return job_status, final_metrics