import math
import time
import logging
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.evaluation.metrics import compute_map_metrics, MAPPred, MAPTarget, TORCHMETRICS_AVAILABLE

logger = logging.getLogger(__name__)


def train_one_epoch(
        model: nn.Module,
        optimizer: Optimizer,
        data_loader: DataLoader,
        device: torch.device,
        epoch: int,
        log_interval: int = 50,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        gradient_clip_norm: Optional[float] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Trains the model for one epoch using standard PyTorch logic.
    Includes optional gradient clipping.
    """
    model.train()
    total_loss = 0.0
    agg_loss_dict = {}
    num_batches = len(data_loader)
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    start_time = time.time()

    if gradient_clip_norm:
        logger.debug(f"Gradient clipping enabled with max_norm={gradient_clip_norm}")

    for i, batch in enumerate(pbar):
        try:
            images, targets = batch
            images = list(img.to(device) for img in images)
            targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets] # Keep original targets on CPU for logging
        except Exception as data_err:
            logger.warning(f"Skipping training batch {i} due to data error: {data_err}")
            continue

        try:
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets_on_device)
                valid_losses = [loss for loss in loss_dict.values() if torch.isfinite(loss)]

                if len(valid_losses) != len(loss_dict):
                    invalid_loss_keys = [k for k, v in loss_dict.items() if not torch.isfinite(v)]
                    logger.warning(f"Non-finite loss detected in components: {invalid_loss_keys} at Epoch {epoch}, Batch {i}. Using only finite losses for sum.")
                    # --- MODIFICATION: Log details if ALL are non-finite ---
                    if not valid_losses:
                        logger.error(f"All loss components non-finite at Epoch {epoch}, Batch {i}. Stopping training.")
                        logger.error(f"Loss components: {loss_dict}")
                        # Log target details for the failing batch
                        try:
                             logger.error("--- Target details for failing batch ---")
                             for idx, target_dict in enumerate(targets): # Use original targets on CPU
                                 logger.error(f" Target {idx}:")
                                 for key, value in target_dict.items():
                                     if isinstance(value, torch.Tensor):
                                         logger.error(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}, values=\n{value}")
                                     elif hasattr(value, 'canvas_size') and hasattr(value, 'data'): # Handle tv_tensors.BoundingBoxes
                                         logger.error(f"  {key}: type={type(value)}, format={getattr(value,'format','N/A')}, canvas={value.canvas_size}, shape={value.shape}, dtype={value.dtype}, device={value.device}, values=\n{value.data}")
                                     else:
                                         logger.error(f"  {key}: {value}")
                             logger.error("--- End target details ---")
                        except Exception as log_ex:
                             logger.error(f"Failed to log target details: {log_ex}")
                        # -----------------------------------------------------
                        raise RuntimeError("All loss components non-finite.")
                    # -----------------------------------------------------
                    losses = sum(valid_losses)
                else:
                    losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()
            if not math.isfinite(loss_value):
                logger.error(f"Loss is {loss_value} at Epoch {epoch}, Batch {i}. Stopping training.")
                logger.error(f"Loss components: {loss_dict}")
                raise RuntimeError(f"Non-finite loss encountered: {loss_value}")

            total_loss += loss_value

            for k, v in loss_dict.items():
                if torch.isfinite(v):
                     agg_loss_dict[k] = agg_loss_dict.get(k, 0.0) + v.item()
                else:
                     agg_loss_dict[k+"_nonfinite_count"] = agg_loss_dict.get(k+"_nonfinite_count", 0) + 1

            if scaler is not None:
                scaler.scale(losses).backward()
                if gradient_clip_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                if gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()

            if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
                batch_loss_str = f"Loss: {loss_value:.4f}"
                pbar.set_postfix_str(batch_loss_str)

        except Exception as e:
            # Add type check to avoid re-logging the specific RuntimeError we raised
            if not isinstance(e, RuntimeError) or "non-finite" not in str(e):
                 logger.error(f"Error during training batch {i}, Epoch {epoch}: {e}", exc_info=True)
            raise e # Re-raise exception

    end_time = time.time()
    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_component_losses = {
        k: v / num_batches for k, v in agg_loss_dict.items() if "_nonfinite_count" not in k
    } if num_batches > 0 else {}
    non_finite_counts = {
         k.replace("_nonfinite_count",""): v for k, v in agg_loss_dict.items() if "_nonfinite_count" in k
    }

    loss_log_str = f"Avg Loss: {avg_epoch_loss:.4f}"
    if avg_component_losses:
         loss_log_str += f" ({', '.join([f'{k}: {v:.4f}' for k, v in avg_component_losses.items()])})"
    if non_finite_counts:
         loss_log_str += f" [NonFiniteCounts: {non_finite_counts}]"

    logger.info(
        f"Epoch {epoch} [Train] {loss_log_str} Time: {end_time - start_time:.2f}s"
    )
    return avg_epoch_loss, avg_component_losses


# (evaluate function remains unchanged)
@torch.no_grad()
def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        epoch: int,
        person_class_id: int
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates the model on the validation set, calculating loss and mAP metrics.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(data_loader)
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    start_time = time.time()

    predictions_for_map: List[MAPPred] = []
    targets_for_map: List[MAPTarget] = []

    for i, batch in enumerate(pbar):
        try:
            images, targets = batch
            images = list(img.to(device) for img in images)
            targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
        except Exception as data_err:
            logger.warning(f"Skipping validation batch {i} due to data error: {data_err}")
            continue

        try:
            original_mode = model.training
            model.train()
            with torch.enable_grad():
                loss_dict = model(images, targets_on_device)
            model.train(original_mode)

            valid_losses = [loss for loss in loss_dict.values() if torch.isfinite(loss)]
            if not valid_losses:
                 logger.warning(f"All validation loss components non-finite in Epoch {epoch}, Batch {i}. Skipping loss accumulation.")
                 loss_value = float('nan')
            else:
                 if len(valid_losses) != len(loss_dict):
                     logger.warning(f"Non-finite validation loss components detected in Epoch {epoch}, Batch {i}.")
                 losses = sum(valid_losses)
                 loss_value = losses.item()
                 if math.isfinite(loss_value):
                      total_loss += loss_value

            if TORCHMETRICS_AVAILABLE:
                model.eval()
                outputs = model(images)
                preds_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

                predictions_for_map.extend(preds_cpu)
                targets_for_map.extend(targets_cpu)

            pbar.set_postfix_str(f"Loss: {loss_value:.4f}")

        except Exception as e:
            logger.error(f"Error during evaluation batch {i}, Epoch {epoch}: {e}", exc_info=True)
            continue

    end_time = time.time()
    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else float('nan')

    map_results = {}
    if TORCHMETRICS_AVAILABLE and predictions_for_map and targets_for_map:
        logger.info(f"Computing mAP metrics for Epoch {epoch} [Val]...")
        try:
            map_results = compute_map_metrics(
                predictions_for_map,
                targets_for_map,
                device=torch.device('cpu'),
                person_class_id=person_class_id
            )
            logger.info(f"Epoch {epoch} [Val] mAP Results: {map_results}")
        except Exception as map_err:
            logger.error(f"Error computing mAP metrics for Epoch {epoch}: {map_err}", exc_info=True)
            map_results = {"eval_map": -1.0, "eval_map_50": -1.0, "eval_map_75": -1.0, "eval_ap_person": -1.0}
    elif TORCHMETRICS_AVAILABLE:
        logger.warning("No predictions/targets collected for mAP calculation in validation.")
        map_results = {"eval_map": -1.0, "eval_map_50": -1.0, "eval_map_75": -1.0, "eval_ap_person": -1.0}

    logger.info(
        f"Epoch {epoch} [Val] Average Loss: {avg_epoch_loss:.4f} "
        f"Time: {end_time - start_time:.2f}s"
    )

    return avg_epoch_loss, map_results