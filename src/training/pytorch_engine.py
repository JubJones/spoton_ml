import math

import time
import logging
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from torch.utils.data import DataLoader
from tqdm import tqdm


from src.evaluation.metrics import compute_map_metrics, MAPPred, MAPTarget, TORCHMETRICS_AVAILABLE

logger = logging.getLogger(__name__)


def train_one_epoch(
        model: nn.Module,
        optimizer: Optimizer,
        data_loader: DataLoader,
        device: torch.device,
        epoch: int,
        log_interval: int = 50,
        scaler: Optional[torch.cuda.amp.GradScaler] = None  # For mixed precision
) -> Tuple[float, Dict[str, float]]:
    """
    Trains the model for one epoch using standard PyTorch logic.
    """
    model.train()
    total_loss = 0.0
    agg_loss_dict = {}
    num_batches = len(data_loader)
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    start_time = time.time()

    for i, batch in enumerate(pbar):
        # Handle potential errors in data loading/collation
        try:
            images, targets = batch
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        except Exception as data_err:
            logger.warning(f"Skipping training batch {i} due to data error: {data_err}")
            continue

        try:
            optimizer.zero_grad()

            # Automatic Mixed Precision (AMP) context
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)  # Model returns dict of losses
                losses = sum(loss for loss in loss_dict.values())

            # Check for invalid loss
            loss_value = losses.item()
            if not math.isfinite(loss_value):
                logger.error(f"Loss is {loss_value} at Epoch {epoch}, Batch {i}. Stopping training.")
                logger.error(f"Loss components: {loss_dict}")
                # Consider raising an error or returning a special value
                raise RuntimeError(f"Non-finite loss encountered: {loss_value}")

            total_loss += loss_value

            # Accumulate component losses
            for k, v in loss_dict.items():
                agg_loss_dict[k] = agg_loss_dict.get(k, 0.0) + v.item()

            # Backward pass & Optimization
            if scaler is not None:
                scaler.scale(losses).backward()
                # Optional: Gradient Clipping
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                # Optional: Gradient Clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Logging
            if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
                batch_loss_str = f"Loss: {loss_value:.4f}"
                pbar.set_postfix_str(batch_loss_str)

        except Exception as e:
            logger.error(f"Error during training batch {i}, Epoch {epoch}: {e}", exc_info=True)
            # Optionally skip batch or re-raise
            raise e  # Re-raise to stop training run

    end_time = time.time()
    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_component_losses = {k: v / num_batches for k, v in agg_loss_dict.items()} if num_batches > 0 else {}

    logger.info(
        f"Epoch {epoch} [Train] Avg Loss: {avg_epoch_loss:.4f} "
        f"({', '.join([f'{k}: {v:.4f}' for k, v in avg_component_losses.items()])}) "
        f"Time: {end_time - start_time:.2f}s"
    )
    return avg_epoch_loss, avg_component_losses


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

    # Accumulators for mAP calculation
    predictions_for_map: List[MAPPred] = []
    targets_for_map: List[MAPTarget] = []

    for i, batch in enumerate(pbar):
        # Handle potential errors in data loading/collation
        try:
            images, targets = batch
            images = list(img.to(device) for img in images)
            targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
        except Exception as data_err:
            logger.warning(f"Skipping validation batch {i} due to data error: {data_err}")
            continue

        try:
            # --- 1. Calculate Loss ---
            # Temporarily switch to train mode *only* if needed to get loss dict
            # Some models might allow loss calculation in eval mode, check model docs
            # For standard torchvision detection models, this switch is often needed.
            with torch.no_grad():  # Ensure no grads outside the temporary switch
                original_mode = model.training
                model.train()
                # Need gradients enabled briefly *within* the model forward for loss calculation
                with torch.enable_grad():
                    loss_dict = model(images, targets_on_device)
                model.train(original_mode)  # Set back to eval mode immediately

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            total_loss += loss_value

            # --- 2. Get Predictions for mAP ---
            if TORCHMETRICS_AVAILABLE:
                # Get predictions in EVAL mode
                outputs = model(images)
                # Move predictions and original targets (before device move) to CPU
                preds_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]
                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

                predictions_for_map.extend(preds_cpu)
                targets_for_map.extend(targets_cpu)

            pbar.set_postfix_str(f"Loss: {loss_value:.4f}")

        except Exception as e:
            logger.error(f"Error during evaluation batch {i}, Epoch {epoch}: {e}", exc_info=True)
            continue  # Continue evaluating other batches

    end_time = time.time()
    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # --- Compute Final mAP Metrics ---
    map_results = {}
    if TORCHMETRICS_AVAILABLE and predictions_for_map and targets_for_map:
        logger.info(f"Computing mAP metrics for Epoch {epoch} [Val]...")
        try:
            # Use CPU for metric calculation usually safer for memory
            map_results = compute_map_metrics(
                predictions_for_map,
                targets_for_map,
                device=torch.device('cpu'),  # Use CPU for metrics
                person_class_id=person_class_id
            )
            logger.info(f"Epoch {epoch} [Val] mAP Results: {map_results}")
        except Exception as map_err:
            logger.error(f"Error computing mAP metrics for Epoch {epoch}: {map_err}", exc_info=True)
            map_results = {  # Provide default error values
                "eval_map": -1.0, "eval_map_50": -1.0, "eval_map_75": -1.0, "eval_ap_person": -1.0
            }
    elif TORCHMETRICS_AVAILABLE:
        logger.warning("No predictions/targets collected for mAP calculation in validation.")
        map_results = {"eval_map": -1.0, "eval_map_50": -1.0, "eval_map_75": -1.0, "eval_ap_person": -1.0}

    logger.info(
        f"Epoch {epoch} [Val] Average Loss: {avg_epoch_loss:.4f} "
        f"Time: {end_time - start_time:.2f}s"
    )

    return avg_epoch_loss, map_results  # Return loss and mAP dict
