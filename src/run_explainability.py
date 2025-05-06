import logging
import sys
import time
import traceback
from pathlib import Path
import random
from typing import Dict, Any, Optional, List, Tuple

import cv2
import torch
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib # Use Agg backend for non-interactive use
matplotlib.use('Agg')

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"PROJECT_ROOT added to sys.path: {PROJECT_ROOT}")
# --- End Project Setup ---

# --- Local Imports ---
try:
    from src.utils.config_loader import load_config
    from src.utils.logging_utils import setup_logging
    from src.utils.device_utils import get_selected_device
    from src.utils.reproducibility import set_seed
    from src.utils.mlflow_utils import setup_mlflow_experiment
    from src.core.runner import log_params_recursive, log_git_info
    from src.inference.detector import load_trained_fasterrcnn, infer_single_image
    from src.training.runner import get_transform
    from src.explainability import (
        explain_detection_gradcam,
        visualize_explanation,
        generate_reasoning_text,
        get_target_layer,
        SUPPORTED_METHODS,
        CV2_COLORMAPS
    )
except ImportError as e:
    print(f"Error importing project modules in src/run_explainability.py: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
# --- End Local Imports ---

logger = logging.getLogger(__name__)

# --- Main Execution Function (Orchestrator) ---
def main():
    """Main execution function for the explainability runner."""
    config_path_str = "configs/explainability_config.yaml"
    final_status = "FAILED"
    run_id = None
    run_name_global = "explainability_run" # Default if not in config

    config = load_config(config_path_str)
    if not config:
        print(f"Failed to load config file: {config_path_str}", file=sys.stderr)
        sys.exit(1)

    mlflow_cfg = config.get("mlflow", {})
    env_config = config.get("environment", {})
    exp_config = config.get("explainability", {})

    run_name_global = config.get("run_name", f"explainability_{time.strftime('%Y%m%d_%H%M%S')}")

    log_dir = PROJECT_ROOT / "logs"
    log_file = setup_logging("explainability", log_dir)
    logger.info(f"--- Starting Explainability Run: {run_name_global} ---")

    seed = env_config.get("seed", random.randint(0, 10000))
    set_seed(seed)
    logger.info(f"Using seed: {seed}")

    device_pref = env_config.get("device", "auto")
    device = get_selected_device(device_pref)
    logger.info(f"Using device: {device}")

    output_dir_str = exp_config.get("output_dir", "outputs/explanations")
    output_dir = PROJECT_ROOT / output_dir_str
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Explainability results will be saved locally to: {output_dir.relative_to(PROJECT_ROOT)}")
    reasoning_log_path = output_dir / f"{run_name_global}_reasoning_log.txt"

    log_to_mlflow = mlflow_cfg.get("log_artifacts", False)
    experiment_id = None
    if log_to_mlflow:
        default_exp_name = mlflow_cfg.get("experiment_name", "Default Explainability")
        experiment_id = setup_mlflow_experiment(config, default_exp_name)
        if not experiment_id:
            logger.warning("MLflow artifact logging enabled but experiment setup failed. Will proceed without MLflow logging.")
            log_to_mlflow = False
        else:
            logger.info(f"MLflow logging enabled. Experiment: {default_exp_name} ({experiment_id})")

    try:
        if log_to_mlflow and experiment_id:
            with mlflow.start_run(run_name=run_name_global, experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                logger.info(f"Started MLflow Run ID: {run_id}")
                log_params_recursive(config)
                mlflow.log_param("environment.seed", seed)
                mlflow.log_param("environment.requested_device", device_pref)
                mlflow.log_param("environment.actual_device", str(device))
                mlflow.set_tag("run_type", "explainability")
                log_git_info()

                cfg_path_abs = (PROJECT_ROOT / config_path_str).resolve()
                if cfg_path_abs.exists():
                     mlflow.log_artifact(str(cfg_path_abs), artifact_path="config")

                success, final_status, metrics = execute_explainability(
                    config=config,
                    device=device,
                    output_dir=output_dir,
                    reasoning_log_path=reasoning_log_path,
                    mlflow_run_id=run_id
                )
                if metrics: mlflow.log_metrics(metrics)
                mlflow.set_tag("run_outcome", final_status)
        else:
             success, final_status, metrics = execute_explainability(
                 config=config,
                 device=device,
                 output_dir=output_dir,
                 reasoning_log_path=reasoning_log_path,
                 mlflow_run_id=None
             )
             if metrics: logger.info(f"Execution Metrics: {metrics}")

        if not success and final_status == "FINISHED":
             final_status = "FAILED"

    except KeyboardInterrupt:
        logger.warning("Explainability run interrupted by user.")
        final_status = "KILLED"
        if run_id and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
             mlflow.set_tag("run_outcome", "Killed by user")
             mlflow.end_run(status="KILLED")
        elif run_id:
             try: MlflowClient().set_terminated(run_id, status="KILLED")
             except Exception: pass

    except Exception as e:
        logger.critical(f"An uncaught error occurred in main: {e}", exc_info=True)
        final_status = "FAILED"
        if run_id and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
            mlflow.set_tag("run_outcome", "Crashed - Outer")
            try: mlflow.log_text(f"Outer Error: {type(e).__name__}\n{e}\n{traceback.format_exc()}", "error_log.txt")
            except Exception: pass
            mlflow.end_run(status="FAILED")
        elif run_id:
            try:
                client = MlflowClient()
                client.set_tag(run_id, "run_outcome", "Crashed - Outer")
                client.set_terminated(run_id, status="FAILED")
            except Exception: pass

    finally:
        logger.info(f"--- Finalizing Explainability Run (Status: {final_status}) ---")
        if log_to_mlflow and run_id and log_file.exists():
            try:
                for handler in logging.getLogger().handlers: handler.flush()
                client = MlflowClient()
                current_run_info = client.get_run(run_id)
                if current_run_info.info.status in ["RUNNING", "SCHEDULED"]:
                    client.log_artifact(run_id, str(log_file), artifact_path="logs")
                    logger.info(f"Logged runner script log to MLflow run {run_id}.")
                else:
                    logger.warning(f"Skipping script log artifact logging as run {run_id} is already terminated ({current_run_info.info.status}).")
            except Exception as log_err:
                logger.warning(f"Could not log script log artifact '{log_file.name}': {log_err}")

        active_run_obj = mlflow.active_run()
        if active_run_obj and active_run_obj.info.run_id == run_id:
             mlflow.end_run(status=final_status)
        elif run_id:
             try:
                 client = MlflowClient()
                 current_run_info = client.get_run(run_id)
                 if current_run_info.info.status in ["RUNNING", "SCHEDULED"]:
                     client.set_terminated(run_id, status=final_status)
                     logger.info(f"Terminated MLflow run {run_id} externally with status {final_status}.")
                 else:
                     logger.info(f"MLflow run {run_id} already terminated ({current_run_info.info.status}).")
             except Exception as term_err:
                  logger.error(f"Failed to terminate run {run_id} externally: {term_err}")

        logger.info(f"--- Explainability Run Completed (Status: {final_status}) ---")
        sys.exit(0 if final_status == "FINISHED" else 1)


# --- Core Explainability Execution Logic ---
def execute_explainability(
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    reasoning_log_path: Path,
    mlflow_run_id: Optional[str] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Core logic for loading model, running inference, and generating explanations.
    Logs artifacts and metrics to MLflow if mlflow_run_id is provided.
    """
    # ... (initial setup, parameter extraction, model loading, etc. remain the same) ...
    model_config = config.get("model", {})
    images_to_explain = config.get("images_to_explain", [])
    exp_config = config.get("explainability", {})
    config_path_str = "configs/explainability_config.yaml"

    run_name = config.get("run_name", "explainability_run")
    metrics = {
        "images_processed_count": 0, "images_failed_count": 0,
        "images_skipped_count": 0, "detections_explained_count": 0,
        "detections_skipped_count": 0, "visualizations_generated_count": 0
    }

    target_layer_name = exp_config.get("target_layer_name")
    person_class_index = model_config.get("person_class_id", 1)
    logger.info(f"Using person_class_index: {person_class_index} for explainability target")
    conf_thresh = exp_config.get("confidence_threshold_for_explanation", 0.6)
    top_n = exp_config.get("top_n_to_explain", 3)
    method = exp_config.get("method", "gradcam").lower()
    alpha = exp_config.get("heatmap_alpha", 0.6)
    colormap_name = exp_config.get("colormap_name", "COLORMAP_JET")
    colormap_int = CV2_COLORMAPS.get(colormap_name.upper())
    if colormap_int is None:
        logger.warning(f"Invalid colormap name '{colormap_name}'. Defaulting to COLORMAP_JET.")
        colormap_int = cv2.COLORMAP_JET

    # --- Input Validation ---
    if not images_to_explain: logger.error("..."); return False, "FAILED", metrics
    checkpoint_path = model_config.get("checkpoint_path")
    if not checkpoint_path: logger.error("..."); return False, "FAILED", metrics
    num_classes = model_config.get("num_classes")
    if not num_classes: logger.error("..."); return False, "FAILED", metrics
    if method not in SUPPORTED_METHODS: logger.error("..."); return False, "FAILED", metrics
    if not target_layer_name and method == "gradcam": logger.error("..."); return False, "FAILED", metrics

    # --- Load Model ---
    try:
        model = load_trained_fasterrcnn( checkpoint_path, device, num_classes, model_config.get('trainable_backbone_layers', 3), mlflow_run_id)
    except Exception as e: logger.critical(f"..."); return False, "FAILED", metrics

    # --- Get Target Layer ---
    target_layer = None
    if method == "gradcam":
        target_layer = get_target_layer(model, target_layer_name)
        if target_layer is None: return False, "FAILED", metrics

    # --- Get Transforms ---
    dummy_run_config_for_transform = {'data': config.get('data', {})}
    transforms = get_transform(train=False, config=dummy_run_config_for_transform)

    # --- Process Images ---
    all_reasoning_texts = []

    for image_path_str in images_to_explain:
        image_path = Path(image_path_str)
        image_name = image_path.stem
        logger.info(f"\n--- Processing Image: {image_path.name} ---")

        if not image_path.is_absolute():
            image_path = (PROJECT_ROOT / image_path).resolve()
        if not image_path.exists():
            logger.error(f"..."); metrics["images_failed_count"] += 1; all_reasoning_texts.append("..."); continue

        detections, original_image_rgb, input_tensor = infer_single_image( model, image_path, transforms, device, conf_thresh, person_class_index)

        if detections is None or original_image_rgb is None or input_tensor is None:
            logger.error(f"..."); metrics["images_failed_count"] += 1; all_reasoning_texts.append("..."); continue

        metrics["images_processed_count"] += 1
        all_reasoning_texts.append(f"--- Image: {image_path_str} ---")

        all_boxes_in_image = [d['box'] for d in detections] # Still need this list

        if not detections:
            logger.warning(f"..."); all_reasoning_texts.append("..."); metrics["images_skipped_count"] +=1; continue

        detections.sort(key=lambda d: d['score'], reverse=True)
        detections_to_explain = detections[:top_n]
        logger.info(f"Found {len(detections)} persons. Explaining top {len(detections_to_explain)}.")
        metrics["detections_skipped_count"] += max(0, len(detections) - len(detections_to_explain))

        for i, det in enumerate(detections_to_explain):
            logger.info(f"  Explaining detection {i+1}/{len(detections_to_explain)} (Score: {det['score']:.3f})...")
            # det_box = det['box'] # No longer needed for highlighting
            det_score = det['score']
            det_label = det['label']
            attribution_map = None
            viz_path = None
            has_visualization = False

            if method == "gradcam":
                if target_layer is None: continue
                original_index = -1
                for orig_idx, orig_det in enumerate(detections):
                    # Simple check - might need refinement if scores/boxes are very close
                    if abs(orig_det['score'] - det_score) < 1e-5 and np.allclose(orig_det['box'], det['box'], atol=1e-2):
                         original_index = orig_idx
                         break
                if original_index == -1: original_index = i
                attribution_map = explain_detection_gradcam( model, input_tensor, target_layer, original_index, person_class_index, device )

            if attribution_map is not None:
                viz_filename = f"{image_name}_explain_{method}_det{i}_score{det_score:.2f}.png"
                viz_path = output_dir / viz_filename
                # --- MODIFICATION: Update visualize_explanation call ---
                visualize_explanation(
                    original_image_rgb=original_image_rgb,
                    attribution_map=attribution_map,
                    output_path=viz_path,
                    # box_to_highlight=None, # Removed
                    boxes_to_draw=all_boxes_in_image, # Pass the full list here
                    score=det_score, # Score of the detection being explained
                    label=det_label, # Label of the detection being explained
                    alpha=alpha,
                    colormap=colormap_int
                )
                # --- END MODIFICATION ---
                has_visualization = True
                metrics["visualizations_generated_count"] += 1
                if mlflow_run_id:
                    try: mlflow.log_artifact(str(viz_path), artifact_path=f"explanations/{image_name}"); logger.info("...")
                    except Exception as mlflow_log_err: logger.warning(f"...")
            else:
                logger.warning(f"...")

            reasoning_text = generate_reasoning_text(det, f"{method.upper()} Focus", has_visualization)
            det_box_coords = det['box'] # Get box coords for logging text
            log_entry = (
                f" Detection {i+1}:"
                f" Box=[{int(det_box_coords[0])},{int(det_box_coords[1])},{int(det_box_coords[2])},{int(det_box_coords[3])}]," # Log box coords
                f" Score={det_score:.4f}, Label={det_label}\n"
                f"   Reasoning: {reasoning_text}\n"
            )
            if viz_path: log_entry += f"   Visualization: {viz_path.relative_to(PROJECT_ROOT)}\n"
            all_reasoning_texts.append(log_entry)
            print(log_entry)
            metrics["detections_explained_count"] += 1

    # ... (Writing summary log remains the same) ...
    logger.info(f"Writing reasoning summary to: {reasoning_log_path}")
    try:
        with open(reasoning_log_path, "w") as f:
            f.write(f"Explainability Run Summary: {run_name}\n")
            f.write(f"Config: {config_path_str}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Method: {method.upper()}\n")
            f.write(f"Person Class Index Used: {person_class_index}\n")
            f.write(f"Processed: {metrics['images_processed_count']} | Failed: {metrics['images_failed_count']} | Skipped: {metrics['images_skipped_count']} (images)\n")
            f.write(f"Explained: {metrics['detections_explained_count']} | Skipped: {metrics['detections_skipped_count']} (detections)\n")
            f.write("="*40 + "\n\n")
            f.write("\n".join(all_reasoning_texts))
        if mlflow_run_id:
            try: mlflow.log_artifact(str(reasoning_log_path), artifact_path="summary"); logger.info("...")
            except Exception as mlflow_log_err: logger.warning(f"...")
    except Exception as write_err:
        logger.error(f"Failed to write reasoning summary log: {write_err}")
        return False, "FAILED", metrics

    # ... (Determine final status remains the same) ...
    if metrics["images_failed_count"] > 0: logger.error("..."); return False, "FAILED", metrics
    elif metrics["images_processed_count"] == 0: logger.warning("..."); return True, "FINISHED", metrics
    else: return True, "FINISHED", metrics


if __name__ == "__main__":
    main()