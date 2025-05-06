# FILE: src/run_explainability.py
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
import matplotlib.pyplot as plt


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
    from src.utils.mlflow_utils import setup_mlflow_experiment # Reused for experiment setup
    from src.inference.detector import load_trained_fasterrcnn, infer_single_image
    from src.training.runner import get_transform # Reuse transform getter
    from src.explainability import (
        explain_detection_gradcam,
        visualize_explanation,
        generate_reasoning_text,
        get_target_layer,
        SUPPORTED_METHODS
    )
except ImportError as e:
    print(f"Error importing project modules in src/run_explainability.py: {e}", file=sys.stderr)
    sys.exit(1)
# --- End Local Imports ---

# Setup logger for this script
logger = logging.getLogger(__name__)

# --- OpenCV Colormap Mapping ---
CV2_COLORMAPS = {name: getattr(cv2, name) for name in dir(cv2) if name.startswith('COLORMAP_')}

def main():
    """Main execution function for the explainability runner."""
    config_path_str = "configs/explainability_config.yaml"
    final_status = "FAILED"
    run_id = None
    run_name_global = "explainability_run" # Default if not in config

    # --- Configuration Loading ---
    config = load_config(config_path_str)
    if not config:
        print(f"Failed to load config file: {config_path_str}", file=sys.stderr)
        sys.exit(1)

    # Extract config sections
    mlflow_cfg = config.get("mlflow", {})
    env_config = config.get("environment", {})
    # No need for model_config, data_config, exp_config here, used in execute_explainability

    # Get run_name from config or generate default
    run_name_global = config.get("run_name", f"explainability_{time.strftime('%Y%m%d_%H%M%S')}")

    # --- Basic Setup ---
    log_dir = PROJECT_ROOT / "logs"
    log_file = setup_logging("explainability", log_dir) # Capture log file path
    logger.info(f"--- Starting Explainability Run: {run_name_global} ---")

    seed = env_config.get("seed", random.randint(0, 10000))
    set_seed(seed)
    logger.info(f"Using seed: {seed}")

    device_pref = env_config.get("device", "auto")
    device = get_selected_device(device_pref)
    logger.info(f"Using device: {device}")

    # --- Output Directory ---
    output_dir_str = config.get("explainability", {}).get("output_dir", "outputs/explanations") # Get from exp_config
    output_dir = PROJECT_ROOT / output_dir_str
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Explainability results will be saved to: {output_dir.relative_to(PROJECT_ROOT)}")
    # Use the run_name_global for the log file name
    reasoning_log_path = output_dir / f"{run_name_global}_reasoning_log.txt"

    # --- MLflow Setup (Optional) ---
    log_to_mlflow = mlflow_cfg.get("log_artifacts", False)
    experiment_id = None
    if log_to_mlflow:
        default_exp_name = mlflow_cfg.get("experiment_name", "Default Explainability")
        experiment_id = setup_mlflow_experiment(config, default_exp_name)
        if not experiment_id:
            logger.warning("MLflow artifact logging enabled but experiment setup failed. Will proceed without MLflow logging.")
            log_to_mlflow = False # Disable if setup failed
        else:
            logger.info(f"MLflow logging enabled. Experiment: {default_exp_name} ({experiment_id})")


    # --- Main Execution Block ---
    try:
        if log_to_mlflow and experiment_id:
            with mlflow.start_run(run_name=run_name_global, experiment_id=experiment_id) as run:
                run_id = run.info.run_id
                logger.info(f"Started MLflow Run ID: {run_id}")
                mlflow.log_param("seed", seed)
                mlflow.log_param("device_preference", device_pref)
                mlflow.log_param("actual_device", str(device))
                mlflow.log_dict(config, "config.yaml") # Log the config used

                # Core logic inside the MLflow run context
                # Pass run_name_global to the execution function
                success, final_status = execute_explainability(
                    config=config,
                    device=device,
                    output_dir=output_dir,
                    reasoning_log_path=reasoning_log_path,
                    mlflow_run_id=run_id,
                    run_name=run_name_global # Pass run_name
                )
                if not success:
                    final_status = "FAILED"
                mlflow.set_tag("run_outcome", final_status)
        else:
             # Run without MLflow context
             # Pass run_name_global to the execution function
             success, final_status = execute_explainability(
                 config=config,
                 device=device,
                 output_dir=output_dir,
                 reasoning_log_path=reasoning_log_path,
                 mlflow_run_id=None,
                 run_name=run_name_global # Pass run_name
             )
             if not success:
                  final_status = "FAILED"


    except KeyboardInterrupt:
        logger.warning("Explainability run interrupted by user.")
        final_status = "KILLED"
        if run_id and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
             mlflow.set_tag("run_outcome", "Killed")
             mlflow.end_run(status="KILLED")
        elif run_id:
             try: MlflowClient().set_terminated(run_id, status="KILLED")
             except Exception: pass

    except Exception as e:
        logger.critical(f"An uncaught error occurred: {e}", exc_info=True)
        final_status = "FAILED"
        if run_id and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
            mlflow.set_tag("run_outcome", "Crashed")
            try: mlflow.log_text(f"Error: {type(e).__name__}\n{e}\n{traceback.format_exc()}", "error_log.txt")
            except Exception: pass
            mlflow.end_run(status="FAILED")
        elif run_id:
            try:
                client = MlflowClient()
                client.set_tag(run_id, "run_outcome", "Crashed")
                client.set_terminated(run_id, status="FAILED")
            except Exception: pass

    finally:
        logger.info(f"--- Finalizing Explainability Run (Status: {final_status}) ---")
        # Log the script's own log file to MLflow if enabled and run exists
        if log_to_mlflow and run_id and log_file.exists():
            try:
                for handler in logging.getLogger().handlers: handler.flush() # Ensure logs written
                client = MlflowClient()
                client.log_artifact(run_id, str(log_file), artifact_path="logs")
                logger.info(f"Logged runner script log to MLflow run {run_id}.")
            except Exception as log_err:
                logger.warning(f"Could not log script log artifact '{log_file.name}': {log_err}")

        # Ensure run termination if not ended properly
        active_run_obj = mlflow.active_run()
        if active_run_obj and active_run_obj.info.run_id == run_id:
             mlflow.end_run(status=final_status)
        elif run_id and not active_run_obj: # Terminate externally if needed
             try: MlflowClient().set_terminated(run_id, status=final_status)
             except Exception as term_err: logger.error(f"Failed to terminate run {run_id} externally: {term_err}")

        logger.info(f"--- Explainability Run Completed (Status: {final_status}) ---")
        sys.exit(0 if final_status == "FINISHED" else 1)

# --- Updated function signature to accept run_name ---
def execute_explainability(
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    reasoning_log_path: Path,
    mlflow_run_id: Optional[str],
    run_name: str # Added argument
) -> Tuple[bool, str]:
    """
    Core logic for loading model, running inference, and generating explanations.
    Separated to be callable within or outside an MLflow run context.

    Returns:
        Tuple[bool, str]: Success status and final execution status string.
    """
    # Extract config sections again for clarity
    model_config = config.get("model", {})
    images_to_explain = config.get("images_to_explain", [])
    exp_config = config.get("explainability", {})
    config_path_str = "configs/explainability_config.yaml" # Assume fixed path for logging


    # --- Explainability Parameters ---
    target_layer_name = exp_config.get("target_layer_name")
    # --- Updated: Get person_class_id from model config ---
    person_class_index = model_config.get("person_class_id", 1) # Default to 1 if not in model config
    logger.info(f"Using person_class_index: {person_class_index} for explainability target")
    # --- End Update ---
    conf_thresh = exp_config.get("confidence_threshold_for_explanation", 0.6)
    top_n = exp_config.get("top_n_to_explain", 3)
    method = exp_config.get("method", "gradcam").lower()
    alpha = exp_config.get("heatmap_alpha", 0.6)
    colormap_name = exp_config.get("colormap_name", "COLORMAP_JET")
    colormap = CV2_COLORMAPS.get(colormap_name.upper())
    if colormap is None:
        logger.warning(f"Invalid colormap name '{colormap_name}'. Defaulting to COLORMAP_JET.")
        colormap = cv2.COLORMAP_JET

    if not images_to_explain:
        logger.error("Configuration error: 'explainability.images_to_explain' list is empty.")
        return False, "FAILED"

    checkpoint_path = model_config.get("checkpoint_path")
    if not checkpoint_path:
        logger.error("Configuration error: 'model.checkpoint_path' must be specified.")
        return False, "FAILED"

    num_classes = model_config.get("num_classes")
    if not num_classes:
        logger.error("Configuration error: 'model.num_classes' must be specified.")
        return False, "FAILED"

    if method not in SUPPORTED_METHODS:
        logger.error(f"Unsupported explainability method: '{method}'. Supported: {SUPPORTED_METHODS}")
        return False, "FAILED"

    if not target_layer_name and method == "gradcam":
        logger.error("Configuration error: 'explainability.target_layer_name' must be specified for Grad-CAM.")
        return False, "FAILED"

    # --- Load Model ---
    try:
        model = load_trained_fasterrcnn(
            checkpoint_path,
            device,
            num_classes,
            trainable_backbone_layers=model_config.get('trainable_backbone_layers', 3), # Use default if not in model config
            mlflow_run_id=mlflow_run_id
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.critical(f"Failed to load model: {e}")
        return False, "FAILED"

    # --- Get Target Layer (if needed) ---
    target_layer = None
    if method == "gradcam":
        target_layer = get_target_layer(model, target_layer_name)
        if target_layer is None:
            return False, "FAILED" # Error logged in get_target_layer

    # --- Get Transforms ---
    dummy_run_config = {'data': {}} # For get_transform compatibility
    transforms = get_transform(train=False, config=dummy_run_config)

    # --- Process Images ---
    all_reasoning_texts = [] # Collect reasoning for summary file
    images_processed_count = 0
    images_failed_count = 0

    for image_path_str in images_to_explain:
        image_path = Path(image_path_str)
        image_name = image_path.stem
        logger.info(f"\n--- Processing Image: {image_path.name} ---")

        # Resolve absolute path if needed (assuming relative paths are from PROJECT_ROOT)
        if not image_path.is_absolute():
            image_path = (PROJECT_ROOT / image_path).resolve()

        if not image_path.exists():
            logger.error(f"Image not found: {image_path}. Skipping.")
            images_failed_count += 1
            all_reasoning_texts.append(f"--- Image: {image_path_str} ---\nERROR: Image file not found.\n")
            continue

        # Run Inference
        detections, original_image_rgb, input_tensor = infer_single_image(
            model,
            image_path,
            transforms,
            device,
            confidence_threshold=conf_thresh,
            person_class_index=person_class_index # Use index from model config
        )

        if detections is None or original_image_rgb is None or input_tensor is None:
            logger.error(f"Inference failed for image: {image_path.name}. Skipping explanation.")
            images_failed_count += 1
            all_reasoning_texts.append(f"--- Image: {image_path_str} ---\nERROR: Inference failed.\n")
            continue

        images_processed_count += 1
        all_reasoning_texts.append(f"--- Image: {image_path_str} ---")

        if not detections:
            logger.warning(f"No persons detected above threshold {conf_thresh} in {image_path.name}.")
            all_reasoning_texts.append(f"No persons detected above threshold {conf_thresh}.\n")
            continue

        # Sort detections and select top N
        detections.sort(key=lambda d: d['score'], reverse=True)
        detections_to_explain = detections[:top_n]
        logger.info(f"Found {len(detections)} persons. Explaining top {len(detections_to_explain)}.")

        # Generate Explanations
        for i, det in enumerate(detections_to_explain):
            logger.info(f"  Explaining detection {i+1}/{len(detections_to_explain)} (Score: {det['score']:.3f})...")
            det_box = det['box']
            det_score = det['score']
            det_label = det['label'] # This should match person_class_index

            attribution_map = None
            viz_path = None
            has_visualization = False

            if method == "gradcam":
                target_detection_index = i # Index within the filtered list
                attribution_map = explain_detection_gradcam(
                    model,
                    input_tensor,
                    target_layer,
                    target_detection_index,
                    person_class_index, # Pass the correct class index
                    device
                )

            if attribution_map is not None:
                viz_filename = f"{image_name}_explain_{method}_det{i}_score{det_score:.2f}.png"
                viz_path = output_dir / viz_filename
                visualize_explanation(
                    original_image_rgb=original_image_rgb,
                    attribution_map=attribution_map,
                    output_path=viz_path,
                    box_to_highlight=det_box,
                    score=det_score,
                    label=det_label,
                    alpha=alpha,
                    colormap=colormap
                )
                has_visualization = True
                if mlflow_run_id:
                    try:
                        mlflow.log_artifact(str(viz_path), artifact_path=f"explanations/{image_name}")
                    except Exception as mlflow_log_err:
                         logger.warning(f"Failed to log explanation artifact to MLflow: {mlflow_log_err}")
            else:
                logger.warning(f"  Could not generate {method} explanation for detection {i}.")

            # Generate Reasoning Text
            reasoning_text = generate_reasoning_text(
                detection_result=det,
                explanation_type=f"{method.upper()} Focus",
                has_visualization=has_visualization
            )

            log_entry = (
                f" Detection {i+1}:"
                f" Box=[{int(det_box[0])},{int(det_box[1])},{int(det_box[2])},{int(det_box[3])}],"
                f" Score={det_score:.4f}, Label={det_label}\n" # Added Label
                f"   Reasoning: {reasoning_text}\n"
            )
            if viz_path:
                log_entry += f"   Visualization: {viz_path.relative_to(PROJECT_ROOT)}\n"
            all_reasoning_texts.append(log_entry)
            print(log_entry) # Also print to console

    # --- Write Summary Log ---
    logger.info(f"Writing reasoning summary to: {reasoning_log_path}")
    try:
        with open(reasoning_log_path, "w") as f:
            # Use the passed run_name here
            f.write(f"Explainability Run Summary: {run_name}\n")
            f.write(f"Config: {config_path_str}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Method: {method.upper()}\n")
            f.write(f"Person Class Index Used: {person_class_index}\n")
            f.write(f"Processed {images_processed_count} images, Failed: {images_failed_count}\n")
            f.write("="*40 + "\n\n")
            f.write("\n".join(all_reasoning_texts))
        if mlflow_run_id:
            try:
                mlflow.log_artifact(str(reasoning_log_path), artifact_path="explanations")
            except Exception as mlflow_log_err:
                logger.warning(f"Failed to log reasoning log artifact to MLflow: {mlflow_log_err}")
    except NameError as ne:
         # This should be fixed now, but keep the catch just in case
         logger.error(f"Failed to write reasoning summary log due to NameError: {ne}", exc_info=True)
         return False, "FAILED" # Indicate failure due to this error
    except Exception as write_err:
        logger.error(f"Failed to write reasoning summary log: {write_err}")

    if images_failed_count > 0:
        logger.error(f"{images_failed_count} images failed during processing.")
        return False, "FAILED"
    elif images_processed_count == 0:
         logger.warning("No images were successfully processed.")
         return True, "FINISHED" # Technically finished, but did nothing
    else:
        return True, "FINISHED"


if __name__ == "__main__":
    main()