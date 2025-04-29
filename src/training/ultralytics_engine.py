import logging
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import mlflow
from ultralytics import RTDETR, YOLO  # Import base YOLO too if needed

logger = logging.getLogger(__name__)


def generate_ultralytics_data_yaml(
        config: Dict[str, Any],
        dataset_dir: Path,
        output_yaml_path: Path
) -> bool:
    """
    Generates the data.yaml file required by Ultralytics train function,
    pointing to the combined dataset from specified scenes/cameras.

    NOTE: This currently assumes a simple structure where 'train' and 'val'
          folders containing images are directly under `dataset_dir`.
          A more robust implementation would need to create symbolic links
          or copy files from the MTMMC structure into this expected format.
          For now, it points to the *parent* of the scene directories,
          assuming Ultralytics can handle discovery within that.
    """
    data_config = config["data"]
    base_path = Path(data_config["base_path"])
    # For Ultralytics, 'path' should be the root containing 'train' and 'val' dirs
    # We'll point it to the *parent* of the scene dirs for now.
    # This is a simplification! Ultralytics might expect a different layout.
    ultralytics_dataset_root = base_path / "train" / "train"  # Points to dir containing sXX dirs

    # Ultralytics typically expects 'train' and 'val' subdirs with images
    # And corresponding labels dirs. This is a mismatch with MTMMC gt.txt.
    # *** This is a MAJOR simplification and likely needs refinement ***
    # We will point train/val to the same root for now and rely on
    # Ultralytics' internal splitting or hope it works.
    # A proper solution involves converting gt.txt to YOLO format labels
    # and restructuring the dataset folders.

    logger.warning("Generating Ultralytics data.yaml based on simplified assumptions.")
    logger.warning("Pointing train/val paths to the root scene directory.")
    logger.warning(
        "This assumes Ultralytics can find images and implies GT conversion to YOLO format is needed separately.")

    # Only one class: 'person'
    num_classes = 1
    class_names = ['person']

    yaml_content = {
        'path': str(ultralytics_dataset_root.resolve()),  # Root dataset directory
        'train': '.',  # Relative path to train images ( Ultralytics searches)
        'val': '.',  # Relative path to val images ( Ultralytics searches)
        # Add test later if needed: 'test': 'path/to/test/images'

        # Classes
        'nc': num_classes,
        'names': class_names,
    }

    try:
        output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)
        logger.info(f"Generated Ultralytics data YAML: {output_yaml_path}")
        logger.info(f"YAML Content:\n{yaml.dump(yaml_content)}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate Ultralytics data YAML: {e}", exc_info=True)
        return False


def train_ultralytics_model(
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        data_config: Dict[str, Any],
        env_config: Dict[str, Any],
        project_root: Path,
        run_dir: Path,  # Directory for this specific MLflow run's artifacts
        device: torch.device
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Trains an Ultralytics model (e.g., RTDETR) using its built-in train method.

    Args:
        model_config: Model specific configuration.
        training_config: Training parameters.
        data_config: Data configuration.
        env_config: Environment configuration.
        project_root: Project root directory.
        run_dir: Directory to store artifacts like the data.yaml for this run.
        device: Target computation device.

    Returns:
        Tuple[bool, Optional[Dict[str, Any]]]: Success status, Final metrics dictionary.
    """
    model_type = model_config.get("type", "unknown").lower()
    model_weights = model_config.get("weights_path", "yolov8n.pt")  # Default needed
    run_name_tag = model_config.get("name_tag", f"ultralytics_{model_type}")

    logger.info(f"--- Starting Ultralytics Training: {run_name_tag} ---")
    logger.info(f"Using base weights: {model_weights}")

    # 1. Prepare Ultralytics Data YAML
    # NOTE: This step is highly dependent on how MTMMC data is preprocessed/structured
    #       for Ultralytics consumption (requires YOLO label format).
    data_yaml_path = run_dir / "ultralytics_mtmmc_data.yaml"
    if not generate_ultralytics_data_yaml(config={'data': data_config}, dataset_dir=run_dir,
                                          output_yaml_path=data_yaml_path):
        logger.error("Failed to prepare data configuration for Ultralytics.")
        return False, None
    # Log the generated yaml to MLflow
    try:
        mlflow.log_artifact(str(data_yaml_path), artifact_path="config")
    except Exception as e:
        logger.warning(f"Failed to log generated data.yaml: {e}")

    # 2. Instantiate Model
    try:
        if model_type == 'rtdetr':
            model = RTDETR(model_weights)
        elif model_type == 'yolo':  # Example if YOLO was added later
            model = YOLO(model_weights)
        else:
            raise ValueError(f"Unsupported Ultralytics model type: {model_type}")
        logger.info(f"Ultralytics model '{model_type}' instantiated from '{model_weights}'.")
    except Exception as e:
        logger.error(f"Failed to instantiate Ultralytics model: {e}", exc_info=True)
        return False, None

    # 3. Prepare Training Arguments for model.train()
    ultralytics_args = {
        "data": str(data_yaml_path.resolve()),
        "epochs": training_config.get("epochs", 10),
        "batch": training_config.get("batch_size", 8),
        "imgsz": training_config.get("imgsz", 640),
        "device": str(device.index) if device.type == 'cuda' else device.type,  # Ultralytics device format
        "project": str(project_root / training_config.get("ultralytics_project_name", "ultralytics_runs")),
        "name": mlflow.active_run().info.run_id if mlflow.active_run() else run_name_tag,
        # Use MLflow run ID for output dir name
        "exist_ok": True,  # Allow reusing project/name directory structure
        # Add other relevant args from training_config: patience, optimizer, lr0 etc.
        # "patience": training_config.get("patience", 50),
        # "optimizer": training_config.get("optimizer", "auto"),
        # "lr0": training_config.get("lr0", 0.01),
        # "seed": env_config.get("seed", 42),
        "save_period": 1,  # Save checkpoint every epoch
        "save_json": True,  # Save results JSON
        "save_hybrid": True,  # Save hybrid format labels
    }
    # Clean None values which ultralytics might not like
    ultralytics_args = {k: v for k, v in ultralytics_args.items() if v is not None}

    logger.info(f"Ultralytics Training Arguments: {ultralytics_args}")

    # 4. Run Training
    final_metrics = None
    success = False
    try:
        # Ultralytics automatically logs to MLflow if detected
        logger.info("Starting model.train()... (Ultralytics handles logging)")
        results = model.train(**ultralytics_args)

        # Training completed, extract final metrics if possible
        if results and hasattr(results, 'results_dict'):
            final_metrics = results.results_dict
            logger.info(f"Ultralytics training finished. Final metrics: {final_metrics}")
        elif results and hasattr(results, 'metrics'):  # Newer attribute?
            final_metrics = results.metrics
            logger.info(f"Ultralytics training finished. Final metrics from .metrics: {final_metrics}")
        else:
            logger.warning("Could not extract final metrics dictionary from Ultralytics results object.")
            final_metrics = {}  # Indicate success but no detailed metrics extracted here

        success = True

        # Log the final results directory from Ultralytics as an artifact (optional)
        # The directory path is usually project/name
        run_output_dir = Path(ultralytics_args['project']) / ultralytics_args['name']
        if run_output_dir.exists():
            try:
                mlflow.log_artifacts(str(run_output_dir), artifact_path="ultralytics_output")
                logger.info(f"Logged Ultralytics output artifacts from: {run_output_dir}")
            except Exception as e:
                logger.warning(f"Failed to log Ultralytics output artifacts: {e}")
        else:
            logger.warning(f"Ultralytics output directory not found: {run_output_dir}")


    except Exception as e:
        logger.error(f"Ultralytics training failed: {e}", exc_info=True)
        success = False
        final_metrics = None  # Ensure metrics are None on failure

    return success, final_metrics
