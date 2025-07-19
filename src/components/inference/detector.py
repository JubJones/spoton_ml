import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import v2 as T
import mlflow # <-- Added import

# Assuming these project modules are accessible
from src.components.training.runner import get_fasterrcnn_model, get_transform

logger = logging.getLogger(__name__)

DetectionOutput = List[Dict[str, Any]] # List of {'box': [x1,y1,x2,y2], 'label': int, 'score': float}


def load_trained_fasterrcnn(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    num_classes: int,
    trainable_backbone_layers: int = 3,
    mlflow_run_id: Optional[str] = None
) -> FasterRCNN:
    """
    Loads a Faster R-CNN model architecture and populates it with trained weights
    from a checkpoint file (local path or MLflow artifact URI).

    Args:
        checkpoint_path: Path to the .pth checkpoint file OR an MLflow artifact URI
                         (e.g., "runs:/<run_id>/path/to/artifact.pth" or
                          "mlflow-artifacts:/<experiment_id>/<run_id>/artifacts/path/to/artifact.pth").
        device: The torch.device to load the model onto.
        num_classes: The number of output classes (including background) the model
                     was trained with.
        trainable_backbone_layers: Number of backbone layers that were trainable
                                   (should match training).
        mlflow_run_id: The ID of the *current* MLflow run (if any). Used for resolving
                       relative artifact paths if needed, though absolute URIs are preferred.

    Returns:
        The loaded Faster R-CNN model in evaluation mode.

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist locally and isn't a valid artifact URI.
        RuntimeError: If loading the state dictionary fails or MLflow download fails.
        ValueError: If the checkpoint doesn't contain a 'model_state_dict'.
    """
    ckpt_path_str = str(checkpoint_path)
    is_mlflow_uri = ckpt_path_str.startswith("runs:/") or ckpt_path_str.startswith("mlflow-artifacts:/")
    local_ckpt_path: Optional[Path] = None

    if is_mlflow_uri:
        logger.info(f"Loading checkpoint from MLflow artifact URI: {ckpt_path_str}")
        try:
            # Download artifact to a temporary directory
            # Note: Using run_id=None might work for absolute URIs (mlflow-artifacts:/...)
            # but providing the current run_id might help resolve relative "runs:/" URIs if used.
            # However, the most robust way is to use the URI directly.
            local_ckpt_path_str = mlflow.artifacts.download_artifacts(
                artifact_uri=ckpt_path_str,
                run_id=mlflow_run_id # Optional, helps resolve relative runs:/ paths if needed
                # dst_path=tempfile.mkdtemp() # Download to temp dir
            )
            local_ckpt_path = Path(local_ckpt_path_str)
            if not local_ckpt_path.is_file():
                raise FileNotFoundError(f"Downloaded MLflow artifact is not a file: {local_ckpt_path}")
            logger.info(f"Successfully downloaded artifact to temporary location: {local_ckpt_path}")
            ckpt_load_path = local_ckpt_path
        except Exception as mlflow_err:
            logger.error(f"Failed to download MLflow artifact {ckpt_path_str}: {mlflow_err}")
            raise RuntimeError(f"MLflow artifact download failed: {mlflow_err}") from mlflow_err
    else:
        ckpt_load_path = Path(ckpt_path_str)
        if not ckpt_load_path.is_file():
            # Check if relative to project root
            project_root = Path(__file__).parent.parent.parent.resolve()
            potential_path = (project_root / ckpt_path_str).resolve()
            if potential_path.is_file():
                ckpt_load_path = potential_path
            else:
                raise FileNotFoundError(f"Checkpoint file not found locally: {ckpt_load_path} or {potential_path}")
        logger.info(f"Loading trained Faster R-CNN model from local path: {ckpt_load_path}")


    # 1. Build the model architecture
    dummy_model_config = {
        "num_classes": num_classes,
        "backbone_weights": "DEFAULT",
        "trainable_backbone_layers": trainable_backbone_layers,
    }
    dummy_config = {"model": dummy_model_config}
    model = get_fasterrcnn_model(dummy_config)
    model.to(device)

    # 2. Load the checkpoint
    try:
        checkpoint = torch.load(ckpt_load_path, map_location=device)
    except Exception as load_err:
        logger.error(f"Failed to load checkpoint file {ckpt_load_path}: {load_err}")
        raise RuntimeError(f"Checkpoint loading failed: {load_err}") from load_err

    # 3. Extract and load the state dictionary
    if "model_state_dict" not in checkpoint:
        if isinstance(checkpoint, dict):
             state_dict = checkpoint
             logger.warning("Checkpoint does not contain 'model_state_dict'. Assuming checkpoint *is* state_dict.")
        else:
            raise ValueError(f"Checkpoint file {ckpt_load_path} does not contain 'model_state_dict' key.")
    else:
        state_dict = checkpoint["model_state_dict"]

    try:
        model.load_state_dict(state_dict)
        logger.info("Successfully loaded model weights from checkpoint.")
    except RuntimeError as e:
        logger.error(f"Failed to load state_dict from {ckpt_load_path}: {e}")
        logger.error("Ensure 'num_classes' and model architecture match the checkpoint.")
        raise RuntimeError(f"State dictionary loading failed: {e}") from e

    model.eval()
    logger.info("Model set to evaluation mode.")

    # Clean up temporary downloaded artifact if necessary
    # Note: mlflow.artifacts.download_artifacts downloads to a temp location by default
    # if dst_path is not specified, which gets cleaned up automatically. If we specified
    # dst_path, we would need to manually clean it up here.
    # if is_mlflow_uri and local_ckpt_path and local_ckpt_path.exists():
    #      try:
    #           # If downloaded to a specific directory, clean it up
    #           # shutil.rmtree(local_ckpt_path.parent)
    #           pass # Assuming default temp dir usage
    #      except Exception as cleanup_err:
    #           logger.warning(f"Could not clean up temporary artifact {local_ckpt_path}: {cleanup_err}")


    return model


@torch.no_grad()
def infer_single_image(
    model: FasterRCNN,
    image_path_or_array: Union[str, Path, np.ndarray],
    transforms: T.Compose,
    device: torch.device,
    confidence_threshold: float = 0.5,
    person_class_index: int = 1
) -> Tuple[Optional[DetectionOutput], Optional[np.ndarray], Optional[torch.Tensor]]:
    """
    Performs inference on a single image using the provided Faster R-CNN model.
    (Function body remains unchanged from previous response)
    """
    logger.debug(f"Performing inference on: {image_path_or_array}")
    original_image_rgb: Optional[np.ndarray] = None
    input_tensor: Optional[torch.Tensor] = None

    try:
        # 1. Load Image
        if isinstance(image_path_or_array, (str, Path)):
            img_path = Path(image_path_or_array)
            # Resolve absolute path if needed (assuming relative paths are from PROJECT_ROOT)
            if not img_path.is_absolute():
                 img_path = (PROJECT_ROOT / img_path).resolve()

            if not img_path.is_file():
                logger.error(f"Image file not found: {img_path}")
                return None, None, None
            # Use imdecode for robustness
            img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
            img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logger.error(f"Failed to decode image: {img_path}")
                return None, None, None
        elif isinstance(image_path_or_array, np.ndarray):
            img_bgr = image_path_or_array
        else:
            logger.error(f"Invalid image input type: {type(image_path_or_array)}")
            return None, None, None

        original_image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(original_image_rgb)

        # 2. Apply Transforms
        input_tensor = transforms(img_pil)
        input_batch = [input_tensor.to(device)] # Model expects a list

        # 3. Perform Inference
        predictions = model(input_batch)

        # 4. Process Results
        processed_detections: DetectionOutput = []
        if predictions and isinstance(predictions, list) and isinstance(predictions[0], dict):
            pred = predictions[0] # Result for the single image
            boxes = pred["boxes"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                if label == person_class_index and score >= confidence_threshold:
                    processed_detections.append({
                        "box": box.tolist(), # [x1, y1, x2, y2]
                        "label": int(label),
                        "score": float(score)
                    })
            logger.info(f"Found {len(processed_detections)} persons above threshold {confidence_threshold}.")
        else:
            logger.warning(f"Model output format unexpected: {type(predictions)}. No detections processed.")

        return processed_detections, original_image_rgb, input_tensor

    except Exception as e:
        logger.error(f"Error during single image inference: {e}", exc_info=True)
        return None, original_image_rgb, input_tensor