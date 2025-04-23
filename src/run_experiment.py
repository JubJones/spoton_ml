import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import subprocess
import time

import dagshub
import mlflow
import torch
from dotenv import load_dotenv
import numpy as np
import cv2
from PIL import Image


# --- Project Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"PROJECT_ROOT added to sys.path: {PROJECT_ROOT}")

# --- Local Imports ---
try:
    from src.utils.config_loader import load_config
    from src.utils.device_utils import get_selected_device
    from src.utils.reproducibility import set_seed
    from src.pipelines.detection_pipeline import DetectionPipeline
    from src.tracking.strategies import (
        DetectionTrackingStrategy,
        YoloStrategy, RTDetrStrategy, FasterRCNNStrategy, RfDetrStrategy
    )
    from src.data.loader import FrameDataLoader
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
except ImportError as e:
    print(f"Error importing local modules or MLflow components: {e}")
    print(f"PYTHONPATH: {sys.path}")
    print("Ensure you are running this script from the project root or that src is in PYTHONPATH.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = PROJECT_ROOT / "experiment.log"
if log_file.exists():
    try:
        open(log_file, 'w').close()
    except OSError as e:
        print(f"Warning: Could not clear log file {log_file}: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Helper Functions ---

def setup_mlflow(config: Dict[str, Any]) -> Optional[str]:
    """Initializes MLflow connection (Dagshub or local) and sets experiment."""
    mlflow_config = config.get("mlflow", {})
    dotenv_path = PROJECT_ROOT / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logger.info("Loaded environment variables from .env file.")
    else:
        logger.info(".env file not found, relying on environment or defaults.")

    try:
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "Jwizzed")
        repo_name = os.getenv("DAGSHUB_REPO_NAME", "spoton_ml")
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        logger.info(f"Dagshub initialized successfully for {repo_owner}/{repo_name}.")
        tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"MLflow tracking URI automatically set by Dagshub: {tracking_uri}")
    except Exception as dag_err:
        logger.warning(f"Dagshub initialization failed: {dag_err}. Attempting manual MLflow URI setup.")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set from environment variable: {tracking_uri}")
        else:
            logger.warning("MLFLOW_TRACKING_URI not set and Dagshub init failed. Using local tracking.")
            local_mlruns = PROJECT_ROOT / "mlruns"
            local_mlruns.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{local_mlruns.resolve()}")
            logger.info(f"MLflow tracking URI set to local: {mlflow.get_tracking_uri()}")

    experiment_name = mlflow_config.get("experiment_name", "Default Experiment")
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to: '{experiment_name}'")

    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.info(f"Experiment '{experiment_name}' not found. Creating...")
            experiment_id = client.create_experiment(experiment_name)
            logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
        elif experiment.lifecycle_stage != 'active':
            logger.error(f"Experiment '{experiment_name}' exists but is deleted or archived.")
            return None
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment ID: {experiment_id}")
        return experiment_id
    except Exception as client_err:
        logger.error(f"Failed to connect to MLflow tracking server or get/create experiment: {client_err}",
                     exc_info=True)
        return None


def log_params_recursive(params_dict: Dict[str, Any], parent_key: str = ""):
    """Recursively logs parameters to MLflow, handling nested dictionaries and lists."""
    for key, value in params_dict.items():
        mlflow_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            log_params_recursive(value, mlflow_key)
        elif isinstance(value, list):
            try:
                # Convert list to string for logging as param (MLflow UI limit ~250 chars)
                param_val_str = json.dumps(value)
                if len(param_val_str) > 250:
                     param_val_str = param_val_str[:247] + "..."
                mlflow.log_param(mlflow_key, param_val_str)
            except TypeError:
                mlflow.log_param(mlflow_key, str(value)[:250]) # Fallback
        else:
            mlflow.log_param(mlflow_key, str(value)[:250])


def log_metrics_dict(metrics: Dict[str, Any]):
    """Logs metrics from a dictionary to MLflow."""
    if not metrics:
        logger.warning("Metrics dictionary is empty. Nothing to log.")
        return
    logger.info(f"Logging {len(metrics)} metrics to MLflow...")
    try:
        # Filter out non-numeric types before logging
        numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, np.number))}
        if len(numeric_metrics) < len(metrics):
            non_numeric_keys = [k for k in metrics if k not in numeric_metrics]
            logger.warning(f"Skipping non-numeric metrics: {non_numeric_keys}")

        if numeric_metrics:
             mlflow.log_metrics(numeric_metrics)
             logger.info(f"Logged {len(numeric_metrics)} numeric metrics successfully.")
        else:
             logger.warning("No numeric metrics found to log.")

    except Exception as e:
        logger.error(f"Failed to log metrics batch: {e}. Attempting individual logging.", exc_info=True)
        for key, value in numeric_metrics.items(): # Log only numeric ones individually too
            try:
                mlflow.log_metric(key, value)
            except Exception as ind_e:
                logger.error(f"Failed to log individual metric '{key}': {value}. Error: {ind_e}")


def get_sample_data_for_signature(
    config: Dict[str, Any],
    strategy: DetectionTrackingStrategy,
    device: torch.device
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads a sample frame and runs inference to get sample input/output
    for MLflow signature inference. Returns (sample_input_np, sample_output_np).
    """
    logger.info("Attempting to generate sample input/output for model signature...")
    sample_input_np = None
    sample_output_np = None
    try:
        # Use FrameDataLoader to get one valid frame
        temp_loader = FrameDataLoader(config)
        sample_frame_bgr = None
        first_frame_path = None
        if temp_loader.image_filenames and temp_loader.active_camera_ids:
             filename = temp_loader.image_filenames[0]
             cam_id = temp_loader.active_camera_ids[0]
             image_path = temp_loader.camera_image_dirs[cam_id] / filename
             if image_path.is_file():
                 img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
                 sample_frame_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                 first_frame_path = str(image_path)
                 logger.info(f"Loaded sample frame: {first_frame_path}")

        del temp_loader # Clean up loader

        if sample_frame_bgr is None:
            logger.warning("Could not load a sample frame for signature generation.")
            return None, None

        # 1. Get Sample Input Tensor using the strategy's method
        sample_input_tensor = strategy.get_sample_input_tensor(sample_frame_bgr)

        if sample_input_tensor is None:
             logger.warning(f"Strategy {strategy.__class__.__name__} did not provide a sample input tensor.")
             if isinstance(strategy, (YoloStrategy, RTDetrStrategy)):
                 sample_input_np = sample_frame_bgr
                 logger.info("Using raw frame as input example for Ultralytics model.")
             else:
                return None, None
        else:
            # Ensure tensor is on the correct device and add batch dim if needed
            if sample_input_tensor.dim() == 3: # CHW
                 sample_input_tensor = sample_input_tensor.unsqueeze(0)
            sample_input_batch = [sample_input_tensor.to(device)]
            sample_input_np = sample_input_batch[0].cpu().numpy()

        # 2. Get Sample Output
        model_object = strategy.get_model()
        if model_object is None:
             logger.warning("Cannot get sample output: Model object not available from strategy.")
             return sample_input_np, None # Return input if available, but no output

        with torch.no_grad():
            if isinstance(model_object, torch.nn.Module):
                logger.info("Running inference on sample data (PyTorch Module)...")
                predictions = model_object(sample_input_batch) # Assumes NCHW input
                # Postprocess based on expected output format (e.g., FasterRCNN)
                if isinstance(predictions, list) and isinstance(predictions[0], dict):
                    # Example for FasterRCNN: use boxes
                    sample_output_np = predictions[0]['boxes'].cpu().numpy()
                elif isinstance(predictions, torch.Tensor): # Generic tensor output
                     sample_output_np = predictions.cpu().numpy()
                else:
                     logger.warning(f"Unexpected output type from PyTorch model: {type(predictions)}")
            elif hasattr(model_object, 'predict') and isinstance(strategy, (YoloStrategy, RTDetrStrategy)):
                logger.info("Running inference on sample data (Ultralytics)...")
                # Ultralytics predict might take the frame directly
                results = model_object.predict(sample_frame_bgr, device=device, verbose=False)
                if results and results[0].boxes is not None:
                    # Use xyxy boxes as output example
                    if hasattr(results[0].boxes, 'xyxy') and results[0].boxes.xyxy is not None:
                        sample_output_np = results[0].boxes.xyxy.cpu().numpy()
                    else: # Fallback to xywh if xyxy not present
                         if hasattr(results[0].boxes, 'xywh') and results[0].boxes.xywh is not None:
                            boxes_xywh = results[0].boxes.xywh.cpu().numpy()
                            # Convert xywh to xyxy for consistency if possible
                            sample_output_np = np.array([[c[0]-c[2]/2, c[1]-c[3]/2, c[0]+c[2]/2, c[1]+c[3]/2] for c in boxes_xywh])
                         else:
                             logger.warning("Could not get boxes (xyxy or xywh) from Ultralytics results.")
                else:
                     logger.warning("Ultralytics predict did not return expected results.")
            elif hasattr(model_object, 'predict') and isinstance(strategy, RfDetrStrategy):
                 logger.info("Running inference on sample data (RFDETR)...")
                 img_rgb_pil = Image.fromarray(cv2.cvtColor(sample_frame_bgr, cv2.COLOR_BGR2RGB))
                 detections = model_object.predict(img_rgb_pil) # Use default threshold for example
                 if detections and hasattr(detections, 'xyxy'):
                      sample_output_np = detections.xyxy
                      if isinstance(sample_output_np, torch.Tensor): sample_output_np = sample_output_np.cpu().numpy()
                 else:
                      logger.warning("RFDETR predict did not return expected detections with xyxy.")

            else:
                 logger.warning(f"Cannot determine how to run inference for model type {type(model_object)} to get sample output.")


        if sample_input_np is not None and sample_output_np is not None:
             logger.info(f"Successfully generated sample input (shape: {sample_input_np.shape}, dtype: {sample_input_np.dtype}) and output (shape: {sample_output_np.shape}, dtype: {sample_output_np.dtype})")
        elif sample_input_np is not None:
            logger.warning("Generated sample input, but failed to generate sample output.")
        else:
            logger.warning("Failed to generate sample input and output.")

        return sample_input_np, sample_output_np

    except Exception as e:
        logger.error(f"Error during sample data generation for signature: {e}", exc_info=True)
        return None, None


# --- Main Execution Orchestrator ---
def main():
    """Main function to orchestrate the experiment run."""
    logger.info("--- Starting Experiment Orchestration ---")
    config_path_str = "configs/experiment_config.yaml"
    run_status = "FAILED"
    run_id = None

    # 1. Load Configuration
    config = load_config(config_path_str)
    if not config:
        logger.critical("Failed to load configuration. Exiting.")
        sys.exit(1)

    # 2. Setup MLflow
    experiment_id = setup_mlflow(config)
    if not experiment_id:
        logger.critical("MLflow setup failed. Exiting.")
        sys.exit(1)

    # 3. Set Seed for Reproducibility
    seed = config.get("environment", {}).get("seed", int(time.time())) # Use time as fallback seed
    set_seed(seed)
    logger.info(f"Using random seed: {seed}")

    # 4. Determine Device
    requested_device_name = config.get("environment", {}).get("device", "auto")
    requested_device = get_selected_device(requested_device_name)

    model_type = config.get("model", {}).get("type", "").lower()
    actual_device = requested_device
    device_override_reason = "None"

    # --- Device Compatibility Checks ---
    # Example: Force FasterRCNN to CPU if no GPU available
    if model_type == "fasterrcnn" and requested_device.type not in ['cuda', 'mps']:
        logger.warning(f"FasterRCNN selected but device is '{requested_device.type}'. Forcing to CPU.")
        actual_device = torch.device('cpu')
        device_override_reason = f"FasterRCNN forced to CPU from {requested_device.type}"
    # Example: Check MPS compatibility (adjust based on real observations)
    elif actual_device.type == 'mps':
        if model_type not in ['yolo', 'rtdetr', 'fasterrcnn']: # Assume these work on MPS, others might not
             logger.warning(f"Model type '{model_type}' may have limited MPS support. Consider CPU or CUDA if available.")
             # Decide whether to force CPU or just warn:
             # actual_device = torch.device('cpu')
             # device_override_reason = f"Model {model_type} compatibility with MPS uncertain, using {actual_device.type}"
             pass # Just warning for now

    logger.info(f"Requested Device: {requested_device} | Actual Device Used: {actual_device} | Reason: {device_override_reason}")

    run_name = config.get("run_name", f"{model_type}_run_{int(time.time())}") # Default run name

    try:
        # 5. Initialize Pipeline (before starting run to potentially get sample data)
        logger.info("Initializing detection pipeline...")
        pipeline = DetectionPipeline(config, actual_device)
        if not pipeline.initialize_components():
             logger.critical("Pipeline initialization failed. Exiting.")
             # Start and immediately fail the run to record parameters?
             with mlflow.start_run(run_name=f"{run_name}_INIT_FAILED", experiment_id=experiment_id) as run:
                 run_id = run.info.run_id
                 logger.info(f"Starting MLflow run for INIT_FAILED: {run_name} (ID: {run_id})")
                 log_params_recursive(config) # Log params even on init failure
                 mlflow.log_param("environment.seed", seed)
                 mlflow.log_param("environment.requested_device", str(requested_device))
                 mlflow.log_param("environment.actual_device_used", str(actual_device))
                 mlflow.log_param("environment.device_override_reason", device_override_reason)
                 mlflow.set_tag("status", "INIT_FAILED")
                 run_status = "FAILED"

             sys.exit(1)

        # 6. Generate Sample Data and Signature (after pipeline init, before run start)
        sample_input_data = None
        sample_output_data = None
        signature = None
        if pipeline.detection_strategy:
             sample_input_data, sample_output_data = get_sample_data_for_signature(
                 config, pipeline.detection_strategy, actual_device
             )
             if sample_input_data is not None and sample_output_data is not None:
                 try:
                     # Infer signature (must happen before run starts if needed by log_model)
                     signature = infer_signature(sample_input_data, sample_output_data)
                     logger.info("Model signature inferred successfully.")
                 except Exception as infer_err:
                     logger.warning(f"Could not infer model signature: {infer_err}", exc_info=True)
                     signature = None # Ensure signature is None if inference fails
        else:
            logger.warning("Detection strategy not available after init, cannot generate sample data/signature.")


        # 7. Start MLflow Run
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            logger.info(f"--- Starting MLflow Run ---")
            logger.info(f"Run Name: {run_name}")
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Experiment ID: {experiment_id}")
            logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")


            # 8. Log Parameters, Config, Seed, Device Info, Tags
            logger.info("Logging parameters...")
            log_params_recursive(config)
            mlflow.log_param("environment.seed", seed) # Log seed used
            mlflow.log_param("environment.requested_device", str(requested_device))
            mlflow.log_param("environment.actual_device_used", str(actual_device))
            mlflow.log_param("environment.device_override_reason", device_override_reason)

            logger.info("Logging tags...")
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("dataset", config['data']['selected_environment'])
            scene_id = config['data'][config['data']['selected_environment']]['scene_id']
            mlflow.set_tag("scene_id", scene_id)
            mlflow.set_tag("mlflow.note.content", f"Run for {model_type} on {config['data']['selected_environment']}/{scene_id}")

            # 9. Log Artifacts (Code State, Config)
            logger.info("Logging code artifacts...")
            # Log config file
            config_artifact_path = PROJECT_ROOT / config_path_str
            if config_artifact_path.is_file():
                mlflow.log_artifact(str(config_artifact_path), artifact_path="config")
                logger.info("Configuration artifact logged.")
            else:
                logger.warning(f"Could not find config file to log as artifact: {config_artifact_path}")

            # Log requirements.txt
            req_path = PROJECT_ROOT / "requirements.txt"
            if req_path.is_file():
                mlflow.log_artifact(str(req_path), artifact_path="code")
                logger.info("requirements.txt logged as artifact.")
            else:
                 logger.warning(f"Could not find requirements.txt at {req_path}")


            # Log Git commit hash
            try:
                commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT).strip().decode('utf-8')
                mlflow.set_tag("git_commit_hash", commit_hash)
                logger.info(f"Git commit hash logged as tag: {commit_hash}")
                # Check for uncommitted changes
                git_status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT).strip().decode('utf-8')
                if git_status:
                    logger.warning("Git working directory has uncommitted changes.")
                    mlflow.set_tag("git_status", "dirty")
                    try:
                         # Log diff (can be large)
                         diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=PROJECT_ROOT).strip().decode('utf-8', errors='ignore')
                         if diff:
                              mlflow.log_text(diff, artifact_file="code/git_diff.diff")
                              logger.info("Git diff logged as artifact.")
                         else:
                              logger.info("No diff detected despite dirty status (might be untracked files).")
                    except Exception as diff_err:
                        logger.warning(f"Could not log git diff: {diff_err}")
                else:
                     mlflow.set_tag("git_status", "clean")
            except Exception as git_err:
                logger.warning(f"Could not get git commit hash or status: {git_err}")
                mlflow.set_tag("git_status", "unknown")


            # 10. Run the Pipeline Processing
            logger.info("Starting pipeline processing...")
            # Pipeline object already initialized
            pipeline_success, metrics, active_cameras, num_frames = pipeline.run() # Reuse initialized pipeline

            # 11. Log Pipeline Info (even if failed)
            if active_cameras is not None:
                mlflow.log_param("data.actual_cameras_used", ",".join(active_cameras))
            if num_frames is not None:
                # Assuming num_frames is the count of unique frame indices processed by pipeline
                mlflow.log_param("data.actual_frame_indices_processed", num_frames)

            # 12. Log Metrics & Metrics Artifact (if successful)
            if pipeline_success and metrics:
                logger.info("Pipeline finished successfully. Logging metrics.")
                log_metrics_dict(metrics)
                run_status = "FINISHED"
                mlflow.set_tag("run_outcome", "Success")

                # Log metrics dictionary as JSON artifact
                try:
                    metrics_path = PROJECT_ROOT / f"run_{run_id}_metrics.json" # Use run_id for unique name
                    with open(metrics_path, 'w') as f:
                        # Convert numpy types for JSON serialization
                        metrics_serializable = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in metrics.items()}
                        json.dump(metrics_serializable, f, indent=4)
                    mlflow.log_artifact(str(metrics_path), artifact_path="results")
                    metrics_path.unlink() # Clean up temp file
                    logger.info("Metrics dictionary logged as JSON artifact.")
                except Exception as json_err:
                    logger.warning(f"Could not log metrics dictionary as JSON: {json_err}")

            elif metrics: # Log metrics even if pipeline failed but metrics were calculated (partial run)
                logger.warning("Pipeline did not finish successfully, but partial metrics were calculated. Logging metrics.")
                log_metrics_dict(metrics)
                run_status = "FAILED"
                mlflow.set_tag("run_outcome", "Partial Execution")

            else:
                logger.error("Pipeline execution failed. No metrics to log.")
                run_status = "FAILED"
                mlflow.set_tag("run_outcome", "Failed Execution")


    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user (KeyboardInterrupt). Terminating run.")
        run_status = "KILLED"
        if run_id: mlflow.set_tag("run_outcome", "Killed by user")
    except Exception as e:
        logger.critical(f"An uncaught error occurred during the experiment run: {e}", exc_info=True)
        run_status = "FAILED"
        if run_id: mlflow.set_tag("run_outcome", "Crashed") # Add tag if run started
    finally:
        # 14. Log Experiment Log File & End MLflow Run (robustly)
        logger.info(f"--- Finalizing MLflow Run (Attempted Status: {run_status}) ---")

        # Check if we are *inside* an active run context managed by `with`
        active_mlflow_run = mlflow.active_run()

        if run_id and active_mlflow_run and active_mlflow_run.info.run_id == run_id:
            # We are inside the `with` block or it exited cleanly/via handled exception
            # Log the experiment log file artifact *before* ending the run
            if log_file.exists():
                try:
                    # Ensure logs are flushed
                    for handler in logging.getLogger().handlers:
                        handler.flush()
                    # Short delay to ensure file system sync? (Usually not needed)
                    # time.sleep(0.5)
                    mlflow.log_artifact(str(log_file), artifact_path="logs")
                    logger.info(f"Experiment log file '{log_file.name}' logged as artifact.")
                except Exception as log_artifact_err:
                    print(f"ERROR: Could not log experiment log file artifact '{log_file}': {log_artifact_err}") # Print directly

            # End the run using the context manager's implicit end or explicitly set status
            logger.info(f"Ending MLflow run {run_id} via context manager.")
            mlflow.end_run(status=run_status) # Explicitly set status on exit
            logger.info(f"MLflow run {run_id} final status set to: {run_status}")

        elif run_id:
            # We are outside the `with` block (e.g., init failed, or very early crash)
            logger.warning(f"Attempting to terminate run {run_id} outside of active context.")
            try:
                 # Try logging the log file even if outside context
                 if log_file.exists():
                     # Need client to log artifact outside run context
                     client = MlflowClient()
                     try:
                        for handler in logging.getLogger().handlers: handler.flush()
                        # time.sleep(0.5)
                        client.log_artifact(run_id, str(log_file), artifact_path="logs")
                        logger.info(f"Experiment log file '{log_file.name}' logged via client for run {run_id}.")
                     except Exception as client_log_err:
                         print(f"ERROR: Could not log experiment log file artifact via client for run {run_id}: {client_log_err}")

                 # Terminate the run
                 client = MlflowClient() # Re-init just in case
                 client.set_terminated(run_id, status=run_status)
                 logger.info(f"MLflow run {run_id} explicitly terminated with status: {run_status}")
            except Exception as client_term_err:
                logger.error(f"Critical: Could not explicitly terminate run {run_id}. Final status might be incorrect. Error: {client_term_err}")
        else:
            # Case where initialization failed *before* run_id was created
            logger.warning("No run ID was generated (likely init failure before run start). Cannot terminate MLflow run.")

    logger.info(f"--- Experiment Orchestration Completed (Final Run Status: {run_status}) ---")
    if run_status != "FINISHED":
        sys.exit(1)  # Exit with error code if run wasn't successful


if __name__ == "__main__":
    main()