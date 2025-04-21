import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import dagshub
import mlflow
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))
print(f"PROJECT_ROOT added to sys.path: {PROJECT_ROOT}")

# Local imports (now that sys.path is set)
try:
    from src.utils.config_loader import load_config
    from src.tracking.device_utils import get_selected_device
    from src.data.loader import FrameDataLoader
    from src.tracking.strategies import get_strategy, DetectionStrategy
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Ensure you are running this script from the project root or that src is in PYTHONPATH.")
    sys.exit(1)

# --- Basic Logging Setup ---
log_file = PROJECT_ROOT / "experiment.log"
if log_file.exists():
    open(log_file, 'w').close()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler(sys.stdout)  # Also print INFO+ to console
    ]
)
logger = logging.getLogger(__name__)

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning)  # Often from numpy/pandas/scipy


# --- MLflow & Dagshub Setup ---
def setup_mlflow(mlflow_config: dict, config_path: str):
    """Initializes Dagshub and sets MLflow tracking URI and experiment."""
    try:
        dotenv_path = PROJECT_ROOT / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
            logger.info("Loaded environment variables from .env file.")
        else:
            logger.info(".env file not found, relying on environment or defaults.")

        try:
            repo_owner = "Jwizzed"
            repo_name = "spoton_ml"
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            logger.info(f"Dagshub initialized successfully for {repo_owner}/{repo_name}.")
            tracking_uri = mlflow.get_tracking_uri()
            logger.info(f"MLflow tracking URI automatically set by Dagshub: {tracking_uri}")

        except Exception as dag_err:
            logger.warning(f"Dagshub initialization failed: {dag_err}. Attempting manual URI setup.")
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                logger.info(f"MLflow tracking URI set from environment variable: {tracking_uri}")
            else:
                logger.warning(
                    "MLFLOW_TRACKING_URI environment variable not set and Dagshub init failed. MLflow will use local tracking './mlruns'.")
                local_mlruns = PROJECT_ROOT / "mlruns"
                local_mlruns.mkdir(exist_ok=True)

        experiment_name = mlflow_config.get("experiment_name", "Default Experiment")
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: '{experiment_name}'")

        # Verify connection and experiment existence
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if not experiment:
                logger.info(f"Experiment '{experiment_name}' not found. Creating it.")
                experiment_id = client.create_experiment(experiment_name)
                logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
                return experiment_id
            else:
                logger.info(f"Found existing experiment ID: {experiment.experiment_id}")
                if experiment.lifecycle_stage != 'active':
                    logger.error(f"Experiment '{experiment_name}' is deleted or archived. Cannot log runs.")
                    return None
                return experiment.experiment_id
        except Exception as client_err:
            logger.error(f"Failed to connect to MLflow tracking server or get/create experiment: {client_err}")
            logger.error("Please ensure the tracking server is running and accessible, or check Dagshub setup.")
            return None

    except Exception as e:
        logger.error(f"Error during MLflow/Dagshub setup: {e}", exc_info=True)
        return None


# --- Main Experiment Logic ---
def run_experiment():
    """Loads config, runs detection, and logs results to MLflow."""
    logger.info("--- Starting Experiment ---")
    config_path_str = "configs/experiment_config.yaml"
    config = load_config(config_path_str)
    if not config:
        logger.critical("Failed to load configuration. Exiting.")
        sys.exit(1)

    # --- Setup MLflow ---
    experiment_id = setup_mlflow(config.get("mlflow", {}), config_path_str)
    if not experiment_id:
        logger.critical("MLflow setup failed. Exiting.")
        sys.exit(1)

    # --- Start MLflow Run ---
    run_name = config.get("run_name", "unnamed_run")
    try:
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            logger.info(f"Starting MLflow run: {run_name} (ID: {run_id})")

            # --- Log Parameters ---
            logger.info("Logging parameters...")
            try:
                # Log parameters recursively (handle nested dicts)
                def log_params_recursive(params_dict, parent_key=""):
                    for key, value in params_dict.items():
                        mlflow_key = f"{parent_key}.{key}" if parent_key else key
                        if isinstance(value, dict):
                            log_params_recursive(value, mlflow_key)
                        elif isinstance(value, list):
                            # Log lists as comma-separated strings or JSON strings
                            try:
                                import json
                                mlflow.log_param(mlflow_key, json.dumps(value))
                            except:  # Fallback for non-serializable lists
                                mlflow.log_param(mlflow_key, str(value)[:250])
                        else:
                            # Truncate long values (e.g., paths) if necessary
                            mlflow.log_param(mlflow_key, str(value)[:250])

                log_params_recursive(config)
                # Log the config file itself as an artifact for full reproducibility
                mlflow.log_artifact(str(PROJECT_ROOT / config_path_str), artifact_path="config")
                logger.info("Parameters and config artifact logged.")
            except Exception as e:
                logger.error(f"Error logging parameters/config artifact: {e}", exc_info=True)

            # --- Setup Device and Strategy ---
            try:
                device_name = config.get("environment", {}).get("device", "auto")
                device = get_selected_device(device_name)

                model_config = config.get("model", {})
                detection_strategy: DetectionStrategy = get_strategy(model_config, device)
                logger.info(f"Using detection strategy: {detection_strategy.__class__.__name__}")

            except (ValueError, ImportError, Exception) as e:
                logger.critical(f"Failed to initialize device or strategy: {e}", exc_info=True)
                mlflow.set_terminated(status="FAILED", state_message=f"Initialization Error: {e}")
                sys.exit(1)

            # --- Setup Data Loader ---
            try:
                data_loader = FrameDataLoader(config)
                total_frame_indices = len(data_loader)
                actual_cameras_used = data_loader.active_camera_ids
                total_frames_to_process = total_frame_indices * len(actual_cameras_used)

                if total_frame_indices == 0:
                    raise ValueError("Data loader found 0 frame indices to process.")
                logger.info(
                    f"Data loader initialized. Processing {total_frame_indices} frame indices across {len(actual_cameras_used)} cameras: {actual_cameras_used}.")
                logger.info(f"Estimated total frames to process: {total_frames_to_process}")

                # Log the *actual* cameras used, in case some were skipped
                mlflow.log_param("data.actual_cameras_used", ",".join(actual_cameras_used))
                mlflow.log_param("data.actual_frame_indices_processed", total_frame_indices)

            except (FileNotFoundError, ValueError, RuntimeError) as e:
                logger.critical(f"Failed to initialize data loader: {e}", exc_info=True)
                mlflow.set_terminated(status="FAILED", state_message=f"Data Loading Error: {e}")
                sys.exit(1)

            # --- Processing Loop ---
            logger.info("Starting frame processing loop...")
            frame_counter = 0  # Counts total frames processed across all cameras
            processed_indices = set()  # Keep track of unique frame indices processed
            total_inference_time_ms = 0
            total_detections = 0
            # Use defaultdict to store per-camera stats easily
            detections_per_camera = defaultdict(int)
            inference_time_per_camera = defaultdict(float)
            frame_count_per_camera = defaultdict(int)

            start_time_total = time.perf_counter()

            # --- Main Loop ---
            for frame_idx, cam_id, filename, frame_bgr in data_loader:
                if frame_bgr is None:
                    logger.debug(f"Skipping Frame {frame_idx} for Cam {cam_id} (Load failed)")
                    continue

                frame_counter += 1
                frame_count_per_camera[cam_id] += 1
                processed_indices.add(frame_idx)
                logger.debug(f"Processing Frame Idx {frame_idx} (Total: {frame_counter}) - Cam: {cam_id} - File: {filename}")

                # --- Run Detection ---
                start_time_inference = time.perf_counter()
                boxes_xyxy, confidences, classes = detection_strategy.process_frame(frame_bgr)
                end_time_inference = time.perf_counter()

                # --- Collect Metrics ---
                inference_time_ms = (end_time_inference - start_time_inference) * 1000
                # Note: Some strategies might already pre-filter, but this is a safeguard
                person_mask = (classes == detection_strategy.person_class_id)
                num_person_detections_frame = np.sum(person_mask)

                total_inference_time_ms += inference_time_ms
                total_detections += num_person_detections_frame
                detections_per_camera[cam_id] += num_person_detections_frame
                inference_time_per_camera[cam_id] += inference_time_ms

                if frame_counter % 100 == 0:  # Log progress periodically
                    logger.info(f"Processed {frame_counter}/{total_frames_to_process} frames... "
                                f"(Index {frame_idx}/{total_frame_indices - 1}) "
                                f"Last frame ({cam_id}): {num_person_detections_frame} pers_dets, {inference_time_ms:.2f} ms")

            end_time_total = time.perf_counter()
            total_processing_time_sec = end_time_total - start_time_total
            logger.info("--- Frame Processing Finished ---")

            # --- Calculate Aggregate Metrics ---
            if frame_counter > 0:
                avg_inference_time_ms = total_inference_time_ms / frame_counter
                avg_detections = total_detections / frame_counter
                processing_fps = frame_counter / total_processing_time_sec if total_processing_time_sec > 0 else 0

                logger.info(f"Total Frames Processed: {frame_counter}")
                logger.info(f"Unique Frame Indices Processed: {len(processed_indices)}")
                logger.info(f"Total Processing Time: {total_processing_time_sec:.2f} seconds")
                logger.info(f"Overall Average Inference Time: {avg_inference_time_ms:.2f} ms/frame")
                logger.info(f"Overall Average Person Detections: {avg_detections:.2f} per frame processed")
                logger.info(f"Overall Processing FPS: {processing_fps:.2f} frames/sec")
                logger.info(f"Total Person Detections Found: {total_detections}")

                # --- Log Metrics to MLflow ---
                logger.info("Logging aggregate metrics to MLflow...")
                mlflow.log_metric("total_frames_processed", frame_counter)
                mlflow.log_metric("unique_frame_indices_processed", len(processed_indices))
                mlflow.log_metric("total_processing_time_sec", round(total_processing_time_sec, 2))
                mlflow.log_metric("avg_inference_time_ms_per_frame", round(avg_inference_time_ms, 2))
                mlflow.log_metric("total_person_detections", total_detections)
                mlflow.log_metric("avg_detections_per_frame", round(avg_detections, 2))
                mlflow.log_metric("processing_fps", round(processing_fps, 2))

                # Log per-camera metrics
                for cam_id in actual_cameras_used:
                    cam_frames = frame_count_per_camera.get(cam_id, 0)
                    cam_dets = detections_per_camera.get(cam_id, 0)
                    cam_inf_time = inference_time_per_camera.get(cam_id, 0)
                    avg_inf_cam = (cam_inf_time / cam_frames) if cam_frames > 0 else 0
                    avg_dets_cam = (cam_dets / cam_frames) if cam_frames > 0 else 0

                    mlflow.log_metric(f"frames_cam_{cam_id}", cam_frames)
                    mlflow.log_metric(f"detections_cam_{cam_id}", cam_dets)
                    mlflow.log_metric(f"avg_inf_ms_cam_{cam_id}", round(avg_inf_cam, 2))
                    mlflow.log_metric(f"avg_dets_cam_{cam_id}", round(avg_dets_cam, 2))

                logger.info("Metrics logged.")
                mlflow.set_terminated(status="FINISHED")

            else:
                logger.warning("No frames were processed successfully. No metrics to log.")
                mlflow.set_terminated(status="FAILED", state_message="No frames processed")

    # --- Handle potential errors during MLflow run context ---
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user (KeyboardInterrupt).")
        print("MLflow run likely marked as KILLED.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An error occurred during the experiment run: {e}", exc_info=True)
        print(f"MLflow run likely marked as FAILED due to: {e}")
        sys.exit(1)  # Exit with error status

    logger.info(f"--- Experiment Run {run_id if 'run_id' in locals() else 'UNKNOWN'} Completed ---")


if __name__ == "__main__":
    run_experiment()
