# FILE: src/run_comparison.py
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import subprocess
import time
import copy # For deep copying config dicts

# --- MLflow Imports ---
import mlflow
try:
    from mlflow.tracking import MlflowClient
    from mlflow.models import infer_signature
except ImportError as mlflow_import_err:
     print(f"FATAL: Failed MLflow component import: {mlflow_import_err}. Install mlflow."); sys.exit(1)
# --- End MLflow Imports ---

import dagshub
import torch
from dotenv import load_dotenv
import numpy as np
import cv2
from PIL import Image

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT)); print(f"PROJECT_ROOT added: {PROJECT_ROOT}")

# --- Local Imports ---
try:
    from src.utils.config_loader import load_config
    from src.utils.device_utils import get_selected_device
    from src.utils.reproducibility import set_seed
    from src.pipelines.detection_pipeline import DetectionPipeline
    from src.tracking.strategies import (
        DetectionTrackingStrategy, YoloStrategy, RTDetrStrategy, FasterRCNNStrategy, RfDetrStrategy
    )
    from src.data.loader import FrameDataLoader
except ImportError as e: print(f"Local import error: {e}. Check PYTHONPATH."); sys.exit(1)

# --- Basic Logging Setup ---
comparison_log_file = PROJECT_ROOT / "comparison_experiment.log"
if comparison_log_file.exists():
    try: open(comparison_log_file, 'w').close()
    except OSError as e: print(f"Warn: Could not clear log {comparison_log_file}: {e}")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[ logging.FileHandler(comparison_log_file, mode='a'), logging.StreamHandler(sys.stdout) ])
logger = logging.getLogger(__name__)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision"); warnings.filterwarnings("ignore", category=FutureWarning)

# --- Helper Functions ---
def setup_mlflow_comparison(config: Dict[str, Any]) -> Optional[str]:
    mlflow_config = config; dotenv_path = PROJECT_ROOT / '.env'
    if dotenv_path.exists(): load_dotenv(dotenv_path=dotenv_path); logger.info("Loaded .env file.")
    else: logger.info(".env file not found.")
    try:
        repo_owner=os.getenv("DAGSHUB_REPO_OWNER", "Jwizzed"); repo_name=os.getenv("DAGSHUB_REPO_NAME", "spoton_ml")
        if not os.getenv("MLFLOW_TRACKING_URI"):
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True); logger.info(f"Dagshub init for {repo_owner}/{repo_name}.")
            logger.info(f"MLflow URI set by Dagshub: {mlflow.get_tracking_uri()}")
        else: logger.info("Using MLFLOW_TRACKING_URI from env.")
    except Exception as dag_err:
        logger.warning(f"Dagshub init failed: {dag_err}. Checking env URI.")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            logger.warning("Env URI not set, Dagshub failed. Using local."); local_mlruns = PROJECT_ROOT / "mlruns"; local_mlruns.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{local_mlruns.resolve()}"); logger.info(f"MLflow URI set to local: {mlflow.get_tracking_uri()}")
        else: logger.info(f"Using env URI: {tracking_uri}")

    # *** MODIFIED: Log experiment name being set ***
    experiment_name = mlflow_config.get("mlflow_experiment_name", "Default Comparison")
    logger.info(f"Attempting to set MLflow experiment to: '{experiment_name}'") # <-- Added Log
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment is set to: '{experiment_name}'") # <-- Added Log Confirmation
        client = MlflowClient(); experiment = client.get_experiment_by_name(experiment_name)
        if not experiment: logger.info("Creating experiment..."); experiment_id = client.create_experiment(experiment_name); logger.info(f"Created exp ID: {experiment_id}")
        elif experiment.lifecycle_stage != 'active': logger.error(f"Exp '{experiment_name}' deleted."); return None
        else: experiment_id = experiment.experiment_id; logger.info(f"Using existing exp ID: {experiment_id}")
        return experiment_id
    except Exception as client_err: logger.error(f"MLflow setup/exp error: {client_err}", exc_info=True); return None

def log_params_recursive(params_dict: Dict[str, Any], parent_key: str = ""):
    for key, value in params_dict.items():
        mlflow_key = f"{parent_key}.{key}" if parent_key else key
        if parent_key == "" and key == "models_to_compare": continue
        if isinstance(value, dict): log_params_recursive(value, mlflow_key)
        elif isinstance(value, list):
            try: p_str=json.dumps(value); p_str=p_str[:247]+"..." if len(p_str)>250 else p_str; mlflow.log_param(mlflow_key, p_str)
            except TypeError: mlflow.log_param(mlflow_key, str(value)[:250])
        else: mlflow.log_param(mlflow_key, str(value)[:250])

def log_metrics_dict(metrics: Dict[str, Any]):
    if not metrics: logger.warning("Metrics dict empty."); return
    num_metrics={k:v for k,v in metrics.items() if isinstance(v,(int,float,np.number))}
    non_num=[k for k in metrics if k not in num_metrics];
    if non_num: logger.warning(f"Skipping non-numeric metrics: {non_num}")
    if num_metrics:
        logger.info(f"Logging {len(num_metrics)} numeric metrics...");
        try: mlflow.log_metrics(num_metrics); logger.info("Metrics logged.")
        except Exception as e:
            logger.error(f"log_metrics failed: {e}. Trying individual.", exc_info=True)
            for k, v in num_metrics.items():
                try: mlflow.log_metric(k,v)
                except Exception as ie: logger.error(f"Log metric '{k}' fail: {ie}")
    else: logger.warning("No numeric metrics found.")

def get_sample_data_for_signature(
    config: Dict[str, Any], strategy: DetectionTrackingStrategy, device: torch.device
) -> Tuple[Optional[Any], Optional[Any]]:
    logger.info("Attempting sample data generation for signature...")
    sample_input_sig, sample_output_sig = None, None
    try:
        temp_loader=FrameDataLoader(config); sample_frame_bgr=None
        if temp_loader.image_filenames and temp_loader.active_camera_ids:
             fn=temp_loader.image_filenames[0]; cam=temp_loader.active_camera_ids[0]; im_path=temp_loader.camera_image_dirs[cam]/fn
             if im_path.is_file(): b=np.fromfile(str(im_path), dtype=np.uint8); sample_frame_bgr=cv2.imdecode(b, cv2.IMREAD_COLOR)
             if sample_frame_bgr is not None: logger.info(f"Loaded sample frame: {im_path}")
             else: logger.warning(f"Decode fail: {im_path}")
        del temp_loader
        if sample_frame_bgr is None: logger.warning("No sample frame."); return None, None

        sample_input_tensor = strategy.get_sample_input_tensor(sample_frame_bgr); inf_batch = None
        if isinstance(strategy, FasterRCNNStrategy) and sample_input_tensor is not None:
            sample_input_sig=sample_input_tensor.cpu().numpy(); inf_batch=[sample_input_tensor.to(device)]; logger.info("Using Tensor (CHW) as input signature for FasterRCNN.")
        elif isinstance(strategy, (YoloStrategy, RTDetrStrategy)):
            sample_input_sig=sample_frame_bgr; inf_batch=sample_frame_bgr; logger.info("Using numpy frame as input signature for Ultralytics.")
        elif isinstance(strategy, RfDetrStrategy):
             sample_input_sig=Image.fromarray(cv2.cvtColor(sample_frame_bgr,cv2.COLOR_BGR2RGB)); inf_batch=sample_input_sig; logger.info("Using PIL Image as input signature for RFDETR.")
        elif sample_input_tensor is not None: # Generic fallback
             sample_input_sig=sample_input_tensor.cpu().numpy(); inf_batch=[sample_input_tensor.to(device)]; logger.info("Using generic Tensor (CHW) as input signature.")
        else: logger.error("Cannot determine sample input for signature."); return None, None
        if inf_batch is None: logger.error("Failed prep input for inference."); return sample_input_sig, None

        model_object = strategy.get_model()
        if model_object is None: logger.warning("Model object unavailable."); return sample_input_sig, None

        with torch.no_grad(): # Inference logic
            if isinstance(strategy, FasterRCNNStrategy): preds=model_object(inf_batch); sample_output_sig=preds[0]['boxes'].cpu().numpy() if isinstance(preds, list) and len(preds)>0 and 'boxes' in preds[0] else None
            elif isinstance(strategy, (YoloStrategy, RTDetrStrategy)):
                results=model_object.predict(inf_batch, device=device, verbose=False) # Pass numpy frame
                if results and results[0].boxes:
                    if hasattr(results[0].boxes, 'xyxy') and results[0].boxes.xyxy is not None: sample_output_sig=results[0].boxes.xyxy.cpu().numpy()
                    elif hasattr(results[0].boxes, 'data') and results[0].boxes.data is not None: sample_output_sig=results[0].boxes.data.cpu().numpy()
            elif isinstance(strategy, RfDetrStrategy):
                 dets=model_object.predict(inf_batch) # Pass PIL image
                 if dets and hasattr(dets, 'xyxy'): out=dets.xyxy; sample_output_sig=out.cpu().numpy() if isinstance(out, torch.Tensor) else out
            elif isinstance(model_object, torch.nn.Module): preds=model_object(inf_batch); sample_output_sig=preds.cpu().numpy() if isinstance(preds, torch.Tensor) else None

        if sample_input_sig is not None and sample_output_sig is not None: logger.info(f"Generated sample input ({type(sample_input_sig)}) and output ({type(sample_output_sig)})")
        elif sample_input_sig is not None: logger.warning("Generated sample input, failed output.")
        else: logger.warning("Failed sample input/output generation.")
        return sample_input_sig, sample_output_sig
    except Exception as e: logger.error(f"Sample data error: {e}", exc_info=True); return None, None


# --- Main Comparison Orchestrator ---
def main():
    logger.info("--- Starting Model Comparison Experiment ---"); config_path_str="configs/comparison_config.yaml"; overall_status="SUCCESS"
    comparison_config = load_config(config_path_str)
    if not comparison_config or "models_to_compare" not in comparison_config: logger.critical(f"Config fail: {config_path_str}. Exit."); sys.exit(1)
    models_to_run: List[Dict[str, Any]] = comparison_config["models_to_compare"]
    if not models_to_run: logger.critical("No models in config. Exit."); sys.exit(1)

    experiment_id = setup_mlflow_comparison(comparison_config)
    if not experiment_id: logger.critical("MLflow setup failed. Exit."); sys.exit(1)
    logger.info(f"All runs will be logged to Experiment ID: {experiment_id}") # <-- Log Experiment ID confirmation

    seed = comparison_config.get("environment", {}).get("seed", int(time.time())); set_seed(seed); logger.info(f"Global random seed: {seed}")
    base_device_preference = comparison_config.get("environment", {}).get("device", "auto"); last_run_id_for_log = None

    for model_config in models_to_run:
        model_name=model_config.get("model_name", model_config.get("type", "unknown")); model_type=model_config.get("type", "unknown").lower()
        logger.info(f"\n--- Starting Run for Model: {model_name} ({model_type}) ---")
        run_status="FAILED"; run_id=None; pipeline=None; signature=None; sample_input_data=None; sample_output_data=None; init_failed=False

        try: # Pre-run Init Block
            current_run_config=copy.deepcopy(comparison_config); del current_run_config["models_to_compare"]; current_run_config["model"]=model_config
            run_name_prefix=comparison_config.get("comparison_run_name_prefix", "comp"); run_name=f"{run_name_prefix}_{model_name}"
            requested_device=get_selected_device(base_device_preference); actual_device=requested_device; device_override_reason="None"
            # Device Logic
            if model_type=="fasterrcnn":
                if requested_device.type=='mps': actual_device=torch.device('mps'); device_override_reason="F-RCNN using req MPS"; logger.info(f"[{model_name}] Using MPS.")
                elif requested_device.type!='cuda': logger.warning(f"[{model_name}] F-RCNN req '{requested_device.type}', not CUDA/MPS. Force CPU."); actual_device=torch.device('cpu'); device_override_reason=f"F-RCNN forced CPU"
            elif actual_device.type=='mps':
                if model_type in ['rfdetr']: logger.warning(f"[{model_name}] '{model_type}' MPS issues. Force CPU."); actual_device=torch.device('cpu'); device_override_reason=f"{model_type} forced CPU"
            logger.info(f"[{model_name}] Device: {actual_device} (Req: {requested_device}, Reason: {device_override_reason})")

            logger.info(f"[{model_name}] Initializing pipeline...")
            pipeline = DetectionPipeline(current_run_config, actual_device)
            if not pipeline.initialize_components(): raise RuntimeError(f"Pipeline init failed")

            if pipeline.detection_strategy: # Signature Gen
                 sample_input_data, sample_output_data = get_sample_data_for_signature(current_run_config, pipeline.detection_strategy, actual_device)
                 if sample_input_data is not None and sample_output_data is not None:
                     try: signature = infer_signature(sample_input_data, sample_output_data); logger.info(f"[{model_name}] Signature inferred.")
                     except Exception as e: logger.warning(f"[{model_name}] Sig infer fail: {e}", exc_info=True); signature = None
                 else: logger.warning(f"[{model_name}] Failed sample data gen.")
            else: logger.warning(f"[{model_name}] No strategy for sample data.")
        except Exception as init_err:
            logger.critical(f"[Model: {model_name}] Pre-run failed: {init_err}", exc_info=True); overall_status="PARTIAL_FAILURE"; init_failed=True

        if not init_failed: # Only proceed if init OK
            try: # MLflow Run Block
                # *** MODIFIED: Log experiment_id before starting run ***
                logger.info(f"[{model_name}] Starting run within Experiment ID: {experiment_id}")
                with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                    run_id = run.info.run_id; last_run_id_for_log = run_id
                    logger.info(f"[Model: {model_name}] MLflow Run Started (ID: {run_id})")

                    logger.info(f"[{model_name}] Logging parameters & tags...") # Params & Tags
                    log_params_recursive(current_run_config)
                    mlflow.log_param("env.seed", seed); mlflow.log_param("env.req_device", str(requested_device))
                    mlflow.log_param("env.actual_device", str(actual_device)); mlflow.log_param("env.override_reason", device_override_reason)
                    mlflow.set_tag("model_name", model_name); mlflow.set_tag("model_type", model_type)
                    mlflow.set_tag("comparison_group", comparison_config.get("comparison_run_name_prefix", "default"))
                    mlflow.set_tag("dataset", current_run_config['data']['selected_environment'])
                    scene_id = current_run_config['data'][current_run_config['data']['selected_environment']]['scene_id']; mlflow.set_tag("scene_id", scene_id)

                    logger.info(f"[{model_name}] Logging common artifacts...") # Artifacts
                    comp_conf_path=PROJECT_ROOT/config_path_str; req_path=PROJECT_ROOT/"requirements.txt"
                    if comp_conf_path.is_file(): mlflow.log_artifact(str(comp_conf_path), artifact_path="config")
                    if req_path.is_file(): mlflow.log_artifact(str(req_path), artifact_path="code")
                    else: logger.warning("requirements.txt not found.")
                    try: # Git Info
                        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT).strip().decode()
                        mlflow.set_tag("git_commit_hash", commit)
                        status=subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT_ROOT).strip().decode()
                        mlflow.set_tag("git_status", "dirty" if status else "clean")
                        if status: diff=subprocess.check_output(['git','diff','HEAD'],cwd=PROJECT_ROOT).strip().decode('utf-8','ignore'); diff and mlflow.log_text(diff, artifact_file="code/git_diff.diff")
                    except Exception as git_err: logger.warning(f"Git info log failed: {git_err}")

                    logger.info(f"[{model_name}] Starting pipeline processing...") # Pipeline Run
                    pipeline_success, metrics, active_cameras, num_frames = pipeline.run()
                    if active_cameras: mlflow.log_param("data.actual_cameras", ",".join(active_cameras))
                    if num_frames is not None: mlflow.log_param("data.actual_frames_processed", num_frames)

                    if metrics: # Metrics & Results
                        logger.info(f"[{model_name}] Logging metrics...")
                        log_metrics_dict(metrics)
                        try: m_path = PROJECT_ROOT/f"run_{run_id}_metrics.json"; ser={k:(v.item() if isinstance(v, np.generic) else v) for k,v in metrics.items()}; f=open(m_path,'w'); json.dump(ser,f,indent=4); f.close(); mlflow.log_artifact(str(m_path),"results"); m_path.unlink()
                        except Exception as json_err: logger.warning(f"Metrics dict log failed: {json_err}")

                    if pipeline_success: run_status="FINISHED"; mlflow.set_tag("run_outcome","Success"); logger.info(f"[{model_name}] Run success.")
                    else: run_status="FAILED"; mlflow.set_tag("run_outcome","Failed Execution"); logger.error(f"[{model_name}] Pipeline failed."); overall_status="PARTIAL_FAILURE"

                    # --- Model Logging (No Flavors) ---
                    if run_status == "FINISHED" and pipeline.detection_strategy:
                        logger.info(f"[{model_name}] Logging model artifact (No Flavors)...")
                        model_obj = pipeline.detection_strategy.get_model()
                        if model_obj:
                             if signature:
                                 try: mlflow.log_dict(signature.to_dict(), "signature.json"); logger.info(f"[{model_name}] Logged signature artifact.")
                                 except Exception as sig_log_err: logger.warning(f"Sig artifact log fail: {sig_log_err}")

                             # *** MODIFIED: Fix log_npy error ***
                             if isinstance(sample_input_data, np.ndarray):
                                 input_example_path = PROJECT_ROOT / f"input_example_{run_id}.npy"
                                 try:
                                     np.save(input_example_path, sample_input_data)
                                     mlflow.log_artifact(str(input_example_path), artifact_path="examples") # Log as artifact
                                     logger.info(f"[{model_name}] Logged numpy input example artifact.")
                                     input_example_path.unlink() # Clean up temp file
                                 except Exception as npy_log_err: logger.warning(f"Input example artifact log fail: {npy_log_err}")

                             logger.info(f"[{model_name}] Flavored logging disabled. Logged sig/example artifacts if available.")
                        else: logger.warning(f"[{model_name}] No model object to log.")
            except KeyboardInterrupt:
                logger.warning(f"[Model: {model_name}] Run interrupted."); run_status="KILLED"; overall_status="KILLED"
                if run_id: mlflow.set_tag("run_outcome","Killed"); break
            except Exception as run_err:
                logger.critical(f"[Model: {model_name}] Uncaught error during run: {run_err}", exc_info=True); run_status="FAILED"; overall_status="PARTIAL_FAILURE"
                if run_id: mlflow.set_tag("run_outcome","Crashed")
            finally: # --- Finalize current run ---
                logger.info(f"--- Finalizing Run: {model_name} (Status: {run_status}, ID: {run_id}) ---")
                try:
                    if run_id:
                        client = MlflowClient()
                        try:
                            run_info = client.get_run(run_id)
                            if run_info.info.lifecycle_stage == "active": logger.warning(f"Run {run_id} still active. Terminating."); client.set_terminated(run_id, status=run_status); logger.info(f"Run {run_id} terminated status: {run_status}")
                            else: logger.info(f"Run {run_id} already terminated ({run_info.info.status})")
                            try: client.set_tag(run_id, "final_status", run_status)
                            except Exception: pass
                        except Exception as get_run_err: logger.error(f"Client error finalizing {run_id}: {get_run_err}", exc_info=True)
                except Exception as final_err: logger.error(f"Error in finalization {run_id}: {final_err}", exc_info=True)
        # --- End of MLflow Run Block or Skip ---

    # --- Comparison Finished ---
    logger.info(f"\n--- Model Comparison Finished ---"); logger.info(f"Overall Status: {overall_status}")
    logger.info(f"Comparison log: {comparison_log_file}")
    if last_run_id_for_log and comparison_log_file.exists():
         try:
              logger.info(f"Logging comparison log to last run {last_run_id_for_log}")
              for handler in logging.getLogger().handlers: handler.flush()
              client = MlflowClient(); client.log_artifact(last_run_id_for_log, str(comparison_log_file), artifact_path="comparison_logs")
              logger.info(f"Logged {comparison_log_file.name} to run {last_run_id_for_log}")
         except Exception as log_err: logger.warning(f"Could not log main log: {log_err}")

    if overall_status != "SUCCESS": logger.error("One or more runs failed/killed."); sys.exit(1)
    else: logger.info("All model runs completed successfully."); sys.exit(0)

if __name__ == "__main__":
    main()