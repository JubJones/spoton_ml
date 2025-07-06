import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

try:
    import dagshub
except ImportError:
    dagshub = None
    logging.warning("Dagshub library not found. MLflow setup will rely on environment variables or local tracking.")

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def setup_mlflow_experiment(config: Dict[str, Any], default_experiment_name: str) -> Optional[str]:
    """
    Initializes MLflow connection (Dagshub or local) and ensures the experiment exists.
    """
    mlflow_config = config.get("mlflow", {})
    dotenv_path = PROJECT_ROOT / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logger.info("Loaded environment variables from .env file.")
    else:
        logger.info(".env file not found, relying on environment or defaults.")

    # Attempt Dagshub initialization first
    dagshub_initialized = False
    if dagshub:
        try:
            repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "DefaultOwner")
            repo_name = os.getenv("DAGSHUB_REPO_NAME", "DefaultRepo")
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            logger.info(f"Dagshub initialized for {repo_owner}/{repo_name}.")
            if mlflow.get_tracking_uri() is not None:
                logger.info(f"MLflow tracking URI set by Dagshub: {mlflow.get_tracking_uri()}")
                dagshub_initialized = True
            else:
                logger.warning("Dagshub init called but MLflow tracking URI is still None.")
        except Exception as dag_err:
            logger.warning(f"Dagshub initialization failed: {dag_err}. Checking MLFLOW_TRACKING_URI.")
    else:
        logger.info("Dagshub library not installed or available. Checking MLFLOW_TRACKING_URI.")

    # Set tracking URI from environment or default to local only if Dagshub didn't set it
    if not dagshub_initialized:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            try:
                mlflow.set_tracking_uri(tracking_uri)
                logger.info(f"MLflow tracking URI set from environment variable: {tracking_uri}")
            except Exception as uri_err:
                logger.error(f"Failed to set tracking URI from environment variable '{tracking_uri}': {uri_err}. Falling back to local.")
                dagshub_initialized = False
        else:
             logger.info("MLFLOW_TRACKING_URI environment variable not found.")


        current_uri = mlflow.get_tracking_uri()
        if not dagshub_initialized and (not tracking_uri or current_uri is None or current_uri.startswith("file:")):
             logger.warning("No remote tracking URI configured (Dagshub/MLFLOW_TRACKING_URI). Using local tracking.")
             local_mlruns = PROJECT_ROOT / "mlruns"
             local_mlruns.mkdir(parents=True, exist_ok=True)
             local_uri = local_mlruns.resolve().as_uri()
             mlflow.set_tracking_uri(local_uri)
             logger.info(f"MLflow tracking URI explicitly set to local: {mlflow.get_tracking_uri()}")


    # Set or Create Experiment
    experiment_name = mlflow_config.get("experiment_name", default_experiment_name)
    logger.info(f"Attempting to set MLflow experiment to: '{experiment_name}'")
    try:
        client = MlflowClient()
        if client.tracking_uri is None:
            logger.critical("MLflow client still has no tracking URI configured after setup attempts. Exiting.")
            return None

        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.info(f"Experiment '{experiment_name}' not found. Creating...")
            experiment_id = client.create_experiment(experiment_name)
            logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
            experiment = client.get_experiment(experiment_id)
            if not experiment:
                 logger.error(f"Failed to fetch experiment '{experiment_name}' immediately after creation.")
                 return None
        elif experiment.lifecycle_stage != 'active':
            logger.error(f"Experiment '{experiment_name}' exists but is deleted or archived (lifecycle_stage: {experiment.lifecycle_stage}).")
            return None
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing active experiment ID: {experiment_id}")

        mlflow.set_experiment(experiment_id=experiment_id)
        logger.info(f"MLflow context set to experiment '{experiment_name}' (ID: {experiment_id})")
        return experiment_id

    except Exception as client_err:
        logger.error(f"Failed to set/get/create MLflow experiment '{experiment_name}': {client_err}", exc_info=True)
        return None


def download_best_model_checkpoint(run_id: str, destination_dir: Path) -> Optional[Path]:
    """
    Downloads the best model checkpoint from a specific MLflow run.

    It searches for an artifact in the 'checkpoints' directory of the run that
    starts with 'ckpt_best_'. If found, it downloads it to the destination
    directory.

    Args:
        run_id: The ID of the MLflow run.
        destination_dir: The local directory where the artifact will be downloaded.

    Returns:
        The local path to the downloaded model checkpoint file, or None if not found.
    """
    logger.info(f"Attempting to download best model checkpoint for run_id: {run_id}")
    client = MlflowClient()
    try:
        artifacts = client.list_artifacts(run_id, path="checkpoints")
        best_model_artifact = None
        for artifact in artifacts:
            if artifact.is_dir:
                continue
            if Path(artifact.path).name.startswith("ckpt_best_"):
                best_model_artifact = artifact
                break

        if not best_model_artifact:
            logger.warning(
                f"No 'best' model checkpoint found in 'checkpoints/' for run {run_id}. "
                "You may need to check the artifacts in the MLflow UI."
            )
            return None

        logger.info(f"Found best model artifact: {best_model_artifact.path}")
        destination_dir.mkdir(parents=True, exist_ok=True)
        local_path_str = client.download_artifacts(
            run_id=run_id,
            path=best_model_artifact.path,
            dst_path=str(destination_dir),
        )
        local_path = Path(local_path_str)
        if local_path.is_file():
            logger.info(f"Successfully downloaded model to: {local_path}")
            return local_path
        else:
            logger.error(f"MLflow client reported download to {local_path_str}, but file not found.")
            return None

    except Exception as e:
        logger.error(f"Failed to download model checkpoint for run {run_id}: {e}", exc_info=True)
        return None
