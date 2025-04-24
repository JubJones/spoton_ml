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

    # Attempt Dagshub initialization if library is available
    if dagshub:
        try:
            repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "DefaultOwner")
            repo_name = os.getenv("DAGSHUB_REPO_NAME", "DefaultRepo")
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            logger.info(f"Dagshub initialized for {repo_owner}/{repo_name}.")
            logger.info(f"MLflow tracking URI set by Dagshub: {mlflow.get_tracking_uri()}")
        except Exception as dag_err:
            logger.warning(f"Dagshub initialization failed: {dag_err}. Checking MLFLOW_TRACKING_URI.")
    else:
        logger.info("Dagshub library not installed. Checking MLFLOW_TRACKING_URI.")

    # Set tracking URI if not already set by Dagshub
    if not mlflow.tracking.utils._is_tracking_uri_set():
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set from environment variable: {tracking_uri}")
        else:
            logger.warning("MLFLOW_TRACKING_URI not set and Dagshub setup failed/skipped. Using local tracking.")
            local_mlruns = PROJECT_ROOT / "mlruns"
            local_mlruns.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{local_mlruns.resolve()}")
            logger.info(f"MLflow tracking URI set to local: {mlflow.get_tracking_uri()}")

    # Set or Create Experiment
    experiment_name = mlflow_config.get("experiment_name", default_experiment_name)
    logger.info(f"Attempting to set MLflow experiment to: '{experiment_name}'")
    try:
        mlflow.set_experiment(experiment_name)
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
        logger.error(f"Failed to set/get/create MLflow experiment '{experiment_name}': {client_err}", exc_info=True)
        return None
