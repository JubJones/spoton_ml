import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/experiment_config.yaml") -> Optional[Dict[str, Any]]:
    """
    Loads the YAML configuration file.
    """
    full_config_path = Path(config_path)
    if not full_config_path.is_file():
        project_root = Path(__file__).parent.parent.parent
        full_config_path = project_root / config_path
        if not full_config_path.is_file():
            logger.error(
                f"Configuration file not found at '{config_path}' or relative to project root '{project_root}'.")
            return None

    try:
        with open(full_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from: {full_config_path}")

        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {full_config_path}: {e}", exc_info=True)
        return None

    except Exception as e:
        logger.error(f"Error loading configuration file {full_config_path}: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
    cfg_path = PROJECT_ROOT_DIR / "configs/experiment_config.yaml"
    print(f"Attempting to load config from: {cfg_path}")
    config = load_config(str(cfg_path))
    if config:
        print("Config loaded successfully:")
        import json

        print(json.dumps(config, indent=2))
    else:
        print("Failed to load config.")
