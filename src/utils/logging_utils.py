import logging
import sys
import time
from pathlib import Path


def setup_logging(log_prefix: str, log_dir: Path, level: int = logging.INFO) -> Path:
    """
    Sets up basic file and stream logging.
    """
    log_filename = f"{log_prefix}_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = log_dir / log_filename
    log_dir.mkdir(parents=True, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    return log_file_path
