"""
Metrics Logger for RF-DETR Training
Advanced metrics collection, analysis, and visualization
"""
import logging
import json
import csv
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import threading
from queue import Queue

logger = logging.getLogger(__name__)


@dataclass
class LoggerConfig:
    """Configuration for metrics logger."""
    
    # Logging configuration
    log_frequency: int = 100
    flush_frequency: int = 1000
    
    # Storage configuration
    json_logging: bool = True
    csv_logging: bool = True
    tensorboard_logging: bool = False  # Optional TensorBoard support
    
    # Performance configuration
    async_logging: bool = True
    buffer_size: int = 10000
    
    # Retention configuration
    max_log_files: int = 50
    max_file_size_mb: int = 100
    
    def __post_init__(self):
        """Validate configuration."""
        if self.log_frequency <= 0:
            raise ValueError(f"log_frequency must be positive, got {self.log_frequency}")
        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {self.buffer_size}")


class MetricsBuffer:
    """Thread-safe buffer for metrics data."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, metrics: Dict[str, Any], timestamp: float, step: int):
        """Add metrics to buffer."""
        with self.lock:
            entry = {
                'metrics': metrics.copy(),
                'timestamp': timestamp,
                'step': step
            }
            self.buffer.append(entry)
    
    def flush(self) -> List[Dict[str, Any]]:
        """Flush buffer and return all entries."""
        with self.lock:
            entries = list(self.buffer)
            self.buffer.clear()
            return entries
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)


class MetricsWriter:
    """Base class for metrics writers."""
    
    def write(self, entries: List[Dict[str, Any]]):
        """Write metrics entries."""
        raise NotImplementedError
    
    def close(self):
        """Close writer and cleanup resources."""
        pass


class JSONWriter(MetricsWriter):
    """JSON metrics writer."""
    
    def __init__(self, log_dir: Path, max_file_size_mb: int = 100):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_mb = max_file_size_mb
        
        self.current_file = None
        self.current_file_path = None
        self.file_counter = 0
        
        self._create_new_file()
    
    def _create_new_file(self):
        """Create new log file."""
        if self.current_file:
            self.current_file.close()
        
        timestamp = int(time.time())
        self.current_file_path = self.log_dir / f"metrics_{timestamp}_{self.file_counter:04d}.jsonl"
        self.current_file = open(self.current_file_path, 'w')
        self.file_counter += 1
        
        logger.debug(f"Created new JSON log file: {self.current_file_path}")
    
    def write(self, entries: List[Dict[str, Any]]):
        """Write entries to JSON log."""
        if not self.current_file:
            self._create_new_file()
        
        for entry in entries:
            json_line = json.dumps(entry, default=str)
            self.current_file.write(json_line + '\n')
        
        self.current_file.flush()
        
        # Check file size and rotate if necessary
        if self.current_file_path.stat().st_size > self.max_file_size_mb * 1024 * 1024:
            self._create_new_file()
    
    def close(self):
        """Close JSON writer."""
        if self.current_file:
            self.current_file.close()
            self.current_file = None


class CSVWriter(MetricsWriter):
    """CSV metrics writer."""
    
    def __init__(self, log_dir: Path, max_file_size_mb: int = 100):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_mb = max_file_size_mb
        
        self.current_file = None
        self.current_writer = None
        self.current_file_path = None
        self.file_counter = 0
        self.fieldnames = set()
        
        self._create_new_file()
    
    def _create_new_file(self):
        """Create new CSV file."""
        if self.current_file:
            self.current_file.close()
        
        timestamp = int(time.time())
        self.current_file_path = self.log_dir / f"metrics_{timestamp}_{self.file_counter:04d}.csv"
        self.current_file = open(self.current_file_path, 'w', newline='')
        self.current_writer = None  # Will be created when we know fieldnames
        self.file_counter += 1
        
        logger.debug(f"Created new CSV log file: {self.current_file_path}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV writing."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def write(self, entries: List[Dict[str, Any]]):
        """Write entries to CSV log."""
        if not entries:
            return
        
        # Flatten all entries and collect fieldnames
        flattened_entries = []
        new_fieldnames = set()
        
        for entry in entries:
            # Flatten the metrics
            flat_entry = {}
            flat_entry['timestamp'] = entry['timestamp']
            flat_entry['step'] = entry['step']
            
            flat_metrics = self._flatten_dict(entry['metrics'])
            flat_entry.update(flat_metrics)
            
            flattened_entries.append(flat_entry)
            new_fieldnames.update(flat_entry.keys())
        
        # Update fieldnames
        self.fieldnames.update(new_fieldnames)
        
        # Create or update CSV writer if fieldnames changed
        if self.current_writer is None or new_fieldnames - set(getattr(self.current_writer, 'fieldnames', [])):
            # Need to recreate writer with new fieldnames
            if self.current_writer is not None:
                # File already has content, need to create new file
                self._create_new_file()
            
            self.current_writer = csv.DictWriter(
                self.current_file,
                fieldnames=sorted(self.fieldnames),
                restval='',
                extrasaction='ignore'
            )
            self.current_writer.writeheader()
        
        # Write entries
        for flat_entry in flattened_entries:
            self.current_writer.writerow(flat_entry)
        
        self.current_file.flush()
        
        # Check file size and rotate if necessary
        if self.current_file_path.stat().st_size > self.max_file_size_mb * 1024 * 1024:
            self._create_new_file()
    
    def close(self):
        """Close CSV writer."""
        if self.current_file:
            self.current_file.close()
            self.current_file = None


class TensorBoardWriter(MetricsWriter):
    """TensorBoard metrics writer (optional)."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.writer = None
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(log_dir))
            logger.info(f"TensorBoard logging enabled: {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available, skipping TensorBoard logging")
    
    def _log_scalar(self, key: str, value: Any, step: int):
        """Log scalar value to TensorBoard."""
        if self.writer and isinstance(value, (int, float)):
            self.writer.add_scalar(key, value, step)
    
    def _flatten_and_log(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """Recursively flatten and log metrics."""
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            
            if isinstance(value, dict):
                self._flatten_and_log(value, step, full_key)
            elif isinstance(value, (int, float, bool)):
                self._log_scalar(full_key, float(value), step)
    
    def write(self, entries: List[Dict[str, Any]]):
        """Write entries to TensorBoard."""
        if not self.writer:
            return
        
        for entry in entries:
            step = entry['step']
            metrics = entry['metrics']
            
            self._flatten_and_log(metrics, step)
        
        self.writer.flush()
    
    def close(self):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
            self.writer = None


class MetricsLogger:
    """
    Advanced metrics logger for RF-DETR training.
    Supports JSON, CSV, and optional TensorBoard logging with async writing.
    """
    
    def __init__(
        self,
        log_dir: str,
        config: Optional[LoggerConfig] = None
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
            config: Logger configuration
        """
        self.config = config or LoggerConfig()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize buffer
        self.buffer = MetricsBuffer(self.config.buffer_size)
        
        # Initialize writers
        self.writers = []
        self._initialize_writers()
        
        # Async logging setup
        self.logging_thread = None
        self.logging_queue = None
        self.stop_logging = False
        
        if self.config.async_logging:
            self._start_async_logging()
        
        # Statistics tracking
        self.total_entries = 0
        self.last_flush_time = time.time()
        
        logger.info(f"Initialized MetricsLogger with log_dir={self.log_dir}")
        logger.info(f"  Writers: {[type(w).__name__ for w in self.writers]}")
        logger.info(f"  Async logging: {self.config.async_logging}")
    
    def _initialize_writers(self):
        """Initialize configured writers."""
        
        if self.config.json_logging:
            json_dir = self.log_dir / "json"
            self.writers.append(JSONWriter(json_dir, self.config.max_file_size_mb))
        
        if self.config.csv_logging:
            csv_dir = self.log_dir / "csv"
            self.writers.append(CSVWriter(csv_dir, self.config.max_file_size_mb))
        
        if self.config.tensorboard_logging:
            tb_dir = self.log_dir / "tensorboard"
            self.writers.append(TensorBoardWriter(tb_dir))
    
    def _start_async_logging(self):
        """Start async logging thread."""
        self.logging_queue = Queue()
        self.logging_thread = threading.Thread(target=self._async_logging_worker, daemon=True)
        self.logging_thread.start()
        
        logger.debug("Started async logging thread")
    
    def _async_logging_worker(self):
        """Async logging worker thread."""
        while not self.stop_logging:
            try:
                # Get entries from queue (blocking with timeout)
                entries = self.logging_queue.get(timeout=1.0)
                if entries is None:  # Shutdown signal
                    break
                
                # Write to all writers
                for writer in self.writers:
                    try:
                        writer.write(entries)
                    except Exception as e:
                        logger.warning(f"Writer {type(writer).__name__} failed: {e}")
                
                self.logging_queue.task_done()
                
            except Exception as e:
                if not self.stop_logging:  # Don't log errors during shutdown
                    logger.warning(f"Async logging error: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        timestamp: Optional[float] = None
    ):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step number
            timestamp: Timestamp (uses current time if None)
        """
        
        if timestamp is None:
            timestamp = time.time()
        
        if step is None:
            step = self.total_entries
        
        # Add to buffer
        self.buffer.add(metrics, timestamp, step)
        self.total_entries += 1
        
        # Check if we should flush
        should_flush = (
            self.total_entries % self.config.flush_frequency == 0 or
            time.time() - self.last_flush_time > 60  # Flush at least every minute
        )
        
        if should_flush:
            self.flush()
    
    def log_event(self, event_name: str, event_data: Optional[Dict[str, Any]] = None):
        """
        Log a specific event.
        
        Args:
            event_name: Name of the event
            event_data: Additional event data
        """
        
        event_metrics = {
            "event": event_name,
            "event_data": event_data or {}
        }
        
        self.log_metrics(event_metrics)
    
    def flush(self):
        """Flush buffered metrics to writers."""
        
        entries = self.buffer.flush()
        if not entries:
            return
        
        if self.config.async_logging and self.logging_queue:
            # Send to async worker
            self.logging_queue.put(entries)
        else:
            # Synchronous writing
            for writer in self.writers:
                try:
                    writer.write(entries)
                except Exception as e:
                    logger.warning(f"Writer {type(writer).__name__} failed: {e}")
        
        self.last_flush_time = time.time()
        
        logger.debug(f"Flushed {len(entries)} metric entries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logger statistics."""
        
        return {
            "total_entries": self.total_entries,
            "buffer_size": self.buffer.size(),
            "writers_count": len(self.writers),
            "writer_types": [type(w).__name__ for w in self.writers],
            "async_logging": self.config.async_logging,
            "log_dir": str(self.log_dir)
        }
    
    def cleanup_old_logs(self):
        """Cleanup old log files based on retention policy."""
        
        for writer_type in ["json", "csv"]:
            writer_dir = self.log_dir / writer_type
            if not writer_dir.exists():
                continue
            
            # Get all log files sorted by modification time
            log_files = sorted(
                writer_dir.glob("metrics_*.jsonl" if writer_type == "json" else "metrics_*.csv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Remove excess files
            if len(log_files) > self.config.max_log_files:
                files_to_remove = log_files[self.config.max_log_files:]
                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        logger.debug(f"Removed old log file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old log file {file_path}: {e}")
    
    def close(self):
        """Close metrics logger and cleanup resources."""
        
        # Flush any remaining metrics
        self.flush()
        
        # Stop async logging
        if self.config.async_logging and self.logging_thread:
            self.stop_logging = True
            if self.logging_queue:
                self.logging_queue.put(None)  # Shutdown signal
                self.logging_thread.join(timeout=5.0)
        
        # Close all writers
        for writer in self.writers:
            try:
                writer.close()
            except Exception as e:
                logger.warning(f"Error closing writer {type(writer).__name__}: {e}")
        
        # Cleanup old logs
        try:
            self.cleanup_old_logs()
        except Exception as e:
            logger.warning(f"Error during log cleanup: {e}")
        
        logger.info(f"Closed MetricsLogger (logged {self.total_entries} entries)")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_metrics_logger(
    log_dir: str,
    json_logging: bool = True,
    csv_logging: bool = True,
    tensorboard_logging: bool = False,
    async_logging: bool = True,
    **kwargs
) -> MetricsLogger:
    """
    Convenience function to create metrics logger.
    
    Args:
        log_dir: Directory for log files
        json_logging: Enable JSON logging
        csv_logging: Enable CSV logging
        tensorboard_logging: Enable TensorBoard logging
        async_logging: Enable async logging
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured metrics logger
    """
    
    config = LoggerConfig(
        json_logging=json_logging,
        csv_logging=csv_logging,
        tensorboard_logging=tensorboard_logging,
        async_logging=async_logging,
        **kwargs
    )
    
    return MetricsLogger(log_dir, config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create metrics logger
    logger_instance = create_metrics_logger(
        log_dir="test_metrics",
        tensorboard_logging=True
    )
    
    # Simulate logging metrics
    with logger_instance:
        for step in range(100):
            metrics = {
                "loss": {
                    "total": 1.0 - step * 0.01,
                    "ce": 0.6 - step * 0.005,
                    "bbox": 0.3 - step * 0.003
                },
                "accuracy": 0.5 + step * 0.005,
                "learning_rate": 1e-4 * (0.99 ** step)
            }
            
            logger_instance.log_metrics(metrics, step=step)
            
            # Log some events
            if step % 20 == 0:
                logger_instance.log_event("checkpoint_saved", {
                    "step": step,
                    "model_size": "123MB"
                })
    
    print("Metrics logging test completed successfully")