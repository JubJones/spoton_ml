import pytest
import torch
from pathlib import Path

@pytest.fixture(scope="session")
def mock_project_root() -> Path:
    """Provides a mock project root Path object."""
    # For simplicity here, we'll just use a dummy path name
    return Path("/fake/project/root")

@pytest.fixture
def mock_fasterrcnn_config() -> dict:
    """Provides a minimal mock config dictionary for Faster R-CNN training."""
    return {
        "data": {
            "base_path": "/fake/data/MTMMC",
            "scenes_to_include": [{"scene_id": "s01", "camera_ids": ["c01"]}],
            "val_split_ratio": 0.2,
            "use_data_subset": False,
            "data_subset_fraction": 1.0,
            "num_workers": 0,
        },
        "model": {
            "type": "fasterrcnn",
            "name_tag": "test_fasterrcnn",
            "backbone_weights": "DEFAULT", # Use string for easier mocking check
            "num_classes": 2,             # person + background
            "trainable_backbone_layers": 3,
            "person_class_id": 1 # Assuming PyTorch model where person is 1
        },
        "training": {
            "engine": "pytorch",
            "epochs": 2,
            "batch_size": 1,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "weight_decay": 0.005,
            "lr_scheduler": "StepLR",
            "lr_scheduler_step_size": 1,
            "lr_scheduler_gamma": 0.1,
            "gradient_clip_norm": None,
            "checkpoint_dir": "mock_checkpoints",
            "save_best_metric": "val_map_50"
        },
        "environment": {
             "device": "cpu",
             "seed": 42
        },
         "preprocessing_visualization": { # Needed by get_transform implicitly?
             "enabled": False
         }
    }

@pytest.fixture
def cpu_device() -> torch.device:
    """Provides a CPU device."""
    return torch.device("cpu")