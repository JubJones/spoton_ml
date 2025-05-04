import pytest
import torch
from torchvision import tv_tensors

from torchvision.tv_tensors import BoundingBoxFormat
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure src is in path
from src.data.training_dataset import MTMMCDetectionDataset
from src.training.runner import get_transform  # Test transform getter here


# Use the mock config fixture
@pytest.fixture
def mock_dataset_config(mock_fasterrcnn_config):
    # Simplify for dataset testing
    config = mock_fasterrcnn_config.copy()
    config["data"]["scenes_to_include"] = [{"scene_id": "s01", "camera_ids": ["c01"]}]
    return config


# Mock data for a single sample
MOCK_IMG_PATH = Path("/fake/data/MTMMC/train/train/s01/c01/rgb/000001.jpg")
MOCK_IMG_ARRAY_BGR = np.random.randint(
    0, 256, (108, 192, 3), dtype=np.uint8
)  # Smaller size
MOCK_ANNOTATIONS = [
    (10, 50.0, 60.0, 20.0, 40.0),  # obj_id, cx, cy, w, h
    (11, 100.0, 50.0, 10.0, 10.0),
]
# Expected format after conversion (XYXY)
EXPECTED_BOXES_XYXY = torch.tensor(
    [
        [40.0, 40.0, 60.0, 80.0],  # 50-10, 60-20, 50+10, 60+20
        [95.0, 45.0, 105.0, 55.0],  # 100-5, 50-5, 100+5, 50+5
    ],
    dtype=torch.float32,
)
EXPECTED_LABELS = torch.tensor([1, 1], dtype=torch.int64)  # person_class_id = 1


# Test the get_transform utility function separately
def test_get_transform(mock_dataset_config):
    transform_train = get_transform(train=True, config=mock_dataset_config)
    transform_val = get_transform(train=False, config=mock_dataset_config)

    # Check for specific transform types more robustly
    assert any(
        "RandomHorizontalFlip" in str(type(t)) for t in transform_train.transforms
    )
    assert not any(
        "RandomHorizontalFlip" in str(type(t)) for t in transform_val.transforms
    )
    assert any("Normalize" in str(type(t)) for t in transform_train.transforms)
    assert any("Normalize" in str(type(t)) for t in transform_val.transforms)
    assert any("ToDtype" in str(type(t)) for t in transform_train.transforms)
    assert any("ToDtype" in str(type(t)) for t in transform_val.transforms)


@patch("src.data.training_dataset.cv2.imdecode")
@patch("src.data.training_dataset.np.fromfile")
@patch("src.data.training_dataset.Path.is_file")
@patch(
    "src.data.training_dataset.MTMMCDetectionDataset._load_data_samples"
)  # Mock initial loading
def test_mtmmc_detection_dataset_getitem(
    mock_load_samples, mock_is_file, mock_fromfile, mock_imdecode, mock_dataset_config
):
    """Tests the __getitem__ method of the dataset."""
    # Setup Mocks
    mock_load_samples.return_value = None  # Prevent actual loading during init
    mock_is_file.return_value = True
    mock_fromfile.return_value = np.array([1, 2, 3], dtype=np.uint8)  # Dummy bytes
    mock_imdecode.return_value = MOCK_IMG_ARRAY_BGR.copy()  # Use a copy

    # Create dataset instance (loading is mocked, splitting needs mock data)
    # Use get_transform(train=False) for basic testing without augmentation randomness
    basic_transform = get_transform(train=False, config=mock_dataset_config)
    dataset = MTMMCDetectionDataset(
        mock_dataset_config, mode="train", transforms=basic_transform
    )
    # Manually set the split data for testing __getitem__
    dataset.samples_split = [(MOCK_IMG_PATH, MOCK_ANNOTATIONS)]
    dataset.class_id = 1  # Ensure class ID is set as expected by config

    # Call __getitem__
    image_out, target_out = dataset[0]

    # Assertions
    mock_imdecode.assert_called_once()
    assert isinstance(image_out, torch.Tensor)
    # Check shape after transforms (should include Normalize, ToDtype, potentially resize if added)
    # Since no resize is in the basic transform, shape reflects ToTensor conversion
    assert image_out.shape[0] == 3  # CHW format
    assert image_out.dtype == torch.float32

    assert isinstance(target_out, dict)
    assert "boxes" in target_out
    assert "labels" in target_out

    assert isinstance(target_out["boxes"], tv_tensors.BoundingBoxes)
    assert target_out["boxes"].format.value == "XYXY"
    # OR explicitly compare with the enum member
    # assert target_out["boxes"].format == BoundingBoxFormat.XYXY

    # Canvas size might change if transforms include resizing, check relative to output image H, W
    assert (
        target_out["boxes"].canvas_size == image_out.shape[1:]
    )  # Should match image H, W
    # Note: Transforms might slightly alter box coordinates due to normalization/clipping. Use allclose.
    # For this basic test without resize, coordinates should be close.
    assert torch.allclose(target_out["boxes"].data, EXPECTED_BOXES_XYXY, atol=1e-5)

    assert isinstance(target_out["labels"], torch.Tensor)
    assert torch.equal(target_out["labels"], EXPECTED_LABELS)


@patch("src.data.training_dataset.cv2.imdecode")
@patch("src.data.training_dataset.np.fromfile")
@patch("src.data.training_dataset.Path.is_file")
@patch("src.data.training_dataset.MTMMCDetectionDataset._load_data_samples")
def test_mtmmc_detection_dataset_getitem_no_annotations(
    mock_load_samples, mock_is_file, mock_fromfile, mock_imdecode, mock_dataset_config
):
    """Tests __getitem__ when a frame has no annotations."""
    # Setup Mocks
    mock_load_samples.return_value = None
    mock_is_file.return_value = True
    mock_fromfile.return_value = np.array([1, 2, 3], dtype=np.uint8)
    mock_imdecode.return_value = MOCK_IMG_ARRAY_BGR.copy()  # Use a copy

    basic_transform = get_transform(train=False, config=mock_dataset_config)
    dataset = MTMMCDetectionDataset(
        mock_dataset_config, mode="train", transforms=basic_transform
    )
    # Manually set split data with EMPTY annotations
    dataset.samples_split = [(MOCK_IMG_PATH, [])]
    dataset.class_id = 1

    # Call __getitem__
    image_out, target_out = dataset[0]

    # Assertions for empty targets
    assert target_out["boxes"].shape == (0, 4)
    assert target_out["labels"].shape == (0,)
    assert target_out["boxes"].canvas_size == image_out.shape[1:]
