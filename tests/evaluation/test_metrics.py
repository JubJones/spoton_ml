import pytest
import torch
from unittest.mock import patch, MagicMock

# Ensure src is in path
from src.evaluation.metrics import compute_map_metrics, TORCHMETRICS_AVAILABLE

# Only run if torchmetrics is actually installed
pytestmark = pytest.mark.skipif(
    not TORCHMETRICS_AVAILABLE, reason="torchmetrics not installed"
)


@patch("src.evaluation.metrics.MeanAveragePrecision")  # Mock the torchmetrics class
def test_compute_map_metrics_basic(MockMeanAveragePrecision, cpu_device):
    """Very basic test for compute_map_metrics interface."""

    # Configure the mock class and its instance
    mock_metric_instance = MagicMock()
    mock_metric_instance.to.return_value = mock_metric_instance
    mock_metric_instance.compute.return_value = {
        "map": torch.tensor(0.6),
        "map_50": torch.tensor(0.8),
        "map_75": torch.tensor(0.7),
        "map_per_class": torch.tensor(
            [0.5, 0.65], device=cpu_device
        ),  # Result likely on CPU
        "classes": torch.tensor([0, 1], device=cpu_device),  # Result likely on CPU
    }
    MockMeanAveragePrecision.return_value = mock_metric_instance

    # Create minimal dummy prediction/target data ON THE TARGET DEVICE
    predictions = [
        {
            "boxes": torch.tensor([[0, 0, 10, 10]], device=cpu_device),
            "scores": torch.tensor([0.9], device=cpu_device),
            "labels": torch.tensor([1], device=cpu_device),  # Person class ID = 1
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0, 0, 9, 9]], device=cpu_device),
            "labels": torch.tensor([1], device=cpu_device),  # Person class ID = 1
        }
    ]

    # Call the function
    map_results = compute_map_metrics(
        predictions, targets, device=cpu_device, person_class_id=1
    )

    # Assertions
    MockMeanAveragePrecision.assert_called_once()  # Was the metric class instantiated?
    mock_metric_instance.to.assert_called_with(
        cpu_device
    )  # Verify .to(device) was called

    mock_metric_instance.update.assert_called_once()  # Was update called?
    # --- Check args passed to update ---
    update_args, _ = mock_metric_instance.update.call_args
    assert len(update_args) == 2
    # Check if the data passed to update matches (or is on the correct device)
    assert (
        isinstance(update_args[0], list)
        and update_args[0][0]["boxes"].device == cpu_device
    )
    assert (
        isinstance(update_args[1], list)
        and update_args[1][0]["boxes"].device == cpu_device
    )
    # More detailed checks if necessary

    mock_metric_instance.compute.assert_called_once()  # Was compute called?

    # Check the output dictionary structure and mocked values
    assert isinstance(map_results, dict)
    assert "eval_map" in map_results
    assert "eval_map_50" in map_results
    assert "eval_ap_person" in map_results
    assert map_results["eval_map"] == 0.6000  # Check rounding if applied
    assert map_results["eval_ap_person"] == 0.6500  # Check extraction
