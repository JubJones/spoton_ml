import torch
from torchvision import tv_tensors
import pytest
from unittest.mock import Mock

# Ensure src is in path for imports (pytest.ini should handle this)
from src.utils.torch_utils import collate_fn, get_optimizer, get_lr_scheduler

def test_collate_fn():
    """Tests the custom collate function."""
    # Mock data similar to dataset output
    img_tensor1 = torch.rand(3, 100, 100)
    target1 = {
        "boxes": tv_tensors.BoundingBoxes(torch.tensor([[10, 10, 50, 50]]), format="XYXY", canvas_size=(100, 100)),
        "labels": torch.tensor([1], dtype=torch.int64)
    }
    img_tensor2 = torch.rand(3, 120, 120)
    target2 = {
        "boxes": tv_tensors.BoundingBoxes(torch.tensor([[20, 30, 60, 80], [0, 0, 10, 10]]), format="XYXY", canvas_size=(120, 120)),
        "labels": torch.tensor([1, 1], dtype=torch.int64)
    }
    batch = [(img_tensor1, target1), (img_tensor2, target2)]

    collated_images, collated_targets = collate_fn(batch)

    # Assertions
    assert isinstance(collated_images, list)
    assert len(collated_images) == 2
    assert torch.equal(collated_images[0], img_tensor1)
    assert torch.equal(collated_images[1], img_tensor2)

    assert isinstance(collated_targets, list)
    assert len(collated_targets) == 2
    assert collated_targets[0] == target1
    assert collated_targets[1] == target2

def test_collate_fn_empty_targets():
    """Tests collate_fn with a sample that has no targets."""
    img_tensor1 = torch.rand(3, 100, 100)
    target1 = {
        "boxes": tv_tensors.BoundingBoxes(torch.empty((0, 4)), format="XYXY", canvas_size=(100, 100)),
        "labels": torch.empty((0,), dtype=torch.int64)
    }
    batch = [(img_tensor1, target1)]

    collated_images, collated_targets = collate_fn(batch)

    assert len(collated_images) == 1
    assert len(collated_targets) == 1
    assert collated_targets[0]["boxes"].shape == (0, 4)
    assert collated_targets[0]["labels"].shape == (0,)


def test_get_optimizer():
    """Tests the optimizer factory function."""
    mock_params = [torch.nn.Parameter(torch.randn(1))]
    lr = 0.01
    wd = 0.001

    opt_adamw = get_optimizer("AdamW", mock_params, lr, wd)
    assert isinstance(opt_adamw, torch.optim.AdamW)

    opt_sgd = get_optimizer("SGD", mock_params, lr, wd)
    assert isinstance(opt_sgd, torch.optim.SGD)

    # Test fallback
    opt_unknown = get_optimizer("UnknownOpt", mock_params, lr, wd)
    assert isinstance(opt_unknown, torch.optim.AdamW)


def test_get_lr_scheduler():
    """Tests the LR scheduler factory function."""
    mock_optimizer = Mock(spec=torch.optim.Optimizer)
    mock_optimizer.param_groups = [{'lr': 0.1}] # Mock attribute used by some schedulers

    sched_step = get_lr_scheduler("StepLR", mock_optimizer, step_size=5, gamma=0.1)
    assert isinstance(sched_step, torch.optim.lr_scheduler.StepLR)

    sched_cosine = get_lr_scheduler("CosineAnnealingLR", mock_optimizer, T_max=10)
    assert isinstance(sched_cosine, torch.optim.lr_scheduler.CosineAnnealingLR)

    # Test fallback
    sched_none = get_lr_scheduler("UnknownSched", mock_optimizer)
    assert sched_none is None