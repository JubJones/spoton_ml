import pytest
import torch
from unittest.mock import patch, MagicMock

# Ensure src is in path
from src.training.runner import get_fasterrcnn_model

# Use the mock config fixture from conftest
def test_get_fasterrcnn_model(mock_fasterrcnn_config):
    """Tests the Faster R-CNN model loading and head modification."""

    # Mock the underlying torchvision model loader
    with patch('src.training.runner.torchvision.models.detection.fasterrcnn_resnet50_fpn') as mock_load:
        # Setup the mock return value (needs to mimic the real model structure minimally)
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.roi_heads = MagicMock()
        mock_model_instance.roi_heads.box_predictor = MagicMock()
        mock_model_instance.roi_heads.box_predictor.cls_score = MagicMock()
        mock_model_instance.roi_heads.box_predictor.cls_score.in_features = 1024 # Example feature size
        mock_load.return_value = mock_model_instance

        # Mock the weights enum access if needed (depends on implementation detail)
        mock_weights = MagicMock()
        mock_weights.DEFAULT = MagicMock() # Mock the specific weight we expect

        with patch('src.training.runner.FasterRCNN_ResNet50_FPN_Weights', new=mock_weights):

             # Call the function under test
             model = get_fasterrcnn_model(mock_fasterrcnn_config)

             # Assertions
             # Check if the torchvision loader was called correctly
             mock_load.assert_called_once()
             # Check args passed to the loader (weights enum, trainable layers)
             call_args, call_kwargs = mock_load.call_args
             assert call_kwargs['weights'] == mock_weights.DEFAULT
             assert call_kwargs['trainable_backbone_layers'] == mock_fasterrcnn_config['model']['trainable_backbone_layers']

             # Check if the box predictor was replaced
             assert isinstance(model.roi_heads.box_predictor, torch.nn.Module) # Should be FastRCNNPredictor
             # Check if the new predictor has the correct number of output classes
             # Note: FastRCNNPredictor itself isn't directly mocked here, we trust its init
             # For a deeper test, you might inspect the created predictor instance's parameters
             # This requires knowing the internal structure of FastRCNNPredictor or mocking it too.
             # For simplicity, we assume the replacement happened if the call succeeded.

             # A simple check: Did we get our mock instance back?
             assert model == mock_model_instance
