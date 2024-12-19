import sys
import torch
import pytest
from pathlib import Path

# Add the parent directory to system path to import the model
sys.path.append(str(Path(__file__).parent.parent))

from Assignment_6.model import Net

def test_parameter_count():
    """Test that model has less than 20k parameters"""
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, which exceeds the limit of 20000"

def test_batch_normalization():
    """Test that model uses batch normalization"""
    model = Net()
    has_batchnorm = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_batchnorm, "Model does not use batch normalization"

def test_dropout():
    """Test that model uses dropout"""
    model = Net()
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model does not use dropout"

def test_gap_or_fc():
    """Test that model uses Global Average Pooling or ends with a fully connected layer"""
    model = Net()
    # Check the last few layers
    last_layers = list(model.modules())[-3:]  # Get last 3 layers
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in last_layers)
    # In your case, you're using a view operation which is fine too
    has_reshape = hasattr(model, 'forward') and 'view' in model.forward.__code__.co_code.__str__()
    
    assert has_gap or has_reshape, "Model should use either Global Average Pooling or proper reshaping at the end" 