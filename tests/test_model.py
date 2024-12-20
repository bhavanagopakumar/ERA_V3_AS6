import sys
import torch
import pytest
from pathlib import Path

# Add the parent directory to system path to import the model
sys.path.append(str(Path(__file__).parent.parent))

from model import Net

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
    # Get all modules
    modules = list(model.modules())
    
    # Check for GAP (through adaptive_avg_pool2d in forward method)
    has_gap = 'adaptive_avg_pool2d' in str(model.forward.__code__.co_code)
    
    # Check for FC layer
    has_fc = any(isinstance(m, torch.nn.Linear) for m in modules)
    
    assert has_gap or has_fc, "Model doesnt use Global Average Pooling or doesnt end with a fully connected layer"