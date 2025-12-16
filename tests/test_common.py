"""
Tests for common utilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from llm_common import get_device, count_parameters, print_model_summary, save_model, load_model


def test_get_device():
    """Test device detection."""
    device = get_device()
    assert device in ["cuda", "cpu", "mps"]
    
    device = get_device("cpu")
    assert device == "cpu"
    
    print("✓ get_device() test passed")


def test_count_parameters():
    """Test parameter counting."""
    model = nn.Linear(10, 5)
    total = count_parameters(model)
    assert total == 55  # 10*5 + 5 (weights + bias)
    
    trainable = count_parameters(model, trainable_only=True)
    assert trainable == total
    
    print("✓ count_parameters() test passed")


def test_save_load_model():
    """Test model saving and loading."""
    import tempfile
    import os
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.linear = nn.Linear(config["input_size"], config["output_size"])
        
        def forward(self, x):
            return self.linear(x)
    
    config = {"input_size": 10, "output_size": 5}
    model = SimpleModel(config)
    
    # Save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        save_model(model, temp_path, config)
        assert os.path.exists(temp_path)
        
        # Load
        loaded_model, loaded_config, _ = load_model(temp_path, SimpleModel)
        assert loaded_config == config
        
        # Test forward pass
        x = torch.randn(2, 10)
        out1 = model(x)
        out2 = loaded_model(x)
        assert torch.allclose(out1, out2)
        
        print("✓ save_model() and load_model() tests passed")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    print("Running common utilities tests...")
    test_get_device()
    test_count_parameters()
    test_save_load_model()
    print("\n✅ All common utilities tests passed!")

