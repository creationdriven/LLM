#!/usr/bin/env python3
"""
Example: Model quantization for reduced memory usage.

This script demonstrates:
- Dynamic quantization
- Model size comparison
- Memory usage reduction
"""

import torch
from gpt_kit import GPTModel, create_model_config, get_device
from llm_common.quantization import (
    quantize_model_dynamic,
    get_model_size_mb,
    compare_model_sizes,
)

# Configuration
MODEL_NAME = "gpt2-small (124M)"
DEVICE = get_device("cpu")  # Quantization works best on CPU


def main():
    print("="*60)
    print("Model Quantization Example")
    print("="*60)
    
    # Create model
    config = create_model_config(MODEL_NAME)
    model = GPTModel(config).to(DEVICE)
    
    # Measure original size
    original_size = get_model_size_mb(model)
    print(f"\nOriginal Model:")
    print(f"  Size: {original_size:.2f} MB")
    
    # Quantize model
    print("\nQuantizing model...")
    quantized_model = quantize_model_dynamic(model, dtype=torch.qint8)
    
    # Compare sizes
    comparison = compare_model_sizes(model, quantized_model)
    
    print(f"\nQuantized Model:")
    print(f"  Size: {comparison['quantized_size_mb']:.2f} MB")
    print(f"  Compression ratio: {comparison['compression_ratio']:.2f}x")
    print(f"  Size reduction: {comparison['size_reduction_percent']:.1f}%")
    
    # Test inference
    print("\nTesting inference...")
    test_input = torch.randint(0, config["vocab_size"], (1, 10))
    
    model.eval()
    with torch.no_grad():
        original_output = model(test_input)
    
    quantized_model.eval()
    with torch.no_grad():
        quantized_output = quantized_model(test_input)
    
    print(f"  Original output shape: {original_output.shape}")
    print(f"  Quantized output shape: {quantized_output.shape}")
    print(f"  Outputs match: {torch.allclose(original_output, quantized_output, atol=1e-2)}")
    
    print("\n✅ Quantization complete!")
    print("   Quantized models use less memory and can run faster on CPU.")


if __name__ == "__main__":
    main()

