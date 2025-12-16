#!/usr/bin/env python3
"""
Example: Distributed/multi-GPU training.

This script demonstrates:
- Multi-GPU training setup
- Model wrapping for DataParallel/DistributedDataParallel
- Distributed utilities
"""

import torch
import torch.nn as nn
from gpt_kit import GPTModel, create_model_config, get_device
from llm_common.distributed import (
    wrap_model_for_distributed,
    get_world_size,
    get_distributed_rank,
    is_main_process,
)

# Configuration
MODEL_NAME = "gpt2-small (124M)"
DEVICE = get_device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("="*60)
    print("Distributed Training Example")
    print("="*60)
    
    # Create model
    config = create_model_config(MODEL_NAME)
    model = GPTModel(config).to(DEVICE)
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    
    # Check distributed status
    world_size = get_world_size()
    rank = get_distributed_rank()
    is_main = is_main_process()
    
    print(f"\nDistributed Status:")
    print(f"  World size: {world_size}")
    print(f"  Rank: {rank}")
    print(f"  Is main process: {is_main}")
    
    # Wrap model for distributed training
    if torch.cuda.device_count() > 1:
        print("\nWrapping model for multi-GPU training...")
        model = wrap_model_for_distributed(model)
        print("  ✓ Model wrapped for parallel training")
    else:
        print("\nSingle GPU/CPU detected - no parallelization needed")
    
    # Example: Training with distributed model
    print("\nExample training step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Dummy input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len)).to(DEVICE)
    
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        input_ids.view(-1),
        ignore_index=-100
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Output shape: {logits.shape}")
    
    print("\n✅ Distributed training setup complete!")
    print("\nNote: For actual distributed training, use:")
    print("  - torch.distributed.launch or torchrun")
    print("  - setup_distributed() before model creation")
    print("  - DistributedDataParallel for best performance")


if __name__ == "__main__":
    main()

