#!/usr/bin/env python3
"""
Example: Loading and using saved model checkpoints.

This script demonstrates:
- Loading saved checkpoints
- Resuming training from checkpoints
- Using checkpoints for inference
- Loading optimizer state
"""

import torch
from gpt_kit import GPTModel, create_model_config, load_model, get_device
from llm_common.checkpoint import save_model

# Configuration
MODEL_NAME = "gpt2-small (124M)"
DEVICE = get_device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "examples/gpt_checkpoint.pt"


def create_and_save_model():
    """Create a model and save it for demonstration."""
    print("Creating and saving model...")
    config = create_model_config(MODEL_NAME)
    model = GPTModel(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Simulate some training
    dummy_input = torch.randint(0, config["vocab_size"], (1, 10)).to(DEVICE)
    logits = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        dummy_input.view(-1)
    )
    loss.backward()
    optimizer.step()
    
    # Save checkpoint
    save_model(
        model, CHECKPOINT_PATH, config,
        optimizer=optimizer, epoch=5, loss=loss.item()
    )
    print(f"✓ Model saved to {CHECKPOINT_PATH}")
    return config


def load_checkpoint_basic():
    """Basic checkpoint loading."""
    print("\n" + "="*60)
    print("Basic Checkpoint Loading")
    print("="*60)
    
    config = create_model_config(MODEL_NAME)
    model, loaded_config, _ = load_model(
        CHECKPOINT_PATH, GPTModel, device=DEVICE
    )
    
    print(f"✓ Model loaded")
    print(f"  Config matches: {config == loaded_config}")
    print(f"  Model on device: {next(model.parameters()).device}")
    
    # Test inference
    test_input = torch.randint(0, config["vocab_size"], (1, 10)).to(DEVICE)
    with torch.no_grad():
        output = model(test_input)
    print(f"  Output shape: {output.shape}")
    
    return model, loaded_config


def load_checkpoint_with_optimizer():
    """Load checkpoint with optimizer state."""
    print("\n" + "="*60)
    print("Loading Checkpoint with Optimizer")
    print("="*60)
    
    model, config, optimizer = load_model(
        CHECKPOINT_PATH, GPTModel, device=DEVICE,
        return_optimizer=True,
        optimizer_class=torch.optim.AdamW,
        lr=3e-4
    )
    
    print(f"✓ Model and optimizer loaded")
    print(f"  Optimizer type: {type(optimizer)}")
    print(f"  Optimizer state keys: {len(optimizer.state_dict()['state'])}")
    
    # Continue training
    model.train()
    dummy_input = torch.randint(0, config["vocab_size"], (1, 10)).to(DEVICE)
    logits = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        dummy_input.view(-1)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"  Continued training - loss: {loss.item():.4f}")
    
    return model, optimizer


def load_checkpoint_metadata():
    """Load checkpoint and access metadata."""
    print("\n" + "="*60)
    print("Accessing Checkpoint Metadata")
    print("="*60)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    print("Checkpoint contents:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"  {key}: Model state dict ({len(checkpoint[key])} keys)")
        elif key == 'optimizer_state_dict':
            print(f"  {key}: Optimizer state dict")
        else:
            print(f"  {key}: {checkpoint[key]}")
    
    print(f"\n  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")


def main():
    print("="*60)
    print("Checkpoint Loading Examples")
    print("="*60)
    
    # Create and save a model first
    import os
    if not os.path.exists(CHECKPOINT_PATH):
        create_and_save_model()
    else:
        print(f"Using existing checkpoint: {CHECKPOINT_PATH}")
    
    # Demonstrate different loading scenarios
    load_checkpoint_basic()
    load_checkpoint_with_optimizer()
    load_checkpoint_metadata()
    
    print("\n" + "="*60)
    print("✅ All checkpoint loading examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()

