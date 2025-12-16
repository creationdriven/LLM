#!/usr/bin/env python3
"""
Example: Training GPT model with advanced features.

This script demonstrates:
- Model initialization
- Data loading
- Training with mixed precision
- Learning rate scheduling
- Early stopping
- Model checkpointing
"""

import torch
import torch.nn as nn
from gpt_kit import (
    GPTModel,
    create_model_config,
    create_pretraining_dataloader,
    AMPTrainer,
    get_cosine_scheduler,
    EarlyStopping,
    save_model,
    get_device,
    count_parameters,
)

# Configuration
MODEL_NAME = "gpt2-small (124M)"
BATCH_SIZE = 4
MAX_LENGTH = 256
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
DEVICE = get_device("cuda" if torch.cuda.is_available() else "cpu")

# Sample training text
TRAINING_TEXT = """
The quick brown fox jumps over the lazy dog.
Machine learning is transforming industries.
Natural language processing enables computers to understand human language.
Deep learning models can learn complex patterns from data.
"""


def main():
    print("="*60)
    print("GPT Training Example")
    print("="*60)
    
    # Create model
    config = create_model_config(MODEL_NAME)
    model = GPTModel(config).to(DEVICE)
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Device: {DEVICE}")
    
    # Create data loader
    train_loader = create_pretraining_dataloader(
        text=TRAINING_TEXT,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        stride=MAX_LENGTH // 2
    )
    
    print(f"\nTraining batches: {len(train_loader)}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_scheduler(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * NUM_EPOCHS
    )
    
    # Setup mixed precision training
    trainer = AMPTrainer(dtype='float16' if DEVICE.startswith('cuda') else None)
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=2, min_delta=0.01)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            loss = trainer.train_step(
                input_batch,
                target_batch,
                model,
                optimizer,
                DEVICE,
                max_grad_norm=1.0
            )
            
            total_loss += loss
            scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")
        
        # Early stopping check
        if early_stopping(avg_loss, model):
            print("Early stopping triggered")
            break
    
    # Save model
    save_model(model, "examples/gpt_model.pt", config, optimizer=optimizer)
    print("\n✅ Training complete! Model saved to examples/gpt_model.pt")


if __name__ == "__main__":
    main()

