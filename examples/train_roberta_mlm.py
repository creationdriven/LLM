#!/usr/bin/env python3
"""
Example: Training RoBERTa for Masked Language Modeling (MLM).

This script demonstrates:
- RoBERTa MLM pretraining
- Dynamic masking (pattern changes each epoch)
- Mixed precision training
- Model checkpointing
"""

import torch
from roberta_kit import (
    RoBERTaForMaskedLM,
    create_model_config,
    create_mlm_dataloader,
    get_tokenizer,
    AMPTrainer,
    compute_mlm_loader_loss,
    save_model,
    get_device,
    count_parameters,
)

# Configuration
MODEL_NAME = "roberta-base (125M)"
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
DEVICE = get_device("cuda" if torch.cuda.is_available() else "cpu")

# Sample training text
TRAINING_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming industries worldwide.",
    "Natural language processing enables computers to understand human language.",
    "Deep learning models can learn complex patterns from data.",
    "RoBERTa improves upon BERT with better training procedures.",
    "Dynamic masking changes the masking pattern each epoch.",
    "Transformer architectures have revolutionized NLP.",
    "Pre-trained language models are powerful tools.",
]


def main():
    print("="*60)
    print("RoBERTa MLM Pretraining Example")
    print("="*60)
    
    # Create model
    config = create_model_config(MODEL_NAME)
    model = RoBERTaForMaskedLM(config).to(DEVICE)
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Device: {DEVICE}")
    print("Note: RoBERTa uses dynamic masking (pattern changes each epoch)")
    
    # Create tokenizer
    tokenizer = get_tokenizer("roberta-base")
    if hasattr(tokenizer, 'encode'):
        def tokenize_fn(text):
            return tokenizer.encode(text, add_special_tokens=False, max_length=510, truncation=True)
    else:
        tokenize_fn = tokenizer
    
    # Create data loader
    train_loader = create_mlm_dataloader(
        texts=TRAINING_TEXTS,
        tokenizer=tokenize_fn,
        batch_size=BATCH_SIZE,
        max_length=128,
        mask_probability=0.15
    )
    
    print(f"\nTraining batches: {len(train_loader)}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Setup mixed precision training
    trainer = AMPTrainer(dtype='float16' if DEVICE.startswith('cuda') else None)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            loss = trainer.train_mlm_step(
                batch["input_ids"],
                batch["labels"],
                model,
                optimizer,
                DEVICE,
                max_grad_norm=1.0
            )
            total_loss += loss
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate perplexity
        perplexity = compute_mlm_loader_loss(model, train_loader, DEVICE)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
    
    # Save model
    save_model(model, "examples/roberta_mlm.pt", config, optimizer=optimizer, epoch=NUM_EPOCHS)
    print("\n✅ Training complete! Model saved to examples/roberta_mlm.pt")


if __name__ == "__main__":
    main()

