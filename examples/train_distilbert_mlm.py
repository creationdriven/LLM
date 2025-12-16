#!/usr/bin/env python3
"""
Example: Training DistilBERT for Masked Language Modeling (MLM).

This script demonstrates:
- DistilBERT MLM pretraining
- Smaller, faster model training
- Mixed precision training
- Performance comparison
"""

import torch
import time
from distilbert_kit import (
    DistilBERTForMaskedLM,
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
MODEL_NAME = "distilbert-base-uncased (66M)"
BATCH_SIZE = 16  # Larger batch due to smaller model
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
DEVICE = get_device("cuda" if torch.cuda.is_available() else "cpu")

# Sample training text
TRAINING_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming industries worldwide.",
    "Natural language processing enables computers to understand human language.",
    "Deep learning models can learn complex patterns from data.",
    "DistilBERT is a smaller, faster version of BERT.",
    "Knowledge distillation reduces model size while maintaining performance.",
    "Transformer architectures have revolutionized NLP.",
    "Pre-trained language models are powerful tools.",
] * 2  # Duplicate for more training data


def main():
    print("="*60)
    print("DistilBERT MLM Pretraining Example")
    print("="*60)
    
    # Create model
    config = create_model_config(MODEL_NAME)
    model = DistilBERTForMaskedLM(config).to(DEVICE)
    
    num_params = count_parameters(model)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Parameters: {num_params:,} (40% smaller than BERT)")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE} (larger than BERT due to smaller model)")
    
    # Create tokenizer
    tokenizer = get_tokenizer("distilbert-base-uncased")
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
    
    # Training loop with timing
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
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
        epoch_time = time.time() - epoch_start
        
        # Evaluate perplexity
        perplexity = compute_mlm_loader_loss(model, train_loader, DEVICE)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total training time: {total_time:.2f} seconds")
    print(f"   Average: {total_time/NUM_EPOCHS:.2f} seconds/epoch")
    
    # Save model
    save_model(model, "examples/distilbert_mlm.pt", config, optimizer=optimizer, epoch=NUM_EPOCHS)
    print("\n✅ Training complete! Model saved to examples/distilbert_mlm.pt")
    print("   DistilBERT is ~60% faster than BERT for inference!")


if __name__ == "__main__":
    main()

