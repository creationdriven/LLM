#!/usr/bin/env python3
"""
Example: Training RoBERTa for classification with dynamic masking.

This script demonstrates:
- RoBERTa model initialization
- Classification training
- Dynamic masking (automatic)
- Mixed precision training
- Evaluation metrics
"""

import torch
from roberta_kit import (
    RoBERTaForSequenceClassification,
    create_model_config,
    create_classification_dataloader,
    get_tokenizer,
    AMPTrainer,
    evaluate_classification_metrics,
    save_model,
    get_device,
)

# Configuration
MODEL_NAME = "roberta-base (125M)"
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = get_device("cuda" if torch.cuda.is_available() else "cpu")

# Sample data
TEXTS = [
    "This movie is fantastic!",
    "Terrible experience, would not recommend.",
    "Amazing product, love it!",
    "Poor quality, very disappointed.",
    "Excellent service, highly recommend.",
    "Worst purchase ever.",
    "Great value for money.",
    "Not worth the price.",
    "Outstanding performance!",
    "Complete waste of time.",
]

LABELS = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative


def main():
    print("="*60)
    print("RoBERTa Classification Training Example")
    print("="*60)
    
    # Create model
    config = create_model_config(MODEL_NAME)
    model = RoBERTaForSequenceClassification(config, num_labels=2).to(DEVICE)
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print("Note: RoBERTa uses dynamic masking (pattern changes each epoch)")
    
    # Create tokenizer
    tokenizer = get_tokenizer("roberta-base")
    if hasattr(tokenizer, 'encode'):
        def tokenize_fn(text):
            return tokenizer.encode(text, add_special_tokens=False, max_length=510, truncation=True)
    else:
        tokenize_fn = tokenizer
    
    # Create data loaders
    split_idx = int(len(TEXTS) * 0.8)
    train_loader = create_classification_dataloader(
        texts=TEXTS[:split_idx],
        labels=LABELS[:split_idx],
        tokenizer=tokenize_fn,
        batch_size=BATCH_SIZE
    )
    val_loader = create_classification_dataloader(
        texts=TEXTS[split_idx:],
        labels=LABELS[split_idx:],
        tokenizer=tokenize_fn,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    print(f"\nTraining samples: {split_idx}")
    print(f"Validation samples: {len(TEXTS) - split_idx}")
    
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
            loss = trainer.train_classification_step(
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
                model,
                optimizer,
                DEVICE,
                max_grad_norm=1.0
            )
            total_loss += loss
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluation
        metrics = evaluate_classification_metrics(model, val_loader, DEVICE, num_classes=2)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Val F1: {metrics['f1']:.4f}")
        print(f"  Val Precision: {metrics['precision']:.4f}")
        print(f"  Val Recall: {metrics['recall']:.4f}")
    
    # Save model
    save_model(model, "examples/roberta_classifier.pt", config)
    print("\n✅ Training complete! Model saved to examples/roberta_classifier.pt")


if __name__ == "__main__":
    main()

