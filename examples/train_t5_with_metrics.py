#!/usr/bin/env python3
"""
Example: Training T5 with BLEU/ROUGE evaluation.

This script demonstrates:
- T5 text-to-text training
- BLEU score computation
- ROUGE score computation
- Mixed precision training
"""

import torch
from t5_kit import (
    T5ForConditionalGeneration,
    create_model_config,
    create_t5_dataloader,
    get_tokenizer,
    AMPTrainer,
    compute_bleu_score,
    compute_rouge_score,
    tokenize_text,
    save_model,
    get_device,
)

# Configuration
MODEL_NAME = "t5-small (60M)"
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
DEVICE = get_device("cuda" if torch.cuda.is_available() else "cpu")

# Sample text-to-text data
INPUTS = [
    "translate English to French: Hello world",
    "summarize: Machine learning is a subset of artificial intelligence.",
    "translate English to French: How are you?",
    "summarize: Deep learning uses neural networks with multiple layers.",
    "translate English to French: Good morning",
    "summarize: Natural language processing enables computers to understand text.",
]

TARGETS = [
    "Bonjour le monde",
    "ML is part of AI",
    "Comment allez-vous?",
    "Deep learning uses multi-layer neural networks.",
    "Bonjour",
    "NLP helps computers understand text.",
]


def main():
    print("="*60)
    print("T5 Training with BLEU/ROUGE Evaluation")
    print("="*60)
    
    # Create model
    config = create_model_config(MODEL_NAME)
    model = T5ForConditionalGeneration(config).to(DEVICE)
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    
    # Create tokenizer
    tokenizer = get_tokenizer("t5-small")
    if hasattr(tokenizer, 'encode'):
        def tokenize_fn(text):
            return tokenizer.encode(text, add_special_tokens=False, max_length=510, truncation=True)
    else:
        tokenize_fn = tokenizer
    
    # Create data loader
    train_loader = create_t5_dataloader(
        inputs=INPUTS,
        targets=TARGETS,
        tokenizer=tokenize_fn,
        batch_size=BATCH_SIZE,
        max_input_length=128,
        max_target_length=64
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
            loss = trainer.train_step(
                batch["input_ids"],
                batch["decoder_input_ids"],
                batch["labels"],
                model,
                optimizer,
                DEVICE,
                max_grad_norm=1.0,
                attention_mask=batch.get("attention_mask"),
                decoder_attention_mask=batch.get("decoder_attention_mask")
            )
            total_loss += loss
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")
    
    # Evaluation with BLEU/ROUGE
    print("\n" + "="*60)
    print("Evaluation with BLEU/ROUGE")
    print("="*60)
    
    # Tokenize references and candidates (simplified - in practice would generate)
    references_tokens = [tokenize_text(target) for target in TARGETS]
    # For demo, use targets as candidates (in practice, generate from model)
    candidates_tokens = [tokenize_text(target) for target in TARGETS]
    
    # Compute BLEU
    bleu_scores = compute_bleu_score(references_tokens, candidates_tokens)
    print(f"\nBLEU Scores:")
    for key, value in bleu_scores.items():
        print(f"  {key}: {value:.4f}")
    
    # Compute ROUGE
    rouge_scores = compute_rouge_score(TARGETS, TARGETS)
    if rouge_scores:
        print(f"\nROUGE Scores:")
        for key, value in rouge_scores.items():
            print(f"  {key}: {value:.4f}")
    else:
        print("\nROUGE scores not available (install rouge-score)")
    
    # Save model
    save_model(model, "examples/t5_model_with_metrics.pt", config)
    print("\n✅ Training complete! Model saved to examples/t5_model_with_metrics.pt")


if __name__ == "__main__":
    main()

