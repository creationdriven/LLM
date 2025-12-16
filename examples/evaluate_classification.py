#!/usr/bin/env python3
"""
Example: Evaluating classification models with comprehensive metrics.

This script demonstrates:
- Loading a trained classification model
- Computing evaluation metrics (accuracy, precision, recall, F1)
- Confusion matrix analysis
- Per-class metrics
"""

import torch
from bert_kit import (
    BERTForSequenceClassification,
    create_model_config,
    create_classification_dataloader,
    get_tokenizer,
    evaluate_classification_metrics,
    compute_confusion_matrix,
    load_model,
    get_device,
)

# Configuration
MODEL_NAME = "bert-base-uncased (110M)"
DEVICE = get_device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "examples/bert_classifier.pt"  # Optional: load from checkpoint
NUM_CLASSES = 2

# Test data
TEST_TEXTS = [
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

TEST_LABELS = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative


def main():
    print("="*60)
    print("Classification Model Evaluation Example")
    print("="*60)
    
    # Load or create model
    import os
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nLoading model from {CHECKPOINT_PATH}...")
        model, config, _ = load_model(CHECKPOINT_PATH, BERTForSequenceClassification, device=DEVICE)
        model.num_labels = NUM_CLASSES  # Ensure num_labels is set
        print("✓ Model loaded from checkpoint")
    else:
        print(f"\nCreating new model (no checkpoint found at {CHECKPOINT_PATH})...")
        config = create_model_config(MODEL_NAME)
        model = BERTForSequenceClassification(config, num_labels=NUM_CLASSES).to(DEVICE)
        print("✓ Model created (untrained - metrics will be random)")
    
    model.eval()
    
    # Create tokenizer
    tokenizer = get_tokenizer("bert-base-uncased")
    if hasattr(tokenizer, 'encode'):
        def tokenize_fn(text):
            return tokenizer.encode(text, add_special_tokens=False, max_length=510, truncation=True)
    else:
        tokenize_fn = tokenizer
    
    # Create test data loader
    test_loader = create_classification_dataloader(
        texts=TEST_TEXTS,
        labels=TEST_LABELS,
        tokenizer=tokenize_fn,
        batch_size=4,
        shuffle=False
    )
    
    print(f"\nTest samples: {len(TEST_TEXTS)}")
    
    # Evaluate with comprehensive metrics
    print("\n" + "="*60)
    print("Computing Evaluation Metrics")
    print("="*60)
    
    metrics = evaluate_classification_metrics(
        model, test_loader, DEVICE, num_classes=NUM_CLASSES
    )
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.append(predictions.cpu())
            all_labels.append(batch["labels"])
    
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    
    confusion_matrix = compute_confusion_matrix(predictions, labels, num_classes=NUM_CLASSES)
    print(f"\n{confusion_matrix}")
    print("\nRows = True labels, Columns = Predicted labels")
    print("Class 0 = Negative, Class 1 = Positive")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    tp = torch.diag(confusion_matrix)
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp
    
    for class_idx in range(NUM_CLASSES):
        precision = tp[class_idx].float() / (tp[class_idx] + fp[class_idx] + 1e-10)
        recall = tp[class_idx].float() / (tp[class_idx] + fn[class_idx] + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        class_name = "Negative" if class_idx == 0 else "Positive"
        print(f"\n  Class {class_idx} ({class_name}):")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    Support:   {confusion_matrix.sum(dim=1)[class_idx].item()}")
    
    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)
    print("\nNote: For meaningful results, train the model first or load a pre-trained checkpoint.")


if __name__ == "__main__":
    main()

