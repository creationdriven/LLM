"""
Integration tests for LLM kits.

Tests complete workflows including training, evaluation, and model saving/loading.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
import shutil


def test_gpt_training_workflow():
    """Test complete GPT training workflow."""
    print("\nTesting GPT training workflow...")
    
    from gpt_kit import (
        GPTModel, create_model_config, create_pretraining_dataloader,
        train_step_with_accumulation, save_model, load_model, get_device
    )
    
    # Setup
    config = create_model_config("gpt2-small (124M)")
    device = get_device("cpu")
    model = GPTModel(config).to(device)
    
    # Create small dataset
    text = "The quick brown fox jumps over the lazy dog. " * 10
    train_loader = create_pretraining_dataloader(
        text=text, batch_size=2, max_length=32, stride=16
    )
    
    # Training step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    step_count = 0
    
    for input_batch, target_batch in list(train_loader)[:2]:  # Just 2 batches
        loss, step_count = train_step_with_accumulation(
            input_batch, target_batch, model, optimizer, device,
            accumulation_steps=1, step_count=step_count
        )
        assert loss > 0
    
    # Save and load
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        save_model(model, temp_path, config, optimizer=optimizer, epoch=1)
        assert os.path.exists(temp_path)
        
        loaded_model, loaded_config, _ = load_model(temp_path, GPTModel, device=device)
        assert loaded_config == config
        
        # Test forward pass
        test_input = torch.randint(0, config["vocab_size"], (1, 10))
        out1 = model(test_input)
        out2 = loaded_model(test_input)
        assert torch.allclose(out1, out2, atol=1e-5)
        
        print("  ✓ GPT training workflow passed")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_bert_classification_workflow():
    """Test complete BERT classification workflow."""
    print("\nTesting BERT classification workflow...")
    
    from bert_kit import (
        BERTForSequenceClassification, create_model_config,
        create_classification_dataloader, evaluate_classification_metrics,
        save_model, get_device
    )
    
    # Setup
    config = create_model_config("bert-base-uncased (110M)")
    device = get_device("cpu")
    model = BERTForSequenceClassification(config, num_labels=2).to(device)
    
    # Create small dataset
    texts = ["Good", "Bad", "Great", "Terrible"]
    labels = [1, 0, 1, 0]
    
    def simple_tokenize(text):
        return [hash(word) % 1000 for word in text.lower().split()]
    
    train_loader = create_classification_dataloader(
        texts=texts, labels=labels, tokenizer=simple_tokenize, batch_size=2
    )
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    for batch in train_loader:
        from bert_kit import compute_classification_loss
        loss = compute_classification_loss(
            batch["input_ids"], batch["attention_mask"],
            batch["labels"], model, device
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        break  # Just one step
    
    # Evaluation
    metrics = evaluate_classification_metrics(model, train_loader, device, num_classes=2)
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    
    print("  ✓ BERT classification workflow passed")
    return True


def test_roberta_with_metrics():
    """Test RoBERTa with evaluation metrics."""
    print("\nTesting RoBERTa with metrics...")
    
    from roberta_kit import (
        RoBERTaForSequenceClassification, create_model_config,
        create_classification_dataloader, evaluate_classification_metrics,
        AMPTrainer, get_device
    )
    
    config = create_model_config("roberta-base (125M)")
    device = get_device("cpu")
    model = RoBERTaForSequenceClassification(config, num_labels=2).to(device)
    
    texts = ["Positive", "Negative"]
    labels = [1, 0]
    
    def simple_tokenize(text):
        return [hash(word) % 1000 for word in text.lower().split()]
    
    train_loader = create_classification_dataloader(
        texts=texts, labels=labels, tokenizer=simple_tokenize, batch_size=2
    )
    
    # Test AMP trainer (disabled for CPU)
    trainer = AMPTrainer(dtype='float16', enabled=False)  # Disable AMP for CPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for batch in train_loader:
        loss = trainer.train_classification_step(
            batch["input_ids"], batch["attention_mask"],
            batch["labels"], model, optimizer, device
        )
        assert loss > 0
        break
    
    # Test metrics
    metrics = evaluate_classification_metrics(model, train_loader, device, num_classes=2)
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    
    print("  ✓ RoBERTa with metrics passed")
    return True


def test_t5_generation():
    """Test T5 text-to-text generation."""
    print("\nTesting T5 generation...")
    
    from t5_kit import (
        T5ForConditionalGeneration, create_model_config,
        create_t5_dataloader, get_device
    )
    
    config = create_model_config("t5-small (60M)")
    device = get_device("cpu")
    model = T5ForConditionalGeneration(config).to(device)
    
    inputs = ["translate: hello"]
    targets = ["bonjour"]
    
    def simple_tokenize(text):
        return [hash(word) % 1000 for word in text.lower().split()]
    
    train_loader = create_t5_dataloader(
        inputs=inputs, targets=targets, tokenizer=simple_tokenize, batch_size=1
    )
    
    # Test forward pass
    for batch in train_loader:
        from t5_kit import compute_t5_loss
        loss = compute_t5_loss(
            batch["input_ids"],
            batch["decoder_input_ids"],
            batch["labels"],
            model,
            device
        )
        assert loss.item() > 0
        break
    
    print("  ✓ T5 generation passed")
    return True


def test_shared_utilities():
    """Test shared utilities package."""
    print("\nTesting shared utilities...")
    
    from llm_common import get_device, count_parameters, save_model, load_model
    from llm_common.quantization import get_model_size_mb, quantize_model_dynamic
    from llm_common.distributed import get_world_size, is_main_process
    
    # Test device
    device = get_device()
    assert device in ["cuda", "cpu", "mps"]
    
    # Test distributed utilities
    assert get_world_size() >= 1
    assert isinstance(is_main_process(), bool)
    
    # Test quantization
    import torch.nn as nn
    test_model = nn.Linear(10, 5)
    size_mb = get_model_size_mb(test_model)
    assert size_mb > 0
    
    quantized = quantize_model_dynamic(test_model)
    assert quantized is not None
    
    print("  ✓ Shared utilities passed")
    return True


if __name__ == "__main__":
    print("="*60)
    print("INTEGRATION TESTS")
    print("="*60)
    
    results = []
    results.append(("GPT Training Workflow", test_gpt_training_workflow()))
    results.append(("BERT Classification Workflow", test_bert_classification_workflow()))
    results.append(("RoBERTa with Metrics", test_roberta_with_metrics()))
    results.append(("T5 Generation", test_t5_generation()))
    results.append(("Shared Utilities", test_shared_utilities()))
    
    print("\n" + "="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Results: {passed}/{total} integration tests passed")
    
    if passed == total:
        print("✅ All integration tests passed!")
        sys.exit(0)
    else:
        print("❌ Some integration tests failed")
        sys.exit(1)

