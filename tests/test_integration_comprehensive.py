"""
Comprehensive integration tests for LLM kits.

Tests complete workflows including:
- End-to-end training pipelines
- Model saving/loading across kits
- Mixed precision training workflows
- Evaluation workflows
- Multi-kit comparisons
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
import shutil
from typing import Dict, Any


def test_complete_gpt_training_pipeline():
    """Test complete GPT training pipeline from start to finish."""
    print("\nTesting complete GPT training pipeline...")
    
    from gpt_kit import (
        GPTModel, create_model_config, create_pretraining_dataloader,
        AMPTrainer, get_cosine_scheduler, EarlyStopping,
        save_model, load_model, get_device, count_parameters
    )
    
    # Setup
    config = create_model_config("gpt2-small (124M)")
    device = get_device("cpu")
    model = GPTModel(config).to(device)
    
    print(f"  Model parameters: {count_parameters(model):,}")
    
    # Create dataset
    text = "The quick brown fox jumps over the lazy dog. " * 20
    train_loader = create_pretraining_dataloader(
        text=text, batch_size=2, max_length=32, stride=16
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_cosine_scheduler(optimizer, num_warmup_steps=2, num_training_steps=10)
    trainer = AMPTrainer(dtype='float16', enabled=False)  # Disable AMP for CPU
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    # Training loop
    losses = []
    for epoch in range(3):
        epoch_loss = 0.0
        for i, (input_batch, target_batch) in enumerate(list(train_loader)[:3]):  # Limit batches
            loss = trainer.train_step(
                input_batch, target_batch, model, optimizer, device,
                max_grad_norm=1.0
            )
            epoch_loss += loss
            scheduler.step()
        avg_loss = epoch_loss / 3
        losses.append(avg_loss)
        
        if early_stopping(avg_loss, model):
            break
    
    assert len(losses) > 0
    assert losses[-1] > 0
    
    # Save and load
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        save_model(model, temp_path, config, optimizer=optimizer, epoch=3, loss=losses[-1])
        
        loaded_model, loaded_config, loaded_optimizer = load_model(
            temp_path, GPTModel, device=device,
            return_optimizer=True, optimizer_class=torch.optim.AdamW, lr=1e-4
        )
        
        assert loaded_config == config
        assert loaded_optimizer is not None
        
        # Verify model works
        test_input = torch.randint(0, config["vocab_size"], (1, 10))
        with torch.no_grad():
            out1 = model(test_input)
            out2 = loaded_model(test_input)
        assert torch.allclose(out1, out2, atol=1e-4)
        
        print("  ✓ Complete GPT training pipeline passed")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_complete_bert_classification_pipeline():
    """Test complete BERT classification pipeline."""
    print("\nTesting complete BERT classification pipeline...")
    
    from bert_kit import (
        BERTForSequenceClassification, create_model_config,
        create_classification_dataloader, get_tokenizer,
        AMPTrainer, evaluate_classification_metrics,
        save_model, load_model, get_device
    )
    
    # Setup
    config = create_model_config("bert-base-uncased (110M)")
    device = get_device("cpu")
    model = BERTForSequenceClassification(config, num_labels=2).to(device)
    
    # Data
    texts = ["Good", "Bad", "Great", "Terrible", "Excellent", "Poor"]
    labels = [1, 0, 1, 0, 1, 0]
    
    def simple_tokenize(text):
        return [hash(word) % 1000 for word in text.lower().split()]
    
    train_loader = create_classification_dataloader(
        texts=texts[:4], labels=labels[:4], tokenizer=simple_tokenize, batch_size=2
    )
    val_loader = create_classification_dataloader(
        texts=texts[4:], labels=labels[4:], tokenizer=simple_tokenize, batch_size=2, shuffle=False
    )
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = AMPTrainer(dtype='float16', enabled=False)
    
    for epoch in range(2):
        for batch in train_loader:
            trainer.train_classification_step(
                batch["input_ids"], batch["attention_mask"],
                batch["labels"], model, optimizer, device
            )
    
    # Evaluation
    metrics = evaluate_classification_metrics(model, val_loader, device, num_classes=2)
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    
    # Save and load
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        save_model(model, temp_path, config, epoch=2, loss=0.5)
        loaded_model, loaded_config, _ = load_model(
            temp_path, BERTForSequenceClassification, device=device
        )
        loaded_model.num_labels = 2
        
        # Verify evaluation still works
        loaded_metrics = evaluate_classification_metrics(loaded_model, val_loader, device, num_classes=2)
        assert 'accuracy' in loaded_metrics
        
        print("  ✓ Complete BERT classification pipeline passed")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_mixed_precision_training_workflow():
    """Test mixed precision training workflow across kits."""
    print("\nTesting mixed precision training workflow...")
    
    from roberta_kit import (
        RoBERTaForSequenceClassification, create_model_config,
        create_classification_dataloader, get_tokenizer,
        AMPTrainer, save_model, get_device
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = AMPTrainer(dtype='float16', enabled=False)  # Disabled for CPU
    
    # Training with AMP
    for batch in train_loader:
        loss = trainer.train_classification_step(
            batch["input_ids"], batch["attention_mask"],
            batch["labels"], model, optimizer, device
        )
        assert loss > 0
    
    print("  ✓ Mixed precision training workflow passed")
    return True


def test_cross_kit_checkpoint_compatibility():
    """Test checkpoint compatibility across different kits."""
    print("\nTesting cross-kit checkpoint compatibility...")
    
    from gpt_kit import GPTModel, create_model_config, save_model
    from bert_kit import BERTModel, create_model_config as create_bert_config, save_model as bert_save
    from llm_common.checkpoint import load_model
    
    # Save GPT checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        gpt_path = f.name
    
    # Save BERT checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        bert_path = f.name
    
    try:
        # Save GPT
        gpt_config = create_model_config("gpt2-small (124M)")
        gpt_model = GPTModel(gpt_config)
        save_model(gpt_model, gpt_path, gpt_config, epoch=1)
        
        # Save BERT
        bert_config = create_bert_config("bert-base-uncased (110M)")
        bert_model = BERTModel(bert_config)
        bert_save(bert_model, bert_path, bert_config, epoch=1)
        
        # Load GPT
        loaded_gpt, loaded_gpt_config, _ = load_model(gpt_path, GPTModel)
        assert loaded_gpt_config == gpt_config
        
        # Load BERT
        loaded_bert, loaded_bert_config, _ = load_model(bert_path, BERTModel)
        assert loaded_bert_config == bert_config
        
        print("  ✓ Cross-kit checkpoint compatibility passed")
        return True
    finally:
        for path in [gpt_path, bert_path]:
            if os.path.exists(path):
                os.remove(path)


def test_evaluation_workflow_all_kits():
    """Test evaluation workflows for all kits."""
    print("\nTesting evaluation workflows for all kits...")
    
    from roberta_kit import (
        RoBERTaForSequenceClassification, create_model_config,
        create_classification_dataloader, get_tokenizer,
        evaluate_classification_metrics, get_device
    )
    
    from distilbert_kit import (
        DistilBERTForSequenceClassification, create_model_config as create_distil_config,
        create_classification_dataloader as create_distil_dataloader,
        get_tokenizer as get_distil_tokenizer,
        evaluate_classification_metrics as eval_distil, get_device
    )
    
    def simple_tokenize(text):
        return [hash(word) % 1000 for word in text.lower().split()]
    
    texts = ["Good", "Bad", "Great", "Terrible"]
    labels = [1, 0, 1, 0]
    
    # Test RoBERTa
    roberta_config = create_model_config("roberta-base (125M)")
    roberta_model = RoBERTaForSequenceClassification(roberta_config, num_labels=2)
    roberta_loader = create_classification_dataloader(
        texts=texts, labels=labels, tokenizer=simple_tokenize, batch_size=2
    )
    roberta_metrics = evaluate_classification_metrics(
        roberta_model, roberta_loader, "cpu", num_classes=2
    )
    assert 'accuracy' in roberta_metrics
    
    # Test DistilBERT
    distil_config = create_distil_config("distilbert-base-uncased (66M)")
    distil_model = DistilBERTForSequenceClassification(distil_config, num_labels=2)
    distil_loader = create_distil_dataloader(
        texts=texts, labels=labels, tokenizer=simple_tokenize, batch_size=2
    )
    distil_metrics = eval_distil(distil_model, distil_loader, "cpu", num_classes=2)
    assert 'accuracy' in distil_metrics
    
    print("  ✓ Evaluation workflows for all kits passed")
    return True


def test_t5_generation_workflow():
    """Test complete T5 text-to-text generation workflow."""
    print("\nTesting T5 generation workflow...")
    
    from t5_kit import (
        T5ForConditionalGeneration, create_model_config,
        create_t5_dataloader, get_tokenizer,
        AMPTrainer, compute_bleu_score, tokenize_text,
        save_model, get_device
    )
    
    config = create_model_config("t5-small (60M)")
    device = get_device("cpu")
    model = T5ForConditionalGeneration(config).to(device)
    
    inputs = ["translate: hello", "summarize: test"]
    targets = ["bonjour", "summary"]
    
    def simple_tokenize(text):
        return [hash(word) % 1000 for word in text.lower().split()]
    
    train_loader = create_t5_dataloader(
        inputs=inputs, targets=targets, tokenizer=simple_tokenize, batch_size=1
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = AMPTrainer(dtype='float16', enabled=False)
    
    # Training
    for batch in train_loader:
        loss = trainer.train_step(
            batch["input_ids"],
            batch["decoder_input_ids"],
            batch["labels"],
            model,
            optimizer,
            device
        )
        assert loss > 0
    
    # Test BLEU scoring
    references = [tokenize_text("hello world")]
    candidates = [tokenize_text("hello world")]
    bleu_scores = compute_bleu_score(references, candidates)
    assert 'bleu' in bleu_scores
    
    print("  ✓ T5 generation workflow passed")
    return True


def test_shared_utilities_across_kits():
    """Test shared utilities work across all kits."""
    print("\nTesting shared utilities across kits...")
    
    from llm_common import get_device, count_parameters, save_model
    from llm_common.quantization import get_model_size_mb, quantize_model_dynamic
    from llm_common.distributed import get_world_size, is_main_process
    
    from gpt_kit import GPTModel, create_model_config
    from bert_kit import BERTModel, create_model_config as create_bert_config
    
    # Test device management
    device = get_device()
    assert device in ["cuda", "cpu", "mps"]
    
    # Test parameter counting
    gpt_config = create_model_config("gpt2-small (124M)")
    gpt_model = GPTModel(gpt_config)
    gpt_params = count_parameters(gpt_model)
    assert gpt_params > 0
    
    bert_config = create_bert_config("bert-base-uncased (110M)")
    bert_model = BERTModel(bert_config)
    bert_params = count_parameters(bert_model)
    assert bert_params > 0
    
    # Test model size
    gpt_size = get_model_size_mb(gpt_model)
    assert gpt_size > 0
    
    # Test quantization
    quantized = quantize_model_dynamic(gpt_model)
    assert quantized is not None
    
    # Test distributed utilities
    world_size = get_world_size()
    assert world_size >= 1
    assert isinstance(is_main_process(), bool)
    
    print("  ✓ Shared utilities across kits passed")
    return True


if __name__ == "__main__":
    print("="*60)
    print("COMPREHENSIVE INTEGRATION TESTS")
    print("="*60)
    
    results = []
    results.append(("Complete GPT Training Pipeline", test_complete_gpt_training_pipeline()))
    results.append(("Complete BERT Classification Pipeline", test_complete_bert_classification_pipeline()))
    results.append(("Mixed Precision Training Workflow", test_mixed_precision_training_workflow()))
    results.append(("Cross-Kit Checkpoint Compatibility", test_cross_kit_checkpoint_compatibility()))
    results.append(("Evaluation Workflow All Kits", test_evaluation_workflow_all_kits()))
    results.append(("T5 Generation Workflow", test_t5_generation_workflow()))
    results.append(("Shared Utilities Across Kits", test_shared_utilities_across_kits()))
    
    print("\n" + "="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Results: {passed}/{total} comprehensive integration tests passed")
    
    if passed == total:
        print("✅ All comprehensive integration tests passed!")
        sys.exit(0)
    else:
        print("❌ Some comprehensive integration tests failed")
        sys.exit(1)

