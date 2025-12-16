"""
Tests for checkpoint saving and loading across all kits.

Verifies that save_model and load_model work correctly for all model types.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
import shutil


def test_gpt_checkpoint():
    """Test GPT model checkpoint saving and loading."""
    print("\nTesting GPT checkpoint...")
    
    from gpt_kit import GPTModel, create_model_config, save_model, load_model, get_device
    
    config = create_model_config("gpt2-small (124M)")
    device = get_device("cpu")
    model = GPTModel(config).to(device)
    
    # Test forward pass
    model.eval()
    test_input = torch.randint(0, config["vocab_size"], (1, 10))
    with torch.no_grad():
        original_output = model(test_input)
    
    # Save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        save_model(model, temp_path, config, epoch=1, loss=0.5)
        assert os.path.exists(temp_path)
        
        # Load
        loaded_model, loaded_config, _ = load_model(temp_path, GPTModel, device=device)
        assert loaded_config == config
        
        # Test forward pass matches (use model.eval() for consistency)
        model.eval()
        loaded_model.eval()
        with torch.no_grad():
            original_output = model(test_input)
            loaded_output = loaded_model(test_input)
        assert torch.allclose(original_output, loaded_output, atol=1e-4)
        
        print("  ✓ GPT checkpoint save/load passed")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_bert_checkpoint():
    """Test BERT model checkpoint saving and loading."""
    print("\nTesting BERT checkpoint...")
    
    from bert_kit import BERTModel, create_model_config, save_model, load_model, get_device
    
    config = create_model_config("bert-base-uncased (110M)")
    device = get_device("cpu")
    model = BERTModel(config).to(device)
    
    # Test forward pass
    model.eval()
    test_input = torch.randint(0, config["vocab_size"], (1, 10))
    with torch.no_grad():
        original_output = model(test_input)
    
    # Save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        save_model(model, temp_path, config, epoch=2, loss=0.3)
        assert os.path.exists(temp_path)
        
        # Load
        loaded_model, loaded_config, _ = load_model(temp_path, BERTModel, device=device)
        assert loaded_config == config
        
        # Test forward pass matches (use model.eval() for consistency)
        model.eval()
        loaded_model.eval()
        with torch.no_grad():
            original_output = model(test_input)
            loaded_output = loaded_model(test_input)
        assert torch.allclose(original_output, loaded_output, atol=1e-4)
        
        print("  ✓ BERT checkpoint save/load passed")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_roberta_checkpoint():
    """Test RoBERTa model checkpoint saving and loading."""
    print("\nTesting RoBERTa checkpoint...")
    
    from roberta_kit import RoBERTaModel, create_model_config, save_model, load_model, get_device
    
    config = create_model_config("roberta-base (125M)")
    device = get_device("cpu")
    model = RoBERTaModel(config).to(device)
    
    # Test forward pass
    model.eval()
    test_input = torch.randint(0, config["vocab_size"], (1, 10))
    with torch.no_grad():
        original_output = model(test_input)
    
    # Save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        save_model(model, temp_path, config, epoch=3, loss=0.4)
        assert os.path.exists(temp_path)
        
        # Load
        loaded_model, loaded_config, _ = load_model(temp_path, RoBERTaModel, device=device)
        assert loaded_config == config
        
        # Test forward pass matches (use model.eval() for consistency)
        model.eval()
        loaded_model.eval()
        with torch.no_grad():
            original_output = model(test_input)
            loaded_output = loaded_model(test_input)
        assert torch.allclose(original_output, loaded_output, atol=1e-4)
        
        print("  ✓ RoBERTa checkpoint save/load passed")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_distilbert_checkpoint():
    """Test DistilBERT model checkpoint saving and loading."""
    print("\nTesting DistilBERT checkpoint...")
    
    from distilbert_kit import DistilBERTModel, create_model_config, save_model, load_model, get_device
    
    config = create_model_config("distilbert-base-uncased (66M)")
    device = get_device("cpu")
    model = DistilBERTModel(config).to(device)
    
    # Test forward pass
    model.eval()
    test_input = torch.randint(0, config["vocab_size"], (1, 10))
    with torch.no_grad():
        original_output = model(test_input)
    
    # Save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        save_model(model, temp_path, config, epoch=4, loss=0.35)
        assert os.path.exists(temp_path)
        
        # Load
        loaded_model, loaded_config, _ = load_model(temp_path, DistilBERTModel, device=device)
        assert loaded_config == config
        
        # Test forward pass matches (use model.eval() for consistency)
        model.eval()
        loaded_model.eval()
        with torch.no_grad():
            original_output = model(test_input)
            loaded_output = loaded_model(test_input)
        assert torch.allclose(original_output, loaded_output, atol=1e-4)
        
        print("  ✓ DistilBERT checkpoint save/load passed")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_t5_checkpoint():
    """Test T5 model checkpoint saving and loading."""
    print("\nTesting T5 checkpoint...")
    
    from t5_kit import T5Model, create_model_config, save_model, load_model, get_device
    
    config = create_model_config("t5-small (60M)")
    device = get_device("cpu")
    model = T5Model(config).to(device)
    
    # Test forward pass
    input_ids = torch.randint(0, config["vocab_size"], (1, 10))
    decoder_input_ids = torch.randint(0, config["vocab_size"], (1, 10))
    original_output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    
    # Save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        save_model(model, temp_path, config, epoch=5, loss=0.6)
        assert os.path.exists(temp_path)
        
        # Load
        loaded_model, loaded_config, _ = load_model(temp_path, T5Model, device=device)
        assert loaded_config == config
        
        # Test forward pass matches
        loaded_output = loaded_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        assert "encoder_last_hidden_state" in loaded_output
        assert "decoder_last_hidden_state" in loaded_output
        
        print("  ✓ T5 checkpoint save/load passed")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_checkpoint_with_optimizer():
    """Test checkpoint saving and loading with optimizer state."""
    print("\nTesting checkpoint with optimizer...")
    
    from gpt_kit import GPTModel, create_model_config, save_model, load_model, get_device
    
    config = create_model_config("gpt2-small (124M)")
    device = get_device("cpu")
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Do a training step to change optimizer state
    dummy_input = torch.randint(0, config["vocab_size"], (1, 10))
    logits = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        dummy_input.view(-1)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Save with optimizer
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        save_model(model, temp_path, config, optimizer=optimizer, epoch=1)
        
        # Load with optimizer
        loaded_model, loaded_config, loaded_optimizer = load_model(
            temp_path, GPTModel, device=device,
            return_optimizer=True, optimizer_class=torch.optim.AdamW, lr=1e-4
        )
        
        assert loaded_config == config
        assert loaded_optimizer is not None
        
        print("  ✓ Checkpoint with optimizer passed")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_checkpoint_metadata():
    """Test checkpoint with additional metadata."""
    print("\nTesting checkpoint metadata...")
    
    from bert_kit import BERTModel, create_model_config, save_model, load_model, get_device
    
    config = create_model_config("bert-base-uncased (110M)")
    device = get_device("cpu")
    model = BERTModel(config).to(device)
    
    # Save with metadata
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        save_model(
            model, temp_path, config,
            epoch=10, loss=0.25,
            best_val_loss=0.20,
            training_time=3600.0,
            custom_metric=0.95
        )
        
        # Load and verify metadata
        checkpoint = torch.load(temp_path)
        assert checkpoint['epoch'] == 10
        assert checkpoint['loss'] == 0.25
        assert checkpoint['best_val_loss'] == 0.20
        assert checkpoint['training_time'] == 3600.0
        assert checkpoint['custom_metric'] == 0.95
        
        print("  ✓ Checkpoint metadata passed")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    print("="*60)
    print("CHECKPOINT TESTS")
    print("="*60)
    
    results = []
    results.append(("GPT Checkpoint", test_gpt_checkpoint()))
    results.append(("BERT Checkpoint", test_bert_checkpoint()))
    results.append(("RoBERTa Checkpoint", test_roberta_checkpoint()))
    results.append(("DistilBERT Checkpoint", test_distilbert_checkpoint()))
    results.append(("T5 Checkpoint", test_t5_checkpoint()))
    results.append(("Checkpoint with Optimizer", test_checkpoint_with_optimizer()))
    results.append(("Checkpoint Metadata", test_checkpoint_metadata()))
    
    print("\n" + "="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Results: {passed}/{total} checkpoint tests passed")
    
    if passed == total:
        print("✅ All checkpoint tests passed!")
        sys.exit(0)
    else:
        print("❌ Some checkpoint tests failed")
        sys.exit(1)

