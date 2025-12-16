"""
Tests for all LLM kits.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_gpt_kit():
    """Test GPT Kit imports and basic functionality."""
    try:
        from gpt_kit import GPTModel, create_model_config, get_device
        
        config = create_model_config("gpt2-small (124M)")
        device = get_device("cpu")
        model = GPTModel(config).to(device)
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, config["vocab_size"])
        print("✓ GPT Kit test passed")
        return True
    except Exception as e:
        print(f"✗ GPT Kit test failed: {e}")
        return False


def test_bert_kit():
    """Test BERT Kit imports and basic functionality."""
    try:
        from bert_kit import BERTModel, create_model_config, get_device
        
        config = create_model_config("bert-base-uncased (110M)")
        device = get_device("cpu")
        model = BERTModel(config).to(device)
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        hidden_states = model(input_ids)
        
        assert hidden_states.shape == (batch_size, seq_len, config["hidden_size"])
        print("✓ BERT Kit test passed")
        return True
    except Exception as e:
        print(f"✗ BERT Kit test failed: {e}")
        return False


def test_roberta_kit():
    """Test RoBERTa Kit imports and basic functionality."""
    try:
        from roberta_kit import RoBERTaModel, create_model_config, get_device
        
        config = create_model_config("roberta-base (125M)")
        device = get_device("cpu")
        model = RoBERTaModel(config).to(device)
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        hidden_states = model(input_ids)
        
        assert hidden_states.shape == (batch_size, seq_len, config["hidden_size"])
        print("✓ RoBERTa Kit test passed")
        return True
    except Exception as e:
        print(f"✗ RoBERTa Kit test failed: {e}")
        return False


def test_distilbert_kit():
    """Test DistilBERT Kit imports and basic functionality."""
    try:
        from distilbert_kit import DistilBERTModel, create_model_config, get_device
        
        config = create_model_config("distilbert-base-uncased (66M)")
        device = get_device("cpu")
        model = DistilBERTModel(config).to(device)
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        hidden_states = model(input_ids)
        
        assert hidden_states.shape == (batch_size, seq_len, config["hidden_size"])
        print("✓ DistilBERT Kit test passed")
        return True
    except Exception as e:
        print(f"✗ DistilBERT Kit test failed: {e}")
        return False


def test_t5_kit():
    """Test T5 Kit imports and basic functionality."""
    try:
        from t5_kit import T5Model, create_model_config, get_device
        
        config = create_model_config("t5-small (60M)")
        device = get_device("cpu")
        model = T5Model(config).to(device)
        
        # Test forward pass - use valid token IDs within vocab_size
        batch_size, seq_len = 2, 10
        vocab_size = config["vocab_size"]
        # Use token IDs in valid range (1 to vocab_size-1 to avoid padding)
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        decoder_input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        
        assert "encoder_last_hidden_state" in outputs
        assert "decoder_last_hidden_state" in outputs
        assert outputs["encoder_last_hidden_state"].shape[0] == batch_size
        assert outputs["decoder_last_hidden_state"].shape[0] == batch_size
        print("✓ T5 Kit test passed")
        return True
    except Exception as e:
        print(f"✗ T5 Kit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running LLM kits tests...\n")
    
    results = []
    results.append(("GPT Kit", test_gpt_kit()))
    results.append(("BERT Kit", test_bert_kit()))
    results.append(("RoBERTa Kit", test_roberta_kit()))
    results.append(("DistilBERT Kit", test_distilbert_kit()))
    results.append(("T5 Kit", test_t5_kit()))
    
    print("\n" + "="*50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All kit tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)

