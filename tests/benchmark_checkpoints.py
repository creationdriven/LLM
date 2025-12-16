"""
Performance benchmarks for checkpoint operations.

Measures save/load performance, checkpoint size, and memory usage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import tempfile
from typing import Dict, Any


def benchmark_save_performance(model, config: Dict[str, Any], num_iterations: int = 10) -> Dict[str, float]:
    """
    Benchmark checkpoint saving performance.
    
    Args:
        model: Model to save
        config: Model configuration
        num_iterations: Number of save operations
        
    Returns:
        Dictionary with timing information
    """
    from llm_common.checkpoint import save_model
    
    times = []
    
    for i in range(num_iterations):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            temp_path = f.name
        
        try:
            start_time = time.time()
            save_model(model, temp_path, config)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    
    return {
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'total_time': sum(times)
    }


def benchmark_load_performance(
    checkpoint_path: str,
    model_class,
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark checkpoint loading performance.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Model class to instantiate
        num_iterations: Number of load operations
        
    Returns:
        Dictionary with timing information
    """
    from llm_common.checkpoint import load_model
    
    times = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        model, config, _ = load_model(checkpoint_path, model_class, device="cpu")
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        # Clean up model
        del model
    
    return {
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'total_time': sum(times)
    }


def get_checkpoint_size(checkpoint_path: str) -> float:
    """Get checkpoint file size in MB."""
    if os.path.exists(checkpoint_path):
        return os.path.getsize(checkpoint_path) / (1024 ** 2)
    return 0.0


def benchmark_gpt_checkpoints():
    """Benchmark GPT checkpoint operations."""
    print("\n" + "="*60)
    print("GPT Checkpoint Benchmarks")
    print("="*60)
    
    from gpt_kit import GPTModel, create_model_config, save_model
    
    config = create_model_config("gpt2-small (124M)")
    model = GPTModel(config)
    
    # Save benchmark
    print("\nSaving benchmark...")
    save_results = benchmark_save_performance(model, config, num_iterations=5)
    print(f"  Average save time: {save_results['avg_time']*1000:.2f} ms")
    print(f"  Min save time: {save_results['min_time']*1000:.2f} ms")
    print(f"  Max save time: {save_results['max_time']*1000:.2f} ms")
    
    # Create checkpoint for load benchmark
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        checkpoint_path = f.name
    
    try:
        save_model(model, checkpoint_path, config)
        checkpoint_size = get_checkpoint_size(checkpoint_path)
        print(f"\nCheckpoint size: {checkpoint_size:.2f} MB")
        
        # Load benchmark
        print("\nLoading benchmark...")
        load_results = benchmark_load_performance(checkpoint_path, GPTModel, num_iterations=5)
        print(f"  Average load time: {load_results['avg_time']*1000:.2f} ms")
        print(f"  Min load time: {load_results['min_time']*1000:.2f} ms")
        print(f"  Max load time: {load_results['max_time']*1000:.2f} ms")
        
        # Throughput
        save_throughput = 1.0 / save_results['avg_time'] if save_results['avg_time'] > 0 else 0
        load_throughput = 1.0 / load_results['avg_time'] if load_results['avg_time'] > 0 else 0
        print(f"\nThroughput:")
        print(f"  Save: {save_throughput:.2f} checkpoints/second")
        print(f"  Load: {load_throughput:.2f} checkpoints/second")
        
    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


def benchmark_bert_checkpoints():
    """Benchmark BERT checkpoint operations."""
    print("\n" + "="*60)
    print("BERT Checkpoint Benchmarks")
    print("="*60)
    
    from bert_kit import BERTModel, create_model_config, save_model
    
    config = create_model_config("bert-base-uncased (110M)")
    model = BERTModel(config)
    
    # Save benchmark
    print("\nSaving benchmark...")
    save_results = benchmark_save_performance(model, config, num_iterations=5)
    print(f"  Average save time: {save_results['avg_time']*1000:.2f} ms")
    
    # Create checkpoint for load benchmark
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        checkpoint_path = f.name
    
    try:
        save_model(model, checkpoint_path, config)
        checkpoint_size = get_checkpoint_size(checkpoint_path)
        print(f"\nCheckpoint size: {checkpoint_size:.2f} MB")
        
        # Load benchmark
        print("\nLoading benchmark...")
        load_results = benchmark_load_performance(checkpoint_path, BERTModel, num_iterations=5)
        print(f"  Average load time: {load_results['avg_time']*1000:.2f} ms")
        
    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


def benchmark_checkpoint_with_optimizer():
    """Benchmark checkpoint operations with optimizer state."""
    print("\n" + "="*60)
    print("Checkpoint with Optimizer Benchmarks")
    print("="*60)
    
    from gpt_kit import GPTModel, create_model_config, save_model
    
    config = create_model_config("gpt2-small (124M)")
    model = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Save with optimizer
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        checkpoint_path = f.name
    
    try:
        start_time = time.time()
        save_model(model, checkpoint_path, config, optimizer=optimizer)
        save_time = time.time() - start_time
        
        checkpoint_size = get_checkpoint_size(checkpoint_path)
        print(f"\nCheckpoint size (with optimizer): {checkpoint_size:.2f} MB")
        print(f"Save time: {save_time*1000:.2f} ms")
        
        # Load with optimizer
        from llm_common.checkpoint import load_model
        start_time = time.time()
        loaded_model, loaded_config, loaded_optimizer = load_model(
            checkpoint_path, GPTModel, device="cpu",
            return_optimizer=True, optimizer_class=torch.optim.AdamW, lr=1e-4
        )
        load_time = time.time() - start_time
        
        print(f"Load time (with optimizer): {load_time*1000:.2f} ms")
        print(f"Optimizer loaded: {loaded_optimizer is not None}")
        
    finally:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


def compare_checkpoint_sizes():
    """Compare checkpoint sizes across different model types."""
    print("\n" + "="*60)
    print("Checkpoint Size Comparison")
    print("="*60)
    
    from gpt_kit import GPTModel, create_model_config, save_model
    from bert_kit import BERTModel, create_model_config as create_bert_config, save_model as bert_save
    from distilbert_kit import DistilBERTModel, create_model_config as create_distil_config, save_model as distil_save
    
    models = [
        ("GPT-2 Small", GPTModel, create_model_config("gpt2-small (124M)"), save_model),
        ("BERT Base", BERTModel, create_bert_config("bert-base-uncased (110M)"), bert_save),
        ("DistilBERT", DistilBERTModel, create_distil_config("distilbert-base-uncased (66M)"), distil_save),
    ]
    
    checkpoint_paths = []
    
    try:
        print("\nModel checkpoint sizes:")
        for name, model_class, config, save_func in models:
            model = model_class(config)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
                path = f.name
            checkpoint_paths.append(path)
            
            save_func(model, path, config)
            size = get_checkpoint_size(path)
            print(f"  {name:20s} {size:>8.2f} MB")
        
    finally:
        for path in checkpoint_paths:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    print("="*60)
    print("CHECKPOINT PERFORMANCE BENCHMARKS")
    print("="*60)
    
    benchmark_gpt_checkpoints()
    benchmark_bert_checkpoints()
    benchmark_checkpoint_with_optimizer()
    compare_checkpoint_sizes()
    
    print("\n" + "="*60)
    print("✅ Benchmarks complete!")
    print("="*60)

