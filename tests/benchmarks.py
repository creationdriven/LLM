"""
Performance benchmarks for LLM kits.

Measures training speed, inference speed, and memory usage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import psutil
import os
from typing import Dict, Any


def measure_training_speed(model, data_loader, device, num_batches: int = 10) -> Dict[str, float]:
    """
    Measure training speed.
    
    Args:
        model: Model to train
        data_loader: DataLoader
        device: Device to run on
        num_batches: Number of batches to measure
        
    Returns:
        Dictionary with timing information
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    times = []
    tokens_processed = 0
    
    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        
        if isinstance(batch, (list, tuple)):
            input_batch, target_batch = batch[0], batch[1]
        elif isinstance(batch, dict):
            input_batch = batch.get("input_ids", batch.get("input"))
            target_batch = batch.get("labels", batch.get("target"))
        else:
            continue
        
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        batch_size, seq_len = input_batch.shape[:2]
        tokens_processed += batch_size * seq_len
        
        start_time = time.time()
        optimizer.zero_grad()
        
        if hasattr(model, '__call__'):
            if isinstance(batch, dict) and "attention_mask" in batch:
                logits = model(input_batch, attention_mask=batch["attention_mask"].to(device))
            else:
                logits = model(input_batch)
        else:
            logits = model(input_batch)
        
        if isinstance(logits, dict):
            loss = logits.get("loss")
            if loss is None:
                import torch.nn.functional as F
                loss = F.cross_entropy(logits["logits"].view(-1, logits["logits"].size(-1)), 
                                      target_batch.view(-1), ignore_index=-100)
        else:
            import torch.nn.functional as F
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                  target_batch.view(-1), ignore_index=-100)
        
        loss.backward()
        optimizer.step()
        
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    avg_time = sum(times) / len(times) if times else 0
    tokens_per_sec = tokens_processed / sum(times) if sum(times) > 0 else 0
    
    return {
        'avg_time_per_batch': avg_time,
        'tokens_per_second': tokens_per_sec,
        'total_tokens': tokens_processed,
        'total_time': sum(times)
    }


def measure_inference_speed(model, input_shape: tuple, device, num_iterations: int = 100) -> Dict[str, float]:
    """
    Measure inference speed.
    
    Args:
        model: Model to test
        input_shape: Input shape (batch_size, seq_len)
        device: Device to run on (str or torch.device)
        num_iterations: Number of iterations
        
    Returns:
        Dictionary with timing information
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    # Measure
    device_str = str(device) if isinstance(device, torch.device) else device
    torch.cuda.synchronize() if device_str.startswith('cuda') else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device_str.startswith('cuda') else None
    elapsed = time.time() - start_time
    
    avg_time = elapsed / num_iterations
    samples_per_sec = num_iterations / elapsed
    
    return {
        'avg_time_per_sample': avg_time,
        'samples_per_second': samples_per_sec,
        'total_time': elapsed
    }


def measure_memory_usage(model, device) -> Dict[str, float]:
    """
    Measure memory usage.
    
    Args:
        model: Model to measure
        device: Device to run on (str or torch.device)
        
    Returns:
        Dictionary with memory information
    """
    process = psutil.Process(os.getpid())
    
    # CPU memory
    cpu_memory_mb = process.memory_info().rss / (1024 ** 2)
    
    # GPU memory
    gpu_memory_mb = 0
    device_str = str(device) if isinstance(device, torch.device) else device
    if device_str.startswith('cuda') and torch.cuda.is_available():
        device_idx = int(device_str.split(':')[1]) if ':' in device_str else 0
        gpu_memory_mb = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
    
    # Model size
    from llm_common.quantization import get_model_size_mb
    model_size_mb = get_model_size_mb(model)
    
    return {
        'cpu_memory_mb': cpu_memory_mb,
        'gpu_memory_mb': gpu_memory_mb,
        'model_size_mb': model_size_mb
    }


def benchmark_gpt():
    """Benchmark GPT model."""
    print("\n" + "="*60)
    print("GPT Benchmark")
    print("="*60)
    
    from gpt_kit import GPTModel, create_model_config, create_pretraining_dataloader, get_device
    
    config = create_model_config("gpt2-small (124M)")
    device = get_device("cpu")
    model = GPTModel(config).to(device)
    
    # Memory
    memory = measure_memory_usage(model, device)
    print(f"\nMemory Usage:")
    print(f"  Model Size: {memory['model_size_mb']:.2f} MB")
    print(f"  CPU Memory: {memory['cpu_memory_mb']:.2f} MB")
    
    # Inference speed
    inference = measure_inference_speed(model, (1, 128), device, num_iterations=50)
    print(f"\nInference Speed:")
    print(f"  Avg time per sample: {inference['avg_time_per_sample']*1000:.2f} ms")
    print(f"  Samples per second: {inference['samples_per_second']:.2f}")
    
    # Training speed
    text = "The quick brown fox jumps over the lazy dog. " * 20
    train_loader = create_pretraining_dataloader(
        text=text, batch_size=4, max_length=128, stride=64
    )
    training = measure_training_speed(model, train_loader, device, num_batches=5)
    print(f"\nTraining Speed:")
    print(f"  Tokens per second: {training['tokens_per_second']:.2f}")
    print(f"  Avg time per batch: {training['avg_time_per_batch']:.4f} s")


def benchmark_bert():
    """Benchmark BERT model."""
    print("\n" + "="*60)
    print("BERT Benchmark")
    print("="*60)
    
    from bert_kit import BERTModel, create_model_config, get_device
    
    config = create_model_config("bert-base-uncased (110M)")
    device = get_device("cpu")
    model = BERTModel(config).to(device)
    
    # Memory
    memory = measure_memory_usage(model, device)
    print(f"\nMemory Usage:")
    print(f"  Model Size: {memory['model_size_mb']:.2f} MB")
    print(f"  CPU Memory: {memory['cpu_memory_mb']:.2f} MB")
    
    # Inference speed
    inference = measure_inference_speed(model, (1, 128), device, num_iterations=50)
    print(f"\nInference Speed:")
    print(f"  Avg time per sample: {inference['avg_time_per_sample']*1000:.2f} ms")
    print(f"  Samples per second: {inference['samples_per_second']:.2f}")


def benchmark_distilbert():
    """Benchmark DistilBERT model."""
    print("\n" + "="*60)
    print("DistilBERT Benchmark")
    print("="*60)
    
    from distilbert_kit import DistilBERTModel, create_model_config, get_device
    
    config = create_model_config("distilbert-base-uncased (66M)")
    device = get_device("cpu")
    model = DistilBERTModel(config).to(device)
    
    # Memory
    memory = measure_memory_usage(model, device)
    print(f"\nMemory Usage:")
    print(f"  Model Size: {memory['model_size_mb']:.2f} MB")
    print(f"  CPU Memory: {memory['cpu_memory_mb']:.2f} MB")
    
    # Inference speed
    inference = measure_inference_speed(model, (1, 128), device, num_iterations=50)
    print(f"\nInference Speed:")
    print(f"  Avg time per sample: {inference['avg_time_per_sample']*1000:.2f} ms")
    print(f"  Samples per second: {inference['samples_per_second']:.2f}")


if __name__ == "__main__":
    print("="*60)
    print("PERFORMANCE BENCHMARKS")
    print("="*60)
    
    benchmark_gpt()
    benchmark_bert()
    benchmark_distilbert()
    
    print("\n" + "="*60)
    print("✅ Benchmarks complete!")
    print("="*60)

