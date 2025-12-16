#!/usr/bin/env python3
"""
Example: Text generation with GPT model.

This script demonstrates:
- Loading a trained GPT model
- Text generation with different sampling strategies
- Temperature and top-k sampling
- Generating from prompts
"""

import torch
from gpt_kit import (
    GPTModel,
    create_model_config,
    generate_text_autoregressive,
    text_to_token_ids,
    token_ids_to_text,
    load_model,
    get_device,
)
import tiktoken

# Configuration
MODEL_NAME = "gpt2-small (124M)"
DEVICE = get_device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "examples/gpt_model.pt"  # Optional: load from checkpoint

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")


def generate_from_prompt(prompt: str, max_tokens: int = 50, temperature: float = 0.8):
    """Generate text from a prompt."""
    print(f"\nPrompt: '{prompt}'")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    print("-" * 60)
    
    # Convert prompt to tokens
    input_ids = text_to_token_ids(prompt, tokenizer)
    
    # Generate
    generated_ids = generate_text_autoregressive(
        model=model,
        token_ids=input_ids.to(DEVICE),
        max_new_tokens=max_tokens,
        context_size=512,
        temperature=temperature,
        top_k=50
    )
    
    # Decode
    generated_text = token_ids_to_text(generated_ids, tokenizer)
    print(f"Generated: {generated_text}")
    print("-" * 60)
    
    return generated_text


def main():
    print("="*60)
    print("GPT Text Generation Example")
    print("="*60)
    
    # Load or create model
    import os
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nLoading model from {CHECKPOINT_PATH}...")
        model, config, _ = load_model(CHECKPOINT_PATH, GPTModel, device=DEVICE)
        print("✓ Model loaded from checkpoint")
    else:
        print(f"\nCreating new model (no checkpoint found at {CHECKPOINT_PATH})...")
        config = create_model_config(MODEL_NAME)
        model = GPTModel(config).to(DEVICE)
        print("✓ Model created (untrained - will generate random text)")
    
    model.eval()
    
    # Different generation examples
    print("\n" + "="*60)
    print("Generation Examples")
    print("="*60)
    
    # Example 1: Creative writing
    generate_from_prompt(
        "Once upon a time, in a distant galaxy",
        max_tokens=100,
        temperature=0.9  # Higher temperature = more creative
    )
    
    # Example 2: Technical explanation
    generate_from_prompt(
        "Machine learning is",
        max_tokens=80,
        temperature=0.7  # Lower temperature = more focused
    )
    
    # Example 3: Question answering style
    generate_from_prompt(
        "The capital of France is",
        max_tokens=50,
        temperature=0.5  # Very low temperature = more deterministic
    )
    
    # Example 4: Story continuation
    generate_from_prompt(
        "The scientist discovered that",
        max_tokens=100,
        temperature=0.8
    )
    
    print("\n" + "="*60)
    print("✅ Text generation examples complete!")
    print("="*60)
    print("\nNote: For better results, train the model first or load a pre-trained checkpoint.")


if __name__ == "__main__":
    main()

