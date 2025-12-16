"""
Data loading module for LLM training and evaluation.

Contains datasets and data loading utilities for:
- GPT pretraining
- Instruction fine-tuning
"""

import torch
import tiktoken
from typing import Dict, List, Tuple, Optional, Callable
from torch.utils.data import Dataset, DataLoader

from .config import PAD_TOKEN_ID, IGNORE_INDEX


class GPTPretrainingDataset(Dataset):
    """
    Dataset for GPT pretraining using sliding window approach.
    
    This dataset creates training examples by sliding a window over the input text.
    Each example consists of a sequence of tokens, and the target is the same
    sequence shifted by one position (next-token prediction).
    
    Sliding Window Strategy:
    - Window size: max_length
    - Stride: How much to move the window (default: max_length for no overlap)
    - Overlap: stride < max_length creates overlapping windows
    
    Example:
        Text: "The cat sat on the mat"
        max_length=5, stride=3
        
        Window 1: "The cat sat on the" → Target: "cat sat on the mat"
        Window 2: "sat on the mat"     → Target: "on the mat [EOS]"
        (with stride=3, windows overlap)
    
    This approach allows the model to see different contexts and learn to
    predict tokens in various positions, improving generalization.
    
    Args:
        text: Raw text string to tokenize
        tokenizer: Tiktoken tokenizer instance (GPT-2 tokenizer)
        max_length: Maximum sequence length (context window size)
        stride: Stride for sliding window
                - max_length: No overlap (non-overlapping windows)
                - < max_length: Overlapping windows (more training data)
                - Smaller stride = more examples but more overlap
    """
    def __init__(
        self, 
        text: str, 
        tokenizer: tiktoken.Encoding, 
        max_length: int, 
        stride: int
    ):
        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []

        # Tokenize the input text into token IDs
        # allowed_special allows the end-of-text token to be included
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Create sliding window sequences
        # Slide window from start to end of token sequence
        # Stop when remaining tokens < max_length (can't form full window)
        for i in range(0, len(token_ids) - max_length, stride):
            # Input chunk: tokens from position i to i+max_length
            input_chunk = token_ids[i:i + max_length]
            
            # Target chunk: same sequence shifted by 1 position
            # This creates next-token prediction targets
            # Position j in target corresponds to position j+1 in input
            target_chunk = token_ids[i + 1: i + max_length + 1]
            
            # Convert to tensors for efficient batching
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_pretraining_dataloader(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for GPT pretraining.
    
    Args:
        text: Raw text string
        batch_size: Batch size
        max_length: Maximum sequence length
        stride: Stride for sliding window
        shuffle: Whether to shuffle the dataset
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader instance
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTPretrainingDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning.
    
    Args:
        data: List of instruction data entries (dicts with 'instruction', 'input', 'output')
        tokenizer: Tiktoken tokenizer instance
        format_instruction_fn: Function to format instruction entries
    """
    def __init__(
        self, 
        data: List[Dict[str, str]], 
        tokenizer: tiktoken.Encoding, 
        format_instruction_fn: Callable[[Dict[str, str]], str]
    ):
        self.data = data
        self.format_instruction = format_instruction_fn
        self.tokenizer = tokenizer
        self.encoded_texts: List[List[int]] = []
        
        # Pre-encode all texts
        for entry in data:
            instruction_text = self.format_instruction(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_text + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index: int) -> List[int]:
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.data)


def create_instruction_collate_fn(
    pad_token_id: int = PAD_TOKEN_ID,
    ignore_index: int = IGNORE_INDEX,
    max_length: Optional[int] = None,
    device: str = "cpu"
) -> Callable:
    """
    Create a custom collate function for instruction fine-tuning.
    
    Args:
        pad_token_id: Token ID to use for padding
        ignore_index: Index to use for padding in targets (ignored in loss)
        max_length: Optional maximum length to truncate sequences
        device: Device to move tensors to
        
    Returns:
        Collate function that can be used with DataLoader
    """
    def collate_fn(batch: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function that pads sequences and creates input/target pairs.
        
        Args:
            batch: List of token ID sequences
            
        Returns:
            Tuple of (input_tensor, target_tensor) both of shape (batch_size, seq_len)
        """
        # Find maximum length in batch
        batch_max_length = max(len(item) + 1 for item in batch)
        if max_length is not None:
            batch_max_length = min(batch_max_length, max_length)
        
        inputs_list: List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []
        
        for item in batch:
            # Add EOS token and pad to batch_max_length
            sequence = item.copy()
            sequence.append(pad_token_id)
            
            # Pad sequence
            padding_length = batch_max_length - len(sequence)
            padded_sequence = sequence + [pad_token_id] * padding_length
            
            # Truncate if necessary
            if max_length is not None and len(padded_sequence) > max_length:
                padded_sequence = padded_sequence[:max_length]
            
            # Create input (all tokens except last) and target (all tokens except first)
            inputs = torch.tensor(padded_sequence[:-1], dtype=torch.long)
            targets = torch.tensor(padded_sequence[1:], dtype=torch.long)
            
            # Replace padding tokens in targets with ignore_index (except first padding token)
            padding_mask = targets == pad_token_id
            padding_indices = torch.nonzero(padding_mask).squeeze()
            if padding_indices.numel() > 0:
                # Keep first padding token, ignore the rest
                if padding_indices.numel() > 1:
                    targets[padding_indices[1:]] = ignore_index
            
            inputs_list.append(inputs)
            targets_list.append(targets)
        
        # Stack into batches
        inputs_tensor = torch.stack(inputs_list).to(device)
        targets_tensor = torch.stack(targets_list).to(device)
        
        return inputs_tensor, targets_tensor
    
    return collate_fn

