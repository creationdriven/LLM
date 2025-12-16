"""
Data loading module for T5 training.

T5 uses text-to-text format where inputs and outputs are both text strings.
"""

import torch
from typing import Dict, List, Optional, Callable
from torch.utils.data import Dataset, DataLoader

from .config import PAD_TOKEN_ID, EOS_TOKEN_ID


class T5Dataset(Dataset):
    """
    Dataset for T5 text-to-text training.
    
    Args:
        inputs: List of input text strings
        targets: List of target text strings
        tokenizer: Tokenizer function
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
    """
    def __init__(
        self,
        inputs: List[str],
        targets: List[str],
        tokenizer: Callable[[str], List[int]],
        max_input_length: int = 512,
        max_target_length: int = 512
    ):
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same length")
        
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        
        input_ids = self.tokenizer(input_text)
        target_ids = self.tokenizer(target_text)
        
        # Truncate
        if len(input_ids) > self.max_input_length:
            input_ids = input_ids[:self.max_input_length]
        if len(target_ids) > self.max_target_length - 1:  # -1 for EOS
            target_ids = target_ids[:self.max_target_length - 1]
        
        # Add EOS token to target
        target_ids = target_ids + [EOS_TOKEN_ID]
        
        # Create decoder input (shifted right)
        decoder_input_ids = [EOS_TOKEN_ID] + target_ids[:-1]
        
        # Pad
        input_attention_mask = [1] * len(input_ids)
        decoder_attention_mask = [1] * len(decoder_input_ids)
        
        while len(input_ids) < self.max_input_length:
            input_ids.append(PAD_TOKEN_ID)
            input_attention_mask.append(0)
        
        while len(decoder_input_ids) < self.max_target_length:
            decoder_input_ids.append(PAD_TOKEN_ID)
            decoder_attention_mask.append(0)
            target_ids.append(-100)  # Ignore padding in loss
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long),
            "attention_mask": torch.tensor(input_attention_mask, dtype=torch.long),
            "decoder_attention_mask": torch.tensor(decoder_attention_mask, dtype=torch.long),
        }


def create_t5_dataloader(
    inputs: List[str],
    targets: List[str],
    tokenizer: Callable[[str], List[int]],
    batch_size: int = 8,
    max_input_length: int = 512,
    max_target_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for T5 training."""
    dataset = T5Dataset(inputs, targets, tokenizer, max_input_length, max_target_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

