"""
Data loading module for RoBERTa training.

Key differences from BERT:
- Dynamic masking (masking pattern changes each epoch)
- No token type IDs (single sentence format)
"""

import torch
from typing import Dict, List, Optional, Callable
from torch.utils.data import Dataset, DataLoader
import random
import logging

from .config import MASK_TOKEN_ID, CLS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID, IGNORE_INDEX, VOCAB_SIZE

logger = logging.getLogger(__name__)


class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling with dynamic masking.
    
    Unlike BERT's static masking, RoBERTa uses dynamic masking where
    the masking pattern changes each time an example is accessed.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer function
        max_length: Maximum sequence length
        mask_probability: Probability of masking a token
        vocab_size: Vocabulary size
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer: Callable[[str], List[int]],
        max_length: int = 512,
        mask_probability: float = 0.15,
        vocab_size: int = VOCAB_SIZE
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example with dynamically masked tokens.
        
        Note: Masking pattern changes each time this is called (dynamic masking).
        """
        text = self.texts[idx]
        token_ids = self.tokenizer(text)
        
        # Truncate if necessary
        if len(token_ids) > self.max_length - 2:
            token_ids = token_ids[:self.max_length - 2]
        
        # Add [CLS] and [SEP] tokens
        token_ids = [CLS_TOKEN_ID] + token_ids + [SEP_TOKEN_ID]
        
        # Create labels
        labels = [-100] * len(token_ids)
        
        # Dynamic masking (pattern changes each time)
        masked_token_ids = token_ids.copy()
        for i in range(1, len(token_ids) - 1):
            if random.random() < self.mask_probability:
                labels[i] = token_ids[i]
                
                # 80% [MASK], 10% random, 10% unchanged
                rand = random.random()
                if rand < 0.8:
                    masked_token_ids[i] = MASK_TOKEN_ID
                elif rand < 0.9:
                    masked_token_ids[i] = random.randint(0, self.vocab_size - 1)
        
        # Pad
        attention_mask = [1] * len(masked_token_ids)
        while len(masked_token_ids) < self.max_length:
            masked_token_ids.append(PAD_TOKEN_ID)
            labels.append(-100)
            attention_mask.append(0)
        
        return {
            "input_ids": torch.tensor(masked_token_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class ClassificationDataset(Dataset):
    """Dataset for sequence classification (no token type IDs)."""
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Callable[[str], List[int]],
        max_length: int = 512
    ):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        token_ids = self.tokenizer(text)
        
        if len(token_ids) > self.max_length - 2:
            token_ids = token_ids[:self.max_length - 2]
        
        token_ids = [CLS_TOKEN_ID] + token_ids + [SEP_TOKEN_ID]
        
        attention_mask = [1] * len(token_ids)
        while len(token_ids) < self.max_length:
            token_ids.append(PAD_TOKEN_ID)
            attention_mask.append(0)
        
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_mlm_dataloader(
    texts: List[str],
    tokenizer: Callable[[str], List[int]],
    batch_size: int = 8,
    max_length: int = 512,
    mask_probability: float = 0.15,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for MLM pretraining.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer function
        batch_size: Batch size
        max_length: Maximum sequence length
        mask_probability: Probability of masking tokens
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    dataset = MLMDataset(texts, tokenizer, max_length, mask_probability)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def create_classification_dataloader(
    texts: List[str],
    labels: List[int],
    tokenizer: Callable[[str], List[int]],
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for classification.
    
    Args:
        texts: List of text strings
        labels: List of label integers
        tokenizer: Tokenizer function
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    dataset = ClassificationDataset(texts, labels, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

