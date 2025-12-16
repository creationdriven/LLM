"""
Data loading module for BERT training and evaluation.

Contains datasets for:
- Masked Language Modeling (MLM) pretraining
- Sequence classification
- Question Answering
"""

import torch
from typing import Dict, List, Tuple, Optional, Callable
from torch.utils.data import Dataset, DataLoader
import random
import logging

from .config import MASK_TOKEN_ID, CLS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID, IGNORE_INDEX, VOCAB_SIZE

logger = logging.getLogger(__name__)


class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling (MLM) pretraining.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer function that converts text to token IDs
        max_length: Maximum sequence length
        mask_probability: Probability of masking a token
        vocab_size: Vocabulary size (for random token replacement)
        seed: Random seed for reproducibility
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer: Callable[[str], List[int]],
        max_length: int = 512,
        mask_probability: float = 0.15,
        vocab_size: int = VOCAB_SIZE,
        seed: Optional[int] = None
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability
        self.vocab_size = vocab_size
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example with masked tokens for MLM training.
        
        This method implements BERT's masking strategy:
        1. Tokenize the text
        2. Add [CLS] and [SEP] tokens
        3. Randomly mask 15% of tokens (excluding special tokens)
        4. For masked tokens: 80% → [MASK], 10% → random, 10% → unchanged
        5. Create labels: original token IDs for masked positions, -100 for others
        
        The -100 in labels tells the loss function to ignore those positions
        (only compute loss for masked tokens).
        
        Returns:
            Dictionary with:
                - input_ids: Token IDs with some tokens masked/replaced
                - labels: Original token IDs (-100 for non-masked tokens)
                - attention_mask: Attention mask (1 for real tokens, 0 for padding)
        """
        text = self.texts[idx]
        # Convert text to token IDs using the tokenizer
        token_ids = self.tokenizer(text)
        
        # Truncate if sequence is too long
        # Reserve 2 positions for [CLS] and [SEP] tokens
        if len(token_ids) > self.max_length - 2:  # -2 for [CLS] and [SEP]
            token_ids = token_ids[:self.max_length - 2]
        
        # Add special tokens: [CLS] at start, [SEP] at end
        # [CLS] is used for classification tasks
        # [SEP] marks the end of the sequence
        token_ids = [CLS_TOKEN_ID] + token_ids + [SEP_TOKEN_ID]
        
        # Initialize labels with -100 (ignore_index)
        # Only masked positions will have actual token IDs
        # This tells the loss function to ignore non-masked tokens
        labels = [-100] * len(token_ids)
        
        # Create a copy for masking (we'll modify this)
        masked_token_ids = token_ids.copy()
        
        # Mask tokens (skip [CLS] at position 0 and [SEP] at last position)
        for i in range(1, len(token_ids) - 1):  # Don't mask [CLS] or [SEP]
            # Randomly decide if this token should be masked (15% probability)
            if random.random() < self.mask_probability:
                # Store the original token ID in labels
                # This is what the model should predict
                labels[i] = token_ids[i]
                
                # BERT's masking strategy (prevents overfitting to [MASK] token):
                # - 80%: Replace with [MASK] token (model learns to predict from context)
                # - 10%: Replace with random token (model learns robustness)
                # - 10%: Keep original (model learns to use context, not just [MASK])
                rand = random.random()
                if rand < 0.8:
                    # Most common: replace with [MASK]
                    masked_token_ids[i] = MASK_TOKEN_ID
                elif rand < 0.9:
                    # Sometimes: replace with random token
                    masked_token_ids[i] = random.randint(0, self.vocab_size - 1)
                # else: keep original (10% of masked tokens)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(masked_token_ids)
        
        # Pad sequences to max_length for batching
        # All sequences in a batch must have the same length
        while len(masked_token_ids) < self.max_length:
            masked_token_ids.append(PAD_TOKEN_ID)  # Add padding token
            labels.append(-100)  # Padding positions are ignored in loss
            attention_mask.append(0)  # Don't attend to padding
        
        return {
            "input_ids": torch.tensor(masked_token_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class ClassificationDataset(Dataset):
    """
    Dataset for sequence classification tasks.
    
    Args:
        texts: List of text strings
        labels: List of label integers
        tokenizer: Tokenizer function
        max_length: Maximum sequence length
    """
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
        """
        Get a single classification example.
        
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        token_ids = self.tokenizer(text)
        
        # Truncate if necessary
        if len(token_ids) > self.max_length - 2:
            token_ids = token_ids[:self.max_length - 2]
        
        # Add [CLS] and [SEP] tokens
        token_ids = [CLS_TOKEN_ID] + token_ids + [SEP_TOKEN_ID]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad to max_length
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
    num_workers: int = 0,
    vocab_size: int = VOCAB_SIZE,
    seed: Optional[int] = None
) -> DataLoader:
    """
    Create a DataLoader for MLM pretraining.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer function
        batch_size: Batch size
        max_length: Maximum sequence length
        mask_probability: Probability of masking tokens
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes
        vocab_size: Vocabulary size
        seed: Random seed for reproducibility
        
    Returns:
        DataLoader instance
    """
    dataset = MLMDataset(texts, tokenizer, max_length, mask_probability, vocab_size, seed)
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
    Create a DataLoader for sequence classification.
    
    Args:
        texts: List of text strings
        labels: List of label integers
        tokenizer: Tokenizer function
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the dataset
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

