"""
Evaluation metrics for T5 text-to-text models.

Contains functions for computing BLEU and ROUGE scores for text generation tasks.
"""

import torch
import logging
from typing import List, Dict, Optional, Union
import re

logger = logging.getLogger(__name__)

# Try to import nltk for BLEU
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    logger.warning(
        "nltk not found. BLEU scores will use simple implementation. "
        "Install with: pip install nltk"
    )

# Try to import rouge-score
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    logger.warning(
        "rouge-score not found. ROUGE scores will not be available. "
        "Install with: pip install rouge-score"
    )


def simple_bleu(reference: List[str], candidate: List[str]) -> float:
    """
    Simple BLEU score implementation (without nltk).
    
    Args:
        reference: Reference sentence tokens
        candidate: Candidate sentence tokens
        
    Returns:
        BLEU score (0-1)
    """
    if len(candidate) == 0:
        return 0.0
    
    # Unigram precision
    reference_counts = {}
    for token in reference:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    
    matches = 0
    for token in candidate:
        if token in reference_counts and reference_counts[token] > 0:
            matches += 1
            reference_counts[token] -= 1
    
    precision = matches / len(candidate) if len(candidate) > 0 else 0.0
    
    # Brevity penalty
    if len(candidate) > len(reference):
        brevity_penalty = len(reference) / len(candidate)
    else:
        brevity_penalty = 1.0
    
    return precision * brevity_penalty


def compute_bleu_score(
    references: List[List[str]],
    candidates: List[List[str]],
    smoothing: bool = True
) -> Dict[str, float]:
    """
    Compute BLEU score for text generation.
    
    Args:
        references: List of reference token sequences
        candidates: List of candidate token sequences
        smoothing: Whether to use smoothing (requires nltk)
        
    Returns:
        Dictionary with 'bleu' score and 'bleu_1' through 'bleu_4' if nltk available
        
    Example:
        >>> refs = [["the", "cat", "sat", "on", "the", "mat"]]
        >>> cands = [["the", "cat", "sat", "on", "the", "mat"]]
        >>> scores = compute_bleu_score(refs, cands)
        >>> print(f"BLEU: {scores['bleu']:.4f}")
    """
    if len(references) != len(candidates):
        raise ValueError("references and candidates must have the same length")
    
    if HAS_NLTK:
        smoothing_fn = SmoothingFunction().method1 if smoothing else None
        bleu_scores = []
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        
        for ref, cand in zip(references, candidates):
            # Full BLEU
            score = sentence_bleu([ref], cand, smoothing_function=smoothing_fn)
            bleu_scores.append(score)
            
            # Individual n-gram BLEU
            from nltk.translate.bleu_score import sentence_bleu
            weights_1 = (1.0, 0.0, 0.0, 0.0)
            weights_2 = (0.5, 0.5, 0.0, 0.0)
            weights_3 = (0.33, 0.33, 0.33, 0.0)
            weights_4 = (0.25, 0.25, 0.25, 0.25)
            
            bleu_1_scores.append(sentence_bleu([ref], cand, weights=weights_1, smoothing_function=smoothing_fn))
            bleu_2_scores.append(sentence_bleu([ref], cand, weights=weights_2, smoothing_function=smoothing_fn))
            bleu_3_scores.append(sentence_bleu([ref], cand, weights=weights_3, smoothing_function=smoothing_fn))
            bleu_4_scores.append(sentence_bleu([ref], cand, weights=weights_4, smoothing_function=smoothing_fn))
        
        return {
            'bleu': sum(bleu_scores) / len(bleu_scores),
            'bleu_1': sum(bleu_1_scores) / len(bleu_1_scores),
            'bleu_2': sum(bleu_2_scores) / len(bleu_2_scores),
            'bleu_3': sum(bleu_3_scores) / len(bleu_3_scores),
            'bleu_4': sum(bleu_4_scores) / len(bleu_4_scores),
        }
    else:
        # Fallback to simple BLEU
        scores = [simple_bleu(ref, cand) for ref, cand in zip(references, candidates)]
        return {
            'bleu': sum(scores) / len(scores),
            'bleu_1': sum(scores) / len(scores),  # Same as simple BLEU
        }


def compute_rouge_score(
    references: List[str],
    candidates: List[str],
    rouge_types: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute ROUGE scores for text generation.
    
    Args:
        references: List of reference text strings
        candidates: List of candidate text strings
        rouge_types: List of ROUGE types to compute (default: ['rouge1', 'rouge2', 'rougeL'])
        
    Returns:
        Dictionary with ROUGE scores
        
    Example:
        >>> refs = ["the cat sat on the mat"]
        >>> cands = ["the cat sat on the mat"]
        >>> scores = compute_rouge_score(refs, cands)
        >>> print(f"ROUGE-L: {scores['rougeL']:.4f}")
    """
    if not HAS_ROUGE:
        logger.warning("rouge-score not available. Install with: pip install rouge-score")
        return {}
    
    if len(references) != len(candidates):
        raise ValueError("references and candidates must have the same length")
    
    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    scores = {rouge_type: [] for rouge_type in rouge_types}
    
    for ref, cand in zip(references, candidates):
        score = scorer.score(ref, cand)
        for rouge_type in rouge_types:
            scores[rouge_type].append(score[rouge_type].fmeasure)
    
    # Average scores
    result = {}
    for rouge_type in rouge_types:
        result[rouge_type] = sum(scores[rouge_type]) / len(scores[rouge_type])
    
    return result


def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenization for BLEU/ROUGE computation.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    # Simple word tokenization
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def evaluate_t5_generation(
    model,
    data_loader,
    device: str,
    tokenizer,
    max_length: int = 128,
    num_beams: int = 1,
    references: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate T5 model generation with BLEU and ROUGE scores.
    
    Args:
        model: T5ForConditionalGeneration model
        data_loader: DataLoader with input-output pairs
        device: Device to run on
        tokenizer: Tokenizer function
        max_length: Maximum generation length
        num_beams: Number of beams for beam search (1 = greedy)
        references: Optional list of reference texts (if not in data_loader)
        
    Returns:
        Dictionary with BLEU and ROUGE scores
    """
    model.eval()
    all_references = []
    all_candidates = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            
            # Generate
            # Note: This is a simplified generation - full implementation would use
            # proper decoding strategies (greedy, beam search, etc.)
            outputs = model.model(
                input_ids=input_ids,
                decoder_input_ids=torch.zeros(input_ids.shape[0], 1, dtype=torch.long).to(device)
            )
            
            # For now, return placeholder - full generation would be implemented here
            # This is a framework for evaluation
    
    if references:
        all_references = [tokenize_text(ref) for ref in references]
    
    if len(all_references) > 0 and len(all_candidates) > 0:
        bleu_scores = compute_bleu_score(all_references, all_candidates)
        rouge_scores = compute_rouge_score(
            [" ".join(ref) for ref in all_references],
            [" ".join(cand) for cand in all_candidates]
        )
        
        return {**bleu_scores, **rouge_scores}
    
    return {}

