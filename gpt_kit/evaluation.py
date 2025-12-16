"""
Evaluation module for LLM responses.

Contains functions for evaluating model responses using Ollama/Llama3 as an evaluator.
"""

import json
import urllib.request
import psutil
import logging
from typing import Dict, List
from tqdm import tqdm

from .config import DEFAULT_OLLAMA_URL, DEFAULT_OLLAMA_MODEL
from .utils import format_alpaca_instruction

logger = logging.getLogger(__name__)


def check_ollama_running() -> bool:
    """
    Check if Ollama process is running.
    
    Returns:
        True if Ollama is running, False otherwise
    """
    for proc in psutil.process_iter(["name"]):
        try:
            if "ollama" in proc.info["name"].lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def query_ollama_model(
    prompt: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 30
) -> str:
    """
    Query Ollama model via REST API.
    
    Args:
        prompt: Input prompt string
        model: Model name to use
        url: Ollama API URL
        timeout: Request timeout in seconds
        
    Returns:
        Model response string
        
    Raises:
        urllib.error.URLError: If request fails
        json.JSONDecodeError: If response is not valid JSON
    """
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }
    
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    
    response_text = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                line = response.readline().decode("utf-8")
                if not line:
                    break
                response_json = json.loads(line)
                response_text += response_json.get("message", {}).get("content", "")
    except urllib.error.URLError as e:
        logger.error(f"Failed to query Ollama: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Ollama response: {e}")
        raise
    
    return response_text


def evaluate_model_responses(
    json_data: List[Dict[str, str]],
    response_key: str = "model_response",
    model: str = DEFAULT_OLLAMA_MODEL,
    url: str = DEFAULT_OLLAMA_URL
) -> List[int]:
    """
    Evaluate model responses using Ollama/Llama3 as an evaluator.
    
    Args:
        json_data: List of dictionaries with 'instruction', 'input', 'output', and response_key
        response_key: Key in json_data entries containing the model response to evaluate
        model: Ollama model to use for evaluation
        url: Ollama API URL
        
    Returns:
        List of scores (0-100) for each entry
        
    Raises:
        RuntimeError: If Ollama is not running
        ValueError: If response_key is missing from entries
    """
    if not check_ollama_running():
        raise RuntimeError(
            "Ollama is not running. Please start Ollama before evaluation."
        )
    
    scores: List[int] = []
    failed_evaluations = 0
    
    for entry in tqdm(json_data, desc="Evaluating responses"):
        # Validate entry
        if response_key not in entry:
            logger.warning(f"Entry missing '{response_key}' key, skipping")
            failed_evaluations += 1
            continue
        
        if "instruction" not in entry or "output" not in entry:
            logger.warning("Entry missing required keys ('instruction' or 'output'), skipping")
            failed_evaluations += 1
            continue
        
        # Create evaluation prompt
        instruction_text = format_alpaca_instruction(entry)
        prompt = (
            f"Given the input `{instruction_text}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[response_key]}` "
            f"on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        
        try:
            score_text = query_ollama_model(prompt, model, url)
            score = int(score_text.strip())
            
            # Validate score range
            if not (0 <= score <= 100):
                logger.warning(f"Score {score} out of range [0, 100], skipping")
                failed_evaluations += 1
                continue
            
            scores.append(score)
        except (ValueError, urllib.error.URLError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to evaluate entry: {e}")
            failed_evaluations += 1
            continue
    
    if failed_evaluations > 0:
        logger.warning(f"Failed to evaluate {failed_evaluations} entries")
    
    return scores

