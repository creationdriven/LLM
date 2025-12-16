# Tests

Unit tests for all LLM kits.

## Running Tests

### Run all tests:
```bash
python tests/test_common.py
python tests/test_kits.py
```

### Run specific test:
```bash
python -m pytest tests/test_common.py -v
python -m pytest tests/test_kits.py -v
```

## Test Coverage

### `test_common.py`
Tests for shared utilities:
- Device detection (`get_device`)
- Parameter counting (`count_parameters`)
- Model saving/loading (`save_model`, `load_model`)

### `test_kits.py`
Tests for all LLM kits:
- GPT Kit: Model initialization and forward pass
- BERT Kit: Model initialization and forward pass
- RoBERTa Kit: Model initialization and forward pass
- DistilBERT Kit: Model initialization and forward pass
- T5 Kit: Model initialization and forward pass

## Adding New Tests

To add new tests:

1. Create a new test file: `tests/test_<feature>.py`
2. Import the module to test
3. Write test functions starting with `test_`
4. Run with: `python tests/test_<feature>.py`

## Requirements

Tests require:
- `torch` (core dependency)
- Optional: `pytest` for advanced test running

