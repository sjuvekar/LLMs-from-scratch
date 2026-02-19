# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
# Or with uv (recommended for faster installs):
uv sync --dev
```

### Run tests
```bash
# Run all tests for the llms_from_scratch package
pytest pkg/llms_from_scratch/tests/

# Run tests for a specific chapter
pytest pkg/llms_from_scratch/tests/test_ch04.py

# Run tests in a chapter directory
pytest ch04/01_main-chapter-code/tests.py

# Validate Jupyter notebooks with nbval
pytest --nbval ch02/01_main-chapter-code/dataloader.ipynb
```

### Linting
```bash
# Run ruff linter (configured in pyproject.toml)
ruff check .
```

## Project Structure

This is the official code repository for "Build a Large Language Model (From Scratch)" by Sebastian Raschka. The code implements a GPT-like LLM step by step.

### Chapter Organization
- `ch02/` - Text data processing (tokenization, dataloaders)
- `ch03/` - Attention mechanisms (self-attention, multi-head attention)
- `ch04/` - GPT model implementation (TransformerBlock, GPTModel)
- `ch05/` - Pretraining on unlabeled data (training loop, loss computation)
- `ch06/` - Finetuning for text classification
- `ch07/` - Instruction finetuning
- `appendix-A/` through `appendix-E/` - Supplementary material (PyTorch intro, training enhancements, LoRA)

Each chapter has a `01_main-chapter-code/` subdirectory with:
- `<chapter>.ipynb` - Main chapter notebook
- `exercise-solutions.ipynb` - Exercise solutions
- Optional standalone `.py` files for key code (e.g., `gpt.py`, `gpt_train.py`)

### Reusable Package
`pkg/llms_from_scratch/` contains the chapter code refactored into importable Python modules:
- `ch02.py`, `ch03.py`, `ch04.py`, `ch05.py`, `ch06.py`, `ch07.py` - Chapter implementations
- `llama3.py`, `qwen3.py` - Alternative LLM architectures
- `kv_cache/` - KV cache implementations for efficient inference
- `tests/` - Unit tests for all modules

## Key Architecture Patterns

### GPT Model Configuration
Models use a configuration dictionary pattern:
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

### Chapter Dependencies
Code builds cumulatively - later chapters import from earlier ones:
- `ch04` imports attention from `ch03`
- `ch05` imports the GPT model from `ch04`
- The package maintains these imports (e.g., `ch04.py` imports from `ch03.py`)

### Tokenization
Uses tiktoken with GPT-2 encoding throughout:
```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
```

## Notes

- Python 3.10-3.13 supported
- PyTorch 2.2.2+ required
- TensorFlow is used for loading pretrained weights only
- Code is designed to run on laptops without specialized hardware
- Bonus materials in chapter subdirectories (e.g., `ch05/07_gpt_to_llama/`) contain optional advanced content
