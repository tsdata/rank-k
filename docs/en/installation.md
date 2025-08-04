# Installation Guide

## Language / 언어
[English](installation.md) | [한국어](../ko/installation.md)

## Install from PyPI

```bash
pip install ranx-k
```

## Install Development Version

To install the latest development version from GitHub:

```bash
pip install git+https://github.com/tsdata/rank-k.git
```

## Development Environment Setup

To contribute to the project or develop locally:

```bash
# Clone repository
git clone https://github.com/tsdata/rank-k.git
cd rank-k

# Create virtual environment (using uv)
uv venv ranx-k-dev
source ranx-k-dev/bin/activate  # Linux/Mac
# ranx-k-dev\Scripts\activate  # Windows

# Install in development mode
uv pip install -e ".[dev]"
```

## Verify Installation

Check if installation was successful:

```python
from ranx_k.tokenizers import KiwiTokenizer

tokenizer = KiwiTokenizer()
print("✅ ranx-k installation successful!")
```

## Dependencies

### Required Dependencies
- Python ≥ 3.8
- kiwipiepy ≥ 0.15.0
- rouge-score ≥ 0.1.2
- sentence-transformers ≥ 2.2.0
- scikit-learn ≥ 1.0.0
- numpy ≥ 1.21.0
- tqdm ≥ 4.62.0
- ranx ≥ 0.3.0

### Development Dependencies
- pytest ≥ 7.0.0
- pytest-cov ≥ 4.0.0
- black ≥ 22.0.0
- isort ≥ 5.0.0
- flake8 ≥ 5.0.0
- mypy ≥ 1.0.0

## Troubleshooting

### Kiwi Installation Error
```bash
# On macOS with Kiwi installation issues
brew install cmake
pip install kiwipiepy
```

### M1 Mac Compatibility
For Apple Silicon (M1/M2) Macs:

```bash
# Native installation without Rosetta
arch -arm64 pip install ranx-k
```

### Memory Issues
If you encounter memory issues with large embedding models:

```python
# Use lightweight models
from ranx_k.evaluation import evaluate_with_ranx_similarity

results = evaluate_with_ranx_similarity(
    # ... other parameters
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Lightweight model
)
```

## Navigation
- [← Home](../index.md) | [Quick Start →](quickstart.md) | [API Reference](api-reference.md)