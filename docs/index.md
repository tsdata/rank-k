# ranx-k Documentation

Welcome to ranx-k - Korean-optimized ranx IR evaluation toolkit!

## Quick Navigation

- [Installation Guide](installation.md)
- [Quick Start](quickstart.md) 
- [API Reference](api-reference.md)

## About ranx-k

ranx-k is a specialized toolkit for evaluating Korean RAG (Retrieval-Augmented Generation) systems with:

- **Korean-optimized tokenizer** using Kiwi morphological analyzer
- **Multiple evaluation methods**: ROUGE, embedding similarity, ranx metrics
- **Comprehensive examples** and documentation
- **Production-ready** with full test coverage

## Features

### 🔤 Korean Tokenization
```python
from ranx_k.tokenizers import KiwiTokenizer

tokenizer = KiwiTokenizer(method='morphs')
tokens = tokenizer.tokenize('한국어 자연어처리 도구입니다.')
print(tokens)  # ['한국어', '자연어', '처리', '도구']
```

### 📊 Evaluation Methods
```python
from ranx_k.evaluation import simple_kiwi_rouge_evaluation

results = simple_kiwi_rouge_evaluation(
    retriever=your_retriever,
    questions=questions,
    reference_contexts=references,
    k=5
)
```

## Getting Started

1. **Install**: `pip install ranx-k`
2. **Follow**: [Quick Start Guide](quickstart.md)
3. **Explore**: [Examples](https://github.com/tsdata/rank-k/tree/main/examples)

## Links

- **GitHub**: https://github.com/tsdata/rank-k
- **PyPI**: https://pypi.org/project/ranx-k/
- **Issues**: https://github.com/tsdata/rank-k/issues