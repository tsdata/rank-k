# ranx-k: Korean-optimized ranx IR Evaluation Toolkit 🇰🇷

[![PyPI version](https://badge.fury.io/py/ranx-k.svg)](https://badge.fury.io/py/ranx-k)
[![Python version](https://img.shields.io/pypi/pyversions/ranx-k.svg)](https://pypi.org/project/ranx-k/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[English](README.md) | [한국어](README.ko.md)**

**ranx-k** is a Korean-optimized Information Retrieval (IR) evaluation toolkit that extends the ranx library with Kiwi tokenizer and Korean embeddings. It provides accurate evaluation for RAG (Retrieval-Augmented Generation) systems.

## 🚀 Key Features

- **Korean-optimized**: Accurate tokenization using Kiwi morphological analyzer
- **ranx-based**: Supports proven IR evaluation metrics (Hit@K, NDCG@K, MRR, MAP@K, etc.)
- **LangChain compatible**: Supports LangChain retriever interface standards
- **Multiple evaluation methods**: ROUGE, embedding similarity, semantic similarity-based evaluation
- **Graded relevance support**: Use similarity scores as relevance grades for NDCG calculation
- **Configurable ROUGE types**: Choose between ROUGE-1, ROUGE-2, and ROUGE-L
- **Strict threshold enforcement**: Documents below similarity threshold are correctly treated as retrieval failures
- **Retrieval order preservation**: Accurate evaluation of reranking systems (v0.0.16+)
- **Practical design**: Supports step-by-step evaluation from prototype to production
- **High performance**: 30-80% improvement in Korean evaluation accuracy over existing methods
- **Bilingual output**: English-Korean output support for international accessibility

## 📦 Installation

```bash
pip install ranx-k
```

Or install development version:

```bash
pip install "ranx-k[dev]"
```

## 🔗 Retriever Compatibility

ranx-k supports **LangChain retriever interface**:

```python
# Retriever must implement invoke() method
class YourRetriever:
    def invoke(self, query: str) -> List[Document]:
        # Return list of Document objects (requires page_content attribute)
        pass

# LangChain Document usage example
from langchain.schema import Document
doc = Document(page_content="Text content")
```

> **Note**: LangChain is distributed under the MIT License. See [documentation](docs/en/quickstart.md#langchain-license) for details.

## 🔧 Quick Start

### Basic Usage

```python
from ranx_k.evaluation import simple_kiwi_rouge_evaluation

# Simple Kiwi ROUGE evaluation
results = simple_kiwi_rouge_evaluation(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5
)

print(f"ROUGE-1: {results['kiwi_rouge1@5']:.3f}")
print(f"ROUGE-2: {results['kiwi_rouge2@5']:.3f}")
print(f"ROUGE-L: {results['kiwi_rougeL@5']:.3f}")
```

### Enhanced Evaluation (Rouge Score + Kiwi)

```python
from ranx_k.evaluation import rouge_kiwi_enhanced_evaluation

# Proven rouge_score library + Kiwi tokenizer
results = rouge_kiwi_enhanced_evaluation(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5,
    tokenize_method='morphs',  # 'morphs' or 'nouns'
    use_stopwords=True
)
```

### Semantic Similarity-based ranx Evaluation

```python
from ranx_k.evaluation import evaluate_with_ranx_similarity

# Reference-based evaluation (recommended for accurate recall)
results = evaluate_with_ranx_similarity(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5,
    method='embedding',
    similarity_threshold=0.6,
    use_graded_relevance=False,        # Binary relevance (default)
    evaluation_mode='reference_based'  # Evaluates against all reference docs
)

print(f"Hit@5: {results['hit_rate@5']:.3f}")
print(f"NDCG@5: {results['ndcg@5']:.3f}")
print(f"MRR: {results['mrr']:.3f}")
print(f"MAP@5: {results['map@5']:.3f}")
```

#### Using Different Embedding Models

```python
# OpenAI embedding model (requires API key)
results = evaluate_with_ranx_similarity(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5,
    method='openai',
    similarity_threshold=0.7,
    embedding_model="text-embedding-3-small"
)

# Latest BGE-M3 model (excellent for Korean)
results = evaluate_with_ranx_similarity(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5,
    method='embedding',
    similarity_threshold=0.6,
    embedding_model="BAAI/bge-m3"
)

# Korean-specialized Kiwi ROUGE method with configurable ROUGE types
results = evaluate_with_ranx_similarity(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5,
    method='kiwi_rouge',
    similarity_threshold=0.3,  # Lower threshold recommended for Kiwi ROUGE
    rouge_type='rougeL',      # Choose 'rouge1', 'rouge2', or 'rougeL'
    tokenize_method='morphs', # Choose 'morphs' or 'nouns'  
    use_stopwords=True        # Configure stopword filtering
)
```

### Comprehensive Evaluation

```python
from ranx_k.evaluation import comprehensive_evaluation_comparison

# Compare all evaluation methods
comparison = comprehensive_evaluation_comparison(
    retriever=your_retriever,
    questions=your_questions,
    reference_contexts=your_reference_contexts,
    k=5
)
```

## 📊 Evaluation Methods

### 1. Kiwi ROUGE Evaluation
- **Advantages**: Fast speed, intuitive interpretation
- **Use case**: Prototyping, quick feedback

### 2. Enhanced ROUGE (Rouge Score + Kiwi)
- **Advantages**: Proven library, stability
- **Use case**: Production environment, reliability-critical evaluation

### 3. Semantic Similarity-based ranx
- **Advantages**: Traditional IR metrics, semantic similarity
- **Use case**: Research, benchmarking, detailed analysis

## 🎯 Performance Improvement Examples

```python
# Existing method (English tokenizer)
basic_rouge1 = 0.234

# ranx-k (Kiwi tokenizer)
ranxk_rouge1 = 0.421  # +79.9% improvement!
```

## 📊 Recommended Embedding Models

| Model | Use Case | Threshold | Features |
|-------|----------|-----------|----------|
| `paraphrase-multilingual-MiniLM-L12-v2` | Default | 0.6 | Fast, lightweight |
| `text-embedding-3-small` (OpenAI) | Accuracy | 0.7 | High accuracy, cost-effective |
| `BAAI/bge-m3` | Korean | 0.6 | Latest, excellent multilingual |
| `text-embedding-3-large` (OpenAI) | Premium | 0.8 | Highest performance |

## 📈 Score Interpretation Guide

| Score Range | Assessment | Recommended Action |
|-------------|------------|-------------------|
| 0.7+ | 🟢 Excellent | Maintain current settings |
| 0.5~0.7 | 🟡 Good | Consider fine-tuning |
| 0.3~0.5 | 🟠 Average | Improvement needed |
| 0.3- | 🔴 Poor | Major revision required |

## 🔍 Advanced Usage

### Graded Relevance Mode

```python
# Graded relevance mode - uses similarity scores as relevance grades
results = evaluate_with_ranx_similarity(
    retriever=your_retriever,
    questions=questions,
    reference_contexts=references,
    method='embedding',
    similarity_threshold=0.6,
    use_graded_relevance=True   # Uses similarity scores as relevance grades
)

print(f"NDCG@5: {results['ndcg@5']:.3f}")
```

> **Note on Graded Relevance**: The `use_graded_relevance` parameter primarily affects NDCG (Normalized Discounted Cumulative Gain) calculation. Other metrics like Hit@K, MRR, and MAP treat relevance as binary in the ranx library. Use graded relevance when you need to distinguish between different levels of document relevance quality.

### Custom Embedding Models

```python
# Use custom embedding model
results = evaluate_with_ranx_similarity(
    retriever=your_retriever,
    questions=questions,
    reference_contexts=references,
    method='embedding',
    embedding_model="your-custom-model-name",
    similarity_threshold=0.6,
    use_graded_relevance=True
)
```

### Configurable ROUGE Types

```python
# Compare different ROUGE metrics
for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
    results = evaluate_with_ranx_similarity(
        retriever=your_retriever,
        questions=questions,
        reference_contexts=references,
        method='kiwi_rouge',
        rouge_type=rouge_type,
        tokenize_method='morphs',
        similarity_threshold=0.3
    )
    print(f"{rouge_type.upper()}: Hit@5 = {results['hit_rate@5']:.3f}")
```

### Threshold Sensitivity Analysis

```python
# Analyze how different thresholds affect evaluation
thresholds = [0.3, 0.5, 0.7]
for threshold in thresholds:
    results = evaluate_with_ranx_similarity(
        retriever=your_retriever,
        questions=questions,
        reference_contexts=references,
        similarity_threshold=threshold
    )
    print(f"Threshold {threshold}: Hit@5={results['hit_rate@5']:.3f}, NDCG@5={results['ndcg@5']:.3f}")
```

## 📚 Examples

- [Basic Tokenizer Example](examples/basic_tokenizer.py)
- [BGE-M3 Evaluation Example](examples/bge_m3_evaluation.py)
- [Embedding Models Comparison](examples/embedding_models_comparison.py)
- [Comprehensive Comparison](examples/comprehensive_comparison.py)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [ranx](https://github.com/AmenRa/ranx) by Elias Bassani
- Korean morphological analysis powered by [Kiwi](https://github.com/bab2min/kiwipiepy)
- Embedding support via [sentence-transformers](https://github.com/UKPLab/sentence-transformers)

## 📞 Support

- 🐛 Issue Tracker: Please submit issues on GitHub
- 📧 Email: ontofinance@gmail.com

---

**ranx-k** - Empowering Korean RAG evaluation with precision and ease!