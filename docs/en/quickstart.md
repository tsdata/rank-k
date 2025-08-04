# Quick Start

## Language / 언어
[English](quickstart.md) | [한국어](../ko/quickstart.md)

This guide introduces the basic usage of ranx-k.

> **Note**: ranx-k provides bilingual output messages (English | Korean) for better international accessibility. See [Bilingual Output System](bilingual-output.md) for details.

## Get Started in 5 Minutes

### 1. Installation

```bash
pip install ranx-k
```

### 2. Basic Tokenizer Usage

```python
from ranx_k.tokenizers import KiwiTokenizer

# Morpheme-based tokenizer
tokenizer = KiwiTokenizer(method='morphs', use_stopwords=True)

text = "Natural language processing is a core technology of artificial intelligence."
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# Output: ['Natural', 'language', 'processing', 'core', 'technology', 'artificial', 'intelligence']
```

### 3. Noun Extraction

```python
# Noun-only tokenizer
noun_tokenizer = KiwiTokenizer(method='nouns')

text = "RAG systems combine retrieval and generation."
nouns = noun_tokenizer.tokenize(text)
print(f"Nouns: {nouns}")
# Output: ['systems', 'retrieval', 'generation']
```

### 4. Custom Stopwords

```python
tokenizer = KiwiTokenizer(use_stopwords=True)

# Add custom stopwords
tokenizer.add_stopwords(['system', 'method'])

# Remove stopwords
tokenizer.remove_stopwords(['technology'])

# Check current stopwords
stopwords = tokenizer.get_stopwords()
print(f"Stopword count: {len(stopwords)}")
```

## Retriever Compatibility

ranx-k is designed to work with **LangChain retriever objects** that implement the `invoke()` method:

```python
# Your retriever must implement:
class YourRetriever:
    def invoke(self, query: str) -> List[Document]:
        # Return list of Document objects with page_content attribute
        pass
```

**LangChain Document Format:**
```python
from langchain.schema import Document

# Documents must have page_content attribute
doc = Document(page_content="Your text content here")
```

> **Note**: ranx-k follows LangChain's retriever interface standards. LangChain is licensed under MIT License.

## Evaluation Functions

### 1. Simple ROUGE Evaluation

```python
from ranx_k.evaluation import simple_kiwi_rouge_evaluation

# Mock retriever and data (replace with your actual implementation)
# retriever = your_retriever
# questions = ["Question 1", "Question 2", ...]
# reference_contexts = [["Answer doc 1", "Answer doc 2"], ...]

# Run evaluation
results = simple_kiwi_rouge_evaluation(
    retriever=retriever,
    questions=questions,
    reference_contexts=reference_contexts,
    k=5
)

print(f"ROUGE-1: {results['kiwi_rouge1@5']:.3f}")
print(f"ROUGE-2: {results['kiwi_rouge2@5']:.3f}")
print(f"ROUGE-L: {results['kiwi_rougeL@5']:.3f}")
```

### 2. Enhanced ROUGE Evaluation

```python
from ranx_k.evaluation import rouge_kiwi_enhanced_evaluation

results = rouge_kiwi_enhanced_evaluation(
    retriever=retriever,
    questions=questions,
    reference_contexts=reference_contexts,
    k=5,
    tokenize_method='morphs',
    use_stopwords=True
)
```

### 3. Semantic Similarity-based ranx Evaluation

```python
from ranx_k.evaluation import evaluate_with_ranx_similarity

results = evaluate_with_ranx_similarity(
    retriever=retriever,
    questions=questions,
    reference_contexts=reference_contexts,
    k=5,
    method='kiwi_rouge',
    similarity_threshold=0.6
)

print(f"Hit@5: {results['hit_rate@5']:.3f}")
print(f"NDCG@5: {results['ndcg@5']:.3f}")
print(f"MRR: {results['mrr']:.3f}")
```

## Real-world Example

### RAG System Evaluation

```python
from ranx_k.tokenizers import KiwiTokenizer
from ranx_k.evaluation import comprehensive_evaluation_comparison

# Initialize tokenizer
tokenizer = KiwiTokenizer(method='morphs')

# Prepare test data
questions = [
    "What is natural language processing?",
    "What are the advantages of RAG systems?",
    "What are the challenges in Korean tokenization?"
]

reference_contexts = [
    ["Natural language processing is a technology that enables computers to understand and process human language."],
    ["RAG combines retrieval and generation to provide more accurate answers."],
    ["Korean is complex for tokenization due to its agglutinative characteristics."]
]

# Run comprehensive evaluation
results = comprehensive_evaluation_comparison(
    retriever=your_retriever,
    questions=questions,
    reference_contexts=reference_contexts,
    k=5
)
```

## Performance Optimization Tips

### 1. Batch Processing
```python
# Process large datasets in batches
batch_size = 100
for i in range(0, len(questions), batch_size):
    batch_questions = questions[i:i+batch_size]
    batch_contexts = reference_contexts[i:i+batch_size]
    # Run evaluation
```

### 2. Lightweight Models
```python
# Use lightweight embedding models to save memory
results = evaluate_with_ranx_similarity(
    # ... other parameters
    method='kiwi_rouge'  # Use ROUGE instead of embeddings
)
```

### 3. Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_tokenize(text):
    return tokenizer.tokenize(text)
```

## Next Steps

- [API Reference](api-reference.md) for complete function documentation
- [GitHub Examples](https://github.com/tsdata/rank-k/tree/main/examples) for real use cases
- [Installation Guide](installation.md) for advanced setup options


## Navigation
- [← Installation](installation.md) | [Home](../index.md) | [API Reference →](api-reference.md)