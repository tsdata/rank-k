# API Reference

## Language / 언어
[English](api-reference.md) | [한국어](../ko/api-reference.md)

Complete documentation for all classes and functions in ranx-k.

## Tokenizers (ranx_k.tokenizers)

### KiwiTokenizer

Korean-specialized tokenizer class.

```python
class KiwiTokenizer:
    def __init__(self, use_stemmer=False, method='morphs', use_stopwords=True):
        """
        Parameters:
            use_stemmer (bool): Compatibility parameter for rouge_score (unused)
            method (str): Tokenization method ('morphs' or 'nouns')
            use_stopwords (bool): Whether to filter Korean stopwords
        """
```

#### Methods

**tokenize(text: str) -> List[str]**
- Tokenizes input text.
- Parameters: `text` - Text to tokenize
- Returns: List of tokens

**add_stopwords(stopwords: List[str]) -> None**
- Adds custom stopwords.
- Parameters: `stopwords` - List of stopwords to add

**remove_stopwords(stopwords: List[str]) -> None**
- Removes stopwords.
- Parameters: `stopwords` - List of stopwords to remove

**get_stopwords() -> Set[str]**
- Returns current stopword set.
- Returns: Set of stopwords

## Evaluation Functions (ranx_k.evaluation)

### simple_kiwi_rouge_evaluation

Performs simple ROUGE evaluation using Kiwi tokenizer.

```python
def simple_kiwi_rouge_evaluation(
    retriever, 
    questions: List[str], 
    reference_contexts: List[List[str]], 
    k: int = 5
) -> Dict[str, float]:
    """
    Parameters:
        retriever: Document retriever object (must have invoke method)
        questions: List of questions
        reference_contexts: List of reference document lists
        k: Evaluate top k documents
        
    Returns:
        Evaluation results dictionary {
            'kiwi_rouge1@k': float,
            'kiwi_rouge2@k': float, 
            'kiwi_rougeL@k': float
        }
    """
```

### rouge_kiwi_enhanced_evaluation

Enhanced evaluation combining validated rouge_score library with Kiwi tokenizer.

```python
def rouge_kiwi_enhanced_evaluation(
    retriever,
    questions: List[str],
    reference_contexts: List[List[str]],
    k: int = 5,
    tokenize_method: str = 'morphs',
    use_stopwords: bool = True
) -> Dict[str, float]:
    """
    Parameters:
        retriever: Document retriever object
        questions: List of questions
        reference_contexts: List of reference document lists
        k: Evaluate top k documents
        tokenize_method: Tokenization method ('morphs' or 'nouns')
        use_stopwords: Whether to filter stopwords
        
    Returns:
        Evaluation results dictionary {
            'enhanced_rouge1@k': float,
            'enhanced_rouge2@k': float,
            'enhanced_rougeL@k': float
        }
    """
```

### evaluate_with_ranx_similarity

Evaluates using ranx metrics with semantic similarity conversion.

```python
def evaluate_with_ranx_similarity(
    retriever,
    questions: List[str],
    reference_contexts: List[List[str]],
    k: int = 5,
    method: str = 'embedding',
    similarity_threshold: float = 0.7
) -> Dict[str, float]:
    """
    Parameters:
        retriever: Document retriever object
        questions: List of questions
        reference_contexts: List of reference document lists
        k: Evaluate top k documents
        method: Similarity calculation method ('embedding' or 'kiwi_rouge')
        similarity_threshold: Threshold for relevant document judgment
        
    Returns:
        ranx evaluation results dictionary {
            'hit_rate@k': float,
            'ndcg@k': float,
            'map@k': float,
            'mrr': float
        }
    """
```

### comprehensive_evaluation_comparison

Comprehensive comparison of all evaluation methods.

```python
def comprehensive_evaluation_comparison(
    retriever,
    questions: List[str],
    reference_contexts: List[List[str]],
    k: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Parameters:
        retriever: Document retriever object
        questions: List of questions
        reference_contexts: List of reference document lists
        k: Evaluate top k documents
        
    Returns:
        Method-wise evaluation results dictionary {
            'Kiwi ROUGE': {...},
            'Enhanced ROUGE': {...},
            'Similarity ranx': {...}
        }
    """
```

## OpenAI Integration (Optional)

### evaluate_with_openai_similarity

Evaluates using OpenAI embeddings (requires `openai` package).

```python
def evaluate_with_openai_similarity(
    retriever,
    questions: List[str],
    reference_contexts: List[List[str]],
    k: int = 5,
    model_name: str = "text-embedding-3-small",
    similarity_threshold: float = 0.7,
    api_key: Optional[str] = None
) -> Dict[str, float]:
    """
    Parameters:
        retriever: Document retriever object
        questions: List of questions
        reference_contexts: List of reference document lists
        k: Evaluate top k documents
        model_name: OpenAI embedding model name
        similarity_threshold: Threshold for relevant document judgment
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
        
    Returns:
        ranx evaluation results dictionary
    """
```

## Utility Classes

### EmbeddingSimilarityCalculator

Embedding-based similarity calculator.

```python
class EmbeddingSimilarityCalculator:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Parameters:
            model_name: sentence-transformers model name
        """
    
    def calculate_similarity_matrix(self, ref_texts: List[str], ret_texts: List[str]) -> np.ndarray:
        """
        Calculates embedding similarity matrix.
        
        Parameters:
            ref_texts: Reference text list
            ret_texts: Retrieved text list
            
        Returns:
            Cosine similarity matrix
        """
```

### KiwiRougeSimilarityCalculator

Kiwi + ROUGE based similarity calculator.

```python
class KiwiRougeSimilarityCalculator:
    def __init__(self):
        """Initialize Kiwi tokenizer and ROUGE metrics."""
    
    def calculate_similarity_matrix(self, ref_texts: List[str], ret_texts: List[str]) -> np.ndarray:
        """
        Calculates Kiwi ROUGE similarity matrix.
        
        Parameters:
            ref_texts: Reference text list
            ret_texts: Retrieved text list
            
        Returns:
            ROUGE-L F1 score matrix
        """
```

## Metric Interpretation

### ROUGE Scores
- **ROUGE-1**: Word-level overlap (0.0-1.0)
- **ROUGE-2**: Bigram-level overlap (0.0-1.0)  
- **ROUGE-L**: Longest common subsequence based (0.0-1.0)

### ranx Metrics
- **Hit@K**: Ratio of relevant documents found in top K (0.0-1.0)
- **NDCG@K**: Normalized Discounted Cumulative Gain (0.0-1.0)
- **MAP@K**: Mean Average Precision (0.0-1.0)
- **MRR**: Mean Reciprocal Rank (0.0-1.0)

## Performance Benchmarks

| Score Range | Rating | Meaning |
|-------------|--------|---------|
| 0.8-1.0 | 🟢 Excellent | Very high accuracy |
| 0.6-0.8 | 🟡 Good | Practical level |
| 0.4-0.6 | 🟠 Fair | Needs improvement |
| 0.0-0.4 | 🔴 Poor | System review required |

## Navigation
- [← Quick Start](quickstart.md) | [Home](../index.md) | [Installation](installation.md)