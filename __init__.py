"""
ranx-k: Korean-optimized ranx IR Evaluation Toolkit

A comprehensive library for evaluating Korean RAG (Retrieval-Augmented Generation) 
systems with Kiwi tokenizer integration and semantic similarity-based evaluation.

Copyright (c) 2025 Pandas Studio
Licensed under MIT License

This library incorporates and builds upon:
- rouge_score (Apache License 2.0)
- ranx (MIT License)
- kiwipiepy (LGPL v3.0)
"""

__version__ = "0.0.1"
__author__ = "Pandas Studio"
__email__ = "ontofinance@gmail.com"
__license__ = "MIT"

# Import main evaluation functions
from ranx_k.evaluation import (
    simple_kiwi_rouge_evaluation,
    rouge_kiwi_enhanced_evaluation,
    evaluate_with_ranx_similarity,
    comprehensive_evaluation_comparison,
)

# Import tokenizers
from ranx_k.tokenizers import KiwiTokenizer

__all__ = [
    # Main evaluation functions
    "simple_kiwi_rouge_evaluation",
    "rouge_kiwi_enhanced_evaluation", 
    "evaluate_with_ranx_similarity",
    "comprehensive_evaluation_comparison",
    # Tokenizers
    "KiwiTokenizer",
    # Version info
    "__version__",
    "__author__",
    "__license__",
]
