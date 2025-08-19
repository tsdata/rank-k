#!/usr/bin/env python3
"""
Realistic test script to verify graded vs binary relevance differences.

This script creates a more realistic scenario with varied similarity scores
to demonstrate the differences between graded and binary relevance.
"""

import numpy as np
from typing import List
from ranx_k.evaluation.similarity_ranx import evaluate_with_ranx_similarity


class MockDocument:
    """Mock document class for testing."""
    def __init__(self, content: str):
        self.page_content = content


class RealisticTestRetriever:
    """Test retriever with more realistic similarity distribution."""
    
    def __init__(self, documents: List[str]):
        self.documents = [MockDocument(doc) for doc in documents]
    
    def invoke(self, query: str) -> List[MockDocument]:
        """Return all documents for any query."""
        return self.documents


def test_realistic_graded_vs_binary():
    """Test with more realistic similarity score distribution."""
    
    print("🧪 Realistic Graded vs Binary Relevance Test | 현실적인 등급별 vs 이진 관련성 테스트")
    print("="*80)
    
    # Create more realistic test scenario
    questions = [
        "What is artificial intelligence and machine learning?",
        "How do neural networks and deep learning work?"
    ]
    
    # Reference documents (ground truth)
    reference_contexts = [
        [
            MockDocument("AI is the simulation of human intelligence in machines that are programmed to think and learn like humans. Machine learning is a subset of AI that uses statistical techniques to give computers the ability to learn from data."),
            MockDocument("Artificial intelligence encompasses various technologies including machine learning, which allows systems to automatically learn and improve from experience without being explicitly programmed.")
        ],
        [
            MockDocument("Neural networks are computing systems inspired by biological neural networks. Deep learning uses multi-layered neural networks to model and understand complex patterns in data."),
            MockDocument("Deep learning networks contain multiple hidden layers that can learn hierarchical representations of data, making them powerful for tasks like image recognition and natural language processing.")
        ]
    ]
    
    # Retrieved documents with varied relevance levels
    retrieved_docs = [
        # High relevance documents (should get high similarity scores)
        "Machine learning is a powerful subset of artificial intelligence that enables computers to learn from data automatically.",
        "Deep learning neural networks use multiple layers to process information and recognize complex patterns in data.",
        
        # Medium relevance documents 
        "Computer science includes various fields like artificial intelligence, databases, and software engineering.",
        "Neural network architectures have evolved significantly with advances in computational power and algorithms.",
        
        # Lower relevance documents
        "Technology companies invest heavily in research and development of new computational methods.",
        "Data science combines statistics, programming, and domain knowledge to extract insights from data.",
        
        # Low relevance documents (should be below threshold)
        "The weather today is sunny with temperatures reaching 75 degrees Fahrenheit.",
        "Programming languages like Python and Java are popular choices for software development.",
        
        # Very low relevance 
        "Cooking recipes often require precise measurements and timing for best results.",
        "Sports teams compete in various leagues and tournaments throughout the season."
    ]
    
    retriever = RealisticTestRetriever(retrieved_docs)
    
    # Test with a threshold that will create varied relevance levels
    threshold = 0.4  # Medium threshold to capture varied relevance levels
    
    # Test 1: Binary relevance
    print("\n1️⃣ Binary Relevance Test | 이진 관련성 테스트")
    print("-" * 40)
    
    binary_results = evaluate_with_ranx_similarity(
        retriever=retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=10,
        method='embedding',
        similarity_threshold=threshold,
        use_graded_relevance=False,
        evaluation_mode='reference_based'
    )
    
    print(f"\n📊 Binary Relevance Results:")
    for metric, score in binary_results.items():
        print(f"  {metric}: {score:.4f}")
    
    # Test 2: Graded relevance
    print(f"\n2️⃣ Graded Relevance Test | 등급별 관련성 테스트")  
    print("-" * 40)
    
    graded_results = evaluate_with_ranx_similarity(
        retriever=retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=10,
        method='embedding',
        similarity_threshold=threshold,
        use_graded_relevance=True,
        evaluation_mode='reference_based'
    )
    
    print(f"\n📊 Graded Relevance Results:")
    for metric, score in graded_results.items():
        print(f"  {metric}: {score:.4f}")
    
    # Test 3: Compare and analyze
    print(f"\n3️⃣ Comparison Analysis | 비교 분석")
    print("-" * 40)
    
    print("Metric Differences (Graded - Binary):")
    for metric in binary_results:
        if metric in graded_results:
            diff = graded_results[metric] - binary_results[metric]
            print(f"  {metric}: {diff:+.4f}")
    
    # Test 4: Test with different thresholds to see sensitivity
    print(f"\n4️⃣ Threshold Sensitivity Analysis | 임계값 민감도 분석")
    print("-" * 50)
    
    thresholds = [0.2, 0.4, 0.6]
    
    print("Threshold | Binary NDCG@10 | Graded NDCG@10 | Difference")
    print("-" * 55)
    
    for test_threshold in thresholds:
        try:
            # Binary
            binary_test = evaluate_with_ranx_similarity(
                retriever=retriever,
                questions=questions[:1],  # Use only first question for speed
                reference_contexts=reference_contexts[:1],
                k=10,
                method='embedding',
                similarity_threshold=test_threshold,
                use_graded_relevance=False,
                evaluation_mode='reference_based'
            )
            
            # Graded
            graded_test = evaluate_with_ranx_similarity(
                retriever=retriever,
                questions=questions[:1],
                reference_contexts=reference_contexts[:1],
                k=10,
                method='embedding',
                similarity_threshold=test_threshold,
                use_graded_relevance=True,
                evaluation_mode='reference_based'
            )
            
            binary_ndcg = binary_test.get('ndcg@10', 0)
            graded_ndcg = graded_test.get('ndcg@10', 0)
            diff = graded_ndcg - binary_ndcg
            
            print(f"   {test_threshold:.1f}   |     {binary_ndcg:.4f}     |     {graded_ndcg:.4f}     |  {diff:+.4f}")
            
        except Exception as e:
            print(f"   {test_threshold:.1f}   | ERROR: {str(e)[:30]}...")
    
    return binary_results, graded_results


if __name__ == "__main__":
    test_realistic_graded_vs_binary()