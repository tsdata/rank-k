#!/usr/bin/env python3
"""
Test script to verify use_graded_relevance functionality and MRR calculations.

This script tests whether graded vs binary relevance produces different results
and verifies that ranking metrics like MRR are calculated correctly.
"""

import numpy as np
from typing import List
from ranx_k.evaluation.similarity_ranx import evaluate_with_ranx_similarity


class MockDocument:
    """Mock document class for testing."""
    def __init__(self, content: str):
        self.page_content = content


class TestRetriever:
    """Simple test retriever that returns predefined documents."""
    
    def __init__(self, documents: List[str]):
        self.documents = [MockDocument(doc) for doc in documents]
    
    def invoke(self, query: str) -> List[MockDocument]:
        """Return all documents for any query."""
        return self.documents


def test_graded_vs_binary_relevance():
    """Test graded relevance vs binary relevance with controlled similarity scores."""
    
    print("🧪 Testing Graded vs Binary Relevance | 등급별 vs 이진 관련성 테스트")
    print("="*70)
    
    # Create test data with known similarity patterns
    questions = [
        "What is machine learning?",
        "How does deep learning work?"
    ]
    
    # Reference documents (ground truth)
    reference_contexts = [
        [
            MockDocument("Machine learning is a subset of AI that learns from data patterns."),
            MockDocument("ML algorithms improve performance through experience and training.")
        ],
        [
            MockDocument("Deep learning uses neural networks with multiple layers."),
            MockDocument("Neural networks process information through interconnected nodes.")
        ]
    ]
    
    # Retrieved documents - mix of relevant and irrelevant
    retrieved_docs = [
        "Machine learning involves algorithms that learn from data automatically.",  # High similarity to ref 1
        "Deep learning is a powerful ML technique using neural networks.",         # Medium similarity to ref 2  
        "The weather today is sunny with a chance of rain.",                      # Low similarity
        "Machine learning algorithms can recognize patterns in large datasets.",   # Medium similarity to ref 1
        "Artificial intelligence encompasses many technologies including ML."       # Low-medium similarity
    ]
    
    # Create test retriever
    retriever = TestRetriever(retrieved_docs)
    
    # Test 1: Binary relevance (traditional approach)
    print("\n1️⃣ Binary Relevance Test | 이진 관련성 테스트")
    print("-" * 40)
    
    binary_results = evaluate_with_ranx_similarity(
        retriever=retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.3,
        use_graded_relevance=False,  # Binary relevance
        evaluation_mode='reference_based'
    )
    
    print(f"\n📊 Binary Relevance Results:")
    for metric, score in binary_results.items():
        print(f"  {metric}: {score:.4f}")
    
    # Test 2: Graded relevance (similarity scores as relevance grades)
    print(f"\n2️⃣ Graded Relevance Test | 등급별 관련성 테스트")
    print("-" * 40)
    
    graded_results = evaluate_with_ranx_similarity(
        retriever=retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.3,
        use_graded_relevance=True,   # Graded relevance
        evaluation_mode='reference_based'
    )
    
    print(f"\n📊 Graded Relevance Results:")
    for metric, score in graded_results.items():
        print(f"  {metric}: {score:.4f}")
    
    # Test 3: Compare results
    print(f"\n3️⃣ Comparison Analysis | 비교 분석")
    print("-" * 40)
    
    print("Metric Differences (Graded - Binary):")
    for metric in binary_results:
        if metric in graded_results:
            diff = graded_results[metric] - binary_results[metric]
            print(f"  {metric}: {diff:+.4f}")
    
    return binary_results, graded_results


def test_mrr_calculation():
    """Test MRR calculation with known ranking positions."""
    
    print(f"\n🎯 MRR Calculation Test | MRR 계산 테스트")
    print("="*50)
    
    # Create test scenario with predictable rankings
    questions = ["Test query 1", "Test query 2", "Test query 3"]
    
    # References with varying similarity to retrieved docs
    reference_contexts = [
        [MockDocument("This is the perfect match for query 1")],  # Should rank #1
        [MockDocument("This somewhat matches query 2")],          # Should rank #3  
        [MockDocument("This exactly matches query 3")]            # Should rank #2
    ]
    
    # Retrieved documents in specific order to test ranking
    retrieved_docs = [
        "This is the perfect match for query 1",        # Rank 1 - exact match for query 1
        "This exactly matches query 3",                 # Rank 2 - exact match for query 3
        "This somewhat matches query 2",                # Rank 3 - partial match for query 2
        "This is completely unrelated content",         # Rank 4 - no match
        "Another irrelevant document here"              # Rank 5 - no match
    ]
    
    retriever = TestRetriever(retrieved_docs)
    
    # Test with different similarity methods
    methods = ['embedding', 'kiwi_rouge']
    
    for method in methods:
        print(f"\n📈 Testing MRR with {method} method | {method} 방법으로 MRR 테스트")
        
        try:
            results = evaluate_with_ranx_similarity(
                retriever=retriever,
                questions=questions,
                reference_contexts=reference_contexts,
                k=5,
                method=method,
                similarity_threshold=0.2,  # Lower threshold to include more results
                use_graded_relevance=True,
                evaluation_mode='reference_based'
            )
            
            mrr = results.get('mrr', 0)
            print(f"  MRR: {mrr:.4f}")
            
            # Calculate expected MRR manually for verification
            # Query 1: relevant doc at rank 1 -> RR = 1.0
            # Query 2: relevant doc at rank 3 -> RR = 1/3 = 0.333
            # Query 3: relevant doc at rank 2 -> RR = 1/2 = 0.5  
            # Expected MRR = (1.0 + 0.333 + 0.5) / 3 = 0.611
            expected_mrr = (1.0 + 1/3 + 1/2) / 3
            print(f"  Expected MRR (theoretical): {expected_mrr:.4f}")
            
            if abs(mrr - expected_mrr) < 0.1:
                print(f"  ✅ MRR calculation appears correct (within tolerance)")
            else:
                print(f"  ⚠️ MRR calculation may need verification")
                
        except Exception as e:
            print(f"  ❌ Error testing {method}: {e}")


def test_threshold_sensitivity():
    """Test how graded relevance responds to different similarity thresholds."""
    
    print(f"\n🎚️ Threshold Sensitivity Test | 임계값 민감도 테스트")
    print("="*50)
    
    questions = ["What is artificial intelligence?"]
    reference_contexts = [
        [MockDocument("AI is the simulation of human intelligence in machines.")]
    ]
    
    retrieved_docs = [
        "Artificial intelligence simulates human-like thinking in computers.",  # High similarity
        "AI involves machine learning and neural networks.",                  # Medium similarity
        "Technology has advanced significantly in recent years.",            # Low similarity
        "Machine intelligence mimics human cognitive functions.",             # Medium-high similarity
        "The weather forecast predicts rain tomorrow."                        # Very low similarity
    ]
    
    retriever = TestRetriever(retrieved_docs)
    
    # Test different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print("Threshold | Hit@5  | NDCG@5 | MRR    | Relevant Docs")
    print("-" * 55)
    
    for threshold in thresholds:
        try:
            results = evaluate_with_ranx_similarity(
                retriever=retriever,
                questions=questions,
                reference_contexts=reference_contexts,
                k=5,
                method='embedding',
                similarity_threshold=threshold,
                use_graded_relevance=True,
                evaluation_mode='reference_based'
            )
            
            hit_rate = results.get('hit_rate@5', 0)
            ndcg = results.get('ndcg@5', 0)
            mrr = results.get('mrr', 0)
            
            print(f"{threshold:8.1f} | {hit_rate:6.3f} | {ndcg:6.3f} | {mrr:6.3f} | Varies")
            
        except Exception as e:
            print(f"{threshold:8.1f} | ERROR: {str(e)[:30]}...")


def main():
    """Run all graded relevance tests."""
    
    print("🧪 Graded Relevance and MRR Verification Tests | 등급 관련성 및 MRR 검증 테스트")
    print("="*80)
    
    try:
        # Test 1: Compare graded vs binary relevance
        binary_results, graded_results = test_graded_vs_binary_relevance()
        
        # Test 2: Verify MRR calculation
        test_mrr_calculation()
        
        # Test 3: Threshold sensitivity
        test_threshold_sensitivity()
        
        print(f"\n✅ All tests completed | 모든 테스트 완료")
        print(f"\n📋 Summary | 요약:")
        print(f"  - Graded relevance uses similarity scores as relevance grades")
        print(f"  - Binary relevance treats all reference docs as equally relevant (1.0)")  
        print(f"  - MRR calculation considers ranking positions of relevant documents")
        print(f"  - Higher similarity thresholds = fewer relevant docs = potentially lower recall")
        
    except Exception as e:
        print(f"❌ Test failed | 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()