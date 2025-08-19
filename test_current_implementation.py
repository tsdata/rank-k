#!/usr/bin/env python3
"""
Test current implementation with unified doc_ ID system.
"""

from ranx_k.evaluation import evaluate_with_ranx_similarity


class MockDocument:
    """Mock document for testing."""
    def __init__(self, content: str):
        self.page_content = content


class TestRetriever:
    """Test retriever with predefined documents."""
    
    def __init__(self, documents):
        self.documents = [MockDocument(doc) for doc in documents]
    
    def invoke(self, query: str):
        """Return all documents for any query."""
        return self.documents


def test_current_implementation():
    """Test the current implementation with doc_ ID system."""
    
    print("🧪 Testing Current Implementation | 현재 구현 테스트")
    print("="*70)
    
    # Test data setup
    questions = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is artificial intelligence?"
    ]
    
    # Reference documents (ground truth)
    reference_contexts = [
        [MockDocument("Machine learning is a subset of AI that learns from data.")],
        [MockDocument("Deep learning uses neural networks with multiple layers.")],
        [MockDocument("AI is the simulation of human intelligence in machines.")]
    ]
    
    # Retrieved documents (mix of relevant and irrelevant)
    retrieved_docs = [
        "Machine learning algorithms learn patterns from data automatically.",  # High relevance to Q1
        "Deep learning networks have multiple hidden layers for processing.",    # High relevance to Q2
        "Artificial intelligence simulates human cognitive abilities.",          # High relevance to Q3
        "Weather forecasting uses statistical models.",                         # Low relevance
        "Sports analytics involves data analysis."                              # Low relevance
    ]
    
    retriever = TestRetriever(retrieved_docs)
    
    # Test 1: Binary relevance with high threshold
    print("\n1️⃣ Binary Relevance (High Threshold) | 이진 관련성 (높은 임계값)")
    print("-" * 50)
    
    results_binary_high = evaluate_with_ranx_similarity(
        retriever=retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.8,
        use_graded_relevance=False,
        evaluation_mode='reference_based'
    )
    
    print("\n📊 Binary Relevance Results (threshold=0.8):")
    for metric, score in results_binary_high.items():
        print(f"  {metric}: {score:.3f}")
    
    # Test 2: Graded relevance with high threshold
    print("\n2️⃣ Graded Relevance (High Threshold) | 등급별 관련성 (높은 임계값)")
    print("-" * 50)
    
    results_graded_high = evaluate_with_ranx_similarity(
        retriever=retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.8,
        use_graded_relevance=True,
        evaluation_mode='reference_based'
    )
    
    print("\n📊 Graded Relevance Results (threshold=0.8):")
    for metric, score in results_graded_high.items():
        print(f"  {metric}: {score:.3f}")
    
    # Test 3: Binary relevance with medium threshold
    print("\n3️⃣ Binary Relevance (Medium Threshold) | 이진 관련성 (중간 임계값)")
    print("-" * 50)
    
    results_binary_med = evaluate_with_ranx_similarity(
        retriever=retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.5,
        use_graded_relevance=False,
        evaluation_mode='reference_based'
    )
    
    print("\n📊 Binary Relevance Results (threshold=0.5):")
    for metric, score in results_binary_med.items():
        print(f"  {metric}: {score:.3f}")
    
    # Test 4: Graded relevance with medium threshold
    print("\n4️⃣ Graded Relevance (Medium Threshold) | 등급별 관련성 (중간 임계값)")
    print("-" * 50)
    
    results_graded_med = evaluate_with_ranx_similarity(
        retriever=retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.5,
        use_graded_relevance=True,
        evaluation_mode='reference_based'
    )
    
    print("\n📊 Graded Relevance Results (threshold=0.5):")
    for metric, score in results_graded_med.items():
        print(f"  {metric}: {score:.3f}")
    
    # Comparison analysis
    print("\n📊 Comparison Analysis | 비교 분석")
    print("="*60)
    
    print("\n🔍 High Threshold (0.8) Comparison:")
    print(f"  Binary NDCG@5:  {results_binary_high.get('ndcg@5', 0):.3f}")
    print(f"  Graded NDCG@5:  {results_graded_high.get('ndcg@5', 0):.3f}")
    print(f"  Difference:     {results_graded_high.get('ndcg@5', 0) - results_binary_high.get('ndcg@5', 0):+.3f}")
    
    print("\n🔍 Medium Threshold (0.5) Comparison:")
    print(f"  Binary NDCG@5:  {results_binary_med.get('ndcg@5', 0):.3f}")
    print(f"  Graded NDCG@5:  {results_graded_med.get('ndcg@5', 0):.3f}")
    print(f"  Difference:     {results_graded_med.get('ndcg@5', 0) - results_binary_med.get('ndcg@5', 0):+.3f}")
    
    print("\n✅ Test completed successfully!")
    return {
        'binary_high': results_binary_high,
        'graded_high': results_graded_high,
        'binary_med': results_binary_med,
        'graded_med': results_graded_med
    }


if __name__ == "__main__":
    test_current_implementation()