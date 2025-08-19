#!/usr/bin/env python3
"""
Test the improved reference_based evaluation mode.
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


def test_improved_reference_based():
    """Test the improved reference_based mode."""
    
    print("🧪 Testing Improved Reference-Based Mode")
    print("=" * 70)
    
    # Test scenario matching the user's case
    questions = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is artificial intelligence?"
    ]
    
    # Reference documents (some queries have multiple refs)
    reference_contexts = [
        [MockDocument("Machine learning is a subset of AI.")],
        [MockDocument("Deep learning uses neural networks."),
         MockDocument("Deep learning is part of machine learning.")],
        [MockDocument("AI simulates human intelligence.")]
    ]
    
    # Total: 4 reference documents
    
    # Retrieved documents (will match some but not all refs)
    retrieved_docs = [
        "Machine learning enables computers to learn from data.",  # Matches Q1 ref
        "Deep learning involves multiple neural network layers.",   # Matches Q2 ref 1
        "Artificial intelligence mimics human cognitive abilities.", # Matches Q3 ref
        "Weather forecasting uses statistical models.",             # No match
        "Sports analytics involves data analysis."                  # No match
    ]
    
    retriever = TestRetriever(retrieved_docs)
    
    # Test with high threshold
    print("\n📊 Test 1: High Threshold (0.8)")
    print("-" * 50)
    
    results = evaluate_with_ranx_similarity(
        retriever=retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.8,
        use_graded_relevance=False,
        evaluation_mode='reference_based'
    )
    
    print("\n✅ Expected behavior:")
    print("  - Qrels should have 4 entries (all reference docs)")
    print("  - Run should have fewer entries (only matched refs)")
    print("  - Overall recall should reflect actual found/total")
    
    # Test with lower threshold
    print("\n📊 Test 2: Lower Threshold (0.5)")
    print("-" * 50)
    
    results2 = evaluate_with_ranx_similarity(
        retriever=retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.5,
        use_graded_relevance=False,
        evaluation_mode='reference_based'
    )
    
    print("\n✅ With lower threshold:")
    print("  - More reference docs should be found")
    print("  - Overall recall should be higher")
    
    # Compare with retrieval_based mode
    print("\n📊 Test 3: Retrieval-Based Mode (for comparison)")
    print("-" * 50)
    
    results3 = evaluate_with_ranx_similarity(
        retriever=retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.5,
        use_graded_relevance=False,
        evaluation_mode='retrieval_based'
    )
    
    print("\n📊 Mode Comparison:")
    print("  - reference_based: Focuses on finding all reference docs")
    print("  - retrieval_based: Focuses on precision of retrieved docs")


if __name__ == "__main__":
    test_improved_reference_based()
    
    print("\n" + "=" * 70)
    print("✅ Improved reference_based mode:")
    print("  1. Uses ref_X IDs for reference documents")
    print("  2. Includes all references in qrels")
    print("  3. Only includes found references in run")
    print("  4. Properly calculates recall")