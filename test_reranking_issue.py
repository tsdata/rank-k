#!/usr/bin/env python3
"""
Analyze why reranking doesn't change evaluation scores.
"""

from ranx_k.evaluation import evaluate_with_ranx_similarity


class MockDocument:
    """Mock document for testing."""
    def __init__(self, content: str, source: str = "test"):
        self.page_content = content
        self.metadata = {"source": source}


class BaseRetriever:
    """Base retriever without reranking."""
    def __init__(self, documents):
        self.documents = [MockDocument(doc, f"doc_{i}") for i, doc in enumerate(documents)]
    
    def invoke(self, query: str):
        """Return documents in original order."""
        return self.documents[:5]  # Return top 5


class RerankingRetriever:
    """Retriever with reranking."""
    def __init__(self, documents):
        self.documents = [MockDocument(doc, f"doc_{i}") for i, doc in enumerate(documents)]
    
    def invoke(self, query: str):
        """Return reranked documents - better docs first."""
        # Simulate reranking by reversing order (assuming better docs were at the end)
        reranked = list(reversed(self.documents[:5]))
        return reranked


def analyze_reranking_impact():
    """
    Analyze why reranking might not change scores.
    """
    
    print("🔍 Analyzing Reranking Impact on Evaluation Scores")
    print("=" * 70)
    
    questions = ["What is machine learning?"]
    
    # Reference documents
    reference_contexts = [
        [MockDocument("Machine learning is a subset of AI that enables learning from data.")]
    ]
    
    # Documents pool - assume doc 4 is the best match
    documents = [
        "Sports involve physical activity and competition.",      # Doc 0 - irrelevant
        "Weather forecasting uses meteorological models.",        # Doc 1 - irrelevant  
        "Cooking requires various ingredients and techniques.",   # Doc 2 - irrelevant
        "AI includes various technologies like robotics.",        # Doc 3 - somewhat relevant
        "ML algorithms learn patterns from training data.",       # Doc 4 - highly relevant
        "Deep learning uses neural networks.",                    # Doc 5 - not retrieved
        "Natural language processing handles text."               # Doc 6 - not retrieved
    ]
    
    # Test without reranking
    print("\n📊 Scenario 1: Without Reranking")
    print("-" * 50)
    print("Retrieved order: [Doc0, Doc1, Doc2, Doc3, Doc4]")
    print("Relevance: [None, None, None, Low, High]")
    
    base_retriever = BaseRetriever(documents)
    results_base = evaluate_with_ranx_similarity(
        retriever=base_retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.7,
        evaluation_mode='retrieval_based'
    )
    
    # Test with reranking
    print("\n📊 Scenario 2: With Reranking")
    print("-" * 50)
    print("Reranked order: [Doc4, Doc3, Doc2, Doc1, Doc0]")
    print("Relevance: [High, Low, None, None, None]")
    
    rerank_retriever = RerankingRetriever(documents)
    results_rerank = evaluate_with_ranx_similarity(
        retriever=rerank_retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.7,
        evaluation_mode='retrieval_based'
    )
    
    print("\n" + "=" * 70)
    print("📊 Score Comparison:")
    print("-" * 50)
    
    # Compare scores
    metrics_to_compare = ['hit_rate@5', 'ndcg@5', 'map@5', 'mrr']
    for metric in metrics_to_compare:
        base_score = results_base.get(metric, 0.0)
        rerank_score = results_rerank.get(metric, 0.0)
        diff = rerank_score - base_score
        
        print(f"{metric}:")
        print(f"  Without reranking: {base_score:.3f}")
        print(f"  With reranking: {rerank_score:.3f}")
        print(f"  Difference: {diff:+.3f}")
        print()


def explain_why_scores_same():
    """
    Explain why reranking might not change scores.
    """
    
    print("\n\n🎯 Why Reranking Might Not Change Scores")
    print("=" * 70)
    
    print("\n1️⃣ **Binary Relevance Problem**")
    print("-" * 50)
    print("If using binary relevance (document is either relevant or not):")
    print("- Hit@k: Same if both retrieve at least one relevant doc")
    print("- Recall: Same if both retrieve the same relevant docs")
    print("- The ORDER doesn't matter for these metrics with binary relevance")
    
    print("\n2️⃣ **All Retrieved Docs Below Threshold**")
    print("-" * 50)
    print("If similarity_threshold is too high (e.g., 0.8):")
    print("- Even the 'best' reranked doc might not exceed threshold")
    print("- Result: 0 relevant docs for both → same scores (all 0)")
    
    print("\n3️⃣ **All Retrieved Docs Above Threshold**")
    print("-" * 50)
    print("If similarity_threshold is too low:")
    print("- All docs might be marked as relevant")
    print("- Result: Perfect scores for both → same scores (all 1.0)")
    
    print("\n4️⃣ **Single Query Pattern**")
    print("-" * 50)
    print("With only one query:")
    print("- Either it finds the reference (score=1) or doesn't (score=0)")
    print("- Binary outcome leads to identical metrics")
    
    print("\n5️⃣ **Reranking Not Actually Working**")
    print("-" * 50)
    print("Check if reranking is actually changing order:")
    print("- Print document order before and after reranking")
    print("- Verify cross-encoder scores are being used")


def test_ranking_sensitive_metrics():
    """
    Test with scenarios where ranking SHOULD matter.
    """
    
    print("\n\n🔬 Testing Ranking-Sensitive Scenarios")
    print("=" * 70)
    
    questions = ["Machine learning query"]
    
    # Multiple reference documents with different relevance
    reference_contexts = [
        [MockDocument("Exact match: Machine learning is AI."),
         MockDocument("Good match: ML algorithms learn."),
         MockDocument("Partial match: AI includes ML.")]
    ]
    
    # Documents where order matters
    documents = [
        "Irrelevant document about sports.",
        "Partial: AI includes ML and robotics.",      # Matches ref 2
        "Irrelevant document about weather.",
        "Good: ML algorithms learn from data.",        # Matches ref 1
        "Best: Machine learning is AI technology.",    # Matches ref 0
    ]
    
    print("\n📊 Test 1: Bad Ranking (best doc last)")
    base_retriever = BaseRetriever(documents)
    
    print("\n📊 Test 2: Good Ranking (best doc first)")
    
    class OptimalRerankingRetriever:
        def __init__(self, documents):
            self.documents = [MockDocument(doc, f"doc_{i}") for i, doc in enumerate(documents)]
        
        def invoke(self, query: str):
            # Return in optimal order: [4, 3, 1, 0, 2]
            return [self.documents[4], self.documents[3], self.documents[1], 
                   self.documents[0], self.documents[2]]
    
    optimal_retriever = OptimalRerankingRetriever(documents)
    
    # Test both
    results_bad = evaluate_with_ranx_similarity(
        retriever=base_retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.5,
        use_graded_relevance=True,  # Use graded to see ranking impact
        evaluation_mode='reference_based'
    )
    
    results_good = evaluate_with_ranx_similarity(
        retriever=optimal_retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.5,
        use_graded_relevance=True,
        evaluation_mode='reference_based'
    )
    
    print("\n📊 Results with Graded Relevance:")
    print("-" * 50)
    for metric in ['mrr', 'map@5', 'ndcg@5']:
        bad = results_bad.get(metric, 0.0)
        good = results_good.get(metric, 0.0)
        print(f"{metric}: Bad={bad:.3f}, Good={good:.3f}, Diff={good-bad:+.3f}")


def provide_debugging_steps():
    """
    Provide debugging steps for the user.
    """
    
    print("\n\n🛠️ Debugging Steps for Your Case")
    print("=" * 70)
    
    print("""
1. **Check Document Order**:
   ```python
   # Before reranking
   base_docs = base_retriever.invoke(query)
   print("Before:", [d.page_content[:30] for d in base_docs])
   
   # After reranking
   reranked_docs = cross_encoder_retriever.invoke(query)
   print("After:", [d.page_content[:30] for d in reranked_docs])
   ```

2. **Check Similarity Scores**:
   ```python
   # Add debug print in evaluate_with_ranx_similarity
   # to see actual similarity scores between retrieved and reference docs
   ```

3. **Lower Threshold**:
   ```python
   # Try lower threshold to see if docs are being matched
   evaluate_with_ranx_similarity(..., similarity_threshold=0.3)
   ```

4. **Use Graded Relevance**:
   ```python
   # This makes ranking matter more
   evaluate_with_ranx_similarity(..., use_graded_relevance=True)
   ```

5. **Check Metrics Individually**:
   ```python
   # MRR and MAP are most sensitive to ranking
   print(f"MRR: {results['mrr']}")
   print(f"MAP: {results['map@5']}")
   ```
""")


if __name__ == "__main__":
    analyze_reranking_impact()
    explain_why_scores_same()
    test_ranking_sensitive_metrics()
    provide_debugging_steps()
    
    print("\n" + "=" * 70)
    print("💡 Key Insights:")
    print("  1. Binary relevance + single match = same scores regardless of position")
    print("  2. Threshold too high/low can mask ranking differences")
    print("  3. Use graded_relevance=True to see ranking impact")
    print("  4. MRR and MAP are most sensitive to ranking changes")
    print("  5. Verify reranking is actually changing document order")