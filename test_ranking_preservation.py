#!/usr/bin/env python3
"""
Test that document ranking is properly preserved in evaluation.
"""

from ranx_k.evaluation import evaluate_with_ranx_similarity
import numpy as np


class MockDocument:
    """Mock document for testing."""
    def __init__(self, content: str):
        self.page_content = content


class OrderedRetriever:
    """Retriever that returns documents in specific order."""
    
    def __init__(self, documents, order):
        self.documents = [MockDocument(doc) for doc in documents]
        self.order = order
    
    def invoke(self, query: str):
        """Return documents in specified order."""
        return [self.documents[i] for i in self.order]


def test_ranking_preservation():
    """
    Test that reranking changes are properly reflected in metrics.
    """
    
    print("🧪 Testing Ranking Preservation in Evaluation")
    print("=" * 70)
    
    questions = ["What is machine learning?"]
    
    # Reference document
    reference_contexts = [
        [MockDocument("Machine learning is a method of data analysis that automates analytical model building.")]
    ]
    
    # Pool of documents with varying relevance
    documents = [
        "Machine learning uses algorithms to learn from data.",           # Doc 0 - High relevance
        "Deep learning is a subset of machine learning.",                 # Doc 1 - Medium relevance  
        "Data analysis involves examining datasets.",                     # Doc 2 - Low relevance
        "Sports analytics uses statistical methods.",                     # Doc 3 - Very low
        "Weather forecasting predicts atmospheric conditions."            # Doc 4 - Very low
    ]
    
    # Test 1: Original order (best doc first)
    print("\n📊 Test 1: Best Document First")
    print("-" * 50)
    original_order = [0, 1, 2, 3, 4]  # Best to worst
    original_retriever = OrderedRetriever(documents, original_order)
    
    print("Document order:")
    for i, idx in enumerate(original_order):
        print(f"  Position {i+1}: Doc {idx} - {documents[idx][:40]}...")
    
    # Test with binary relevance
    print("\n🔹 Binary Relevance (use_graded_relevance=False):")
    results_original_binary = evaluate_with_ranx_similarity(
        retriever=original_retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.5,
        use_graded_relevance=False,
        evaluation_mode='retrieval_based'
    )
    
    # Test with graded relevance
    print("\n🔹 Graded Relevance (use_graded_relevance=True):")
    results_original_graded = evaluate_with_ranx_similarity(
        retriever=original_retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.5,
        use_graded_relevance=True,
        evaluation_mode='retrieval_based'
    )
    
    # Test 2: Reversed order (worst doc first) 
    print("\n\n📊 Test 2: Worst Document First (Simulating Bad Retrieval)")
    print("-" * 50)
    reversed_order = [4, 3, 2, 1, 0]  # Worst to best
    reversed_retriever = OrderedRetriever(documents, reversed_order)
    
    print("Document order:")
    for i, idx in enumerate(reversed_order):
        print(f"  Position {i+1}: Doc {idx} - {documents[idx][:40]}...")
    
    # Test with binary relevance
    print("\n🔹 Binary Relevance (use_graded_relevance=False):")
    results_reversed_binary = evaluate_with_ranx_similarity(
        retriever=reversed_retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.5,
        use_graded_relevance=False,
        evaluation_mode='retrieval_based'
    )
    
    # Test with graded relevance
    print("\n🔹 Graded Relevance (use_graded_relevance=True):")
    results_reversed_graded = evaluate_with_ranx_similarity(
        retriever=reversed_retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.5,
        use_graded_relevance=True,
        evaluation_mode='retrieval_based'
    )
    
    # Test 3: Reranked order (simulating cross-encoder improvement)
    print("\n\n📊 Test 3: Reranked Order (Simulating Cross-Encoder)")
    print("-" * 50)
    reranked_order = [0, 2, 1, 3, 4]  # Best doc still first, but 2 and 1 swapped
    reranked_retriever = OrderedRetriever(documents, reranked_order)
    
    print("Document order:")
    for i, idx in enumerate(reranked_order):
        print(f"  Position {i+1}: Doc {idx} - {documents[idx][:40]}...")
    
    results_reranked_binary = evaluate_with_ranx_similarity(
        retriever=reranked_retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.5,
        use_graded_relevance=False,
        evaluation_mode='retrieval_based'
    )
    
    results_reranked_graded = evaluate_with_ranx_similarity(
        retriever=reranked_retriever,
        questions=questions,
        reference_contexts=reference_contexts,
        k=5,
        method='embedding',
        similarity_threshold=0.5,
        use_graded_relevance=True,
        evaluation_mode='retrieval_based'
    )
    
    # Compare all results
    print("\n\n" + "=" * 70)
    print("📊 RESULTS COMPARISON")
    print("=" * 70)
    
    metrics = ['mrr', 'map@5', 'ndcg@5', 'hit_rate@5', 'precision@5']
    
    print("\n🔹 Binary Relevance Results:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Best First':<15} {'Worst First':<15} {'Reranked':<15}")
    print("-" * 50)
    for metric in metrics:
        orig = results_original_binary.get(metric, 0)
        rev = results_reversed_binary.get(metric, 0)
        rerank = results_reranked_binary.get(metric, 0)
        print(f"{metric:<15} {orig:<15.3f} {rev:<15.3f} {rerank:<15.3f}")
    
    print("\n🔹 Graded Relevance Results:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Best First':<15} {'Worst First':<15} {'Reranked':<15}")
    print("-" * 50)
    for metric in metrics:
        orig = results_original_graded.get(metric, 0)
        rev = results_reversed_graded.get(metric, 0)
        rerank = results_reranked_graded.get(metric, 0)
        print(f"{metric:<15} {orig:<15.3f} {rev:<15.3f} {rerank:<15.3f}")
    
    # Analysis
    print("\n\n📈 Analysis:")
    print("-" * 50)
    
    # Check if ranking matters
    binary_mrr_diff = abs(results_original_binary.get('mrr', 0) - results_reversed_binary.get('mrr', 0))
    graded_mrr_diff = abs(results_original_graded.get('mrr', 0) - results_reversed_graded.get('mrr', 0))
    
    print(f"\n1. MRR Sensitivity to Ranking:")
    print(f"   Binary: {binary_mrr_diff:.3f} difference")
    print(f"   Graded: {graded_mrr_diff:.3f} difference")
    
    binary_map_diff = abs(results_original_binary.get('map@5', 0) - results_reversed_binary.get('map@5', 0))
    graded_map_diff = abs(results_original_graded.get('map@5', 0) - results_reversed_graded.get('map@5', 0))
    
    print(f"\n2. MAP@5 Sensitivity to Ranking:")
    print(f"   Binary: {binary_map_diff:.3f} difference")
    print(f"   Graded: {graded_map_diff:.3f} difference")
    
    binary_ndcg_diff = abs(results_original_binary.get('ndcg@5', 0) - results_reversed_binary.get('ndcg@5', 0))
    graded_ndcg_diff = abs(results_original_graded.get('ndcg@5', 0) - results_reversed_graded.get('ndcg@5', 0))
    
    print(f"\n3. NDCG@5 Sensitivity to Ranking:")
    print(f"   Binary: {binary_ndcg_diff:.3f} difference")
    print(f"   Graded: {graded_ndcg_diff:.3f} difference")
    
    if binary_mrr_diff > 0.01 or binary_map_diff > 0.01 or binary_ndcg_diff > 0.01:
        print("\n✅ Binary relevance now shows ranking sensitivity!")
    else:
        print("\n⚠️ Binary relevance still not sensitive to ranking")
    
    if graded_mrr_diff > binary_mrr_diff:
        print("✅ Graded relevance shows MORE ranking sensitivity than binary!")
    
    # Check reranking effect
    rerank_improvement = results_reranked_binary.get('map@5', 0) - results_original_binary.get('map@5', 0)
    if abs(rerank_improvement) > 0.001:
        print(f"\n✅ Reranking effect detected: MAP@5 changed by {rerank_improvement:+.3f}")
    else:
        print("\n⚠️ Reranking had no effect on MAP@5")


if __name__ == "__main__":
    test_ranking_preservation()
    
    print("\n\n" + "=" * 70)
    print("💡 KEY IMPROVEMENTS:")
    print("=" * 70)
    print("""
1. Position-Aware Scoring:
   - Documents maintain their retrieval order
   - Earlier positions get slight score boost
   - Reranking improvements are measurable

2. Binary Relevance Fixed:
   - Now sensitive to document positions
   - MRR, MAP, NDCG reflect ranking quality
   - Reranking effects are visible

3. Graded Relevance Enhanced:
   - Combines similarity and position
   - More sensitive to ranking changes
   - Better differentiation between good/bad ranking

4. Use Cases:
   - Binary: When you only care if docs are relevant or not
   - Graded: When similarity degree matters (recommended for reranking)
""")