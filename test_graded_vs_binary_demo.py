#!/usr/bin/env python3
"""
Comprehensive test demonstrating Binary vs Graded relevance differences.
This test creates scenarios where the differences become more apparent.
"""

from ranx import Qrels, Run, evaluate
import numpy as np


def test_with_varying_positions():
    """Test where relevant documents appear at different positions."""
    
    print("🔬 Test 1: Relevant Documents at Different Positions")
    print("=" * 60)
    
    # Scenario: 3 queries with relevant docs at different ranking positions
    # This will show differences in NDCG since position matters
    
    # Binary: all relevant docs get 1.0
    qrels_binary = {
        "q_1": {"doc_1": 1.0, "doc_3": 1.0, "doc_5": 1.0},  # 3 relevant docs
        "q_2": {"doc_2": 1.0, "doc_4": 1.0},                # 2 relevant docs  
        "q_3": {"doc_0": 1.0, "doc_3": 1.0, "doc_4": 1.0}   # 3 relevant docs
    }
    
    # Graded: different relevance scores (simulating similarity scores)
    qrels_graded = {
        "q_1": {"doc_1": 9.0, "doc_3": 5.0, "doc_5": 2.0},  # High, medium, low relevance
        "q_2": {"doc_2": 7.0, "doc_4": 3.0},                # High, low relevance
        "q_3": {"doc_0": 10.0, "doc_3": 4.0, "doc_4": 1.0}  # Very high, medium, very low
    }
    
    # Run with documents retrieved (some relevant docs at lower positions)
    run_dict = {
        "q_1": {
            "doc_0": 0.95,  # Not relevant
            "doc_1": 0.90,  # Relevant (high grade)
            "doc_2": 0.85,  # Not relevant
            "doc_3": 0.80,  # Relevant (medium grade)
            "doc_4": 0.75,  # Not relevant
            "doc_5": 0.70   # Relevant (low grade) - at position 6
        },
        "q_2": {
            "doc_0": 0.88,  # Not relevant
            "doc_1": 0.82,  # Not relevant
            "doc_2": 0.76,  # Relevant (high grade) - at position 3
            "doc_3": 0.70,  # Not relevant
            "doc_4": 0.64   # Relevant (low grade) - at position 5
        },
        "q_3": {
            "doc_0": 0.92,  # Relevant (very high grade) - at position 1
            "doc_1": 0.86,  # Not relevant
            "doc_2": 0.80,  # Not relevant
            "doc_3": 0.74,  # Relevant (medium grade) - at position 4
            "doc_4": 0.68   # Relevant (very low grade) - at position 5
        }
    }
    
    # Evaluate
    metrics = ["hit_rate@5", "ndcg@5", "map@5", "mrr", "recall@5", "precision@5"]
    
    qrels_b = Qrels(qrels_binary)
    qrels_g = Qrels(qrels_graded)
    run = Run(run_dict)
    
    print("\n📊 Binary Relevance Results:")
    results_binary = evaluate(qrels_b, run, metrics)
    for metric, score in results_binary.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n📊 Graded Relevance Results:")
    results_graded = evaluate(qrels_g, run, metrics)
    for metric, score in results_graded.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n🔍 Differences (Graded - Binary):")
    for metric in metrics:
        diff = results_graded[metric] - results_binary[metric]
        symbol = "✅" if abs(diff) > 0.001 else "❌"
        print(f"  {symbol} {metric}: {diff:+.4f}")
    
    print("\n💡 Explanation:")
    print("  - NDCG@5 differs because graded relevance considers importance levels")
    print("  - MAP@5 may differ when highly relevant docs are ranked higher")
    print("  - Hit rate and recall are the same (both find the same docs)")


def test_with_partial_retrieval():
    """Test where not all relevant documents are retrieved."""
    
    print("\n\n🔬 Test 2: Partial Retrieval (Not All Relevant Docs Found)")
    print("=" * 60)
    
    # Binary relevance
    qrels_binary = {
        "q_1": {"doc_0": 1.0, "doc_1": 1.0, "doc_2": 1.0, "doc_6": 1.0},  # 4 relevant
        "q_2": {"doc_1": 1.0, "doc_3": 1.0, "doc_7": 1.0},                 # 3 relevant
        "q_3": {"doc_2": 1.0, "doc_4": 1.0, "doc_8": 1.0}                  # 3 relevant
    }
    
    # Graded relevance with varying importance
    qrels_graded = {
        "q_1": {"doc_0": 10.0, "doc_1": 7.0, "doc_2": 4.0, "doc_6": 1.0},  # Varying importance
        "q_2": {"doc_1": 9.0, "doc_3": 5.0, "doc_7": 2.0},
        "q_3": {"doc_2": 8.0, "doc_4": 6.0, "doc_8": 3.0}
    }
    
    # Run - only retrieves top 5, missing some relevant docs
    run_dict = {
        "q_1": {
            "doc_0": 0.9,   # Relevant (highest grade)
            "doc_1": 0.85,  # Relevant (high grade)
            "doc_3": 0.8,   # Not relevant
            "doc_4": 0.75,  # Not relevant
            "doc_5": 0.7    # Not relevant
            # Missing: doc_2 (medium grade), doc_6 (low grade)
        },
        "q_2": {
            "doc_0": 0.88,  # Not relevant
            "doc_1": 0.82,  # Relevant (highest grade)
            "doc_2": 0.76,  # Not relevant
            "doc_3": 0.70,  # Relevant (medium grade)
            "doc_4": 0.64   # Not relevant
            # Missing: doc_7 (low grade)
        },
        "q_3": {
            "doc_0": 0.92,  # Not relevant
            "doc_1": 0.86,  # Not relevant
            "doc_2": 0.80,  # Relevant (highest grade)
            "doc_3": 0.74,  # Not relevant
            "doc_5": 0.68   # Not relevant
            # Missing: doc_4 (high grade), doc_8 (medium grade)
        }
    }
    
    metrics = ["hit_rate@5", "ndcg@5", "map@5", "mrr", "recall@5", "precision@5"]
    
    qrels_b = Qrels(qrels_binary)
    qrels_g = Qrels(qrels_graded)
    run = Run(run_dict)
    
    print("\n📊 Binary Relevance Results:")
    results_binary = evaluate(qrels_b, run, metrics)
    for metric, score in results_binary.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n📊 Graded Relevance Results:")
    results_graded = evaluate(qrels_g, run, metrics)
    for metric, score in results_graded.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n🔍 Differences (Graded - Binary):")
    for metric in metrics:
        diff = results_graded[metric] - results_binary[metric]
        symbol = "✅" if abs(diff) > 0.001 else "❌"
        print(f"  {symbol} {metric}: {diff:+.4f}")
    
    print("\n💡 Explanation:")
    print("  - Graded relevance better captures that we found the most important docs")
    print("  - Binary treats all relevant docs equally (missing any is equally bad)")
    print("  - NDCG@5 shows bigger difference when important docs are retrieved")


def test_realistic_rag_scenario():
    """Test simulating a realistic RAG evaluation scenario."""
    
    print("\n\n🔬 Test 3: Realistic RAG Scenario")
    print("=" * 60)
    print("Simulating similarity scores from actual retrieval...")
    
    # Simulate 5 queries with varying retrieval quality
    similarity_threshold = 0.7
    
    # Query 1: Perfect retrieval (best doc has highest similarity)
    # Query 2: Good retrieval (most relevant docs found)
    # Query 3: Poor retrieval (low similarities)
    # Query 4: Mixed retrieval (some good, some bad)
    # Query 5: No relevant docs found
    
    qrels_binary = {}
    qrels_graded = {}
    run_dict = {}
    
    # Define similarity scores for each query
    similarities = {
        "q_1": [0.95, 0.88, 0.75, 0.65, 0.50],  # 3 docs above threshold
        "q_2": [0.82, 0.78, 0.71, 0.68, 0.55],  # 3 docs above threshold (lower scores)
        "q_3": [0.73, 0.69, 0.65, 0.60, 0.45],  # 1 doc above threshold
        "q_4": [0.91, 0.72, 0.68, 0.55, 0.40],  # 2 docs above threshold
        "q_5": [0.65, 0.60, 0.55, 0.50, 0.45]   # 0 docs above threshold
    }
    
    for query_id, scores in similarities.items():
        qrels_binary[query_id] = {}
        qrels_graded[query_id] = {}
        run_dict[query_id] = {}
        
        for doc_idx, score in enumerate(scores):
            doc_id = f"doc_{doc_idx}"
            # All docs go into run with their similarity scores
            run_dict[query_id][doc_id] = score
            
            if score >= similarity_threshold:
                # Binary: use 1.0 for all relevant
                qrels_binary[query_id][doc_id] = 1.0
                
                # Graded: scale similarity to relevance grade
                # Scale from [0.7, 1.0] to [1.0, 10.0]
                scaled_score = 1.0 + (score - similarity_threshold) * 9.0 / (1.0 - similarity_threshold)
                qrels_graded[query_id][doc_id] = scaled_score
    
    print("\n📋 Retrieval Statistics:")
    for query_id, scores in similarities.items():
        relevant_count = sum(1 for s in scores if s >= similarity_threshold)
        max_score = max(scores)
        print(f"  {query_id}: {relevant_count} relevant docs, max similarity: {max_score:.2f}")
    
    metrics = ["hit_rate@5", "ndcg@5", "map@5", "mrr", "recall@5", "precision@5"]
    
    qrels_b = Qrels(qrels_binary)
    qrels_g = Qrels(qrels_graded)
    run = Run(run_dict)
    
    print("\n📊 Binary Relevance Results:")
    results_binary = evaluate(qrels_b, run, metrics)
    for metric, score in results_binary.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n📊 Graded Relevance Results:")
    results_graded = evaluate(qrels_g, run, metrics)
    for metric, score in results_graded.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n🔍 Differences (Graded - Binary):")
    significant_diffs = []
    for metric in metrics:
        diff = results_graded[metric] - results_binary[metric]
        symbol = "✅" if abs(diff) > 0.001 else "❌"
        print(f"  {symbol} {metric}: {diff:+.4f}")
        if abs(diff) > 0.001:
            significant_diffs.append(metric)
    
    print("\n💡 Key Insights:")
    if significant_diffs:
        print(f"  ✅ Metrics with significant differences: {', '.join(significant_diffs)}")
        print("  - Graded relevance captures quality differences between retrieved docs")
        print("  - Binary relevance treats all relevant docs as equally important")
    else:
        print("  ⚠️ No significant differences found in this scenario")
        print("  - This can happen when all relevant docs are retrieved at top positions")
        print("  - Or when the retrieval quality is uniformly good/bad across queries")


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Comprehensive Binary vs Graded Relevance Comparison")
    print("=" * 70)
    
    test_with_varying_positions()
    test_with_partial_retrieval()
    test_realistic_rag_scenario()
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("\n📝 Summary:")
    print("  - Binary relevance: Simple 0/1 for irrelevant/relevant")
    print("  - Graded relevance: Uses actual similarity scores as grades")
    print("  - Differences are most visible in NDCG when:")
    print("    • Relevant docs have varying importance")
    print("    • Important docs are ranked higher")
    print("    • Not all relevant docs are retrieved")