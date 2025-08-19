#!/usr/bin/env python3
"""
Test to verify Binary vs Graded relevance produces different results.
"""

from ranx import Qrels, Run, evaluate
import numpy as np


def test_binary_vs_graded_difference():
    """Test that binary and graded relevance produce different NDCG scores."""
    
    print("🧪 Testing Binary vs Graded Relevance Difference")
    print("="*60)
    
    # Test scenario with multiple relevant documents at different positions
    qrels_binary = {
        "q_1": {
            "doc_0": 1.0,  # Binary: all relevant docs get 1.0
            "doc_1": 1.0,
            "doc_3": 1.0
        },
        "q_2": {
            "doc_1": 1.0,
            "doc_2": 1.0
        }
    }
    
    qrels_graded = {
        "q_1": {
            "doc_0": 3.0,  # Graded: different relevance scores
            "doc_1": 2.0,
            "doc_3": 1.5
        },
        "q_2": {
            "doc_1": 2.5,
            "doc_2": 1.8
        }
    }
    
    # Run with documents at various positions
    run_dict = {
        "q_1": {
            "doc_0": 0.9,  # Rank 1
            "doc_1": 0.8,  # Rank 2
            "doc_2": 0.7,  # Rank 3 (not relevant)
            "doc_3": 0.6,  # Rank 4
            "doc_4": 0.5   # Rank 5 (not relevant)
        },
        "q_2": {
            "doc_0": 0.85,  # Rank 1 (not relevant)
            "doc_1": 0.75,  # Rank 2
            "doc_2": 0.65,  # Rank 3
            "doc_3": 0.55,  # Rank 4 (not relevant)
            "doc_4": 0.45   # Rank 5 (not relevant)
        }
    }
    
    # Evaluate binary
    qrels_b = Qrels(qrels_binary)
    run = Run(run_dict)
    metrics = ["hit_rate@5", "ndcg@5", "map@5", "mrr", "recall@5", "precision@5"]
    
    print("\n📊 Binary Relevance Results:")
    results_binary = evaluate(qrels_b, run, metrics)
    for metric, score in results_binary.items():
        print(f"  {metric}: {score:.4f}")
    
    # Evaluate graded
    qrels_g = Qrels(qrels_graded)
    
    print("\n📊 Graded Relevance Results:")
    results_graded = evaluate(qrels_g, run, metrics)
    for metric, score in results_graded.items():
        print(f"  {metric}: {score:.4f}")
    
    # Compare differences
    print("\n🔍 Differences (Graded - Binary):")
    for metric in metrics:
        if metric in results_binary and metric in results_graded:
            diff = results_graded[metric] - results_binary[metric]
            symbol = "✅" if abs(diff) > 0.001 else "❌"
            print(f"  {symbol} {metric}: {diff:+.4f}")
    
    # Check if metrics are all the same
    print("\n📈 Metric Variation Analysis:")
    
    binary_values = list(results_binary.values())
    graded_values = list(results_graded.values())
    
    binary_unique = len(set(binary_values))
    graded_unique = len(set(graded_values))
    
    print(f"  Binary: {binary_unique} unique values out of {len(binary_values)} metrics")
    print(f"  Graded: {graded_unique} unique values out of {len(graded_values)} metrics")
    
    if binary_unique == 1:
        print("  ⚠️ WARNING: All binary metrics have the same value!")
    if graded_unique == 1:
        print("  ⚠️ WARNING: All graded metrics have the same value!")


def test_realistic_scenario():
    """Test with a more realistic scenario."""
    
    print("\n\n🔬 Realistic Scenario Test")
    print("="*60)
    
    # Simulate similarity scores from real retrieval
    similarity_threshold = 0.6
    
    # Simulated similarity scores for 3 queries
    similarities = {
        "q_1": [0.85, 0.72, 0.45, 0.68, 0.30],  # 3 docs above threshold
        "q_2": [0.55, 0.78, 0.82, 0.40, 0.35],  # 2 docs above threshold  
        "q_3": [0.91, 0.30, 0.25, 0.20, 0.15]   # 1 doc above threshold
    }
    
    # Build qrels and run based on similarities
    qrels_binary = {}
    qrels_graded = {}
    run_dict = {}
    
    for query_id, scores in similarities.items():
        qrels_binary[query_id] = {}
        qrels_graded[query_id] = {}
        run_dict[query_id] = {}
        
        for doc_idx, score in enumerate(scores):
            doc_id = f"doc_{doc_idx}"
            run_dict[query_id][doc_id] = score
            
            if score >= similarity_threshold:
                # Binary: use 1.0
                qrels_binary[query_id][doc_id] = 1.0
                
                # Graded: scale similarity score
                # Scale from [0.6, 1.0] to [1.0, 10.0]
                scaled_score = 1.0 + (score - similarity_threshold) * 9.0 / (1.0 - similarity_threshold)
                qrels_graded[query_id][doc_id] = scaled_score
    
    # Print data structures
    print("\n📋 Data Structures:")
    print("\nBinary Qrels:")
    for q, docs in qrels_binary.items():
        print(f"  {q}: {docs}")
    
    print("\nGraded Qrels:")
    for q, docs in qrels_graded.items():
        print(f"  {q}: {docs}")
    
    print("\nRun:")
    for q, docs in run_dict.items():
        print(f"  {q}: {docs}")
    
    # Evaluate
    metrics = ["hit_rate@5", "ndcg@5", "map@5", "mrr", "recall@5", "precision@5"]
    
    qrels_b = Qrels(qrels_binary)
    qrels_g = Qrels(qrels_graded)
    run = Run(run_dict)
    
    print("\n📊 Binary Results:")
    results_binary = evaluate(qrels_b, run, metrics)
    for metric, score in results_binary.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n📊 Graded Results:")
    results_graded = evaluate(qrels_g, run, metrics)
    for metric, score in results_graded.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n🔍 Differences (Graded - Binary):")
    for metric in metrics:
        if metric in results_binary and metric in results_graded:
            diff = results_graded[metric] - results_binary[metric]
            symbol = "✅" if abs(diff) > 0.001 else "❌"
            print(f"  {symbol} {metric}: {diff:+.4f}")


if __name__ == "__main__":
    test_binary_vs_graded_difference()
    test_realistic_scenario()