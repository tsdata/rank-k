#!/usr/bin/env python3
"""
Analyze why all metrics show identical values in certain scenarios.
"""

from ranx import Qrels, Run, evaluate
import numpy as np


def analyze_metric_convergence():
    """
    Analyze scenarios where different metrics converge to the same value.
    """
    
    print("🔍 Analyzing Metric Convergence Scenarios")
    print("=" * 60)
    
    # Scenario 1: Each query has exactly 1 relevant document at position 1
    print("\n📊 Scenario 1: One relevant doc per query, always at top")
    print("-" * 50)
    
    qrels1 = {
        "q_1": {"doc_0": 1.0},
        "q_2": {"doc_1": 1.0}, 
        "q_3": {"doc_2": 1.0},
        "q_4": {"doc_3": 1.0},
        "q_5": {"doc_4": 1.0}
    }
    
    run1 = {
        "q_1": {"doc_0": 0.95, "doc_1": 0.8, "doc_2": 0.7, "doc_3": 0.6, "doc_4": 0.5},
        "q_2": {"doc_1": 0.94, "doc_2": 0.8, "doc_3": 0.7, "doc_4": 0.6, "doc_5": 0.5},
        "q_3": {"doc_2": 0.93, "doc_3": 0.8, "doc_4": 0.7, "doc_5": 0.6, "doc_6": 0.5},
        "q_4": {"doc_3": 0.92, "doc_4": 0.8, "doc_5": 0.7, "doc_6": 0.6, "doc_7": 0.5},
        "q_5": {"doc_4": 0.91, "doc_5": 0.8, "doc_6": 0.7, "doc_7": 0.6, "doc_8": 0.5}
    }
    
    qrels_obj = Qrels(qrels1)
    run_obj = Run(run1)
    metrics = ["hit_rate@5", "ndcg@5", "map@5", "mrr"]
    results = evaluate(qrels_obj, run_obj, metrics)
    
    for metric, score in results.items():
        print(f"  {metric}: {score:.3f}")
    
    print("\n💡 Analysis: All metrics = 1.000 because:")
    print("  - Hit@5: All relevant docs are in top 5 (5/5)")
    print("  - NDCG@5: All relevant docs at position 1 (perfect ranking)")
    print("  - MAP@5: Average precision is 1.0 for all queries")
    print("  - MRR: All relevant docs at rank 1 (reciprocal rank = 1.0)")
    
    # Scenario 2: 92% of queries succeed, 8% fail completely
    print("\n📊 Scenario 2: 92% success rate (46/50 queries)")
    print("-" * 50)
    
    qrels2 = {}
    run2 = {}
    
    # 46 successful queries (92%)
    for i in range(46):
        query_id = f"q_{i+1}"
        doc_id = f"doc_{i}"
        qrels2[query_id] = {doc_id: 1.0}
        # Relevant doc always at top
        run2[query_id] = {
            doc_id: 0.95 - i*0.001,
            f"doc_{i+100}": 0.7,
            f"doc_{i+200}": 0.6,
            f"doc_{i+300}": 0.5,
            f"doc_{i+400}": 0.4
        }
    
    # 4 failed queries (8%) - no relevant docs found
    for i in range(46, 50):
        query_id = f"q_{i+1}"
        qrels2[query_id] = {f"doc_{i}": 1.0}
        # Relevant doc not in retrieved results
        run2[query_id] = {
            f"doc_{i+100}": 0.7,
            f"doc_{i+200}": 0.6,
            f"doc_{i+300}": 0.5,
            f"doc_{i+400}": 0.4,
            f"doc_{i+500}": 0.3
        }
    
    qrels_obj2 = Qrels(qrels2)
    run_obj2 = Run(run2)
    results2 = evaluate(qrels_obj2, run_obj2, metrics)
    
    for metric, score in results2.items():
        print(f"  {metric}: {score:.3f}")
    
    print("\n💡 Analysis: All metrics ≈ 0.920 because:")
    print("  - 46/50 = 0.92 success rate")
    print("  - Hit@5: 46 hits out of 50 queries = 0.920")
    print("  - NDCG@5: 46 perfect + 4 zeros, averaged = 0.920")
    print("  - MAP@5: 46 queries with AP=1.0, 4 with AP=0.0 = 0.920")
    print("  - MRR: 46 queries with RR=1.0, 4 with RR=0.0 = 0.920")
    
    # Scenario 3: Multiple relevant docs with varying positions
    print("\n📊 Scenario 3: Multiple relevant docs at different positions")
    print("-" * 50)
    
    qrels3 = {
        "q_1": {"doc_0": 1.0, "doc_1": 1.0, "doc_2": 1.0},
        "q_2": {"doc_3": 1.0, "doc_4": 1.0},
        "q_3": {"doc_5": 1.0, "doc_6": 1.0, "doc_7": 1.0, "doc_8": 1.0},
        "q_4": {"doc_9": 1.0},
        "q_5": {"doc_10": 1.0, "doc_11": 1.0}
    }
    
    run3 = {
        "q_1": {"doc_0": 0.9, "doc_100": 0.8, "doc_1": 0.7, "doc_200": 0.6, "doc_2": 0.5},
        "q_2": {"doc_300": 0.9, "doc_3": 0.8, "doc_4": 0.7, "doc_400": 0.6, "doc_500": 0.5},
        "q_3": {"doc_5": 0.9, "doc_6": 0.8, "doc_600": 0.7, "doc_7": 0.6, "doc_8": 0.5},
        "q_4": {"doc_700": 0.9, "doc_800": 0.8, "doc_9": 0.7, "doc_900": 0.6, "doc_1000": 0.5},
        "q_5": {"doc_10": 0.9, "doc_1100": 0.8, "doc_1200": 0.7, "doc_11": 0.6, "doc_1300": 0.5}
    }
    
    qrels_obj3 = Qrels(qrels3)
    run_obj3 = Run(run3)
    results3 = evaluate(qrels_obj3, run_obj3, metrics)
    
    for metric, score in results3.items():
        print(f"  {metric}: {score:.3f}")
    
    print("\n💡 Analysis: Metrics differ when:")
    print("  - Multiple relevant docs exist per query")
    print("  - Relevant docs appear at different positions")
    print("  - Not all relevant docs are retrieved")


def explain_metric_calculation():
    """
    Explain how each metric is calculated and why they might converge.
    """
    
    print("\n\n📚 Understanding Metric Calculations")
    print("=" * 60)
    
    print("\n1️⃣ Hit Rate @ K")
    print("  Formula: % of queries with at least 1 relevant doc in top K")
    print("  Range: [0, 1]")
    print("  Converges when: Each query has 0 or 1 relevant doc")
    
    print("\n2️⃣ NDCG @ K (Normalized Discounted Cumulative Gain)")
    print("  Formula: DCG@K / IDCG@K")
    print("  Range: [0, 1]")
    print("  Converges when: All relevant docs at same position across queries")
    
    print("\n3️⃣ MAP @ K (Mean Average Precision)")
    print("  Formula: Average of AP scores across queries")
    print("  Range: [0, 1]")
    print("  Converges when: Each query has 1 relevant doc at position 1")
    
    print("\n4️⃣ MRR (Mean Reciprocal Rank)")
    print("  Formula: Average of 1/rank_of_first_relevant")
    print("  Range: [0, 1]")
    print("  Converges when: First relevant always at same position")
    
    print("\n⚠️ Key Insight:")
    print("When each query has exactly 1 relevant document and it's either:")
    print("  • Always found at position 1 → All metrics = 1.0")
    print("  • Found at position 1 in X% of queries → All metrics ≈ X%")
    print("  • Never found → All metrics = 0.0")


def simulate_your_scenario():
    """
    Simulate the exact scenario from the user's output.
    """
    
    print("\n\n🎯 Simulating Your Exact Scenario")
    print("=" * 60)
    print("50 queries, 58 relevant docs found out of 76 references")
    print("Average 1.16 relevant per query, 92% success rate")
    print("-" * 50)
    
    qrels = {}
    run = {}
    
    # Simulate: 46 queries with 1 relevant doc found, 4 queries with 0
    # Total: 46 successful (92%), some with extra relevant docs
    
    success_count = 46
    total_queries = 50
    
    # First 46 queries: successful retrieval
    for i in range(success_count):
        query_id = f"q_{i+1}"
        doc_id = f"doc_{i}"
        
        # Most queries have 1 relevant doc
        if i < 40:
            qrels[query_id] = {doc_id: 1.0}
        # Some queries have 2 relevant docs (to reach 58 total)
        elif i < 46:
            qrels[query_id] = {doc_id: 1.0, f"doc_{i+100}": 1.0}
        
        # Relevant doc always at top position
        run[query_id] = {
            doc_id: 0.95,
            f"doc_{i+100}": 0.85,  # Sometimes relevant
            f"doc_{i+200}": 0.75,
            f"doc_{i+300}": 0.65,
            f"doc_{i+400}": 0.55
        }
    
    # Last 4 queries: failed retrieval
    for i in range(success_count, total_queries):
        query_id = f"q_{i+1}"
        # Has relevant docs in ground truth
        qrels[query_id] = {f"doc_{i}": 1.0}
        # But not retrieved
        run[query_id] = {
            f"doc_{i+500}": 0.7,
            f"doc_{i+600}": 0.6,
            f"doc_{i+700}": 0.5,
            f"doc_{i+800}": 0.4,
            f"doc_{i+900}": 0.3
        }
    
    qrels_obj = Qrels(qrels)
    run_obj = Run(run)
    metrics = ["hit_rate@5", "ndcg@5", "map@5", "mrr"]
    results = evaluate(qrels_obj, run_obj, metrics)
    
    print("\n📊 Results:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.3f}")
    
    print("\n✅ Explanation for 0.920 across all metrics:")
    print("  • 46 out of 50 queries (92%) have their relevant doc at rank 1")
    print("  • 4 queries (8%) completely fail to retrieve relevant docs")
    print("  • When successful, the first relevant doc is always at position 1")
    print("  • This creates a binary outcome: perfect (1.0) or failure (0.0)")
    print("  • Average: 0.92 * 1.0 + 0.08 * 0.0 = 0.920")
    
    print("\n📝 This is NOT a bug, but indicates:")
    print("  1. Your retrieval is very binary (works perfectly or fails)")
    print("  2. When it works, the best doc is always ranked first")
    print("  3. Consider using graded relevance for more nuanced evaluation")


if __name__ == "__main__":
    analyze_metric_convergence()
    explain_metric_calculation()
    simulate_your_scenario()
    
    print("\n" + "=" * 60)
    print("💡 Recommendations:")
    print("  1. This pattern suggests high-quality retrieval when successful")
    print("  2. Focus on improving the 8% failure cases")
    print("  3. Consider using graded relevance for NDCG differentiation")
    print("  4. Examine which queries are failing completely")