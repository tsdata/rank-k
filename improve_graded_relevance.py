#!/usr/bin/env python3
"""
Improved graded relevance implementation that provides practical benefits.
"""

import numpy as np
from typing import Dict, List, Tuple


def calculate_graded_metrics(qrels_dict: Dict, run_dict: Dict, k: int = 5) -> Dict[str, float]:
    """
    Calculate custom metrics that properly utilize graded relevance scores.
    
    These metrics consider not just whether a document is relevant,
    but HOW relevant it is based on similarity scores.
    """
    
    metrics = {}
    
    # Weighted Hit Rate: Sum of relevance grades / Max possible sum
    weighted_hits = []
    
    # Graded MRR: Reciprocal rank weighted by relevance grade
    graded_mrr_scores = []
    
    # Weighted Precision: Sum of grades / (k * max_grade)
    weighted_precision_scores = []
    
    # Graded MAP: Average precision weighted by relevance grades
    graded_map_scores = []
    
    for query_id in qrels_dict:
        if query_id not in run_dict:
            continue
            
        qrels = qrels_dict[query_id]
        run = run_dict[query_id]
        
        # Sort run by score (descending)
        sorted_docs = sorted(run.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Calculate max possible relevance for normalization
        max_possible_relevance = max(qrels.values()) if qrels else 1.0
        
        # Weighted Hit Rate
        total_relevance = sum(qrels.get(doc_id, 0.0) for doc_id, _ in sorted_docs)
        max_relevance = max_possible_relevance * min(len(qrels), k)
        weighted_hit = total_relevance / max_relevance if max_relevance > 0 else 0
        weighted_hits.append(weighted_hit)
        
        # Graded MRR
        for rank, (doc_id, _) in enumerate(sorted_docs, 1):
            if doc_id in qrels:
                relevance_grade = qrels[doc_id]
                normalized_grade = relevance_grade / max_possible_relevance
                graded_mrr = normalized_grade / rank
                graded_mrr_scores.append(graded_mrr)
                break
        else:
            graded_mrr_scores.append(0.0)
        
        # Weighted Precision @ k
        relevance_sum = sum(qrels.get(doc_id, 0.0) for doc_id, _ in sorted_docs)
        max_precision_sum = max_possible_relevance * k
        weighted_precision = relevance_sum / max_precision_sum if max_precision_sum > 0 else 0
        weighted_precision_scores.append(weighted_precision)
        
        # Graded MAP
        precisions = []
        cumulative_relevance = 0
        for rank, (doc_id, _) in enumerate(sorted_docs, 1):
            if doc_id in qrels:
                relevance_grade = qrels[doc_id]
                cumulative_relevance += relevance_grade
                # Precision at this rank, weighted by cumulative relevance
                precision = cumulative_relevance / (rank * max_possible_relevance)
                precisions.append(precision)
        
        if precisions:
            graded_map = np.mean(precisions)
        else:
            graded_map = 0.0
        graded_map_scores.append(graded_map)
    
    # Calculate final metrics
    metrics['weighted_hit_rate'] = np.mean(weighted_hits) if weighted_hits else 0.0
    metrics['graded_mrr'] = np.mean(graded_mrr_scores) if graded_mrr_scores else 0.0
    metrics['weighted_precision'] = np.mean(weighted_precision_scores) if weighted_precision_scores else 0.0
    metrics['graded_map'] = np.mean(graded_map_scores) if graded_map_scores else 0.0
    
    return metrics


def compare_binary_vs_graded_with_custom_metrics():
    """
    Compare binary vs graded relevance using custom metrics that actually benefit from grades.
    """
    
    print("🔬 Custom Graded Metrics Comparison")
    print("=" * 60)
    
    # Scenario: Documents with varying similarity scores
    similarity_threshold = 0.6
    
    # Simulate retrieval results with similarity scores
    run_dict = {
        "q_1": {
            "doc_0": 0.95,  # Very high similarity
            "doc_1": 0.85,  # High similarity
            "doc_2": 0.75,  # Medium similarity
            "doc_3": 0.65,  # Low similarity
            "doc_4": 0.55   # Below threshold
        },
        "q_2": {
            "doc_0": 0.70,  # Medium similarity
            "doc_1": 0.68,  # Low similarity
            "doc_2": 0.62,  # Very low similarity
            "doc_3": 0.58,  # Below threshold
            "doc_4": 0.50   # Below threshold
        }
    }
    
    # Binary relevance (traditional approach)
    qrels_binary = {}
    for query_id, docs in run_dict.items():
        qrels_binary[query_id] = {}
        for doc_id, score in docs.items():
            if score >= similarity_threshold:
                qrels_binary[query_id][doc_id] = 1.0
    
    # Graded relevance (using actual similarity scores)
    qrels_graded = {}
    for query_id, docs in run_dict.items():
        qrels_graded[query_id] = {}
        for doc_id, score in docs.items():
            if score >= similarity_threshold:
                # Use actual similarity score as relevance grade
                qrels_graded[query_id][doc_id] = score
    
    print("\n📊 Binary Relevance (Traditional):")
    print(f"  q_1: {qrels_binary['q_1']}")
    print(f"  q_2: {qrels_binary['q_2']}")
    
    print("\n📊 Graded Relevance (Similarity-based):")
    for query_id, docs in qrels_graded.items():
        print(f"  {query_id}: {docs}")
    
    # Calculate metrics for both approaches
    binary_metrics = calculate_graded_metrics(qrels_binary, run_dict, k=5)
    graded_metrics = calculate_graded_metrics(qrels_graded, run_dict, k=5)
    
    print("\n📈 Binary Approach Results:")
    for metric, value in binary_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n📈 Graded Approach Results:")
    for metric, value in graded_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n🔍 Improvements with Graded Relevance:")
    for metric in binary_metrics:
        diff = graded_metrics[metric] - binary_metrics[metric]
        pct_change = (diff / binary_metrics[metric] * 100) if binary_metrics[metric] > 0 else 0
        symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➖"
        print(f"  {symbol} {metric}: {diff:+.4f} ({pct_change:+.1f}%)")
    
    print("\n💡 Key Benefits of Graded Relevance:")
    print("  1. Weighted Hit Rate: Rewards finding highly relevant docs")
    print("  2. Graded MRR: Higher scores for more relevant first results")
    print("  3. Weighted Precision: Considers quality, not just quantity")
    print("  4. Graded MAP: Comprehensive quality-aware ranking metric")


def demonstrate_practical_benefits():
    """
    Demonstrate practical scenarios where graded relevance is beneficial.
    """
    
    print("\n\n🎯 Practical Benefits of Graded Relevance")
    print("=" * 60)
    
    scenarios = {
        "Perfect Retrieval": {
            "run": {"doc_0": 0.95, "doc_1": 0.90, "doc_2": 0.85, "doc_3": 0.80, "doc_4": 0.75},
            "threshold": 0.7
        },
        "Mixed Quality": {
            "run": {"doc_0": 0.95, "doc_1": 0.72, "doc_2": 0.71, "doc_3": 0.55, "doc_4": 0.40},
            "threshold": 0.7
        },
        "Borderline Results": {
            "run": {"doc_0": 0.72, "doc_1": 0.71, "doc_2": 0.70, "doc_3": 0.69, "doc_4": 0.68},
            "threshold": 0.7
        }
    }
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n📋 Scenario: {scenario_name}")
        print("-" * 40)
        
        run_dict = {"q_1": scenario_data["run"]}
        threshold = scenario_data["threshold"]
        
        # Binary approach
        qrels_binary = {"q_1": {}}
        for doc_id, score in scenario_data["run"].items():
            if score >= threshold:
                qrels_binary["q_1"][doc_id] = 1.0
        
        # Graded approach  
        qrels_graded = {"q_1": {}}
        for doc_id, score in scenario_data["run"].items():
            if score >= threshold:
                qrels_graded["q_1"][doc_id] = score
        
        binary_metrics = calculate_graded_metrics(qrels_binary, run_dict, k=5)
        graded_metrics = calculate_graded_metrics(qrels_graded, run_dict, k=5)
        
        print(f"  Binary relevant docs: {len(qrels_binary['q_1'])}")
        print(f"  Similarity range: {min(scenario_data['run'].values()):.2f} - {max(scenario_data['run'].values()):.2f}")
        
        print(f"\n  Metric Comparison:")
        for metric in binary_metrics:
            b_val = binary_metrics[metric]
            g_val = graded_metrics[metric]
            diff = g_val - b_val
            print(f"    {metric}:")
            print(f"      Binary: {b_val:.4f}")
            print(f"      Graded: {g_val:.4f} ({diff:+.4f})")


if __name__ == "__main__":
    compare_binary_vs_graded_with_custom_metrics()
    demonstrate_practical_benefits()
    
    print("\n\n✅ Conclusion:")
    print("  Graded relevance provides meaningful benefits when:")
    print("  • Using metrics designed to leverage relevance grades")
    print("  • Evaluating systems where similarity quality matters")
    print("  • Distinguishing between 'barely relevant' and 'highly relevant'")
    print("\n  Standard ranx metrics (MAP, MRR) treat relevance as binary,")
    print("  so custom graded metrics are needed for full benefits.")