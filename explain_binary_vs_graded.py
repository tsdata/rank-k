#!/usr/bin/env python3
"""
Explain why order doesn't matter with binary relevance (use_graded_relevance=False).
"""

from ranx import Qrels, Run, evaluate
import numpy as np


def demonstrate_binary_relevance():
    """
    Show that with binary relevance, order doesn't affect most metrics.
    """
    
    print("🔍 Binary Relevance (use_graded_relevance=False)")
    print("=" * 70)
    print("\n📌 With binary relevance, documents are either:")
    print("  - Relevant (1.0) if similarity >= threshold")
    print("  - Not relevant (0.0) if similarity < threshold")
    print("\n⚠️ The actual similarity score is NOT used for relevance grading!")
    print("=" * 70)
    
    # Same relevant documents, different order
    qrels = {
        "q1": {
            "doc_0": 1.0,  # Relevant (binary)
            "doc_2": 1.0,  # Relevant (binary)
            # doc_1, doc_3, doc_4 are not in qrels (not relevant)
        }
    }
    
    print("\n📊 Scenario: 2 relevant docs out of 5 retrieved")
    print("Qrels (ground truth):", qrels["q1"])
    
    # Test 1: Bad ranking (relevant docs at end)
    print("\n" + "-" * 50)
    print("❌ Bad Ranking: Relevant docs at positions 4 and 5")
    run_bad = {
        "q1": {
            "doc_1": 0.9,  # Position 1 - NOT relevant (high score but not in qrels)
            "doc_3": 0.8,  # Position 2 - NOT relevant
            "doc_4": 0.7,  # Position 3 - NOT relevant
            "doc_0": 0.6,  # Position 4 - RELEVANT ✓
            "doc_2": 0.5   # Position 5 - RELEVANT ✓
        }
    }
    
    print("Run (ranked list):")
    for doc_id, score in sorted(run_bad["q1"].items(), key=lambda x: x[1], reverse=True):
        relevant = "✓ RELEVANT" if doc_id in qrels["q1"] else "✗ Not relevant"
        print(f"  {doc_id}: {score:.1f} - {relevant}")
    
    qrels_obj = Qrels(qrels)
    run_obj = Run(run_bad)
    
    metrics = ["hit_rate@5", "recall@5", "precision@5", "ndcg@5", "map@5", "mrr"]
    results_bad = evaluate(qrels_obj, run_obj, metrics)
    
    print("\nMetrics:")
    for metric, score in results_bad.items():
        print(f"  {metric}: {score:.3f}")
    
    # Test 2: Good ranking (relevant docs at beginning)
    print("\n" + "-" * 50)
    print("✅ Good Ranking: Relevant docs at positions 1 and 2")
    run_good = {
        "q1": {
            "doc_0": 0.95,  # Position 1 - RELEVANT ✓
            "doc_2": 0.90,  # Position 2 - RELEVANT ✓
            "doc_1": 0.4,   # Position 3 - NOT relevant
            "doc_3": 0.3,   # Position 4 - NOT relevant
            "doc_4": 0.2    # Position 5 - NOT relevant
        }
    }
    
    print("Run (ranked list):")
    for doc_id, score in sorted(run_good["q1"].items(), key=lambda x: x[1], reverse=True):
        relevant = "✓ RELEVANT" if doc_id in qrels["q1"] else "✗ Not relevant"
        print(f"  {doc_id}: {score:.1f} - {relevant}")
    
    run_obj2 = Run(run_good)
    results_good = evaluate(qrels_obj, run_obj2, metrics)
    
    print("\nMetrics:")
    for metric, score in results_good.items():
        print(f"  {metric}: {score:.3f}")
    
    # Compare results
    print("\n" + "=" * 70)
    print("📊 COMPARISON: Bad Ranking vs Good Ranking")
    print("-" * 70)
    print(f"{'Metric':<15} {'Bad Ranking':<15} {'Good Ranking':<15} {'Difference':<15} {'Order Matters?'}")
    print("-" * 70)
    
    for metric in metrics:
        bad = results_bad[metric]
        good = results_good[metric]
        diff = good - bad
        matters = "YES ✓" if abs(diff) > 0.001 else "NO ✗"
        print(f"{metric:<15} {bad:<15.3f} {good:<15.3f} {diff:+15.3f} {matters}")
    
    print("\n💡 Key Insights:")
    print("  • Hit@5: SAME (1.0) - Both found at least one relevant doc within top 5")
    print("  • Recall@5: SAME (1.0) - Both found all 2 relevant docs")
    print("  • Precision@5: SAME (0.4) - Both have 2 relevant out of 5")
    print("  • MRR: DIFFERENT - First relevant doc position matters!")
    print("  • MAP@5: DIFFERENT - Average precision at each relevant doc matters!")
    print("  • NDCG@5: DIFFERENT - Position-weighted relevance matters!")


def demonstrate_graded_relevance():
    """
    Show that with graded relevance, order DOES matter significantly.
    """
    
    print("\n\n🔍 Graded Relevance (use_graded_relevance=True)")
    print("=" * 70)
    print("\n📌 With graded relevance, documents get scores from 1.0 to 10.0:")
    print("  - Based on actual similarity score")
    print("  - Higher similarity = higher relevance grade")
    print("  - Position of higher-grade docs matters MORE")
    print("=" * 70)
    
    # Graded relevance based on similarity
    qrels_graded = {
        "q1": {
            "doc_0": 8.0,  # High relevance (similarity was 0.9)
            "doc_2": 5.0,  # Medium relevance (similarity was 0.7)
        }
    }
    
    print("\n📊 Scenario: 2 relevant docs with different grades")
    print("Qrels (graded):", qrels_graded["q1"])
    
    # Same runs as before but now grades matter
    print("\n" + "-" * 50)
    print("❌ Bad Ranking: High-grade doc at position 4")
    run_bad_graded = {
        "q1": {
            "doc_1": 0.95,  # Position 1 - not relevant
            "doc_3": 0.85,  # Position 2 - not relevant
            "doc_4": 0.75,  # Position 3 - not relevant
            "doc_0": 0.65,  # Position 4 - HIGH GRADE (8.0)
            "doc_2": 0.55   # Position 5 - MEDIUM GRADE (5.0)
        }
    }
    
    qrels_obj = Qrels(qrels_graded)
    run_obj = Run(run_bad_graded)
    results_bad_graded = evaluate(qrels_obj, run_obj, ["ndcg@5", "map@5"])
    
    print("\n✅ Good Ranking: High-grade doc at position 1")
    run_good_graded = {
        "q1": {
            "doc_0": 0.95,  # Position 1 - HIGH GRADE (8.0)
            "doc_2": 0.85,  # Position 2 - MEDIUM GRADE (5.0)
            "doc_1": 0.45,  # Position 3 - not relevant
            "doc_3": 0.35,  # Position 4 - not relevant
            "doc_4": 0.25   # Position 5 - not relevant
        }
    }
    
    run_obj2 = Run(run_good_graded)
    results_good_graded = evaluate(qrels_obj, run_obj2, ["ndcg@5", "map@5"])
    
    print("\n📊 Graded Relevance Impact:")
    print(f"NDCG@5: Bad={results_bad_graded['ndcg@5']:.3f}, Good={results_good_graded['ndcg@5']:.3f}")
    print(f"MAP@5: Bad={results_bad_graded['map@5']:.3f}, Good={results_good_graded['map@5']:.3f}")
    print("\n✅ With graded relevance, order matters MUCH MORE!")


def explain_ranx_k_implementation():
    """
    Explain how ranx-k implements binary vs graded relevance.
    """
    
    print("\n\n📚 How ranx-k Handles This")
    print("=" * 70)
    
    print("\n1️⃣ Binary Relevance (use_graded_relevance=False):")
    print("-" * 50)
    print("""
In similarity_ranx.py:
```python
if max_similarity >= similarity_threshold:
    qrels_dict[query_id][doc_id] = 1.0  # Always 1.0 for binary
```

Result:
- Doc A: similarity=0.9 → relevance=1.0
- Doc B: similarity=0.8 → relevance=1.0  
- Doc C: similarity=0.7 → relevance=1.0
- ALL get same relevance score (1.0)!
- Order doesn't affect Hit@k, Recall, Precision
""")
    
    print("\n2️⃣ Graded Relevance (use_graded_relevance=True):")
    print("-" * 50)
    print("""
In similarity_ranx.py:
```python
if max_similarity >= similarity_threshold:
    # Scale similarity [threshold, 1.0] to relevance [1.0, 10.0]
    scaled_score = 1.0 + (max_similarity - threshold) * 9.0 / (1.0 - threshold)
    qrels_dict[query_id][doc_id] = scaled_score
```

Result with threshold=0.7:
- Doc A: similarity=0.9 → relevance=7.0
- Doc B: similarity=0.8 → relevance=4.0
- Doc C: similarity=0.7 → relevance=1.0
- Different grades → order MATTERS for NDCG, MAP!
""")


def show_metric_formulas():
    """
    Show why some metrics care about order and others don't.
    """
    
    print("\n\n📐 Why Some Metrics Care About Order")
    print("=" * 70)
    
    print("\n🎯 Order-INSENSITIVE with Binary Relevance:")
    print("-" * 50)
    
    print("\n1. Hit Rate@k:")
    print("   Formula: 1 if any relevant doc in top k, else 0")
    print("   Example: [0,0,1,0,0] = 1.0, [1,0,0,0,0] = 1.0")
    print("   → SAME result regardless of position!")
    
    print("\n2. Recall@k:")
    print("   Formula: (relevant docs found) / (total relevant docs)")
    print("   Example: Found 2 of 3 = 0.667")
    print("   → Doesn't matter WHERE they are in top k!")
    
    print("\n3. Precision@k:")
    print("   Formula: (relevant docs in top k) / k")
    print("   Example: 2 relevant in top 5 = 0.4")
    print("   → Doesn't matter WHERE in top k!")
    
    print("\n\n🎯 Order-SENSITIVE even with Binary Relevance:")
    print("-" * 50)
    
    print("\n1. MRR (Mean Reciprocal Rank):")
    print("   Formula: 1 / (position of first relevant doc)")
    print("   Example: First relevant at position 1 → MRR = 1.0")
    print("           First relevant at position 4 → MRR = 0.25")
    print("   → Position of FIRST relevant doc matters!")
    
    print("\n2. MAP (Mean Average Precision):")
    print("   Formula: Average of precision at each relevant doc position")
    print("   Example: Relevant at positions [1,2] → MAP = (1/1 + 2/2)/2 = 1.0")
    print("           Relevant at positions [4,5] → MAP = (1/4 + 2/5)/2 = 0.325")
    print("   → Position of ALL relevant docs matters!")
    
    print("\n3. NDCG (Normalized Discounted Cumulative Gain):")
    print("   Formula: Sum of (relevance / log2(position+1)) normalized")
    print("   With binary: relevance=1, but position discount still applies")
    print("   Example: Relevant at [1,2] gets higher score than [4,5]")
    print("   → Position weighting affects score!")


if __name__ == "__main__":
    demonstrate_binary_relevance()
    demonstrate_graded_relevance()
    explain_ranx_k_implementation()
    show_metric_formulas()
    
    print("\n\n" + "=" * 70)
    print("🎯 SUMMARY")
    print("=" * 70)
    print("""
With use_graded_relevance=False (Binary):
- All relevant docs get score = 1.0
- Hit@k, Recall, Precision: Order DOESN'T matter
- MRR, MAP, NDCG: Order still matters (but less)

With use_graded_relevance=True (Graded):
- Relevant docs get scores 1.0-10.0 based on similarity
- ALL metrics become more sensitive to order
- Better docs at top = much higher scores

💡 For reranking evaluation, ALWAYS use:
   - use_graded_relevance=True
   - Focus on MRR, MAP, NDCG metrics
""")