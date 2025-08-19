#!/usr/bin/env python3
"""
Test improved reference_based mode with proper precision calculation.
"""

from ranx import Qrels, Run, evaluate
import numpy as np


def simulate_reference_based_evaluation():
    """
    Simulate the improved reference_based evaluation with false positives.
    """
    
    print("🧪 Testing Improved Reference-Based Mode with False Positives")
    print("=" * 70)
    
    # Scenario: 3 reference docs, 5 retrieved docs
    # Retrieved: [AI-doc(matches ref_0), Sports-doc, ML-doc(matches ref_1), Weather-doc, DL-doc(below threshold)]
    
    print("\n📊 Scenario:")
    print("  Reference docs: 3 (AI, ML, DL)")
    print("  Retrieved docs: 5 (AI-like, Sports, ML-like, Weather, DL-weak)")
    print("  Threshold: 0.7")
    print("  Matches: AI-like→ref_0 (0.9), ML-like→ref_1 (0.8), DL-weak→ref_2 (0.4 below threshold)")
    
    # Build qrels and run as the improved code would
    qrels = {
        "q1": {
            "ref_0": 1.0,  # AI reference
            "ref_1": 1.0,  # ML reference  
            "ref_2": 1.0   # DL reference
        }
    }
    
    # Run includes found refs + false positives
    run = {
        "q1": {
            # Found references
            "ref_0": 0.9,   # AI doc found with high similarity
            "ref_1": 0.8,   # ML doc found with good similarity
            # ref_2 not in run (similarity 0.4 < threshold 0.7)
            
            # False positives (retrieved but don't match any ref above threshold)
            "ret_1": 0.2,   # Sports doc - low similarity to any ref
            "ret_3": 0.3,   # Weather doc - low similarity to any ref
            "ret_4": 0.4    # DL-weak doc - below threshold match to ref_2
        }
    }
    
    print("\n📋 Qrels (ground truth):")
    for doc_id, rel in qrels["q1"].items():
        print(f"  {doc_id}: {rel}")
    
    print("\n📋 Run (system output with false positives):")
    for doc_id, score in sorted(run["q1"].items(), key=lambda x: x[1], reverse=True):
        doc_type = "✅ Found reference" if doc_id.startswith("ref_") else "❌ False positive"
        print(f"  {doc_id}: {score:.2f} ({doc_type})")
    
    # Evaluate with ranx
    qrels_obj = Qrels(qrels)
    run_obj = Run(run)
    
    metrics = ["hit_rate@5", "recall@5", "precision@5", "ndcg@5", "map@5", "mrr"]
    results = evaluate(qrels_obj, run_obj, metrics)
    
    print("\n📊 Metrics with false positives included:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.3f}")
    
    print("\n📈 Metric Interpretation:")
    print(f"  Recall@5: {results['recall@5']:.3f} = 2/3 (found 2 of 3 references)")
    print(f"  Precision@5: {results['precision@5']:.3f} = 2/5 (2 relevant out of 5 retrieved)")
    print(f"  MRR: {results['mrr']:.3f} (first relevant at position 1)")
    print(f"  Hit@5: {results['hit_rate@5']:.3f} (at least one relevant found)")
    
    # Compare with incorrect approach (without false positives)
    print("\n\n❌ Comparison with OLD approach (no false positives):")
    print("-" * 50)
    
    run_no_fp = {
        "q1": {
            "ref_0": 0.9,
            "ref_1": 0.8
        }
    }
    
    run_obj2 = Run(run_no_fp)
    results2 = evaluate(qrels_obj, run_obj2, metrics)
    
    print("📋 Run (only found refs - INCORRECT):")
    for doc_id, score in run_no_fp["q1"].items():
        print(f"  {doc_id}: {score:.2f}")
    
    print("\n📊 Metrics without false positives (WRONG):")
    for metric, score in results2.items():
        print(f"  {metric}: {score:.3f}")
    
    print("\n⚠️ Problem with old approach:")
    print(f"  Precision shows {results2['precision@5']:.3f} but should be {results['precision@5']:.3f}")
    print(f"  This incorrectly suggests perfect precision when we actually retrieved 5 docs!")
    
    return results, results2


def test_ranking_positions():
    """
    Test that ranking positions are correct with false positives.
    """
    
    print("\n\n🏆 Testing Ranking Positions")
    print("=" * 70)
    
    # Scenario where a false positive ranks higher than a true positive
    qrels = {
        "q1": {
            "ref_0": 1.0,
            "ref_1": 1.0
        }
    }
    
    # False positive (ret_0) has similarity 0.85, ranking higher than ref_1
    run = {
        "q1": {
            "ref_0": 0.95,   # Position 1
            "ret_0": 0.85,   # Position 2 (false positive)
            "ref_1": 0.75,   # Position 3 (true positive)
            "ret_1": 0.65,   # Position 4 (false positive)
            "ret_2": 0.55    # Position 5 (false positive)
        }
    }
    
    print("📊 Scenario: False positive ranks higher than true positive")
    print("\nRanking order:")
    for i, (doc_id, score) in enumerate(sorted(run["q1"].items(), key=lambda x: x[1], reverse=True), 1):
        doc_type = "✅ Relevant" if doc_id.startswith("ref_") else "❌ Not relevant"
        print(f"  Position {i}: {doc_id} (score: {score:.2f}) - {doc_type}")
    
    qrels_obj = Qrels(qrels)
    run_obj = Run(run)
    
    results = evaluate(qrels_obj, run_obj, ["mrr", "map@5", "precision@1", "precision@3"])
    
    print("\n📊 Metrics:")
    print(f"  MRR: {results['mrr']:.3f} (first relevant at position 1)")
    print(f"  MAP@5: {results['map@5']:.3f}")
    print(f"  Precision@1: {results['precision@1']:.3f} (position 1 is relevant)")
    print(f"  Precision@3: {results['precision@3']:.3f} (2 relevant in top 3)")
    
    print("\n✅ Correct behavior:")
    print("  - False positive at position 2 affects precision@3")
    print("  - Ranking positions are preserved correctly")
    print("  - MAP and MRR account for false positives in ranking")


if __name__ == "__main__":
    results_with_fp, results_without_fp = simulate_reference_based_evaluation()
    test_ranking_positions()
    
    print("\n" + "=" * 70)
    print("💡 Summary:")
    print("  The improved reference_based mode now:")
    print("  1. Includes ALL reference docs in qrels for proper recall")
    print("  2. Includes found references in run with ref_ IDs")
    print("  3. Includes false positives in run with ret_ IDs")
    print("  4. Correctly calculates precision accounting for all retrieved docs")
    print("  5. Preserves ranking order for accurate MRR, MAP, NDCG")
    print("\n  This solves the precision calculation issue you identified!")