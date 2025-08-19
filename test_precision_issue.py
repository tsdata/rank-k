#!/usr/bin/env python3
"""
Test precision calculation issue in reference_based mode.
"""

from ranx import Qrels, Run, evaluate


def test_current_approach():
    """Test current approach - only found refs in run."""
    print("❌ Current Approach: Only found refs in run")
    print("-" * 50)
    
    # 2 reference docs
    qrels = {
        "q1": {"ref_0": 1.0, "ref_1": 1.0}
    }
    
    # Only found refs in run (but we retrieved 5 docs!)
    run = {
        "q1": {"ref_0": 0.9, "ref_1": 0.8}
    }
    
    qrels_obj = Qrels(qrels)
    run_obj = Run(run)
    
    results = evaluate(qrels_obj, run_obj, ["precision@5", "recall@5"])
    print(f"Qrels: {qrels}")
    print(f"Run: {run}")
    print(f"Precision@5: {results['precision@5']:.3f}")  # Will be 2/2 = 1.0 (WRONG!)
    print(f"Recall@5: {results['recall@5']:.3f}")  # Will be 2/2 = 1.0
    print("⚠️ Problem: Precision shows 1.0 but we retrieved 5 docs, only 2 relevant!")


def test_proper_approach():
    """Test proper approach - include all retrieved docs."""
    print("\n\n✅ Proper Approach: Include all retrieved docs")
    print("-" * 50)
    
    # 2 reference docs
    qrels = {
        "q1": {"ref_0": 1.0, "ref_1": 1.0}
    }
    
    # Include found refs AND false positives
    run = {
        "q1": {
            "ref_0": 0.9,    # Found ref
            "ref_1": 0.8,    # Found ref
            "ret_2": 0.7,    # False positive
            "ret_3": 0.6,    # False positive  
            "ret_4": 0.5     # False positive
        }
    }
    
    qrels_obj = Qrels(qrels)
    run_obj = Run(run)
    
    results = evaluate(qrels_obj, run_obj, ["precision@5", "recall@5"])
    print(f"Qrels: {qrels}")
    print(f"Run: {run}")
    print(f"Precision@5: {results['precision@5']:.3f}")  # Will be 2/5 = 0.4 (CORRECT!)
    print(f"Recall@5: {results['recall@5']:.3f}")  # Will be 2/2 = 1.0
    print("✅ Correct: Precision reflects that only 2 of 5 retrieved were relevant")


def demonstrate_ranking_issue():
    """Show how ranking is affected."""
    print("\n\n📊 Ranking Impact")
    print("-" * 50)
    
    # Scenario: False positive has high similarity
    qrels = {
        "q1": {"ref_0": 1.0, "ref_1": 1.0}
    }
    
    # Include all docs with their similarities
    run = {
        "q1": {
            "ret_0": 0.95,   # False positive but highest similarity!
            "ref_0": 0.85,   # Found ref at position 2
            "ret_1": 0.75,   # False positive
            "ref_1": 0.65,   # Found ref at position 4
            "ret_2": 0.55    # False positive
        }
    }
    
    qrels_obj = Qrels(qrels)
    run_obj = Run(run)
    
    results = evaluate(qrels_obj, run_obj, ["mrr", "map@5", "ndcg@5", "precision@5"])
    print(f"Qrels: {qrels}")
    print(f"Run (with false positives): {run}")
    print(f"MRR: {results['mrr']:.3f}")  # First relevant at position 2
    print(f"MAP@5: {results['map@5']:.3f}")  
    print(f"NDCG@5: {results['ndcg@5']:.3f}")
    print(f"Precision@5: {results['precision@5']:.3f}")  # 2/5 = 0.4
    
    print("\n📊 Without false positives (current approach):")
    run_no_fp = {
        "q1": {
            "ref_0": 0.85,   # Position 1 (but was actually position 2!)
            "ref_1": 0.65    # Position 2 (but was actually position 4!)
        }
    }
    
    run_obj2 = Run(run_no_fp)
    results2 = evaluate(qrels_obj, run_obj2, ["mrr", "map@5", "ndcg@5", "precision@5"])
    print(f"Run (refs only): {run_no_fp}")
    print(f"MRR: {results2['mrr']:.3f}")  # First relevant at position 1 (WRONG!)
    print(f"MAP@5: {results2['map@5']:.3f}")  
    print(f"NDCG@5: {results2['ndcg@5']:.3f}")
    print(f"Precision@5: {results2['precision@5']:.3f}")  # 2/2 = 1.0 (WRONG!)
    
    print("\n⚠️ Problem: Removing false positives changes ranking positions!")


if __name__ == "__main__":
    test_current_approach()
    test_proper_approach()
    demonstrate_ranking_issue()
    
    print("\n" + "=" * 60)
    print("💡 Conclusion:")
    print("  We MUST include false positives in run_dict for:")
    print("  1. Correct precision calculation")
    print("  2. Correct ranking positions")
    print("  3. Proper MRR, MAP, NDCG calculations")
    print("\n  Solution: Add non-matching retrieved docs as 'ret_X' in run")