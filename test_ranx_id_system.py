#!/usr/bin/env python3
"""
Test how ranx handles different ID systems.
"""

from ranx import Qrels, Run, evaluate


def test_matching_ids():
    """Test with matching IDs between qrels and run."""
    print("✅ Test 1: Matching IDs")
    print("-" * 40)
    
    qrels = {
        "q1": {"doc_0": 1.0, "doc_1": 1.0, "doc_2": 1.0}
    }
    
    run = {
        "q1": {"doc_0": 0.9, "doc_1": 0.8, "doc_3": 0.7}  # doc_2 not found, doc_3 is false positive
    }
    
    qrels_obj = Qrels(qrels)
    run_obj = Run(run)
    
    results = evaluate(qrels_obj, run_obj, ["recall@5", "precision@5"])
    print(f"Qrels: {qrels}")
    print(f"Run: {run}")
    print(f"Recall@5: {results['recall@5']:.3f}")  # Should be 2/3 = 0.667
    print(f"Precision@5: {results['precision@5']:.3f}")  # Should be 2/3 = 0.667


def test_mismatched_ids():
    """Test with different ID systems - THIS WON'T WORK."""
    print("\n\n❌ Test 2: Mismatched IDs (ref_ vs doc_)")
    print("-" * 40)
    
    qrels = {
        "q1": {"ref_0": 1.0, "ref_1": 1.0}  # Reference IDs
    }
    
    run = {
        "q1": {"doc_0": 0.9, "doc_1": 0.8}  # Retrieved IDs
    }
    
    qrels_obj = Qrels(qrels)
    run_obj = Run(run)
    
    results = evaluate(qrels_obj, run_obj, ["recall@5", "precision@5"])
    print(f"Qrels: {qrels}")
    print(f"Run: {run}")
    print(f"Recall@5: {results['recall@5']:.3f}")  # Will be 0.0 - no matching IDs!
    print(f"Precision@5: {results['precision@5']:.3f}")  # Will be 0.0


def test_proper_reference_based():
    """Test proper reference-based evaluation with same IDs."""
    print("\n\n✅ Test 3: Proper Reference-Based (same IDs)")
    print("-" * 40)
    
    # All reference documents in qrels
    qrels = {
        "q1": {
            "ref_0": 1.0,  # Reference doc 0
            "ref_1": 1.0,  # Reference doc 1
            "ref_2": 1.0   # Reference doc 2
        }
    }
    
    # Only found reference docs in run (with same IDs)
    run = {
        "q1": {
            "ref_0": 0.95,  # Found ref 0 with high similarity
            "ref_2": 0.85   # Found ref 2 with good similarity
            # ref_1 not found - will affect recall
        }
    }
    
    qrels_obj = Qrels(qrels)
    run_obj = Run(run)
    
    results = evaluate(qrels_obj, run_obj, ["recall@5", "precision@5", "hit_rate@5"])
    print(f"Qrels (all refs): {qrels}")
    print(f"Run (found refs): {run}")
    print(f"Recall@5: {results['recall@5']:.3f}")  # Should be 2/3 = 0.667
    print(f"Precision@5: {results['precision@5']:.3f}")  # Should be 2/2 = 1.000
    print(f"Hit@5: {results['hit_rate@5']:.3f}")  # Should be 1.0


def demonstrate_solution():
    """Demonstrate the recommended solution."""
    print("\n\n🚀 Recommended Solution")
    print("=" * 60)
    
    print("\nFor reference_based mode:")
    print("1. Use 'ref_X' IDs for all documents (both qrels and run)")
    print("2. Qrels: Include ALL reference documents")
    print("3. Run: Include only FOUND reference documents")
    print("4. This allows proper recall calculation")
    
    print("\nExample with 3 reference docs, 2 found:")
    
    # Scenario: 3 references, but only 2 are found
    qrels = {
        "q1": {
            "ref_0": 1.0,  # AI document
            "ref_1": 1.0,  # ML document  
            "ref_2": 1.0   # DL document
        }
    }
    
    # Retrieved: [AI-like, Sports, ML-like, News, Weather]
    # Only AI-like and ML-like match references
    run = {
        "q1": {
            "ref_0": 0.95,  # AI document found
            "ref_1": 0.88,  # ML document found
            # ref_2 not found (no match above threshold)
        }
    }
    
    qrels_obj = Qrels(qrels)
    run_obj = Run(run)
    
    metrics = ["hit_rate@5", "recall@5", "precision@5", "ndcg@5", "map@5", "mrr"]
    results = evaluate(qrels_obj, run_obj, metrics)
    
    print(f"\nQrels: {qrels}")
    print(f"Run: {run}")
    print("\nMetrics:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.3f}")
    
    print("\n📊 Interpretation:")
    print(f"  - Recall: 2/3 = 0.667 (found 2 of 3 references)")
    print(f"  - Precision: 2/2 = 1.000 (all found items are relevant)")
    print(f"  - This properly reflects incomplete retrieval")


if __name__ == "__main__":
    test_matching_ids()
    test_mismatched_ids()
    test_proper_reference_based()
    demonstrate_solution()
    
    print("\n" + "=" * 60)
    print("💡 Key Insight:")
    print("  ranx requires matching IDs between qrels and run.")
    print("  We should use 'ref_X' for reference-based evaluation")
    print("  and include all refs in qrels, only found refs in run.")