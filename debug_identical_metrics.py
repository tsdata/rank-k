#!/usr/bin/env python3
"""
Debug why all metrics return identical values.
"""

from ranx import Qrels, Run, evaluate


def debug_identical_metrics():
    """Debug why all ranx metrics return the same value."""
    
    print("🔍 Debugging Identical Metrics Issue | 동일한 메트릭 값 문제 디버깅")
    print("="*70)
    
    # Test case 1: Normal case (should have different metrics)
    print("\n1️⃣ Normal Test Case | 정상 테스트 케이스:")
    
    qrels_dict_normal = {
        "q_1": {"doc_0": 1.0, "doc_1": 1.0},
        "q_2": {"doc_2": 1.0},
        "q_3": {"doc_0": 1.0, "doc_3": 1.0}
    }
    
    run_dict_normal = {
        "q_1": {"doc_0": 0.9, "doc_1": 0.8, "doc_2": 0.7, "doc_3": 0.6, "doc_4": 0.5},
        "q_2": {"doc_0": 0.6, "doc_1": 0.7, "doc_2": 0.9, "doc_3": 0.5, "doc_4": 0.4},
        "q_3": {"doc_0": 0.8, "doc_1": 0.6, "doc_2": 0.5, "doc_3": 0.7, "doc_4": 0.4}
    }
    
    test_ranx_evaluation(qrels_dict_normal, run_dict_normal, "Normal")
    
    # Test case 2: Perfect case (might cause identical metrics)
    print("\n2️⃣ Perfect Test Case | 완벽한 테스트 케이스:")
    
    qrels_dict_perfect = {
        "q_1": {"doc_0": 1.0, "doc_1": 1.0, "doc_2": 1.0, "doc_3": 1.0, "doc_4": 1.0},
        "q_2": {"doc_0": 1.0, "doc_1": 1.0, "doc_2": 1.0, "doc_3": 1.0, "doc_4": 1.0},
        "q_3": {"doc_0": 1.0, "doc_1": 1.0, "doc_2": 1.0, "doc_3": 1.0, "doc_4": 1.0}
    }
    
    run_dict_perfect = {
        "q_1": {"doc_0": 0.9, "doc_1": 0.8, "doc_2": 0.7, "doc_3": 0.6, "doc_4": 0.5},
        "q_2": {"doc_0": 0.9, "doc_1": 0.8, "doc_2": 0.7, "doc_3": 0.6, "doc_4": 0.5},
        "q_3": {"doc_0": 0.9, "doc_1": 0.8, "doc_2": 0.7, "doc_3": 0.6, "doc_4": 0.5}
    }
    
    test_ranx_evaluation(qrels_dict_perfect, run_dict_perfect, "Perfect")
    
    # Test case 3: Graded relevance case
    print("\n3️⃣ Graded Relevance Test Case | 등급별 관련성 테스트 케이스:")
    
    qrels_dict_graded = {
        "q_1": {"doc_0": 3.0, "doc_1": 2.0},
        "q_2": {"doc_2": 2.5},
        "q_3": {"doc_0": 1.5, "doc_3": 3.0}
    }
    
    test_ranx_evaluation(qrels_dict_graded, run_dict_normal, "Graded")


def test_ranx_evaluation(qrels_dict, run_dict, case_name):
    """Test ranx evaluation with given qrels and run."""
    
    try:
        qrels = Qrels(qrels_dict)
        run = Run(run_dict)
        
        metrics = ["hit_rate@5", "ndcg@5", "map@5", "mrr"]
        results = evaluate(qrels, run, metrics)
        
        print(f"\n{case_name} Results:")
        for metric, score in results.items():
            print(f"  {metric}: {score:.4f}")
        
        # Check if all metrics are identical
        values = list(results.values())
        if len(set(values)) == 1:
            print(f"  ⚠️ WARNING: All metrics are identical!")
        else:
            print(f"  ✅ Metrics show variation (good)")
            
        # Show qrels and run structure
        print(f"\nQrels structure:")
        for query_id, docs in qrels_dict.items():
            print(f"  {query_id}: {len(docs)} docs, values: {list(docs.values())}")
            
        print(f"Run structure:")
        for query_id, docs in run_dict.items():
            print(f"  {query_id}: {len(docs)} docs, scores: {[f'{v:.2f}' for v in list(docs.values())[:3]]}...")
        
    except Exception as e:
        print(f"❌ Error in {case_name}: {e}")


if __name__ == "__main__":
    debug_identical_metrics()