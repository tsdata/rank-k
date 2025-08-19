#!/usr/bin/env python3
"""
Debug script to diagnose graded relevance issue.
"""

import numpy as np
from typing import List, Dict
from ranx import Qrels, Run, evaluate


def debug_ranx_data_structures():
    """Debug what qrels and run should look like for ranx to work properly."""
    
    print("🔍 Debugging ranx data structures | ranx 데이터 구조 디버깅")
    print("="*60)
    
    # Test 1: Simple working example
    print("\n1️⃣ Simple Working Example | 간단한 작동 예제")
    
    qrels_dict = {
        "q_1": {
            "doc_0": 1.0,  # Binary relevance
            "doc_1": 1.0
        },
        "q_2": {
            "doc_0": 1.0,
            "doc_2": 1.0
        }
    }
    
    run_dict = {
        "q_1": {
            "doc_0": 0.9,  # High score
            "doc_1": 0.7,  # Medium score
            "doc_2": 0.5,  # Low score
            "doc_3": 0.3   # Very low score
        },
        "q_2": {
            "doc_0": 0.8,
            "doc_1": 0.6,
            "doc_2": 0.9,  # High score
            "doc_3": 0.4
        }
    }
    
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)
    
    metrics = ["hit_rate@5", "ndcg@5", "map@5", "mrr"]
    results = evaluate(qrels, run, metrics)
    
    print("Binary relevance results:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}")
    
    # Test 2: Graded relevance example (continuous scores)
    print("\n2️⃣ Graded Relevance Example (continuous) | 등급별 관련성 예제 (연속값)")
    
    qrels_dict_graded = {
        "q_1": {
            "doc_0": 0.9,  # High relevance (similarity score)
            "doc_1": 0.7   # Medium relevance (similarity score)
        },
        "q_2": {
            "doc_0": 0.8,  # High relevance
            "doc_2": 0.9   # Very high relevance
        }
    }
    
    # Same run data
    qrels_graded = Qrels(qrels_dict_graded)
    run_graded = Run(run_dict)
    
    results_graded = evaluate(qrels_graded, run_graded, metrics)
    
    print("Graded relevance results (continuous scores):")
    for metric, score in results_graded.items():
        print(f"  {metric}: {score:.4f}")
    
    # Test 2b: Graded relevance example (integer grades)
    print("\n2b️⃣ Graded Relevance Example (integer) | 등급별 관련성 예제 (정수값)")
    
    qrels_dict_graded_int = {
        "q_1": {
            "doc_0": 3,  # High relevance (grade 3)
            "doc_1": 2   # Medium relevance (grade 2)
        },
        "q_2": {
            "doc_0": 2,  # Medium relevance
            "doc_2": 3   # High relevance
        }
    }
    
    qrels_graded_int = Qrels(qrels_dict_graded_int)
    results_graded_int = evaluate(qrels_graded_int, run_graded, metrics)
    
    print("Graded relevance results (integer grades):")
    for metric, score in results_graded_int.items():
        print(f"  {metric}: {score:.4f}")
        
    # Test 2c: Try scaled similarity scores (multiply by 10)
    print("\n2c️⃣ Graded Relevance Example (scaled) | 등급별 관련성 예제 (스케일링)")
    
    qrels_dict_scaled = {
        "q_1": {
            "doc_0": 9.0,  # 0.9 * 10
            "doc_1": 7.0   # 0.7 * 10
        },
        "q_2": {
            "doc_0": 8.0,  # 0.8 * 10
            "doc_2": 9.0   # 0.9 * 10
        }
    }
    
    qrels_scaled = Qrels(qrels_dict_scaled)
    results_scaled = evaluate(qrels_scaled, run_graded, metrics)
    
    print("Graded relevance results (scaled scores):")
    for metric, score in results_scaled.items():
        print(f"  {metric}: {score:.4f}")
    
    # Test 3: Show the data structures
    print("\n3️⃣ Data Structure Analysis | 데이터 구조 분석")
    print("\nQrels (binary):")
    for query_id, docs in qrels_dict.items():
        print(f"  {query_id}: {docs}")
    
    print("\nQrels (graded):")
    for query_id, docs in qrels_dict_graded.items():
        print(f"  {query_id}: {docs}")
        
    print("\nRun:")
    for query_id, docs in run_dict.items():
        print(f"  {query_id}: {docs}")
    
    # Test 4: Key insights
    print("\n4️⃣ Key Insights | 핵심 인사이트")
    print("✅ Both qrels and run must have matching document IDs")
    print("✅ Qrels contains only relevant documents (above threshold)")  
    print("✅ Run contains all retrieved documents with their scores")
    print("✅ Graded relevance uses similarity scores in qrels instead of 1.0")


if __name__ == "__main__":
    debug_ranx_data_structures()