#!/usr/bin/env python3
"""
Debug evaluation metrics issue where all metrics return the same value.
"""

from ranx_k.evaluation import evaluate_with_ranx_similarity

def debug_evaluation_metrics(retriever, questions, reference_contexts):
    """Debug the evaluation metrics issue."""
    
    print("🔍 Debugging Evaluation Metrics Issue | 평가 메트릭 문제 디버깅")
    print("="*70)
    
    # Test with different thresholds
    thresholds = [0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        print(f"\n📊 Testing with threshold: {threshold}")
        print("-" * 40)
        
        try:
            results = evaluate_with_ranx_similarity(
                retriever=retriever,
                questions=questions[:5],  # Use fewer questions for debugging
                reference_contexts=reference_contexts[:5],
                k=5,
                method='embedding',  
                embedding_model="BAAI/bge-m3",
                similarity_threshold=threshold,
                use_graded_relevance=False,  # Start with binary
                evaluation_mode='reference_based'
            )
            
            print("Results:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
                
        except Exception as e:
            print(f"❌ Error with threshold {threshold}: {e}")
    
    # Test graded vs binary
    print(f"\n🆚 Binary vs Graded Relevance Comparison")
    print("-" * 50)
    
    for use_graded in [False, True]:
        mode = "Graded" if use_graded else "Binary"
        print(f"\n{mode} Relevance:")
        
        try:
            results = evaluate_with_ranx_similarity(
                retriever=retriever,
                questions=questions[:3],
                reference_contexts=reference_contexts[:3],
                k=5,
                method='embedding',
                embedding_model="BAAI/bge-m3", 
                similarity_threshold=0.7,  # More reasonable threshold
                use_graded_relevance=use_graded,
                evaluation_mode='reference_based'
            )
            
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
                
        except Exception as e:
            print(f"❌ Error with {mode}: {e}")

if __name__ == "__main__":
    print("Place this function in your evaluation script and call:")
    print("debug_evaluation_metrics(base_retriever, questions, reference_contexts)")