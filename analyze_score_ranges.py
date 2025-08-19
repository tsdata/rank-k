#!/usr/bin/env python3
"""
Analyze the range of final_score values in different scenarios.
"""

def analyze_score_ranges():
    """
    Calculate the possible ranges of final_score values.
    """
    
    print("📊 Analysis of final_score Ranges in ranx-k v0.0.15")
    print("=" * 70)
    
    # Parameters
    base_score = 1000.0
    position_decay = 10.0
    similarity_threshold = 0.5
    
    print(f"\n📌 Parameters:")
    print(f"  base_score = {base_score}")
    print(f"  position_decay = {position_decay}")
    print(f"  similarity_threshold = {similarity_threshold}")
    
    # ============================================================
    # RETRIEVAL_BASED MODE
    # ============================================================
    print("\n\n1️⃣ RETRIEVAL_BASED MODE")
    print("=" * 70)
    print("\nFormula: order_preserving_score = base_score - (j * position_decay) + max_similarity")
    
    print("\n📊 Score Ranges by Position:")
    print("-" * 50)
    print(f"{'Position':<10} {'Min Score':<20} {'Max Score':<20} {'Range'}")
    print("-" * 50)
    
    for j in range(5):  # Positions 0-4 (top 5 documents)
        # max_similarity ranges from 0.0 to 1.0
        min_similarity = 0.0
        max_similarity = 1.0
        
        min_score = base_score - (j * position_decay) + min_similarity
        max_score = base_score - (j * position_decay) + max_similarity
        
        print(f"{j+1:<10} {min_score:<20.1f} {max_score:<20.1f} {max_score - min_score:.1f}")
    
    print("\n💡 Insights:")
    print("  - Position 1: 1000.0 - 1001.0 (highest scores)")
    print("  - Position 5: 960.0 - 961.0 (lowest scores)")
    print("  - Position dominates: 10-point gap between positions")
    print("  - Similarity adds only 0-1 point variation within position")
    
    # ============================================================
    # REFERENCE_BASED MODE - Binary Relevance
    # ============================================================
    print("\n\n2️⃣ REFERENCE_BASED MODE - Binary Relevance")
    print("=" * 70)
    print("\nFormula: final_score = base_score - (ret_idx * position_decay) + best_similarity")
    
    print("\n📊 For Matched References (similarity >= threshold):")
    print("-" * 50)
    print(f"{'Position':<10} {'Min Score':<20} {'Max Score':<20}")
    print("-" * 50)
    
    for ret_idx in range(5):
        # Only documents with similarity >= threshold are added
        min_similarity = similarity_threshold  # 0.5
        max_similarity = 1.0
        
        min_score = base_score - (ret_idx * position_decay) + min_similarity
        max_score = base_score - (ret_idx * position_decay) + max_similarity
        
        print(f"{ret_idx+1:<10} {min_score:<20.1f} {max_score:<20.1f}")
    
    print("\n📊 For False Positives (similarity < threshold):")
    print("-" * 50)
    print(f"{'Position':<10} {'Min Score':<20} {'Max Score':<20}")
    print("-" * 50)
    
    for ret_idx in range(5):
        # False positives have similarity < threshold
        min_similarity = 0.0
        max_similarity = similarity_threshold - 0.001  # Just below threshold
        
        min_score = base_score - (ret_idx * position_decay) + min_similarity
        max_score = base_score - (ret_idx * position_decay) + max_similarity
        
        print(f"{ret_idx+1:<10} {min_score:<20.1f} {max_score:<20.1f}")
    
    # ============================================================
    # REFERENCE_BASED MODE - Graded Relevance
    # ============================================================
    print("\n\n3️⃣ REFERENCE_BASED MODE - Graded Relevance")
    print("=" * 70)
    print("\nFormula: final_score = base_score - (ret_idx * position_decay) + scaled_score")
    print("Where: scaled_score = 1.0 + (similarity - threshold) * 9.0 / (1.0 - threshold)")
    
    print("\n📊 Scaled Score Calculation:")
    print("-" * 50)
    
    similarities = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for sim in similarities:
        if sim >= similarity_threshold:
            scaled = 1.0 + (sim - similarity_threshold) * 9.0 / (1.0 - similarity_threshold)
            print(f"Similarity {sim:.1f} → Scaled score: {scaled:.2f}")
    
    print("\n📊 Final Score Ranges by Position (Graded):")
    print("-" * 50)
    print(f"{'Position':<10} {'Min Score':<25} {'Max Score':<25}")
    print("-" * 50)
    
    for ret_idx in range(5):
        # Min: similarity = threshold (0.5) → scaled = 1.0
        min_scaled = 1.0
        min_score = base_score - (ret_idx * position_decay) + min_scaled
        
        # Max: similarity = 1.0 → scaled = 10.0
        max_scaled = 10.0
        max_score = base_score - (ret_idx * position_decay) + max_scaled
        
        print(f"{ret_idx+1:<10} {min_score:<25.1f} {max_score:<25.1f}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n\n📈 SUMMARY OF SCORE RANGES")
    print("=" * 70)
    
    print("\n🎯 Key Observations:")
    print("-" * 50)
    print("1. Base score (1000) ensures all scores are positive")
    print("2. Position decay (10) creates clear separation between positions")
    print("3. Documents at same position can vary by:")
    print("   - Binary: 0-1 point (similarity difference)")
    print("   - Graded: 0-9 points (scaled relevance difference)")
    print("4. Position 1 scores: 1000-1001 (binary) or 1001-1010 (graded)")
    print("5. Position 5 scores: 960-961 (binary) or 961-970 (graded)")
    
    print("\n🔄 Ranking Implications:")
    print("-" * 50)
    print("✅ Documents ALWAYS maintain retrieval order")
    print("✅ Position 1 doc always scores higher than position 2")
    print("✅ Reranking effects are preserved perfectly")
    print("✅ Within-position variation allows similarity-based fine-tuning")
    
    print("\n⚠️ Important Notes:")
    print("-" * 50)
    print("• ranx uses run_dict scores to rank documents")
    print("• Higher score = better rank")
    print("• Our scoring ensures retrieval order is preserved")
    print("• Similarity provides minor adjustments within positions")


def compare_with_without_position():
    """
    Compare scores with and without position preservation.
    """
    
    print("\n\n🔄 COMPARISON: With vs Without Position Preservation")
    print("=" * 70)
    
    print("\n📊 Example: 3 documents with different similarities")
    print("-" * 50)
    
    docs = [
        ("Doc A (pos 1)", 0, 0.6),  # Position 1, similarity 0.6
        ("Doc B (pos 2)", 1, 0.9),  # Position 2, similarity 0.9 (higher sim!)
        ("Doc C (pos 3)", 2, 0.7),  # Position 3, similarity 0.7
    ]
    
    base_score = 1000.0
    position_decay = 10.0
    
    print("\n❌ Without Position Preservation (similarity only):")
    print("-" * 50)
    for name, pos, sim in docs:
        score = sim  # Only similarity
        print(f"{name}: similarity={sim:.1f} → score={score:.1f}")
    print("Ranking: Doc B (0.9) > Doc C (0.7) > Doc A (0.6)")
    print("⚠️ PROBLEM: Original order lost! Doc B jumped to first!")
    
    print("\n✅ With Position Preservation (our approach):")
    print("-" * 50)
    for name, pos, sim in docs:
        score = base_score - (pos * position_decay) + sim
        print(f"{name}: pos={pos+1}, sim={sim:.1f} → score={score:.1f}")
    print("Ranking: Doc A (1000.6) > Doc B (990.9) > Doc C (980.7)")
    print("✅ SUCCESS: Original retrieval order preserved!")
    
    print("\n💡 The 10-point position decay ensures:")
    print("  - Even highest similarity (1.0) can't overcome position gap")
    print("  - Document at position N always ranks higher than position N+1")
    print("  - Reranking by retriever is faithfully preserved")


if __name__ == "__main__":
    analyze_score_ranges()
    compare_with_without_position()
    
    print("\n\n" + "=" * 70)
    print("🎯 FINAL ANSWER:")
    print("=" * 70)
    print("""
final_score ranges:

1. RETRIEVAL_BASED mode:
   - Position 1: 1000.0 - 1001.0
   - Position 2: 990.0 - 991.0
   - Position 3: 980.0 - 981.0
   - Position 4: 970.0 - 971.0
   - Position 5: 960.0 - 961.0

2. REFERENCE_BASED (Binary):
   - Same as above for matched docs
   - Slightly lower for false positives

3. REFERENCE_BASED (Graded):
   - Position 1: 1001.0 - 1010.0
   - Position 2: 991.0 - 1000.0
   - Position 3: 981.0 - 990.0
   - Position 4: 971.0 - 980.0
   - Position 5: 961.0 - 970.0

Key: Position dominates, similarity fine-tunes within position
""")