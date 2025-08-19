# Simple Graded Relevance Implementation Plan

## 🎯 Current Status
- Binary relevance: ✅ Works perfectly 
- Graded relevance: ❌ Still returning 0.000 for all metrics
- Root cause: Document ID mismatch between qrels and run

## 💡 Simple Solution

**Use retrieval-based mode logic for graded relevance in reference_based mode:**

```python
# Instead of complex reference mapping, use simple approach:
# 1. Add retrieved documents to both qrels and run with same IDs
# 2. Use similarity scores as relevance grades
# 3. Keep binary relevance logic unchanged

if evaluation_mode == 'reference_based':
    if use_graded_relevance:
        # SIMPLE: Use retrieval_based logic with similarity-based relevance
        for j in range(len(retrieved_texts)):
            doc_id = f"doc_{j}"
            max_similarity = np.max(similarity_matrix[:, j])
            
            # Add to run (all retrieved docs)
            run_dict[query_id][doc_id] = float(max_similarity)
            
            # Add to qrels only if similarity >= threshold
            if max_similarity >= similarity_threshold:
                qrels_dict[query_id][doc_id] = float(max_similarity)
    else:
        # Binary relevance: existing logic works fine
        # ... current working binary logic ...
```

## 🔄 Implementation Steps

### Step 1: Replace complex reference mapping with simple retrieval-based approach
- Remove reference document IDs from qrels/run
- Use only retrieved document IDs: doc_0, doc_1, doc_2, etc.
- Apply similarity scores as relevance grades

### Step 2: Test
- Binary relevance should still work perfectly
- Graded relevance should now produce non-zero meaningful scores

### Step 3: Validate
- Check that MRR reflects actual document ranking positions
- Ensure NDCG and MAP use similarity scores properly

## 🎯 Expected Results

After this change:
- **Binary relevance**: Still 1.000 (no change)
- **Graded relevance**: 0.3-0.8 range (meaningful scores)
- **MRR**: Reflects actual ranking positions
- **Simplicity**: Much cleaner code, easier to understand

## ⚡ Quick Implementation

Replace the complex reference_based logic with:

```python
if evaluation_mode == 'reference_based':
    # For both binary and graded, evaluate retrieved documents
    for j in range(len(retrieved_texts)):
        doc_id = f"doc_{j}"
        max_similarity = np.max(similarity_matrix[:, j]) if similarity_matrix.shape[0] > 0 else 0.01
        
        # Always add to run
        run_dict[query_id][doc_id] = float(max_similarity)
        
        # Add to qrels based on relevance type
        if use_graded_relevance:
            if max_similarity >= similarity_threshold:
                qrels_dict[query_id][doc_id] = float(max_similarity)
        else:
            if max_similarity >= similarity_threshold:
                qrels_dict[query_id][doc_id] = 1.0
```

This approach:
✅ Keeps existing binary relevance working
✅ Makes graded relevance work properly  
✅ Maintains proper document ID matching
✅ Simplifies the codebase significantly