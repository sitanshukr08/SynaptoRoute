import os
import sys
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils import load_datasets, init_synaptoroute

def run_threshold_evaluations(model_name="BAAI/bge-small-en-v1.5"):
    print("=== Phase 1.75: Investigation 2 - Threshold Sweeps ===\n")
    
    dataset_version, routes_data, test_queries = load_datasets()
    expected = [q["expected_route"] for q in test_queries]
    queries = [q["query"] for q in test_queries]
    
    print(f"Loaded {len(test_queries)} test queries.")
    
    # We use a single model. Let's use the baseline for the sweep
    # or whichever won Investigation 1, but for now we stick to baseline
    router = init_synaptoroute(routes_data, model_name=model_name, storage_path="bench_thresholds.sqlite")
    
    thresholds_to_test = [0.0, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    print(f"{'Threshold':<10} | {'F1':<6} | {'Precision':<10} | {'Recall':<6} | {'OOD FPR':<8} | {'OOD FNR':<8}")
    print("-" * 65)
    
    # Precompute all top-1 scores to save time since threshold is just a post-hoc filter
    # on the search results.
    # Actually, we can just run a single query sweep and capture the top-1 score and name.
    
    query_results = [] # list of (top1_name, top1_score)
    
    for q in queries:
        emb = router.encoder.encode(q)
        results = router.index.search(np.array([emb]), top_k=1)[0]
        if results:
            score, name = results[0]
            query_results.append((name, score))
        else:
            query_results.append((None, -1.0))
            
    y_true = [e if e is not None else 'OOD' for e in expected]
    
    for t in thresholds_to_test:
        preds = []
        ood_total = 0
        ood_false_positives = 0
        
        valid_total = 0
        valid_false_negatives = 0
        
        for i, (name, score) in enumerate(query_results):
            exp = expected[i]
            
            if score >= t:
                final_pred = name
            else:
                final_pred = None
                
            preds.append(final_pred)
            
            # OOD stats
            if exp is None:
                ood_total += 1
                if final_pred is not None:
                    ood_false_positives += 1
            else:
                valid_total += 1
                if final_pred is None:
                    valid_false_negatives += 1
                    
        y_pred = [p if p is not None else 'OOD' for p in preds]
        
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        ood_fpr = (ood_false_positives / ood_total) if ood_total > 0 else 0.0
        ood_fnr = (valid_false_negatives / valid_total) if valid_total > 0 else 0.0
        
        print(f"{t:<10.2f} | {f1:.4f} | {prec:.4f}     | {rec:.4f} | {ood_fpr:.4f}   | {ood_fnr:.4f}")

if __name__ == "__main__":
    run_threshold_evaluations()
