import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils import load_datasets, init_synaptoroute

def run_failure_analysis(model_name="BAAI/bge-small-en-v1.5"):
    print("=== Phase 1.75: Investigation 4 - Failure Analysis ===\n")
    
    dataset_version, routes_data, test_queries = load_datasets()
    expected = [q["expected_route"] for q in test_queries]
    queries = [q["query"] for q in test_queries]
    
    print(f"Loaded {len(test_queries)} test queries.")
    
    router = init_synaptoroute(routes_data, model_name=model_name, storage_path="bench_failures.sqlite")
    
    # Store failures as {expected_route: [(query, predicted_route, score)]}
    failures = defaultdict(list)
    
    for q, exp in zip(queries, expected):
        emb = router.encoder.encode(q)
        results = router.index.search(np.array([emb]), top_k=1)[0]
        
        pred = results[0][1] if results else None
        score = results[0][0] if results else -1.0
        
        if pred != exp:
            failures[str(exp)].append((q, str(pred), score))
            
    print("\n--- Failure Hotspots ---")
    # Sort by number of failures
    sorted_failures = sorted(failures.items(), key=lambda x: len(x[1]), reverse=True)
    
    for exp_route, fails in sorted_failures:
        print(f"\nExpected Route: '{exp_route}' ({len(fails)} failures)")
        
        # What did it predict instead?
        pred_counts = defaultdict(int)
        for q, pred, score in fails:
            pred_counts[pred] += 1
            
        print("  Common Misclassifications:")
        for pred, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    -> {pred}: {count} times")
            
        print("  Sample Failing Queries:")
        for q, pred, score in fails[:5]:
            print(f"    - '{q}' (Routed to {pred} with score {score:.3f})")

if __name__ == "__main__":
    run_failure_analysis()
