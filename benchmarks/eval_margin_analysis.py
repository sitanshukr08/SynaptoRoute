import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils import load_datasets, init_synaptoroute

def run_margin_analysis(model_name="BAAI/bge-small-en-v1.5"):
    print("=== Phase 1.75: Investigation 3 - Margin Analysis ===\n")
    
    dataset_version, routes_data, test_queries = load_datasets()
    expected = [q["expected_route"] for q in test_queries]
    queries = [q["query"] for q in test_queries]
    
    print(f"Loaded {len(test_queries)} test queries.")
    
    router = init_synaptoroute(routes_data, model_name=model_name, storage_path="bench_margin.sqlite")
    
    margins_correct = []
    margins_incorrect = []
    
    print("\n--- Example Margins ---")
    
    for i, q in enumerate(queries):
        emb = router.encoder.encode(q)
        # Get top 2
        results = router.index.search(np.array([emb]), top_k=2)[0]
        
        if not results:
            continue
            
        top1_score, top1_name = results[0]
        
        if len(results) > 1:
            top2_score, top2_name = results[1]
        else:
            top2_score, top2_name = 0.0, "None"
            
        margin = top1_score - top2_score
        
        exp = expected[i]
        is_correct = (top1_name == exp)
        
        if is_correct:
            margins_correct.append(margin)
        else:
            margins_incorrect.append(margin)
            
        # Print a few examples
        if i % 100 == 0:
            print(f"Query: '{q[:40]}...' | Correct: {is_correct}")
            print(f"  Top1: {top1_name} ({top1_score:.3f})")
            print(f"  Top2: {top2_name} ({top2_score:.3f})")
            print(f"  Margin: {margin:.3f}\n")
            
    avg_margin_correct = np.mean(margins_correct) if margins_correct else 0.0
    avg_margin_incorrect = np.mean(margins_incorrect) if margins_incorrect else 0.0
    
    print("\n--- Aggregate Margin Analysis ---")
    print(f"Average Margin for Correct Predictions:   {avg_margin_correct:.4f}")
    print(f"Average Margin for Incorrect Predictions: {avg_margin_incorrect:.4f}")

if __name__ == "__main__":
    run_margin_analysis()
