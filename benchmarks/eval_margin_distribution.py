import sys
import os
import json
import numpy as np

# Add parent dir to path to import synaptoroute
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import init_synaptoroute

def analyze_margin_distribution(dataset_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    print(f"\n{'='*50}")
    print(f"Margin Distribution Analysis")
    print(f"Model: {model_name}")
    print(f"{'='*50}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    routes_data = data["routes"]
    eval_queries = data["eval_queries"]

    print("Initializing SynaptoRoute...")
    synapto = init_synaptoroute(routes_data, model_name=model_name, storage_path=f"bench_margin.sqlite")

    correct_margins = []
    incorrect_margins = []

    for item in eval_queries:
        query = item["query"]
        expected = item["expected_route"]
        
        # Get query embedding manually to calculate top2 margin
        query_emb = synapto.encoder.encode(query)
        search_results = synapto.index.search(np.array([query_emb]), top_k=2)[0]
        
        top1_score = -1.0
        top1_route = None
        top2_score = -1.0
        
        if len(search_results) > 0:
            top1_score, top1_route = search_results[0]
            # Verify threshold
            if top1_route in synapto._route_map and top1_score < synapto._route_map[top1_route].threshold:
                top1_route = "None"
        else:
            top1_route = "None"
            
        if len(search_results) > 1:
            top2_score = search_results[1][0]
            
        margin = top1_score - top2_score if top2_score != -1.0 else 0.0
        
        predicted = top1_route
        
        if predicted == expected:
            # We don't want to track margins for OOD rejections that were naturally "None" due to threshold, 
            # as they are not margin-based rejections, but let's track everything that passed the threshold.
            if predicted != "None":
                correct_margins.append(margin)
        else:
            # If it predicted something but it was wrong (or it was supposed to predict something but predicted None)
            # Actually, margin is only relevant when the router makes an active prediction (not None).
            if predicted != "None":
                incorrect_margins.append(margin)
            elif expected != "None":
                # Expected a route, but top1 was below threshold (OOD)
                pass

    print("\n--- Margin Statistics (Active Predictions Only) ---")
    
    if correct_margins:
        print(f"\nCorrect Predictions ({len(correct_margins)}):")
        print(f"  Mean Margin: {np.mean(correct_margins):.4f}")
        print(f"  Median Margin: {np.median(correct_margins):.4f}")
        print(f"  Min Margin: {np.min(correct_margins):.4f}")
        print(f"  Max Margin: {np.max(correct_margins):.4f}")
        
    if incorrect_margins:
        print(f"\nIncorrect Predictions ({len(incorrect_margins)}):")
        print(f"  Mean Margin: {np.mean(incorrect_margins):.4f}")
        print(f"  Median Margin: {np.median(incorrect_margins):.4f}")
        print(f"  Min Margin: {np.min(incorrect_margins):.4f}")
        print(f"  Max Margin: {np.max(incorrect_margins):.4f}")
        
if __name__ == "__main__":
    dataset_file = os.path.join(os.path.dirname(__file__), "datasets", "stress_dataset.json")
    if not os.path.exists(dataset_file):
        print(f"Dataset not found: {dataset_file}")
        sys.exit(1)
        
    analyze_margin_distribution(dataset_file, "sentence-transformers/all-MiniLM-L6-v2")
    analyze_margin_distribution(dataset_file, "nomic-ai/nomic-embed-text-v1.5")
