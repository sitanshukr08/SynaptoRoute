import sys
import os
import time
import json
from collections import defaultdict
import random

# Add parent dir to path to import synaptoroute
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import init_synaptoroute, init_semantic_router

def evaluate_on_stress_dataset(dataset_path: str, model_names: list):
    print(f"\n{'='*50}")
    print(f"Adversarial & Hard-Negative Stress Test")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*50}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    routes_data = data["routes"]
    eval_queries = data["eval_queries"]

    print(f"Routes: {len(routes_data)}")
    print(f"Total Eval Queries: {len(eval_queries)}")
    
    categories = list(set(q["category"] for q in eval_queries))
    print(f"Categories: {', '.join(categories)}")

    for model_name in model_names:
        print(f"\n\n{'*'*40}")
        print(f"MODEL: {model_name}")
        print(f"{'*'*40}")

        print("\n--- Initializing SynaptoRoute ---")
        synapto = init_synaptoroute(routes_data, model_name=model_name, storage_path=f"bench_synapto_{model_name.replace('/', '_')}.sqlite")
        
        print("\n--- Initializing Semantic Router ---")
        sr_layer = init_semantic_router(routes_data, model_name=model_name)

        # Structure to track results
        # dict structure: {"synapto": {"category": [correct, total]}, "sr": {"category": [correct, total]}}
        results = {
            "synapto": defaultdict(lambda: {"correct": 0, "total": 0}),
            "sr": defaultdict(lambda: {"correct": 0, "total": 0})
        }
        
        # Track overall OOD rejection (where expected is "None")
        ood_total = 0
        ood_synapto_correct = 0
        ood_sr_correct = 0

        # Evaluate SynaptoRoute
        t0_syn = time.perf_counter()
        for item in eval_queries:
            query = item["query"]
            expected = item["expected_route"]
            cat = item["category"]
            
            res = synapto(query)
            predicted = res.name if res else "None"
            
            results["synapto"][cat]["total"] += 1
            if predicted == expected:
                results["synapto"][cat]["correct"] += 1
                if expected == "None":
                    ood_synapto_correct += 1
            
            if expected == "None":
                ood_total += 1
                
        t1_syn = time.perf_counter()
        
        # Evaluate Semantic Router
        t0_sr = time.perf_counter()
        for item in eval_queries:
            query = item["query"]
            expected = item["expected_route"]
            cat = item["category"]
            
            res = sr_layer(query)
            predicted = res.name if res.name else "None"
            
            results["sr"][cat]["total"] += 1
            if predicted == expected:
                results["sr"][cat]["correct"] += 1
                if expected == "None":
                    ood_sr_correct += 1
        t1_sr = time.perf_counter()

        syn_latency = ((t1_syn - t0_syn) / len(eval_queries)) * 1000
        sr_latency = ((t1_sr - t0_sr) / len(eval_queries)) * 1000

        print(f"\n[SynaptoRoute] Overall Avg Latency: {syn_latency:.2f} ms")
        print(f"[Semantic Router] Overall Avg Latency: {sr_latency:.2f} ms")
        print("\nAccuracy by Category:")
        print(f"{'Category':<20} | {'SynaptoRoute':<15} | {'Semantic Router':<15}")
        print("-" * 55)
        
        total_syn_correct = 0
        total_sr_correct = 0
        total_queries = 0
        
        for cat in sorted(categories):
            syn_c = results["synapto"][cat]["correct"]
            syn_t = results["synapto"][cat]["total"]
            sr_c = results["sr"][cat]["correct"]
            
            total_syn_correct += syn_c
            total_sr_correct += sr_c
            total_queries += syn_t
            
            syn_acc = (syn_c / syn_t) * 100 if syn_t > 0 else 0
            sr_acc = (sr_c / syn_t) * 100 if syn_t > 0 else 0
            
            print(f"{cat:<20} | {syn_acc:>5.1f}% ({syn_c}/{syn_t}) | {sr_acc:>5.1f}% ({sr_c}/{syn_t})")
            
        print("-" * 55)
        syn_overall = (total_syn_correct / total_queries) * 100
        sr_overall = (total_sr_correct / total_queries) * 100
        print(f"{'OVERALL':<20} | {syn_overall:>5.1f}% ({total_syn_correct}/{total_queries}) | {sr_overall:>5.1f}% ({total_sr_correct}/{total_queries})")

        if ood_total > 0:
            syn_ood = (ood_synapto_correct / ood_total) * 100
            sr_ood = (ood_sr_correct / ood_total) * 100
            print(f"\nOOD Rejection Accuracy:")
            print(f"SynaptoRoute:    {syn_ood:.1f}%")
            print(f"Semantic Router: {sr_ood:.1f}%")

if __name__ == "__main__":
    random.seed(42)
    models_to_test = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "nomic-ai/nomic-embed-text-v1.5"
    ]
    
    dataset_file = os.path.join(os.path.dirname(__file__), "datasets", "stress_dataset.json")
    if not os.path.exists(dataset_file):
        print(f"Dataset not found: {dataset_file}")
        print("Please run generate_hard_datasets.py first.")
        sys.exit(1)
        
    evaluate_on_stress_dataset(dataset_file, models_to_test)
