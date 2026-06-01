import sys
import os
import json
import time
from collections import defaultdict

# Add parent dir to path to import synaptoroute
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import init_synaptoroute
from synaptoroute.reranker import CrossEncoderReranker

def evaluate_router(router, eval_queries, name):
    correct = 0
    total = len(eval_queries)
    ood_correct = 0
    ood_total = 0
    
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    
    start_time = time.perf_counter()
    
    for item in eval_queries:
        query = item["query"]
        expected = item["expected_route"]
        category = item["category"]
        
        category_total[category] += 1
        if expected == "None":
            ood_total += 1
            
        route = router(query)
        predicted = route.name if route else "None"
        
        if predicted == expected:
            correct += 1
            category_correct[category] += 1
            if expected == "None":
                ood_correct += 1

    latency = (time.perf_counter() - start_time) / total * 1000
    acc = correct / total if total > 0 else 0
    ood_acc = ood_correct / ood_total if ood_total > 0 else 0
    
    print(f"\n{name} Results:")
    print(f"Overall Accuracy:  {acc*100:.1f}% ({correct}/{total})")
    print(f"OOD Rejection Acc: {ood_acc*100:.1f}% ({ood_correct}/{ood_total})")
    print(f"Avg Latency:       {latency:.2f} ms")
    
    print("By Category:")
    for cat in sorted(category_total.keys()):
        c_acc = category_correct[cat] / category_total[cat]
        print(f"  {cat.ljust(18)} | {c_acc*100:5.1f}%")
        
    return acc, ood_acc

def main():
    dataset_file = os.path.join(os.path.dirname(__file__), "datasets", "stress_dataset.json")
    if not os.path.exists(dataset_file):
        print(f"Dataset not found: {dataset_file}")
        sys.exit(1)

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    routes_data = data["routes"]
    eval_queries = data["eval_queries"]

    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "nomic-ai/nomic-embed-text-v1.5"
    ]
    
    margins = [0.0, 0.05, 0.10]

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")
        
        # 1. Evaluate Margins
        print("\n--- Evaluating Margin Gating ---")
        for margin in margins:
            synapto = init_synaptoroute(routes_data, model_name=model_name, storage_path=f"bench_v2_{margin}.sqlite")
            synapto.margin = margin
            evaluate_router(synapto, eval_queries, f"SynaptoRoute (Margin={margin})")
            
        # 2. Evaluate Reranker
        print("\n--- Evaluating Cross-Encoder Reranker ---")
        synapto = init_synaptoroute(routes_data, model_name=model_name, storage_path=f"bench_v2_rerank.sqlite")
        try:
            # We use a zero threshold for margin here because reranker takes over
            synapto.margin = 0.0
            
            print("\n--- Evaluating MS-Marco Cross-Encoder Reranker ---")
            synapto.reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", threshold=0.0) 
            evaluate_router(synapto, eval_queries, f"SynaptoRoute (MS-Marco Reranker)")
            
            # Evaluate with NLI Reranker
            print("\n--- Evaluating NLI Cross-Encoder Reranker ---")
            synapto.reranker = CrossEncoderReranker(model_name="cross-encoder/nli-deberta-v3-small", threshold=0.0) 
            evaluate_router(synapto, eval_queries, f"SynaptoRoute (NLI Reranker)")
        except ImportError as e:
            print(f"Skipping Reranker eval: {e}")

if __name__ == "__main__":
    main()
