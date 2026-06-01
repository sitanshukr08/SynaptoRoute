import os
import sys
import numpy as np
import time
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils import load_datasets, init_synaptoroute

MODELS_TO_TEST = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/multilingual-e5-large",
    "nomic-ai/nomic-embed-text-v1.5"
]

def run_embedding_evaluations():
    print("=== Phase 1.75: Investigation 1 - Embedding Models ===\n")
    
    # Load dataset once
    dataset_version, routes_data, test_queries = load_datasets()
    expected = [q["expected_route"] for q in test_queries]
    queries = [q["query"] for q in test_queries]
    
    print(f"Loaded {len(test_queries)} test queries.\n")
    print(f"{'Model':<40} | {'F1':<6} | {'Top1':<6} | {'Top3':<6} | {'OOD FPR':<8} | {'Latency (ms)':<12}")
    print("-" * 90)
    
    for model_name in MODELS_TO_TEST:
        try:
            # Initialize router (this will download model weights if missing)
            router = init_synaptoroute(routes_data, model_name=model_name, storage_path=f"bench_{model_name.replace('/', '_')}.sqlite")
        except Exception as e:
            print(f"{model_name:<40} | FAILED TO LOAD: {e}")
            continue
            
        preds = []
        latencies = []
        top1_hits = 0
        top3_hits = 0
        ood_total = 0
        ood_false_positives = 0
        
        # Warmup
        try:
            router("warmup")
        except:
            pass
            
        for q, exp in zip(queries, expected):
            start = time.perf_counter()
            # We want top 3
            emb = router.encoder.encode(q)
            results = router.index.search(np.array([emb]), top_k=3)[0]
            latencies.append((time.perf_counter() - start) * 1000) # ms
            
            names = [r_name for score, r_name in results]
            top1 = names[0] if names else None
            preds.append(top1)
            
            if exp is None:
                ood_total += 1
                if top1 is not None:
                    ood_false_positives += 1
                else:
                    # Technically if top1 is None it's correct
                    top1_hits += 1
                    top3_hits += 1
            else:
                if top1 == exp:
                    top1_hits += 1
                if exp in names:
                    top3_hits += 1
                    
        y_true = [e if e is not None else 'OOD' for e in expected]
        y_pred = [p if p is not None else 'OOD' for p in preds]
        
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        top1_acc = top1_hits / len(queries)
        top3_acc = top3_hits / len(queries)
        ood_fpr = (ood_false_positives / ood_total) if ood_total > 0 else 0.0
        avg_latency = np.mean(latencies)
        
        print(f"{model_name:<40} | {f1:.4f} | {top1_acc:.4f} | {top3_acc:.4f} | {ood_fpr:.4f}   | {avg_latency:.2f}")

if __name__ == "__main__":
    run_embedding_evaluations()
