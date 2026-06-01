import sys
import os
import time
from collections import defaultdict
import random

# Add parent dir to path to import synaptoroute
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets

def evaluate_on_dataset(dataset_name, subset_name=None, num_train_per_intent=20, max_test_samples=500):
    print(f"\n{'='*50}")
    print(f"Evaluating on {dataset_name} (subset: {subset_name})")
    print(f"{'='*50}")

    if subset_name:
        ds = datasets.load_dataset(dataset_name, subset_name, trust_remote_code=True)
    else:
        ds = datasets.load_dataset(dataset_name, trust_remote_code=True)

    # Use 'train' for routes, 'test' for evaluation (avoid data leakage)
    train_ds = ds["train"]
    test_ds = ds["test"]

    # CLINC uses 'intent', Banking77 uses 'label'
    intent_col = 'intent' if 'intent' in train_ds.column_names else 'label'
    
    # Map label IDs to string names if needed (banking77 uses integer labels)
    if hasattr(train_ds.features[intent_col], 'int2str'):
        def get_intent_name(idx):
            return train_ds.features[intent_col].int2str(idx)
    else:
        def get_intent_name(idx):
            return str(idx)

    # Collect training utterances per intent
    intent_to_utterances = defaultdict(list)
    for item in train_ds:
        intent = get_intent_name(item[intent_col])
        text = item['text']
        intent_to_utterances[intent].append(text)

    from utils import init_synaptoroute, init_semantic_router
    
    print(f"Building routers with {len(intent_to_utterances)} intents...")
    
    routes_data = []
    for intent, utterances in intent_to_utterances.items():
        # Subsample to keep memory/time reasonable
        sampled_utterances = random.sample(utterances, min(num_train_per_intent, len(utterances)))
        routes_data.append({"name": intent, "utterances": sampled_utterances})

    print("Fitting SynaptoRoute index...")
    synapto = init_synaptoroute(routes_data)

    print("Building Semantic Router index...")
    sr_layer = init_semantic_router(routes_data)

    # Prepare Test Data
    test_samples = list(test_ds)
    if len(test_samples) > max_test_samples:
        test_samples = random.sample(test_samples, max_test_samples)

    print(f"\nEvaluating on {len(test_samples)} test queries...")

    # Evaluate SynaptoRoute
    synapto_correct = 0
    synapto_latencies = []
    
    for item in test_samples:
        query = item['text']
        expected = get_intent_name(item[intent_col])
        
        t0 = time.perf_counter()
        res = synapto(query)
        t1 = time.perf_counter()
        
        synapto_latencies.append((t1 - t0) * 1000)
        predicted = res.name if res else None
        
        if predicted == expected:
            synapto_correct += 1

    # Evaluate Semantic Router
    sr_correct = 0
    sr_latencies = []
    
    for item in test_samples:
        query = item['text']
        expected = get_intent_name(item[intent_col])
        
        t0 = time.perf_counter()
        res = sr_layer(query)
        t1 = time.perf_counter()
        
        sr_latencies.append((t1 - t0) * 1000)
        predicted = res.name if res.name else None
        
        if predicted == expected:
            sr_correct += 1

    # Print Results
    synapto_acc = synapto_correct / len(test_samples)
    sr_acc = sr_correct / len(test_samples)
    
    synapto_avg_lat = sum(synapto_latencies) / len(synapto_latencies)
    sr_avg_lat = sum(sr_latencies) / len(sr_latencies)

    print("\n--- Results ---")
    print(f"[SynaptoRoute]    Accuracy: {synapto_acc:.4f} | Avg Latency: {synapto_avg_lat:.2f} ms")
    print(f"[Semantic Router] Accuracy: {sr_acc:.4f} | Avg Latency: {sr_avg_lat:.2f} ms")


if __name__ == "__main__":
    random.seed(42)
    # 1. CLINC150 (small subset)
    # evaluate_on_dataset("clinc/clinc_oos", "small", num_train_per_intent=20, max_test_samples=1000)
    
    # 2. Banking77
    evaluate_on_dataset("mteb/banking77", subset_name=None, num_train_per_intent=20, max_test_samples=1000)
