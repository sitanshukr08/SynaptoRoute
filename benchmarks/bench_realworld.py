import time
import json
import argparse
from typing import List, Dict, Set
from collections import defaultdict

from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

from synaptoroute import AdaptiveRouter, Route
from synaptoroute.encoder import FastEmbedEncoder
from synaptoroute.storage import SQLiteStorage

def calculate_metrics(y_true, y_pred, y_pred_topk, num_intents, is_ood_true=None, is_ood_pred=None):
    # Top-1
    top1 = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    
    # Top-K
    top3 = sum(1 for t, pk in zip(y_true, y_pred_topk) if t in pk[:3]) / len(y_true)
    top5 = sum(1 for t, pk in zip(y_true, y_pred_topk) if t in pk[:5]) / len(y_true)
    
    # Precision, Recall, F1
    # For clean PRF1, we ignore OOD metrics if OOD is handled separately, but we can compute macro
    # Replace None with a string for sklearn compatibility
    y_pred_sklearn = ["OOD_LABEL_NONE" if p is None else p for p in y_pred]
    y_true_sklearn = ["OOD_LABEL_NONE" if t is None else t for t in y_true]
    p, r, f1, _ = precision_recall_fscore_support(y_true_sklearn, y_pred_sklearn, average='macro', zero_division=0)
    
    # Route Coverage
    predicted_intents = set(y_pred)
    coverage = len(predicted_intents - {None}) / num_intents
    
    # OOD Rejection
    ood_acc = None
    if is_ood_true is not None and is_ood_pred is not None:
        correct_ood = sum(1 for t, p in zip(is_ood_true, is_ood_pred) if t and p)
        total_ood = sum(1 for t in is_ood_true if t)
        if total_ood > 0:
            ood_acc = correct_ood / total_ood
            
    return top1, top3, top5, p, r, f1, coverage, ood_acc

async def evaluate_dataset(name: str, ds_name: str, ds_config: str, intent_key: str, text_key: str, ood_label: int = None, model_name="BAAI/bge-small-en-v1.5"):
    print(f"\n============================================================")
    print(f"BENCHMARK: {name}")
    print(f"============================================================")
    
    print(f"Loading {name} from HuggingFace...")
    ds = load_dataset(ds_name, ds_config) if ds_config else load_dataset(ds_name)
    
    train_ds = ds["train"]
    test_ds = ds["test"]
    
    features = train_ds.features[intent_key]
    if hasattr(features, 'names'):
        intent_names = features.names
    else:
        # If it's not a ClassLabel, but we know it's Banking77 where label_text has the names
        # Let's map label -> label_text
        import re
        label_to_text = {}
        for row in train_ds:
            if row[intent_key] not in label_to_text:
                # Sanitize the route name to match ^[a-zA-Z0-9_-]+$
                clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', row["label_text"])
                label_to_text[row[intent_key]] = clean_name
        intent_names = [label_to_text[i] for i in range(max(label_to_text.keys()) + 1)]
    
    valid_intents = {i for i in range(len(intent_names)) if i != ood_label}
    
    route_utterances = defaultdict(list)
    for row in train_ds:
        intent_id = row[intent_key]
        if intent_id in valid_intents:
            route_utterances[intent_names[intent_id]].append(row[text_key])
            
    print(f"Constructing {len(route_utterances)} Routes from Train Split...")
    routes = []
    for intent_name, utterances in route_utterances.items():
        routes.append(Route(
            name=intent_name,
            utterances=utterances,
            threshold=0.60
        ))
        
    db_path = f"bench_{name.lower()}_memory.sqlite"
    import os
    db_exists = os.path.exists(db_path)
        
    storage = SQLiteStorage(db_path)
    encoder = FastEmbedEncoder(model_name=model_name)
    router = AdaptiveRouter(storage=storage, encoder=encoder)
    
    t0 = time.perf_counter()
    await router.start()
    
    if not db_exists:
        print("Building SynaptoRoute Index (this encodes all train utterances)...")
        for route in routes:
            router.add_route(route)
        print(f"Index built in {time.perf_counter() - t0:.2f}s")
    else:
        print("DB already exists. Using cached routes.")
    
    print(f"Evaluating on Test Split ({len(test_ds)} queries)...")
    y_true = []
    y_pred = []
    y_pred_topk = []
    
    is_ood_true = []
    is_ood_pred = []
    
    latencies = []
    
    import numpy as np
    
    for row in test_ds:
        intent_id = row[intent_key]
        text = row[text_key]
        
        t_s = time.perf_counter()
        best_route = await router.aquery(text)
        latencies.append(time.perf_counter() - t_s)
        
        true_name = intent_names[intent_id]
        pred_name = best_route.name if best_route else None
        
        # Manually compute Top-K from Faiss for Top-K metrics (bypasses threshold/margin, measures pure retrieval)
        emb = router.encoder.encode(text)
        raw_results = router.index.search(np.array([emb]), top_k=5)
        top_names = [route_name for score, route_name in raw_results[0]]
        
        y_true.append(true_name)
        y_pred.append(pred_name)
        y_pred_topk.append(top_names)
        
        if ood_label is not None:
            is_ood_true.append(intent_id == ood_label)
            is_ood_pred.append(pred_name is None)
                
    avg_latency = (sum(latencies) / len(latencies)) * 1000
    
    top1, top3, top5, p, r, f1, coverage, ood_acc = calculate_metrics(
        y_true, y_pred, y_pred_topk, len(valid_intents), 
        is_ood_true if ood_label is not None else None,
        is_ood_pred if ood_label is not None else None
    )
    
    print("\n[Metrics]")
    print(f"Top-1 Accuracy:  {top1*100:.2f}%")
    print(f"Top-3 Accuracy:  {top3*100:.2f}%")
    print(f"Top-5 Accuracy:  {top5*100:.2f}%")
    print(f"Precision:       {p*100:.2f}%")
    print(f"Recall:          {r*100:.2f}%")
    print(f"F1 Score:        {f1*100:.2f}%")
    print(f"Route Coverage:  {coverage*100:.2f}%")
    if ood_acc is not None:
        print(f"OOD Rejection:   {ood_acc*100:.2f}%")
    print(f"Avg Latency:     {avg_latency:.2f}ms")

async def main():
    # CLINC150
    # print("Testing CLINC150...")
    # ds = load_dataset("clinc/clinc_oos", "plus")
    # features = ds["train"].features["intent"]
    # ood_id = features.names.index("oos")
    # await evaluate_dataset("CLINC150", "clinc/clinc_oos", "plus", "intent", "text", ood_label=ood_id)
    
    # Banking77
    print("Testing Banking77...")
    await evaluate_dataset("Banking77", "mteb/banking77", None, "label", "text")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
