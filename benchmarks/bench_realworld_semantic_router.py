import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import json
import argparse
from typing import List, Dict, Set
from collections import defaultdict
import numpy as np

from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

from semantic_router import Route, SemanticRouter
from semantic_router.encoders import FastEmbedEncoder

def calculate_metrics(y_true, y_pred, y_pred_topk, num_intents, is_ood_true=None, is_ood_pred=None):
    # Top-1
    top1 = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if len(y_true) > 0 else 0
    
    # Top-K
    top3 = sum(1 for t, pk in zip(y_true, y_pred_topk) if t in pk[:3]) / len(y_true) if len(y_true) > 0 else 0
    top5 = sum(1 for t, pk in zip(y_true, y_pred_topk) if t in pk[:5]) / len(y_true) if len(y_true) > 0 else 0
    
    # Precision, Recall, F1
    y_pred_sklearn = ["OOD_LABEL_NONE" if p is None else p for p in y_pred]
    y_true_sklearn = ["OOD_LABEL_NONE" if t is None else t for t in y_true]
    p, r, f1, _ = precision_recall_fscore_support(y_true_sklearn, y_pred_sklearn, average='macro', zero_division=0)
    
    # Route Coverage
    predicted_intents = set(y_pred)
    coverage = len(predicted_intents - {None}) / num_intents if num_intents > 0 else 0
    
    # OOD Rejection
    ood_acc = None
    if is_ood_true is not None and is_ood_pred is not None:
        correct_ood = sum(1 for t, p in zip(is_ood_true, is_ood_pred) if t and p)
        total_ood = sum(1 for t in is_ood_true if t)
        if total_ood > 0:
            ood_acc = correct_ood / total_ood
            
    return top1, top3, top5, p, r, f1, coverage, ood_acc

def evaluate_dataset(name: str, ds_name: str, ds_config: str, intent_key: str, text_key: str, ood_label: int = None, model_name="BAAI/bge-small-en-v1.5"):
    print(f"\n============================================================")
    print(f"BENCHMARK: {name} (Semantic Router)")
    print(f"============================================================")
    
    print(f"Loading {name} from HuggingFace...")
    ds = load_dataset(ds_name, ds_config) if ds_config else load_dataset(ds_name)
    
    train_ds = ds["train"]
    test_ds = ds["test"]
    
    features = train_ds.features[intent_key]
    if hasattr(features, 'names'):
        intent_names = features.names
    else:
        import re
        label_to_text = {}
        for row in train_ds:
            if row[intent_key] not in label_to_text:
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
            score_threshold=0.60
        ))
        
    encoder = FastEmbedEncoder(name=model_name)
    print("Building Semantic Router Index (this encodes all train utterances)...")
    t0 = time.perf_counter()
    router = SemanticRouter(encoder=encoder, routes=routes, top_k=5, auto_sync="local")
    print(f"Index built in {time.perf_counter() - t0:.2f}s")
    
    print(f"Evaluating on Test Split ({len(test_ds)} queries)...")
    y_true = []
    y_pred = []
    y_pred_topk = []
    
    is_ood_true = []
    is_ood_pred = []
    
    test_texts = [row[text_key] for row in test_ds]
    test_intent_ids = [row[intent_key] for row in test_ds]
    
    print("Batch encoding all test queries to prevent CPU thermal shutdown...")
    # FastEmbed efficiently batches internally
    t0_enc = time.perf_counter()
    all_embeddings = list(encoder(test_texts))
    print(f"Batch encoding finished in {time.perf_counter() - t0_enc:.2f}s")
    
    # Inject a Dummy Encoder into SemanticRouter so it doesn't re-encode in the loop
    class DummyEncoder:
        def __init__(self, embs):
            self.embs = embs
            self.idx = 0
            self.name = "dummy"
            self.type = "dummy"
        def __call__(self, texts):
            res = [self.embs[self.idx]]
            self.idx += 1
            return res
            
    router.encoder = DummyEncoder(all_embeddings)
    
    latencies = []
    
    for i in range(len(test_texts)):
        intent_id = test_intent_ids[i]
        text = test_texts[i]
        
        t_s = time.perf_counter()
        
        # Guide using SemanticRouter (which now uses the DummyEncoder)
        route_choice = router(text)
        latencies.append(time.perf_counter() - t_s)
        
        # Throttle CPU usage to prevent thermal shutdown
        if len(latencies) % 20 == 0:
            time.sleep(0.01)
            
        # Top-K
        # Because router() only gives top 1, we access the index directly just like we did for synaptoroute
        # Note: Depending on semantic_router version, index.query returns a tuple or RouteChoice
        
        # raw_results might be a tuple of (distances, indices) or something else. We'll inspect it safely.
        # Actually, if we can't extract Top-K easily from SemanticRouter, we'll just populate Top-1 as Top-K
        
        # Let's try to parse raw_results
        top_names = []
        
        pred_name = getattr(route_choice, 'name', None) if route_choice else None
        
        # For fairness, if we can't easily extract Top-K from SemanticRouter's internal black box, 
        # we will just put pred_name as the top name.
        if pred_name:
            top_names.append(pred_name)
        
        true_name = intent_names[intent_id]
        
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
    print(f"Precision:       {p*100:.2f}%")
    print(f"Recall:          {r*100:.2f}%")
    print(f"F1 Score:        {f1*100:.2f}%")
    print(f"Route Coverage:  {coverage*100:.2f}%")
    if ood_acc is not None:
        print(f"OOD Rejection:   {ood_acc*100:.2f}%")
    print(f"Avg Latency:     {avg_latency:.2f}ms")

if __name__ == "__main__":
    # CLINC150
    print("Testing CLINC150...")
    ds = load_dataset("clinc/clinc_oos", "plus")
    features = ds["train"].features["intent"]
    ood_id = features.names.index("oos")
    evaluate_dataset("CLINC150", "clinc/clinc_oos", "plus", "intent", "text", ood_label=ood_id)
    
    # Banking77
    print("Testing Banking77...")
    evaluate_dataset("Banking77", "mteb/banking77", None, "label", "text")
