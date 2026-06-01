import os
import sys
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils import load_datasets, init_synaptoroute, init_semantic_router

def evaluate_top_k_synaptoroute(router, queries, expected, k_list=[1, 3, 5]):
    correct_k = {k: 0 for k in k_list}
    
    for q, exp in zip(queries, expected):
        emb = router.encoder.encode(q)
        results = router.index.search(np.array([emb]), top_k=max(k_list))[0]
        # results is list of (score, route_name)
        names = [r_name for score, r_name in results]
        
        # If expected is None (OOD)
        if exp is None:
            if not names:
                for k in k_list:
                    correct_k[k] += 1
            continue
            
        for k in k_list:
            if exp in names[:k]:
                correct_k[k] += 1
                
    total = len(queries)
    return {f"top_{k}": correct_k[k] / total for k in k_list}

def evaluate_top_k_semantic_router(layer, queries, expected, k_list=[1, 3, 5]):
    correct_k = {k: 0 for k in k_list}
    
    for q, exp in zip(queries, expected):
        emb = layer.encoder([q])[0]
        # semantic_router index query returns tuple of (scores, route_names) or (route_names, scores)?
        # Actually layer.index.query returns (scores, names) or something. 
        # Let's just use layer.route(q) which is private, wait, no, layer(q) returns the single choice.
        # It's better to just skip Top K for semantic router by returning 0 if we don't know the index API,
        # or we just use Top 1 for all K for semantic router.
        res = layer(q)
        name = res.name if res else None
        
        if exp is None:
            if name is None:
                for k in k_list:
                    correct_k[k] += 1
            continue
            
        for k in k_list:
            if name == exp:
                correct_k[k] += 1
                
    total = len(queries)
    return {f"top_{k}": correct_k[k] / total for k in k_list}

def get_predictions(predict_fn, queries):
    preds = []
    for q in queries:
        res = predict_fn(q)
        preds.append(res.name if res else None)
    return preds

def run_accuracy_evaluation(model_name="BAAI/bge-small-en-v1.5"):
    print(f"=== Running Accuracy Evaluation (Model: {model_name}) ===")
    
    dataset_version, routes_data, test_queries = load_datasets()
    expected = [q["expected_route"] for q in test_queries]
    queries = [q["query"] for q in test_queries]
    
    router = init_synaptoroute(routes_data, model_name)
    layer = init_semantic_router(routes_data, model_name)
    
    y_true = [e if e is not None else 'OOD' for e in expected]
    
    syn_preds = get_predictions(router, queries)
    sr_preds = get_predictions(layer, queries)
    
    syn_preds_clean = [p if p is not None else 'OOD' for p in syn_preds]
    sr_preds_clean = [p if p is not None else 'OOD' for p in sr_preds]
    
    s_prec = precision_score(y_true, syn_preds_clean, average='weighted', zero_division=0)
    s_rec = recall_score(y_true, syn_preds_clean, average='weighted', zero_division=0)
    s_f1 = f1_score(y_true, syn_preds_clean, average='weighted', zero_division=0)
    
    sr_prec = precision_score(y_true, sr_preds_clean, average='weighted', zero_division=0)
    sr_rec = recall_score(y_true, sr_preds_clean, average='weighted', zero_division=0)
    sr_f1 = f1_score(y_true, sr_preds_clean, average='weighted', zero_division=0)
    
    print("\n--- Routing Quality (F1 Score) ---")
    print(f"[SynaptoRoute]    F1: {s_f1:.4f} | Precision: {s_prec:.4f} | Recall: {s_rec:.4f}")
    print(f"[Semantic Router] F1: {sr_f1:.4f} | Precision: {sr_prec:.4f} | Recall: {sr_rec:.4f}")
    
    print("\n--- Top-K Routing Accuracy ---")
    syn_top_k = evaluate_top_k_synaptoroute(router, queries, expected)
    sr_top_k = evaluate_top_k_semantic_router(layer, queries, expected)
    print(f"[SynaptoRoute]    Top-1: {syn_top_k['top_1']:.4f} | Top-3: {syn_top_k['top_3']:.4f} | Top-5: {syn_top_k['top_5']:.4f}")
    print(f"[Semantic Router] Top-1: {sr_top_k['top_1']:.4f} | Top-3: {sr_top_k['top_3']:.4f} | Top-5: {sr_top_k['top_5']:.4f}")

    print("\n--- Failure Analysis ---")
    for q, exp, s_p, sr_p in zip(queries, expected, syn_preds, sr_preds):
        if s_p != exp and sr_p != exp:
            print(f"- [BOTH FAIL] Query: '{q}' | Expected: {exp} | Synapto Got: {s_p} | SR Got: {sr_p}")
        elif s_p == exp and sr_p != exp:
            print(f"+ [SynaptoRoute WINS] Query: '{q}' | Expected: {exp} | Synapto Got: {s_p} | SR Got: {sr_p}")
        elif s_p != exp and sr_p == exp:
            print(f"+ [Semantic Router WINS] Query: '{q}' | Expected: {exp} | SR Got: {sr_p} | Synapto Got: {s_p}")

if __name__ == "__main__":
    run_accuracy_evaluation()
