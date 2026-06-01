import os
import sys
import glob
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from synaptoroute.encoder import FastEmbedEncoder

def check_exact_duplicates(all_queries):
    # all_queries: list of (query_text, expected_route)
    seen = defaultdict(set)
    duplicates = []
    
    for q_text, route in all_queries:
        if q_text in seen:
            if route not in seen[q_text]:
                duplicates.append((q_text, list(seen[q_text]) + [route]))
        seen[q_text].add(route)
        
    return duplicates

def validate_datasets(dataset_dir="benchmarks/datasets/standard"):
    print("=== Validating Dataset Integrity ===\n")
    
    files = glob.glob(os.path.join(dataset_dir, "*.json"))
    if not files:
        print("No datasets found!")
        return
        
    all_queries = [] # list of (text, route_name)
    route_queries = defaultdict(list)
    
    for f in files:
        with open(f, "r") as f_obj:
            data = json.load(f_obj)
            for item in data.get("test_queries", []):
                q = item["query"]
                exp = item["expected_route"]
                if exp is None:
                    exp = "None"
                all_queries.append((q, exp))
                route_queries[exp].append(q)
                
    print(f"Total queries loaded: {len(all_queries)}")
    print(f"Total routes loaded: {len(route_queries)}")
    
    # 1. Exact Duplicates across routes
    exact_dups = check_exact_duplicates(all_queries)
    if exact_dups:
        print("\n[FAIL] Found exact duplicates across different routes:")
        for q_text, routes in exact_dups[:10]:
            print(f"  - '{q_text}' appears in: {routes}")
        sys.exit(1)
    else:
        print("\n[PASS] No exact duplicates found across routes.")
        
    # 2. Embedding overlap
    print("\nEncoding all queries to check for semantic overlap...")
    encoder = FastEmbedEncoder(model_name="BAAI/bge-small-en-v1.5")
    
    route_names = list(route_queries.keys())
    route_embeddings = {}
    
    for r in route_names:
        qs = route_queries[r]
        if not qs:
            continue
        embs = encoder.encode_batch(qs)
        route_embeddings[r] = embs
        
    # Intra vs Inter route similarity
    print("\nCalculating Route Overlap Scores (Cosine Similarity)...")
    
    warnings = 0
    for i, r1 in enumerate(route_names):
        embs1 = route_embeddings.get(r1)
        if embs1 is None or len(embs1) == 0:
            continue
            
        # Intra
        intra_sim = cosine_similarity(embs1, embs1)
        # Exclude self-similarity diagonal
        np.fill_diagonal(intra_sim, 0)
        avg_intra = np.mean(intra_sim) * (len(embs1) / (len(embs1) - 1)) if len(embs1) > 1 else 1.0
        
        max_inter = 0.0
        max_inter_route = ""
        
        for j, r2 in enumerate(route_names):
            if i == j:
                continue
            embs2 = route_embeddings.get(r2)
            if embs2 is None or len(embs2) == 0:
                continue
                
            inter_sim = cosine_similarity(embs1, embs2)
            avg_inter = np.mean(inter_sim)
            
            # Check near duplicates
            near_dups = np.where(inter_sim > 0.95)
            if len(near_dups[0]) > 0:
                print(f"[WARN] Found {len(near_dups[0])} near-duplicates (>0.95 sim) between '{r1}' and '{r2}'")
                idx1 = near_dups[0][0]
                idx2 = near_dups[1][0]
                print(f"       Ex: '{route_queries[r1][idx1]}' vs '{route_queries[r2][idx2]}'")
                warnings += 1
                
            if avg_inter > max_inter:
                max_inter = avg_inter
                max_inter_route = r2
                
        print(f"Route '{r1}':")
        print(f"  Intra-similarity: {avg_intra:.3f}")
        print(f"  Max Inter-similarity: {max_inter:.3f} (with '{max_inter_route}')")
        
        if max_inter >= avg_intra:
            print(f"  [FAIL] Route '{r1}' overlaps more with '{max_inter_route}' than with itself!")
            sys.exit(1)
            
    print(f"\nValidation Complete. {warnings} warnings.")

if __name__ == "__main__":
    validate_datasets()
