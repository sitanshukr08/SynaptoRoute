import os
import json
import glob
from pathlib import Path
from collections import defaultdict

def calculate_similarity(s1, s2):
    # Simple Jaccard similarity for near-duplicate filtering
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def is_near_duplicate(text, existing_texts, threshold=0.8):
    for ex in existing_texts:
        if calculate_similarity(text, ex) >= threshold:
            return True
    return False

def validate_and_filter(data):
    validated_routes = []
    all_utterances = set()
    
    # Process Routes
    for route in data.get("routes", []):
        filtered_utterances = []
        for utt in route.get("utterances", []):
            if not utt.strip():
                continue
            # Deduplication & Near-duplicate filtering
            if not is_near_duplicate(utt, filtered_utterances):
                filtered_utterances.append(utt)
                all_utterances.add(utt.lower())
                
        validated_routes.append({
            "name": route["name"],
            "utterances": filtered_utterances
        })
        
    # Route balance checks
    if validated_routes:
        min_count = min(len(r["utterances"]) for r in validated_routes)
        # Cap all routes to max of (min_count * 1.5) to maintain balance
        max_allowed = int(max(min_count * 1.5, 10))
        for r in validated_routes:
            if len(r["utterances"]) > max_allowed:
                r["utterances"] = r["utterances"][:max_allowed]
                
    # Process Test Queries and check for leakage
    validated_queries = []
    for q in data.get("test_queries", []):
        query_text = q.get("query", "").lower()
        if not query_text:
            continue
        # Leakage detection (exact match with any training utterance)
        if query_text in all_utterances:
            continue
        
        validated_queries.append(q)
        
    return {"routes": validated_routes, "test_queries": validated_queries}

def split_datasets(data):
    # Very basic simulation of splitting into standard, hard_negative, and adversarial
    # In practice, this would rely on metadata or more complex heuristics
    standard = {"routes": data["routes"], "test_queries": []}
    hard_negative = {"routes": data["routes"], "test_queries": []}
    adversarial = {"routes": data["routes"], "test_queries": []}
    
    for i, q in enumerate(data.get("test_queries", [])):
        if i % 3 == 0:
            adversarial["test_queries"].append(q)
        elif i % 3 == 1:
            hard_negative["test_queries"].append(q)
        else:
            standard["test_queries"].append(q)
            
    return standard, hard_negative, adversarial

def main():
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "benchmarks" / "datasets"
    intermediate_dir = dataset_dir / "intermediate"
    
    out_dirs = {
        "standard": dataset_dir / "standard",
        "hard_negative": dataset_dir / "hard_negative",
        "adversarial": dataset_dir / "adversarial"
    }
    
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    if not intermediate_dir.exists():
        print(f"Intermediate directory not found: {intermediate_dir}")
        return
        
    candidate_files = glob.glob(str(intermediate_dir / "*.json"))
    
    for cand_file in candidate_files:
        print(f"Validating {cand_file}...")
        with open(cand_file, "r") as f:
            data = json.load(f)
            
        validated_data = validate_and_filter(data)
        standard, hard_negative, adversarial = split_datasets(validated_data)
        
        filename = Path(cand_file).name
        with open(out_dirs["standard"] / filename, "w") as f:
            json.dump(standard, f, indent=2)
            
        with open(out_dirs["hard_negative"] / filename, "w") as f:
            json.dump(hard_negative, f, indent=2)
            
        with open(out_dirs["adversarial"] / filename, "w") as f:
            json.dump(adversarial, f, indent=2)
            
        print(f"Saved validated datasets for {filename}")

if __name__ == "__main__":
    main()
