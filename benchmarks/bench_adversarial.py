import random
import string
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage

def introduce_typo(text):
    if len(text) < 3:
        return text
    
    chars = list(text)
    num_typos = random.randint(1, 2)
    for _ in range(num_typos):
        idx = random.randint(0, len(chars) - 1)
        action = random.choice(["replace", "delete", "insert", "swap"])
        if action == "replace":
            chars[idx] = random.choice(string.ascii_lowercase)
        elif action == "delete" and len(chars) > 2:
            chars.pop(idx)
        elif action == "insert":
            chars.insert(idx, random.choice(string.ascii_lowercase))
        elif action == "swap" and idx < len(chars) - 1:
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            
    return "".join(chars)

def main():
    print("Loading router...")
    encoder = Encoder(model_name="BAAI/bge-small-en-v1.5")
    storage = SQLiteStorage(db_path="data/router_memory.sqlite")
    router = AdaptiveRouter(encoder, storage)
    
    queries = []
    for route_name, route in router._route_map.items():
        for utt in route.utterances:
            queries.append((utt, route_name))
            
    random.shuffle(queries)
    queries = queries[:50]
    
    print(f"Selected {len(queries)} valid queries. Benchmarking Adversarial Typo Robustness...")
    
    correct_original = 0
    correct_typo = 0
    
    for q_text, expected_route in queries:
        match_orig = router(q_text)
        if match_orig and match_orig.name == expected_route:
            correct_original += 1
            
        typo_text = introduce_typo(q_text)
        match_typo = router(typo_text)
        if match_typo and match_typo.name == expected_route:
            correct_typo += 1
            
    print(f"Original Accuracy: {correct_original}/{len(queries)} ({(correct_original/len(queries))*100:.1f}%)")
    print(f"Typo Accuracy: {correct_typo}/{len(queries)} ({(correct_typo/len(queries))*100:.1f}%)")
    if correct_original > 0:
        deg = (correct_original - correct_typo) / correct_original * 100
        print(f"Accuracy Degradation: {deg:.1f}%")

if __name__ == '__main__':
    main()
