import os
import random
import time
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage

def main():
    print("Loading router...")
    encoder = Encoder(model_name="BAAI/bge-small-en-v1.5")
    storage = SQLiteStorage(db_path="data/router_memory.sqlite")
    router = AdaptiveRouter(encoder, storage)
    
    ood_queries = [
        "how do I bake a cake", "what is the capital of France", "tell me a joke",
        "who won the world cup in 2018", "can dogs eat grapes", "how to tie a tie",
        "what is the meaning of life", "where is the nearest gas station",
        "how many ounces in a cup", "how to solve a rubik's cube",
        "what time is it in Tokyo", "what's the weather like today",
        "how to cook rice", "who is the president of the US",
        "how to draw a dog", "do aliens exist", "how to lose weight fast",
        "what is bitcoin", "how to make pancakes", "how to tie shoes",
        "what is a noun", "how to get pregnant", "what are the 7 wonders of the world",
        "how to lower blood pressure", "who is elon musk", "what is lupus",
        "how to play chess", "what is the largest ocean", "who wrote romeo and juliet",
        "what is the speed of light", "how to boil an egg", "what is my IP address",
        "how to screenshot on mac", "what is the longest river in the world",
        "who painted the mona lisa", "what is a black hole", "how to make money online",
        "what is the richest country in the world", "how to block a number",
        "who is the richest person in the world", "what is area 51",
        "how to write a cover letter", "what is the highest mountain in the world",
        "who invented the light bulb", "what is the smallest country in the world",
        "how to make slime", "what is the best movie of all time",
        "who is the fastest runner in the world", "what is the strongest animal",
        "how to get rid of hiccups"
    ]
    
    calibration_samples = []
    calibration_labels = []
    in_domain_queries = []
    
    for route_name, route in router._route_map.items():
        # First 5 for testing
        in_domain_queries.extend(route.utterances[:5])
        # Next 5 for calibration
        cal_utts = route.utterances[5:10]
        calibration_samples.extend(cal_utts)
        calibration_labels.extend([route_name] * len(cal_utts))
    
    # Use 20 OOD queries for calibration
    calibration_samples.extend(ood_queries[-20:])
    calibration_labels.extend(["OOD"] * 20)
    
    # Remaining 30 OOD queries for testing
    ood_queries = ood_queries[:-20]
    
    random.shuffle(in_domain_queries)
    in_domain_queries = in_domain_queries[:50]
    
    print("Calibrating thresholds to improve OOD rejection...")
    router.fit_thresholds(calibration_samples, calibration_labels)
    
    print(f"Testing with {len(in_domain_queries)} In-Domain and {len(ood_queries)} OOD queries.")
    
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    
    for q in in_domain_queries:
        if router(q) is not None:
            tp += 1
        else:
            fn += 1
            
    for q in ood_queries:
        if router(q) is not None:
            fp += 1
        else:
            tn += 1
            
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"True Positives: {tp}")
    print(f"False Negatives: {fn}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == '__main__':
    main()
