import time
import random
import statistics
import numpy as np
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage

def main():
    print("Loading router...")
    encoder = Encoder(model_name="BAAI/bge-small-en-v1.5")
    storage = SQLiteStorage(db_path="data/router_memory.sqlite")
    router = AdaptiveRouter(encoder, storage)
    
    queries = ["hello", "cancel my order", "I need a refund", "where is my package", "speak to a human"] * 200
    random.shuffle(queries)
    import os
    num_queries = 100 if os.environ.get("CI") else 1000
    queries = queries[:num_queries]
    
    latencies = []
    print("Benchmarking Inference Latency...")
    # warm up
    router("warm up query")
    for q in queries:
        start = time.perf_counter()
        router(q)
        latencies.append((time.perf_counter() - start) * 1000) # ms
        
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"Inference Latency: P50={p50:.2f}ms, P95={p95:.2f}ms, P99={p99:.2f}ms")
    
    print("Benchmarking Hot-Reload Latency...")
    route_name = list(router._route_map.keys())[0] if router._route_map else "test_route"
    if not router._route_map:
        from synaptoroute.models import Route
        router.add_route(Route(name=route_name, utterances=["dummy"], threshold=0.5))
        
    hot_reload_latencies = []
    for i in range(100):
        utterance = f"new utterance {i} for {route_name}"
        start = time.perf_counter()
        router.add_utterance(route_name, utterance)
        hot_reload_latencies.append((time.perf_counter() - start) * 1000)
        
    hr_avg = sum(hot_reload_latencies) / len(hot_reload_latencies)
    print(f"Hot-Reload Latency (add_utterance): Avg={hr_avg:.2f}ms")

if __name__ == '__main__':
    main()
