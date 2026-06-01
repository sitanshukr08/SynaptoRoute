import os
import sys
import time
import asyncio
from unittest.mock import patch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils import load_datasets, init_synaptoroute, init_semantic_router

async def profile_synaptoroute(router, queries):
    print("\n--- Profiling SynaptoRoute Components ---")
    
    timings = {"encode": [], "search": [], "total": []}
    
    original_encode = router.encoder.encode
    def mock_encode(text):
        t0 = time.perf_counter()
        res = original_encode(text)
        timings["encode"].append(time.perf_counter() - t0)
        return res
        
    original_encode_batch = router.encoder.encode_batch
    def mock_encode_batch(texts):
        t0 = time.perf_counter()
        res = original_encode_batch(texts)
        elapsed = (time.perf_counter() - t0) / len(texts)
        for _ in texts:
            timings["encode"].append(elapsed)
        return res
        
    with patch.object(router.encoder, 'encode', side_effect=mock_encode), \
         patch.object(router.encoder, 'encode_batch', side_effect=mock_encode_batch):
        
        await router.start()
        t0 = time.perf_counter()
        await asyncio.gather(*(router.aquery(q) for q in queries))
        total_time = (time.perf_counter() - t0) / len(queries)
        timings["total"] = [total_time] * len(queries)
        await router.stop()
        
    avg_encode = np.mean(timings["encode"]) * 1000
    avg_total = np.mean(timings["total"]) * 1000
    avg_overhead = avg_total - avg_encode
    
    print(f"Embedding Generation: {avg_encode:.2f} ms / query")
    print(f"Search & Queueing:  {avg_overhead:.2f} ms / query")
    print(f"Total Latency:      {avg_total:.2f} ms / query")

def profile_semantic_router(layer, queries):
    print("\n--- Profiling Semantic Router Components ---")
    
    timings = {"encode": [], "total": []}
    
    original_call = layer.encoder.__call__
    def mock_encode(docs):
        t0 = time.perf_counter()
        res = original_call(docs)
        elapsed = (time.perf_counter() - t0) / len(docs)
        for _ in docs:
            timings["encode"].append(elapsed)
        return res
        
    with patch.object(layer.encoder, '__call__', side_effect=mock_encode):
        t0 = time.perf_counter()
        for q in queries:
            layer(q)
        total_time = (time.perf_counter() - t0) / len(queries)
        timings["total"] = [total_time] * len(queries)
        
    avg_encode = np.mean(timings["encode"]) * 1000
    avg_total = np.mean(timings["total"]) * 1000
    avg_overhead = avg_total - avg_encode
    
    print(f"Embedding Generation: {avg_encode:.2f} ms / query")
    print(f"Search & Overhead:  {avg_overhead:.2f} ms / query")
    print(f"Total Latency:      {avg_total:.2f} ms / query")

async def main():
    print("=== Component-Level Performance Profiling ===")
    dataset_version, routes_data, test_queries = load_datasets()
    query_texts = [q["query"] for q in test_queries]
    
    query_texts = (query_texts * (100 // len(query_texts) + 1))[:100]
    
    router = init_synaptoroute(routes_data)
    layer = init_semantic_router(routes_data)
    
    await profile_synaptoroute(router, query_texts)
    profile_semantic_router(layer, query_texts)

if __name__ == "__main__":
    asyncio.run(main())
