import os
import sys
import numpy as np
import time
import asyncio

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils import load_datasets, init_synaptoroute, init_semantic_router
from stats_utils import calculate_statistics, print_statistics_report

async def measure_latency(predict_fn, queries, is_async=False):
    latencies = []
    
    if is_async:
        t0 = time.perf_counter()
        await asyncio.gather(*(predict_fn(q) for q in queries))
        total = time.perf_counter() - t0
        # For pure concurrent latency without returning per-query wait times, 
        # we can just use total / len, but wait, the prompt asks for percentile profiles under load.
        # So we actually measure the time taken to process the batch of N queries.
        # But wait, the standard way is to record the end-to-end time for each query.
        
        async def timed_query(q):
            start = time.perf_counter()
            await predict_fn(q)
            return time.perf_counter() - start
            
        latencies = await asyncio.gather(*(timed_query(q) for q in queries))
    else:
        for q in queries:
            start = time.perf_counter()
            predict_fn(q)
            latencies.append(time.perf_counter() - start)
            
    return latencies

async def main():
    print("=== Running Latency Evaluation (Model: BAAI/bge-small-en-v1.5) ===")
    print("Initializing Routers...")
    dataset_version, routes_data, test_queries = load_datasets()
    query_texts = [q["query"] for q in test_queries]
    
    router = init_synaptoroute(routes_data)
    layer = init_semantic_router(routes_data)
    
    await router.start()
    
    # Warmup
    print("Warming up...")
    await router.aquery("warmup")
    layer("warmup")
    
    load_profiles = [1, 100, 1000]
    for count in load_profiles:
        print(f"\n--- Load Profile: {count} Concurrent Queries ---")
        
        # Test queries expanded to match profile count
        test_queries_expanded = (query_texts * (count // len(query_texts) + 1))[:count]
        
        # Semantic Router Latency
        sr_latencies = []
        chunk_size = 100
        for i in range(0, len(test_queries_expanded), chunk_size):
            chunk = test_queries_expanded[i:i+chunk_size]
            async def sr_predict(q):
                return await asyncio.to_thread(layer, q)
            chunk_lat = await measure_latency(sr_predict, chunk, is_async=True)
            sr_latencies.extend(chunk_lat)
        sr_p50 = np.percentile(sr_latencies, 50) * 1000
        sr_p90 = np.percentile(sr_latencies, 90) * 1000
        sr_p95 = np.percentile(sr_latencies, 95) * 1000
        sr_p99 = np.percentile(sr_latencies, 99) * 1000
        sr_worst = np.max(sr_latencies) * 1000
        
        # SynaptoRoute Latency
        s_latencies = await measure_latency(router.aquery, test_queries_expanded, is_async=True)
        s_p50 = np.percentile(s_latencies, 50) * 1000
        s_p90 = np.percentile(s_latencies, 90) * 1000
        s_p95 = np.percentile(s_latencies, 95) * 1000
        s_p99 = np.percentile(s_latencies, 99) * 1000
        s_worst = np.max(s_latencies) * 1000
        
        print(f"[Semantic Router] P50: {sr_p50:.2f}ms | P90: {sr_p90:.2f}ms | P95: {sr_p95:.2f}ms | P99: {sr_p99:.2f}ms | Worst: {sr_worst:.2f}ms")
        print(f"[SynaptoRoute]    P50: {s_p50:.2f}ms | P90: {s_p90:.2f}ms | P95: {s_p95:.2f}ms | P99: {s_p99:.2f}ms | Worst: {s_worst:.2f}ms")
        
        stats_res = calculate_statistics(s_latencies, sr_latencies)
        print_statistics_report(stats_res, name_a="SynaptoRoute", name_b="Semantic Router")

    await router.stop()

if __name__ == "__main__":
    asyncio.run(main())
