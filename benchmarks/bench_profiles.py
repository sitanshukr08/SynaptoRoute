import asyncio
import time
import numpy as np
import concurrent.futures
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.models import Route
from synaptoroute.profile import get_profile, ProfileType

class DummyStorage:
    def __init__(self): pass
    def save_route(self, route, embeddings=None): pass
    def delete_route(self, route_name): pass
    def load_all_routes(self): return [], []
    def update_threshold(self, route_name, threshold): pass

async def bench_async_blast(router: AdaptiveRouter, num_requests: int):
    await router.start()
    start_time = time.time()
    
    tasks = [router.aquery(f"dummy query {i}") for i in range(num_requests)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    duration = time.time() - start_time
    await router.stop()
    return duration

def bench_sync_blast(router: AdaptiveRouter, num_requests: int):
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(router.__call__, f"dummy query {i}") for i in range(num_requests)]
        concurrent.futures.wait(futures)
        
    duration = time.time() - start_time
    return duration

def bench_sequential(router: AdaptiveRouter, num_requests: int):
    start_time = time.time()
    for i in range(num_requests):
        router(f"dummy query {i}")
    duration = time.time() - start_time
    return duration

async def main():
    print("Initializing components...")
    # Use CPU explicitly to stress thread locks
    encoder_latency = Encoder(providers=["CPUExecutionProvider"], threads=get_profile(ProfileType.LATENCY).threads)
    router_latency = AdaptiveRouter(encoder_latency, DummyStorage(), profile=get_profile(ProfileType.LATENCY), max_capacity=5000)

    encoder_throughput = Encoder(providers=["CPUExecutionProvider"], threads=get_profile(ProfileType.THROUGHPUT).threads)
    router_throughput = AdaptiveRouter(encoder_throughput, DummyStorage(), profile=get_profile(ProfileType.THROUGHPUT), max_capacity=5000)
    
    # Warmup and add routes
    print("Loading 1,000 dummy routes...")
    for i in range(1000):
        route = Route(name=f"route_{i}", utterances=[f"this is utterance {i}"])
        router_latency.add_route(route)
        router_throughput.add_route(route)
        
    print("\n=========================================")
    print("BRUTAL STRESS TEST: PROFILES")
    print("=========================================\n")
    
    # Test 1: Isolated Sequential Latency
    reqs = 100
    print(f"[1] Sequential P50 Latency ({reqs} iterations)")
    lat_seq = bench_sequential(router_latency, reqs)
    thru_seq = bench_sequential(router_throughput, reqs)
    print(f"LATENCY Profile:    {(lat_seq/reqs)*1000:.2f} ms/query (Expected: FASTER)")
    print(f"THROUGHPUT Profile: {(thru_seq/reqs)*1000:.2f} ms/query (Expected: SLOWER)")
    
    # Test 2: Synchronous Thread Thrashing
    reqs = 500
    print(f"\n[2] Sync ThreadPool Blast ({reqs} queries, 50 workers)")
    lat_sync = bench_sync_blast(router_latency, reqs)
    thru_sync = bench_sync_blast(router_throughput, reqs)
    print(f"LATENCY Profile:    {reqs/lat_sync:.2f} QPS")
    print(f"THROUGHPUT Profile: {reqs/thru_sync:.2f} QPS")
    
    # Test 3: Asynchronous Batch Queue Overload
    reqs = 2000
    print(f"\n[3] Asyncio Gather Blast ({reqs} queries)")
    lat_async = await bench_async_blast(router_latency, reqs)
    thru_async = await bench_async_blast(router_throughput, reqs)
    print(f"LATENCY Profile:    {reqs/lat_async:.2f} QPS (Expected: BOTTLENECKED)")
    print(f"THROUGHPUT Profile: {reqs/thru_async:.2f} QPS (Expected: MASSIVE MULTIPLIER)")

if __name__ == "__main__":
    asyncio.run(main())
