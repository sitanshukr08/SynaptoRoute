import time
import asyncio
import numpy as np

# SynaptoRoute imports
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder as SREncoder
from synaptoroute.storage import BaseStorage
from synaptoroute.models import Route as SR_ModelRoute

# Semantic Router imports
try:
    from semantic_router import Route
    from semantic_router import SemanticRouter
    from semantic_router.encoders import FastEmbedEncoder
except ImportError as e:
    print(f"Warning: Ensure you have semantic-router installed. Error: {e}")
    exit(1)

class DummyStorage(BaseStorage):
    def load_all_routes(self): return []
    def save_route(self, route): pass
    def add_utterance(self, name, utt): pass
    def get_route(self, name): return None

def bench_hot_reload():
    print("--- Test 1: Hot-Reload O(N) Degradation ---")
    
    print("Initializing Semantic-Router (Eager Compilation)...")
    sr_encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5")
    sr_layer = SemanticRouter(encoder=sr_encoder, routes=[])
    
    sr_times = []
    # Add 500 routes dynamically
    for i in range(500):
        start = time.perf_counter()
        route = Route(name=f"route_{i}", utterances=[f"dummy utterance {i}"])
        sr_layer.add(route)
        sr_times.append((time.perf_counter() - start) * 1000)
    
    print("Initializing SynaptoRoute (Lazy Compilation)...")
    try:
        syn_encoder = SREncoder(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except:
        syn_encoder = SREncoder()
        
    syn_storage = DummyStorage()
    syn_router = AdaptiveRouter(syn_encoder, syn_storage)
    
    syn_times = []
    # Add 500 routes dynamically
    for i in range(500):
        start = time.perf_counter()
        route = SR_ModelRoute(name=f"route_{i}", utterances=[f"dummy utterance {i}"])
        syn_router.add_route(route)
        syn_times.append((time.perf_counter() - start) * 1000)

    print(f"\n[Semantic-Router]")
    print(f"10th Route Addition: {np.mean(sr_times[5:15]):.2f} ms")
    print(f"490th Route Addition: {np.mean(sr_times[485:495]):.2f} ms")
    print(f"Degradation: {np.mean(sr_times[485:495]) - np.mean(sr_times[5:15]):.2f} ms")
    
    print(f"\n[SynaptoRoute]")
    print(f"10th Route Addition: {np.mean(syn_times[5:15]):.2f} ms")
    print(f"490th Route Addition: {np.mean(syn_times[485:495]):.2f} ms")
    print(f"Degradation: {np.mean(syn_times[485:495]) - np.mean(syn_times[5:15]):.2f} ms")

async def bench_concurrency():
    print("\n--- Test 2: Concurrency & Throughput ---")
    sr_encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5")
    sr_layer = SemanticRouter(encoder=sr_encoder, routes=[Route(name="dummy", utterances=["test"])])
    
    try:
        syn_encoder = SREncoder(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except:
        syn_encoder = SREncoder()
        
    syn_storage = DummyStorage()
    syn_router = AdaptiveRouter(syn_encoder, syn_storage)
    syn_router.add_route(SR_ModelRoute(name="dummy", utterances=["test"]))
    await syn_router.start()
    
    queries = [f"test query {i}" for i in range(100)]
    
    print("\nFiring 100 queries at Semantic-Router (Sequential)...")
    start = time.perf_counter()
    for q in queries:
        sr_layer(q)
    sr_total = time.perf_counter() - start
    
    print("Firing 100 concurrent queries at SynaptoRoute (Dynamic Batching)...")
    start = time.perf_counter()
    tasks = [syn_router.aquery(q) for q in queries]
    await asyncio.gather(*tasks)
    syn_total = time.perf_counter() - start
    
    print(f"\n[Semantic-Router] Total Time: {sr_total:.2f} s")
    print(f"[SynaptoRoute] Total Time: {syn_total:.2f} s")
    print(f"Speedup: {sr_total / syn_total:.1f}x faster")
    
    await syn_router.stop()

if __name__ == "__main__":
    bench_hot_reload()
    asyncio.run(bench_concurrency())
