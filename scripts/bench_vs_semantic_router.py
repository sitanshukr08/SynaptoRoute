import time
import asyncio
import os

import random
from typing import List

# Semantic Router imports
from semantic_router import Route as SemanticRoute
from semantic_router import SemanticRouter as RouteLayer
from semantic_router.encoders import FastEmbedEncoder as SemanticFastEmbedEncoder

# SynaptoRoute imports
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder as SynaptoEncoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.models import Route as SynaptoRoute
from synaptoroute.profile import get_profile, ProfileType


def generate_synthetic_routes(num_routes: int, utterances_per_route: int) -> List[dict]:
    """Generates random routes to populate the routers."""
    routes = []
    for i in range(num_routes):
        utterances = [f"This is a sample utterance {j} for intent {i} about something specific." for j in range(utterances_per_route)]
        routes.append({"name": f"intent_{i}", "utterances": utterances})
    return routes

async def benchmark_synaptoroute(routes_data: List[dict], test_queries: List[str]):
    print("\n--- Benchmarking SynaptoRoute ---")
    storage_path = "bench_synapto.sqlite"
    if os.path.exists(storage_path):
        os.remove(storage_path)

    encoder = SynaptoEncoder(model_name="BAAI/bge-small-en-v1.5", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    storage = SQLiteStorage(storage_path)
    profile = get_profile(ProfileType.THROUGHPUT)
    router = AdaptiveRouter(encoder=encoder, storage=storage, profile=profile)

    # 1. Load routes
    print(f"Loading {len(routes_data)} routes...")
    start_load = time.time()
    for route_data in routes_data:
        route = SynaptoRoute(name=route_data["name"], utterances=route_data["utterances"])
        router.add_route(route)
    load_time = time.time() - start_load
    print(f"Initial Load Time: {load_time:.2f}s")

    # 2. Hot Reload Penalty
    new_route = SynaptoRoute(name="hot_reload_test", utterances=["This is a brand new utterance testing reload." * 5])
    start_reload = time.time()
    router.add_route(new_route)
    reload_time = time.time() - start_reload
    print(f"Hot-Reload Penalty (add_route): {reload_time * 1000:.2f} ms")

    # 3. Concurrent Throughput
    await router.start()
    
    print(f"Executing {len(test_queries)} concurrent async queries...")
    start_query = time.time()
    tasks = [router.aquery(q) for q in test_queries]
    await asyncio.gather(*tasks)
    query_time = time.time() - start_query
    print(f"Total Async Query Time: {query_time:.2f}s")
    print(f"Sustained QPS: {len(test_queries) / query_time:.2f}")

    await router.stop()
    if os.path.exists(storage_path):
        os.remove(storage_path)

def benchmark_semantic_router(routes_data: List[dict], test_queries: List[str]):
    print("\n--- Benchmarking semantic-router ---")
    encoder = SemanticFastEmbedEncoder(name="BAAI/bge-small-en-v1.5")
    
    # 1. Load routes
    print(f"Loading {len(routes_data)} routes...")
    start_load = time.time()
    s_routes = []
    for route_data in routes_data:
        s_routes.append(SemanticRoute(name=route_data["name"], utterances=route_data["utterances"]))
    layer = RouteLayer(encoder=encoder, routes=s_routes)
    load_time = time.time() - start_load
    print(f"Initial Load Time: {load_time:.2f}s")

    # 2. Hot Reload Penalty
    new_route = SemanticRoute(name="hot_reload_test", utterances=["This is a brand new utterance testing reload." * 5])
    start_reload = time.time()
    layer.add(new_route)
    reload_time = time.time() - start_reload
    print(f"Hot-Reload Penalty (add_route): {reload_time * 1000:.2f} ms")

    # 3. Synchronous Throughput (Semantic router is blocking)
    print(f"Executing {len(test_queries)} sequential queries (no async batching)...")
    start_query = time.time()
    for q in test_queries:
        layer(q)
    query_time = time.time() - start_query
    print(f"Total Query Time: {query_time:.2f}s")
    print(f"Sustained QPS: {len(test_queries) / query_time:.2f}")


async def main():
    NUM_ROUTES = 100
    UTTERANCES_PER_ROUTE = 5
    NUM_QUERIES = 500

    print(f"Generating benchmark dataset ({NUM_ROUTES} routes, {NUM_QUERIES} queries)...")
    routes_data = generate_synthetic_routes(NUM_ROUTES, UTTERANCES_PER_ROUTE)
    
    # Build queries
    test_queries = [f"This is test query {i} representing random user intent." for i in range(NUM_QUERIES)]

    # Warmup encoders (important for ONNX)
    print("Warming up encoders...")
    warmup_encoder = SynaptoEncoder(model_name="BAAI/bge-small-en-v1.5", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    warmup_encoder.encode_batch(["Warmup string to compile ONNX graph"])
    
    # Run benchmarks
    benchmark_semantic_router(routes_data, test_queries)
    await benchmark_synaptoroute(routes_data, test_queries)

if __name__ == "__main__":
    asyncio.run(main())
