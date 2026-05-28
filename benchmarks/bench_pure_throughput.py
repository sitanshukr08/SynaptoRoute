import asyncio
import time
import os
import random
import string
import numpy as np

from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.models import Route

NUM_ROUTES = 50
UTTERANCES_PER_ROUTE = 100
NUM_QUERIES = 2000

def generate_random_string(length=50):
    return ''.join(random.choices(string.ascii_letters + " ", k=length))

def setup_router():
    print("Pre-loading database...")
    if os.path.exists("data/throughput.sqlite"):
        os.remove("data/throughput.sqlite")
    if os.path.exists("data/throughput.sqlite-wal"):
        os.remove("data/throughput.sqlite-wal")
    if os.path.exists("data/throughput.sqlite-shm"):
        os.remove("data/throughput.sqlite-shm")
        
    storage = SQLiteStorage("data/throughput.sqlite")
    encoder = Encoder()
    router = AdaptiveRouter(encoder, storage)
    
    # Pre-seed routes
    for i in range(NUM_ROUTES):
        utterances = [generate_random_string() for _ in range(UTTERANCES_PER_ROUTE)]
        router.add_route(Route(name=f"route_{i}", utterances=utterances))
        
    return router

async def run_async_benchmark(router, queries):
    print(f"\n[Async Benchmark] Firing {len(queries)} queries concurrently via Asyncio queue...")
    start_time = time.time()
    
    tasks = [router.aquery(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    duration = time.time() - start_time
    qps = len(queries) / duration
    
    print(f"Async Duration: {duration:.2f}s")
    print(f"Async Throughput: {qps:.2f} Queries Per Second (QPS)")
    print(f"Average Route Time: {(duration/len(queries))*1000:.2f} ms")
    
def run_sync_benchmark(router, queries):
    print(f"\n[Sync Benchmark] Routing {len(queries)} queries sequentially...")
    start_time = time.time()
    
    for q in queries:
        _ = router(q)
        
    duration = time.time() - start_time
    qps = len(queries) / duration
    
    print(f"Sync Duration: {duration:.2f}s")
    print(f"Sync Throughput: {qps:.2f} Queries Per Second (QPS)")
    print(f"Average Route Time: {(duration/len(queries))*1000:.2f} ms")

async def main():
    print(f"--- Pure Throughput Benchmark ---")
    print(f"Database Size: {NUM_ROUTES * UTTERANCES_PER_ROUTE} vectors ({NUM_ROUTES} routes)")
    
    router = setup_router()
    await router.start()
    
    # Generate Queries
    print(f"Generating {NUM_QUERIES} queries...")
    queries = [generate_random_string() for _ in range(NUM_QUERIES)]
    
    # Run Benchmarks
    run_sync_benchmark(router, queries)
    await run_async_benchmark(router, queries)
    
    await router.stop()
    router.storage.close()
    
if __name__ == "__main__":
    asyncio.run(main())
