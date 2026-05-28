import asyncio
import time
import os
import random
import string
import psutil

from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.models import Route

NUM_QUERIES = 10000

def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def generate_random_string(length=50):
    return ''.join(random.choices(string.ascii_letters + " ", k=length))

def setup_db(db_path, num_vectors):
    print(f"Generating DB with {num_vectors} vectors...")
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(db_path + "-wal"):
        os.remove(db_path + "-wal")
    if os.path.exists(db_path + "-shm"):
        os.remove(db_path + "-shm")
        
    storage = SQLiteStorage(db_path)
    # Use CPU for fast generation to avoid OOM if GPU isn't available, but we can try CUDA if installed. 
    # Let's use CPU for safe generation, the router's batch size protects it anyway.
    encoder = Encoder(providers=["CPUExecutionProvider"])
    router = AdaptiveRouter(encoder, storage)
    
    # Pre-seed routes. We'll use 50 routes, and math out the utterances per route
    routes = 50
    utterances_per = num_vectors // routes
    
    for i in range(routes):
        utterances = [generate_random_string() for _ in range(utterances_per)]
        router.add_route(Route(name=f"route_{i}", utterances=utterances))
        
    router.storage.close()
    print("Database built.")

async def run_workload(db_path, provider_name, queries):
    print(f"  [Init] Booting router with {provider_name}...")
    
    start_ram = get_ram_mb()
    storage = SQLiteStorage(db_path)
    encoder = Encoder(providers=[provider_name])
    router = AdaptiveRouter(encoder, storage)
    await router.start()
    
    print(f"  [Bench] Routing {len(queries)} concurrent requests...")
    start_time = time.time()
    
    tasks = [router.aquery(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    duration = time.time() - start_time
    qps = len(queries) / duration
    avg_latency = (duration / len(queries)) * 1000
    peak_ram = get_ram_mb() - start_ram
    
    await router.stop()
    router.storage.close()
    
    print(f"  [Results] Duration: {duration:.2f}s | QPS: {qps:.2f} | Avg Latency: {avg_latency:.2f}ms | RAM Footprint: {peak_ram:.2f} MB")
    
async def main():
    print("=== Extreme Scale CPU vs GPU Benchmark ===")
    queries = [generate_random_string() for _ in range(NUM_QUERIES)]
    db_path = "data/extreme_bench.sqlite"
    
    scales = [10000, 25000, 50000]
    
    for scale in scales:
        print(f"\n==========================================")
        print(f"       TESTING SCALE: {scale} VECTORS       ")
        print(f"==========================================")
        
        setup_db(db_path, scale)
        
        print("\n--- PASS 1: CPU EXECUTION ---")
        await run_workload(db_path, "CPUExecutionProvider", queries)
        
        print("\n--- PASS 2: GPU EXECUTION ---")
        try:
            await run_workload(db_path, "CUDAExecutionProvider", queries)
        except Exception as e:
            print(f"GPU Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
