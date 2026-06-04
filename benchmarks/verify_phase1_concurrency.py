import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import threading
import time
from synaptoroute import AdaptiveRouter
from synaptoroute.models import Route
from synaptoroute.storage import SQLiteStorage

async def run_test():
    router = AdaptiveRouter(storage=SQLiteStorage(':memory:'))
    await router.start()
    
    # 1. Verify thread pool exists
    assert hasattr(router, '_thread_pool'), "ThreadPoolExecutor not found on router"
    print("[PASS] ThreadPoolExecutor is initialized.")
    
    # 2. Add an initial route to test duplicates
    router.add_route(Route(name="test_route", utterances=["hello world"]))
    
    start = time.perf_counter()
    
    # We will simulate 50 concurrent requests trying to add the SAME utterance
    # If our lock logic is flawed (TOCTOU), the utterance will be added multiple times.
    def add_concurrently():
        router.add_utterance("test_route", "this is a concurrent test")
        
    threads = []
    for _ in range(50):
        t = threading.Thread(target=add_concurrently)
        threads.append(t)
        
    for t in threads:
        t.start()
        
    for t in threads:
        t.join()
        
    duration = time.perf_counter() - start
    print(f"[INFO] 50 concurrent additions completed in {duration:.2f}s")
    
    # 3. Verify exactly one insertion occurred despite 50 threads racing
    route = router._route_map["test_route"]
    utterance_count = len([u for u in route.utterances if u == "this is a concurrent test"])
    
    if utterance_count == 1:
        print("[PASS] TOCTOU Lock logic is safe. No duplicate insertions.")
    else:
        print(f"[FAIL] Found {utterance_count} duplicate utterances! The lock is flawed.")
        exit(1)
        
    await router.stop()

if __name__ == "__main__":
    asyncio.run(run_test())
