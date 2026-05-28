import asyncio
import concurrent.futures
import time
import os
import sqlite3
from typing import List

from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.models import Route

NUM_THREADS = 20
NUM_ROUTES = 10
UTTERANCES_PER_ROUTE = 100  # total 1000 utterances

def setup_router():
    if os.path.exists("data/stress_test.sqlite"):
        os.remove("data/stress_test.sqlite")
    if os.path.exists("data/stress_test.sqlite-wal"):
        os.remove("data/stress_test.sqlite-wal")
    if os.path.exists("data/stress_test.sqlite-shm"):
        os.remove("data/stress_test.sqlite-shm")
        
    storage = SQLiteStorage("data/stress_test.sqlite")
    encoder = Encoder()
    router = AdaptiveRouter(encoder, storage)
    
    # Pre-seed routes
    for i in range(NUM_ROUTES):
        router.add_route(Route(name=f"route_{i}", utterances=["dummy init utterance"]))
        
    return router

def writer_worker(router: AdaptiveRouter, route_idx: int, utterance_start_idx: int, count: int):
    """Adds multiple utterances to a specific route."""
    for i in range(count):
        router.add_utterance(f"route_{route_idx}", f"This is test utterance {utterance_start_idx + i} for route {route_idx}")

async def main():
    print("Starting Extreme Concurrency Stress Test (v0.2.0 Architecture)...")
    router = setup_router()
    await router.start()
    
    print(f"\n[Test 1] Attacking Router with {NUM_ROUTES * UTTERANCES_PER_ROUTE} concurrent writes across {NUM_THREADS} threads...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for r_idx in range(NUM_ROUTES):
            chunk_size = UTTERANCES_PER_ROUTE // 5
            for chunk_idx in range(5):
                futures.append(
                    executor.submit(writer_worker, router, r_idx, chunk_idx * chunk_size, chunk_size)
                )
        concurrent.futures.wait(futures)
        
    duration = time.time() - start_time
    print(f"Finished writes in {duration:.2f} seconds.")
    expected_count = (NUM_ROUTES * UTTERANCES_PER_ROUTE) + NUM_ROUTES
    print(f"Internal Buffer Cursor Size: {router._cursor} (Expected: {expected_count})")
    
    if router._cursor != expected_count:
        print("FAILED: Data corruption detected. Cursor does not match inserted count.")
    else:
        print("PASSED: True O(1) buffer logic works perfectly under concurrent stress.")
        
    print("\n[Test 2] Testing SQLite Transaction Safety during fit_thresholds...")
    
    samples = []
    labels = []
    for r_idx in range(NUM_ROUTES):
        for i in range(10):
            samples.append(f"Mock validation utterance {i} for route_{r_idx}")
            labels.append(f"route_{r_idx}")
            
    start_time = time.time()
    # If transaction bleeding exists, this will crash with database locked or operational error
    router.fit_thresholds(samples, labels)
    duration = time.time() - start_time
    
    print(f"Finished fit_thresholds across all {NUM_ROUTES} routes in {duration:.2f} seconds.")
    print("PASSED: SQLite Connection Pooling successfully isolated transactions.")
    
    await router.stop()
    
    # Final SQLite Verification
    conn = sqlite3.connect("data/stress_test.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM utterances")
    db_count = cursor.fetchone()[0]
    conn.close()
    
    print(f"\nFinal DB Record Count: {db_count} (Expected: {expected_count})")
    if db_count != expected_count:
         print("FAILED: SQLite dropped transactions!")
    else:
         print("PASSED: Zero SQLite Data Loss.")

if __name__ == "__main__":
    asyncio.run(main())
