import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import time
import random
from synaptoroute import AdaptiveRouter
from synaptoroute.models import Route
from synaptoroute.storage import SQLiteStorage
from synaptoroute.sync import RedisSyncManager

async def run_stress_test():
    db_path = 'deep_stress.db'
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass

    storage = SQLiteStorage(db_path)
    # 5,000 capacity. We will push around 3,000 total vectors with high tombstone churn to force multiple GC cycles.
    router = AdaptiveRouter(storage=storage, max_capacity=5000)
    
    # Mock Redis to isolate system test
    class MockPubSub:
        async def subscribe(self, channel): pass
        async def unsubscribe(self, channel): pass
        async def close(self): pass
        async def listen(self):
            while True:
                await asyncio.sleep(1)
                
    class MockRedis:
        def pubsub(self): return MockPubSub()
        async def publish(self, channel, message): pass
        async def aclose(self): pass

    try:
        import redis.asyncio as redis
        redis.from_url = lambda url: MockRedis()
    except ImportError:
        pass

    sync_manager = RedisSyncManager("redis://localhost", sync_worker_count=8)
    sync_manager.register(router)
    sync_manager._inbound_queue = asyncio.Queue(maxsize=10000)
    
    await router.start()
    workers = [asyncio.create_task(sync_manager._dispatch_worker_loop()) for _ in range(8)]
    
    print("[INFO] Seeding baseline of 500 routes...")
    for i in range(500):
        router.add_route(Route(name=f"base_route_{i}", utterances=[f"hello base {i}"]))
        
    start_time = time.perf_counter()
    errors = []
    
    stats = {"reads": 0, "writes": 0, "syncs": 0}
    
    async def heavy_writer(worker_id):
        try:
            # 50 writers, each doing 20 operations
            for i in range(20):
                action = random.choices(["add_route", "add_utterance", "delete_route"], weights=[50, 30, 20])[0]
                target_route = f"stress_route_{worker_id}_{i}"
                
                if action == "add_route":
                    router.add_route(Route(name=target_route, utterances=[f"stress test utterance {i}"]))
                    stats["writes"] += 1
                elif action == "add_utterance":
                    base = f"base_route_{random.randint(0, 499)}"
                    router.add_utterance(base, f"new utterance {worker_id} {i}")
                    stats["writes"] += 1
                elif action == "delete_route":
                    base = f"base_route_{random.randint(0, 499)}"
                    router.delete_route(base)
                    stats["writes"] += 1
                    
                await asyncio.sleep(0.001) # Yield to event loop
        except Exception as e:
            if "not found" not in str(e).lower():
                errors.append(f"Writer error: {str(e)}")
            
    async def heavy_reader(worker_id):
        try:
            # 100 readers, each doing 50 queries
            for i in range(50):
                await router.aquery(f"query looking for {worker_id} {i}")
                stats["reads"] += 1
                await asyncio.sleep(0.001)
        except Exception as e:
            errors.append(f"Reader error: {str(e)}")
            
    async def heavy_sync_flooder(worker_id):
        try:
            # 5 flooders, each blasting 200 payload deduplication messages
            for i in range(200):
                msg = {
                    "action": "add_route",
                    "payload": {
                        "name": f"sync_route_{i%20}",
                        "utterances": [f"sync utter {i}"]
                    }
                }
                await sync_manager._inbound_queue.put(msg)
                stats["syncs"] += 1
                await asyncio.sleep(0.001)
        except Exception as e:
            errors.append(f"Sync error: {str(e)}")

    print("[INFO] Launching Deep Stress Test...")
    print("       - 50 Concurrent Writers")
    print("       - 100 Concurrent Readers")
    print("       - 5 Redis Sync Flooders")
    
    tasks = []
    tasks.extend(asyncio.create_task(heavy_writer(i)) for i in range(50))
    tasks.extend(asyncio.create_task(heavy_reader(i)) for i in range(100))
    tasks.extend(asyncio.create_task(heavy_sync_flooder(i)) for i in range(5))
    
    await asyncio.gather(*tasks)
    
    # Wait for background components to settle
    print("[INFO] Load complete, allowing WAL buffers and background workers to flush (2s)...")
    await asyncio.sleep(2.0)
    
    for worker in workers:
        worker.cancel()
    await router.stop()
    
    duration = time.perf_counter() - start_time
    
    routes, _ = storage.load_all_routes()
    print(f"\n[RESULTS] Deep Stress Test finished in {duration:.2f}s")
    print(f"   - Total Operations: {stats['reads']} reads, {stats['writes']} writes, {stats['syncs']} syncs")
    print(f"   - Operations/sec: {(stats['reads'] + stats['writes'] + stats['syncs']) / duration:.2f}")
    print(f"[VERIFY] Final SQLite Routes: {len(routes)}")
    print(f"[VERIFY] Final FAISS Index Size: {router.index.total_vectors}")
    print(f"[VERIFY] Exceptions caught: {len(errors)}")
    
    if errors:
        print("\nERRORS DETECTED:")
        for e in set(errors):
            print(f" - {e}")
            
    if len(errors) == 0:
        print("\n[PASS] No critical failures detected. The architecture held.")

if __name__ == "__main__":
    asyncio.run(run_stress_test())
