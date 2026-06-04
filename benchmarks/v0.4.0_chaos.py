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

async def run_test():
    db_path = 'chaos_v0.4.0.db'
    if os.path.exists(db_path):
        try: os.remove(db_path)
        except: pass

    storage = SQLiteStorage(db_path)
    # Extremely small capacity to force constant rebuilds (GC) while chaos happens
    router = AdaptiveRouter(storage=storage, max_capacity=50)
    
    # Mock Redis Sync Manager to simulate massive incoming broadcast noise
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

    sync_manager = RedisSyncManager("redis://localhost")
    sync_manager.register(router)
    sync_manager._inbound_queue = asyncio.Queue()
    
    await router.start()
    worker = asyncio.create_task(sync_manager._dispatch_worker_loop())
    
    # Pre-populate some routes
    for i in range(5):
        router.add_route(Route(name=f"base_route_{i}", utterances=[f"hello base {i}"]))
    
    start_time = time.perf_counter()
    errors = []
    
    async def chaotic_writer(worker_id):
        try:
            for i in range(15):
                action = random.choice(["add_route", "add_utterance", "delete_route"])
                target_route = f"route_{worker_id}_{i}"
                
                if action == "add_route":
                    router.add_route(Route(name=target_route, utterances=[f"some utterance {i}"]))
                elif action == "add_utterance":
                    # Add to a base route
                    base = f"base_route_{random.randint(0, 4)}"
                    router.add_utterance(base, f"new utterance {worker_id} {i}")
                elif action == "delete_route":
                    base = f"base_route_{random.randint(0, 4)}"
                    router.delete_route(base)
                    
                await asyncio.sleep(random.uniform(0.001, 0.01))
        except Exception as e:
            errors.append(f"Writer error: {str(e)}")
            
    async def chaotic_reader(worker_id):
        try:
            for i in range(30):
                # Reads hit the SQLite semaphore concurrently
                await router(f"query {worker_id} {i}")
                await asyncio.sleep(random.uniform(0.001, 0.005))
        except Exception as e:
            errors.append(f"Reader error: {str(e)}")
            
    async def chaotic_sync_flooder():
        try:
            for i in range(50):
                msg = {
                    "action": "add_route",
                    "payload": {
                        "name": f"sync_route_{i%5}",
                        "utterances": [f"sync utter {i}"]
                    }
                }
                await sync_manager._inbound_queue.put(msg)
                await asyncio.sleep(random.uniform(0.001, 0.01))
        except Exception as e:
            errors.append(f"Sync error: {str(e)}")

    print("[INFO] Igniting v0.4.0 Chaos Simulation...")
    print("       - 10 Concurrent Writers (triggering constant GC WAL buffers)")
    print("       - 20 Concurrent Readers (pounding SQLite Semaphore)")
    print("       - 1 Redis Sync Flooder (triggering deduplication logic)")
    
    writers = [asyncio.create_task(chaotic_writer(i)) for i in range(10)]
    readers = [asyncio.create_task(chaotic_reader(i)) for i in range(20)]
    sync_flooder = asyncio.create_task(chaotic_sync_flooder())
    
    await asyncio.gather(*(writers + readers + [sync_flooder]))
    
    # Allow final WAL flushes
    await asyncio.sleep(1.0)
    
    worker.cancel()
    await router.stop()
    
    duration = time.perf_counter() - start_time
    
    routes, _ = storage.load_all_routes()
    print(f"\n[RESULTS] Chaos Simulation survived in {duration:.2f}s")
    print(f"[VERIFY] Final SQLite Routes: {len(routes)}")
    print(f"[VERIFY] Final FAISS Index Size: {router.index.total_vectors}")
    print(f"[VERIFY] Errors caught: {len(errors)}")
    
    if errors:
        print("\nERRORS DETECTED:")
        for e in set(errors):
            print(f" - {e}")
            
    if len(errors) == 0:
        print("\n[PASS] No deadlocks, no exceptions, no capacity overloads (except intended limits if any).")

if __name__ == "__main__":
    asyncio.run(run_test())
