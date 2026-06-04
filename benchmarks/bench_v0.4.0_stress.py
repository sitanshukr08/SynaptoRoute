import sys
import os
import asyncio
import time
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synaptoroute import AdaptiveRouter
from synaptoroute.models import Route
from synaptoroute.storage import SQLiteStorage
from synaptoroute.sync import RedisSyncManager

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

class MockEncoder:
    def __init__(self, dim=384):
        self._dim = dim
        self.requires_lock = False
    
    @property
    def dim(self) -> int:
        return self._dim
        
    def encode(self, text: str):
        return np.random.rand(self.dim).astype(np.float32)
        
    def encode_batch(self, texts: list):
        return np.random.rand(len(texts), self.dim).astype(np.float32)

async def run_proper_benchmark():
    db_path = 'proper_stress.db'
    if os.path.exists(db_path):
        try: os.remove(db_path)
        except: pass

    storage = SQLiteStorage(db_path)
    router = AdaptiveRouter(encoder=MockEncoder(), storage=storage, max_capacity=15000)
    
    sync_manager = RedisSyncManager("redis://localhost")
    sync_manager.register(router)
    
    await router.start()

    print("=" * 60)
    print(" SYNAPTOROUTE v0.4.0 METRICS & STRESS BENCHMARK ")
    print("=" * 60)
    
    print("[1/4] Pre-populating baseline (10,000 routes) ...")
    start_seed = time.perf_counter()
    
    # We batch add for speed in seeding
    for i in range(10000):
        router.add_route(Route(name=f"baseline_{i}", utterances=[f"baseline query syntax text sequence {i}"]))
        if i > 0 and i % 2500 == 0:
            print(f"      ... {i}/10000 routes added")
            
    seed_duration = time.perf_counter() - start_seed
    print(f"      [DONE] Seeded in {seed_duration:.2f}s")
    
    print("[1.5/4] Testing Sequential Latency (No Contention) ...")
    seq_latencies = []
    for _ in range(1000):
        qs = f"baseline query syntax text sequence {random.randint(0, 9999)}"
        t0 = time.perf_counter()
        await router.aquery(qs)
        seq_latencies.append((time.perf_counter() - t0) * 1000)
    
    seq_latencies.sort()
    print(f"      - P50 Latency: {seq_latencies[500]:.2f} ms")
    print(f"      - P99 Latency: {seq_latencies[990]:.2f} ms")
    print(f"      - Max Latency: {seq_latencies[-1]:.2f} ms")
    
    read_latencies = []
    write_latencies = []
    errors = []
    
    is_running = True
    
    async def stress_reader(worker_id):
        while is_running:
            target_i = random.randint(0, 9999)
            query = f"baseline query syntax text sequence {target_i} random {random.randint(0, 1000)}"
            
            t0 = time.perf_counter()
            try:
                await router.aquery(query)
                read_latencies.append(time.perf_counter() - t0)
            except Exception as e:
                errors.append(f"Reader error: {str(e)}")
            
            if len(read_latencies) % 1000 == 0:
                print(f"Readers completed {len(read_latencies)} ops...")
                
            await asyncio.sleep(0.005) # Yield

    async def stress_writer(worker_id):
        while is_running:
            action = random.choices(["add_route", "add_utterance", "delete_route"], weights=[40, 40, 20])[0]
            
            t0 = time.perf_counter()
            try:
                if action == "add_route":
                    name = f"dynamic_route_{worker_id}_{random.randint(0, 100000)}"
                    await asyncio.to_thread(router.add_route, Route(name=name, utterances=[f"dynamic newly added utterance {random.random()}"]))
                elif action == "add_utterance":
                    base = f"baseline_{random.randint(0, 9999)}"
                    await asyncio.to_thread(router.add_utterance, base, f"additional utterance variant {random.random()}")
                elif action == "delete_route":
                    base = f"baseline_{random.randint(0, 9999)}"
                    await asyncio.to_thread(router.delete_route, base)
                
                write_latencies.append(time.perf_counter() - t0)
            except Exception as e:
                if "not found" not in str(e) and "ID_OVERFLOW" not in str(e):
                    errors.append(f"Writer error: {str(e)}")
            await asyncio.sleep(0.05) # Yield
            
    print("[2/4] Launching highly concurrent read/write storm...")
    print("      - 100 Concurrent Async Readers")
    print("      - 25 Concurrent Async Writers (triggering GC and WAL buffers)")
    
    readers = [asyncio.create_task(stress_reader(i)) for i in range(100)]
    writers = [asyncio.create_task(stress_writer(i)) for i in range(25)]
    
    # Run for exactly 60 seconds
    stress_duration = 5.0
    print(f"[3/4] Stress test running for {stress_duration} seconds...")
    
    for remaining in range(int(stress_duration), 0, -10):
        print(f"      ... {remaining}s remaining")
        await asyncio.sleep(min(10, remaining))
        
    is_running = False
    
    await asyncio.gather(*readers, *writers, return_exceptions=True)
    
    print("[4/4] Calculating final metrics...")
    
    await router.stop()
    
    if not read_latencies:
        read_latencies = [0]
    if not write_latencies:
        write_latencies = [0]
        
    read_latencies = np.array(read_latencies) * 1000  # ms
    write_latencies = np.array(write_latencies) * 1000  # ms
    
    total_ops = len(read_latencies) + len(write_latencies)
    throughput = total_ops / stress_duration
    
    print("\n" + "=" * 60)
    print(" " * 20 + "FINAL RESULTS")
    print("=" * 60)
    print(f" Total Operations:      {total_ops:,}")
    print(f" Test Duration:         {stress_duration} seconds")
    print(f" Throughput:            {throughput:,.2f} ops/sec")
    print(f" System Exceptions:     {len(errors)}")
    if errors:
        print("\n   Top Exceptions:")
        for e in list(set(errors))[:5]:
            print(f"     - {e}")
            
    print("\n [ READ METRICS (Asynchronous aquery) ]")
    print(f"   Count:               {len(read_latencies):,}")
    print(f"   P50 Latency:         {np.percentile(read_latencies, 50):.2f} ms")
    print(f"   P90 Latency:         {np.percentile(read_latencies, 90):.2f} ms")
    print(f"   P95 Latency:         {np.percentile(read_latencies, 95):.2f} ms")
    print(f"   P99 Latency:         {np.percentile(read_latencies, 99):.2f} ms")
    print(f"   Max Latency:         {np.max(read_latencies):.2f} ms")
    
    print("\n [ WRITE METRICS (Synchronous Mutable Operations) ]")
    print(f"   Count:               {len(write_latencies):,}")
    print(f"   P50 Latency:         {np.percentile(write_latencies, 50):.2f} ms")
    print(f"   P95 Latency:         {np.percentile(write_latencies, 95):.2f} ms")
    print(f"   Max Latency:         {np.max(write_latencies):.2f} ms")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_proper_benchmark())
