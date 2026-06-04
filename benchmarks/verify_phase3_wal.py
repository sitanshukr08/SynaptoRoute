import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import time
from synaptoroute import AdaptiveRouter
from synaptoroute.models import Route
from synaptoroute.storage import SQLiteStorage

async def run_test():
    if os.path.exists('test_wal.db'):
        try: os.remove('test_wal.db')
        except: pass
    storage = SQLiteStorage('test_wal.db')
    router = AdaptiveRouter(storage=storage, max_capacity=10)
    await router.start()
    
    router.add_route(Route(name="route1", utterances=["test1", "test2"]))
    
    # Mock the index rebuild to take 2 seconds, simulating a heavy index rebuild
    original_rebuild = router._rebuild_index
    async def slow_rebuild():
        print("[INFO] Index Rebuild started...")
        await asyncio.sleep(2)
        await original_rebuild()
        print("[INFO] Index Rebuild finished. WAL buffer flushed.")
    
    router._rebuild_index = slow_rebuild

    # Trigger rebuild asynchronously
    router._rebuild_pending = True
    asyncio.create_task(slow_rebuild())
    
    time.sleep(0.1) # ensure rebuild started

    # During rebuild, add a new route and an utterance
    print("[INFO] Adding route2 and utterance to route1 DURING rebuild lock...")
    start_add = time.perf_counter()
    router.add_route(Route(name="route2", utterances=["test3"]))
    router.add_utterance("route1", "test4")
    duration = time.perf_counter() - start_add
    
    if duration > 1.0:
        print(f"[FAIL] Adding during rebuild BLOCKED for {duration:.2f}s instead of queueing in WAL!")
        exit(1)
    else:
        print(f"[PASS] Non-blocking WAL queue worked! Time: {duration:.4f}s")
        
    print(f"WAL length is {len(router._pending_rebuild_mutations)}")
    print(f"_rebuild_pending is {router._rebuild_pending}")
    assert len(router._pending_rebuild_mutations) == 2, "Mutations not appended to WAL"
    print("[PASS] Mutations successfully queued in WAL.")
    
    # Wait for rebuild to finish
    await asyncio.sleep(2.5)
    
    assert len(router._pending_rebuild_mutations) == 0, "WAL was not cleared after rebuild"
    
    # Verify index has the vectors
    assert router.index.total_vectors == 4, f"Index has {router.index.total_vectors} vectors instead of 4"
    print("[PASS] WAL flushed into new index successfully!")

    await router.stop()
    
    if os.path.exists('test_wal.db'):
        try:
            os.remove('test_wal.db')
        except:
            pass

if __name__ == "__main__":
    asyncio.run(run_test())
