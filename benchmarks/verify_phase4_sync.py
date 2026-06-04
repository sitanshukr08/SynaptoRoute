import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import time
from synaptoroute import AdaptiveRouter
from synaptoroute.sync import RedisSyncManager
from synaptoroute.storage import SQLiteStorage

async def run_test():
    storage = SQLiteStorage(':memory:')
    router = AdaptiveRouter(storage=storage)
    
    # Create sync manager
    sync_manager = RedisSyncManager("redis://localhost")
    sync_manager.register(router)
    
    # We will directly insert into inbound queue to test deduplication
    sync_manager._inbound_queue = asyncio.Queue()
    
    # Start dispatcher worker
    worker = asyncio.create_task(sync_manager._dispatch_worker_loop())
    
    start = time.perf_counter()
    
    # Send 100 identical add_route messages
    for i in range(100):
        msg = {
            "action": "add_route",
            "payload": {
                "name": "target_route",
                "utterances": ["hello world"]
            }
        }
        await sync_manager._inbound_queue.put(msg)
        
    await sync_manager._inbound_queue.join()
    
    duration = time.perf_counter() - start
    worker.cancel()
    
    if duration > 1.0:
        print(f"[FAIL] Deduplication failed! Processing took {duration:.2f}s because it re-added 100 times.")
        exit(1)
        
    print(f"[PASS] Deduplication successful! Ignored 99 duplicates in {duration:.4f}s.")
    
    # Check that route was added
    assert "target_route" in router._route_map
    assert "target_route" in sync_manager._synced_routes
    print("[PASS] Route successfully added and tracked in _synced_routes.")

if __name__ == "__main__":
    asyncio.run(run_test())
