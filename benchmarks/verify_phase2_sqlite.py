import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import threading
import time
from synaptoroute.storage import SQLiteStorage

def run_test():
    storage = SQLiteStorage('test_sqlite_pool.db')
    
    # Init DB
    storage._init_db()
    
    print("[INFO] Testing SQLite Semaphore boundaries...")
    
    active_connections = 0
    max_active = 0
    lock = threading.Lock()
    
    def worker():
        nonlocal active_connections, max_active
        with storage._get_connection():
            with lock:
                active_connections += 1
                if active_connections > max_active:
                    max_active = active_connections
            
            # Simulate read delay
            time.sleep(0.1)
            
            with lock:
                active_connections -= 1
                
    threads = []
    for _ in range(50):
        t = threading.Thread(target=worker)
        threads.append(t)
        
    start = time.perf_counter()
    for t in threads:
        t.start()
        
    for t in threads:
        t.join()
        
    time.perf_counter() - start
    
    if max_active <= 10:
        print(f"[PASS] Bounded Semaphore worked! Max concurrent connections was {max_active} (limit is 10).")
    else:
        print(f"[FAIL] Semaphore failed! Max concurrent connections reached {max_active}.")
        exit(1)
        
    if os.path.exists('test_sqlite_pool.db'):
        os.remove('test_sqlite_pool.db')

if __name__ == "__main__":
    run_test()
