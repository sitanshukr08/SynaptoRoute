import tracemalloc
import time
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.models import Route

def main():
    print("--- Memory Leak Endurance Test (2,000 Hot-Reloads) ---")
    try:
        encoder = Encoder(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except:
        encoder = Encoder()
        
    storage = SQLiteStorage("data/test_leak.sqlite")
    router = AdaptiveRouter(encoder, storage)
    
    # Pre-allocate one route
    router.add_route(Route(name="base", utterances=["base utterance"]))
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    for i in range(2000):
        # We overwrite the exact same route to trigger the is_overwrite logic
        router.add_route(Route(name="volatile_route", utterances=[f"volatile {i}"]))
        
        if i % 500 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Iteration {i} | Current RAM: {current / 10**6:.2f} MB | Peak RAM: {peak / 10**6:.2f} MB")
            
    current, peak = tracemalloc.get_traced_memory()
    total_time = time.perf_counter() - start_time
    
    print(f"Final | Current RAM: {current / 10**6:.2f} MB | Peak RAM: {peak / 10**6:.2f} MB")
    print(f"Total Time for 2,000 Reloads: {total_time:.2f} s")
    tracemalloc.stop()

if __name__ == "__main__":
    main()
