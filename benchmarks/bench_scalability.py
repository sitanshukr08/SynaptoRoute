import tracemalloc
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
import gc
import os

def measure_memory_for_utterances(num_utterances):
    db_path = f"data/temp_bench_mem_{num_utterances}.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    gc.collect()
    tracemalloc.start()
    
    encoder = Encoder(model_name="BAAI/bge-small-en-v1.5")
    storage = SQLiteStorage(db_path=db_path)
    router = AdaptiveRouter(encoder, storage)
    
    from synaptoroute.models import Route
    router.add_route(Route(name="scale_route", utterances=["base"], threshold=0.5))
    
    for i in range(num_utterances):
        router.add_utterance("scale_route", f"scale utterance number {i}")
        
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    storage.conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)
        
    return peak / (1024 * 1024)

def main():
    print("Benchmarking Scalability (Peak RAM)...")
    for n in [100, 1000, 5000]:
        peak_mb = measure_memory_for_utterances(n)
        print(f"Peak RAM for {n} utterances: {peak_mb:.2f} MB")

if __name__ == '__main__':
    main()
