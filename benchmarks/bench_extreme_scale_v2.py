import os
import sys
import json
import time
import asyncio
import psutil
import argparse

# Add parent dir to path to import synaptoroute
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import init_synaptoroute
from synaptoroute import AdaptiveRouter, Route

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to synthetic dataset")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="Embedding model")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SCALE TEST: {args.dataset}")
    print(f"MODEL: {args.model}")
    print(f"{'='*60}")

    print(f"[Memory] Base footprint: {get_memory_mb():.2f} MB")

    # 1. Load Dataset
    print(f"\nLoading {args.dataset} from disk...")
    t0 = time.perf_counter()
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    routes_data = data["routes"]
    num_routes = len(routes_data)
    
    # Pre-parse routes to prevent timing overhead during build
    routes = []
    for r in routes_data:
        routes.append(Route(name=r["name"], utterances=r["utterances"]))
    
    t_load = time.perf_counter() - t0
    print(f"Loaded {num_routes} routes ({len(routes[0].utterances)} utterances each) in {t_load:.2f}s")
    print(f"[Memory] After loading JSON: {get_memory_mb():.2f} MB")

    # 2. Build Router
    print("\nBuilding SynaptoRoute Index...")
    db_path = f"scale_test_{num_routes}.sqlite"
    if os.path.exists(db_path) and os.path.getsize(db_path) > 100000:
        from synaptoroute.storage import SQLiteStorage
        from synaptoroute.encoder import Encoder
        print("DB already exists and is populated. Skipping bulk insert.")
        storage = SQLiteStorage(db_path)
        encoder = Encoder(model_name=args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        router = AdaptiveRouter(
            storage=storage,
            encoder=encoder,
            max_capacity=300000
        )
        await router.start()
    else:
        # Building DB
        from synaptoroute.storage import SQLiteStorage
        from synaptoroute.encoder import Encoder
        
        storage = SQLiteStorage(db_path)
        encoder = Encoder(model_name=args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        router = AdaptiveRouter(
            storage=storage,
            encoder=encoder,
            max_capacity=300000
        )
        
        await router.start()
        
        print("Encoding all routes (batching)...")
        t_enc = time.perf_counter()
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=OFF;')
        
        storage._init_db()
        
        all_utterances = []
        for r in routes:
            all_utterances.extend(r.utterances)
            
        print(f"Encoding {len(all_utterances)} utterances in chunks...")
        all_embeddings = []
        chunk_size = 5000
        for i in range(0, len(all_utterances), chunk_size):
            chunk = all_utterances[i:i+chunk_size]
            if getattr(encoder, 'requires_lock', True):
                with router._encoder_lock:
                    emb_chunk = encoder.encode_batch(chunk)
            else:
                emb_chunk = encoder.encode_batch(chunk)
            all_embeddings.extend(emb_chunk)
            print(f"  Encoded {min(i+chunk_size, len(all_utterances))}/{len(all_utterances)}")
            
        print(f"Encoding took {time.perf_counter() - t_enc:.2f}s")
        
        print("Bulk inserting into SQLite...")
        t_sql = time.perf_counter()
        
        route_inserts = [(r.name, r.threshold, json.dumps(r.metadata) if r.metadata else None) for r in routes]
        conn.executemany('INSERT INTO routes (name, threshold, metadata) VALUES (?, ?, ?)', route_inserts)
        
        utt_inserts = []
        idx = 0
        for r in routes:
            for u in r.utterances:
                utt_inserts.append((r.name, u, all_embeddings[idx].tobytes()))
                idx += 1
                
        conn.executemany('INSERT INTO utterances (route_name, utterance, embedding) VALUES (?, ?, ?)', utt_inserts)
        conn.commit()
        conn.close()
        
        print(f"SQL Insert took {time.perf_counter() - t_sql:.2f}s")
    
    # Now trigger router to load from DB
    print("Loading SQLite into Faiss...")
    t_faiss = time.perf_counter()
    router._load_routes()
    print(f"Faiss build took {time.perf_counter() - t_faiss:.2f}s")
    
    await router.aquery("warmup compile")

    t_build = time.perf_counter() - t0
    print(f"Build complete in {t_build:.2f}s")
    print(f"[Memory] After Build: {get_memory_mb():.2f} MB")
    
    # 3. Test Latency
    print("\nTesting Latency (1000 sequential queries)...")
    test_queries = ["fetch the document locally 42"] * 1000
    
    latencies = []
    for q in test_queries:
        t0 = time.perf_counter()
        router(q)
        latencies.append((time.perf_counter() - t0) * 1000)
        
    p50 = sorted(latencies)[len(latencies)//2]
    p99 = sorted(latencies)[int(len(latencies)*0.99)]
    avg = sum(latencies)/len(latencies)
    
    print(f"Sequential Latency: Avg={avg:.2f}ms | P50={p50:.2f}ms | P99={p99:.2f}ms")
    
    print("\nTesting Latency (1000 concurrent queries)...")
    t0 = time.perf_counter()
    await asyncio.gather(*(router.aquery(q) for q in test_queries))
    t_concurrent = time.perf_counter() - t0
    print(f"Concurrent Throughput: {t_concurrent:.2f}s for 1000 queries")
    
    await router.stop()
    print(f"\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
