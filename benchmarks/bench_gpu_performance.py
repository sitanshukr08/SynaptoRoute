import time
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.router import AdaptiveRouter
from synaptoroute.models import Route

def run_gpu_benchmark():
    print("Initializing Encoder on CUDA GPU...")
    # Initialize with CUDA Provider
    encoder = Encoder(providers=["CUDAExecutionProvider"])
    storage = SQLiteStorage(":memory:")
    router = AdaptiveRouter(encoder, storage)

    # Add dummy routes
    route1 = Route(name="support", utterances=["I need help", "Contact support", "Help me"])
    route2 = Route(name="billing", utterances=["Invoice issue", "Payment failed", "Billing error"])
    router.add_route(route1)
    router.add_route(route2)

    # Warmup
    print("Warming up GPU...")
    for _ in range(5):
        router("Warmup query")

    # Benchmark single inference latency
    num_queries = 1000
    print(f"Running {num_queries} queries through GPU...")
    latencies = []
    
    for _ in range(num_queries):
        start = time.perf_counter()
        _ = router("I want a refund for my last invoice")
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    latencies.sort()
    p50 = latencies[int(num_queries * 0.50)]
    p95 = latencies[int(num_queries * 0.95)]
    p99 = latencies[int(num_queries * 0.99)]
    
    print("\n--- GPU BENCHMARK RESULTS ---")
    print(f"P50 Latency: {p50:.2f} ms")
    print(f"P95 Latency: {p95:.2f} ms")
    print(f"P99 Latency: {p99:.2f} ms")
    
if __name__ == "__main__":
    run_gpu_benchmark()
