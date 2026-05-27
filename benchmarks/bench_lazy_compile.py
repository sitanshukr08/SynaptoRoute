import time
import numpy as np
from synaptoroute.router import AdaptiveRouter
from synaptoroute.models import Route
from synaptoroute.storage import SQLiteStorage

class DummyEncoder:
    def encode(self, text: str) -> np.ndarray:
        return np.random.rand(384)
        
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.random.rand(len(texts), 384)

def run_benchmark(num_utterances):
    print(f"Setting up router with {num_utterances} utterances...")
    encoder = DummyEncoder()
    storage = SQLiteStorage(':memory:')
    router = AdaptiveRouter(encoder=encoder, storage=storage)
    
    route = Route(
        name="dummy_route",
        utterances=[f"This is utterance number {i}" for i in range(num_utterances)]
    )
    router.add_route(route)
    
    # Force initial compile
    _ = router("test query")
    
    # Add one more utterance to dirty the state
    router.add_utterance("dummy_route", "new utterance for triggering lazy compile")
    
    # Measure the next __call__ which will trigger _compile_vectors_locked
    start_time = time.perf_counter()
    _ = router("new utterance for triggering lazy compile")
    end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000
    print(f"Latency for N={num_utterances}: {latency_ms:.4f} ms")
    print("-" * 50)

if __name__ == "__main__":
    run_benchmark(1000)
    run_benchmark(5000)
