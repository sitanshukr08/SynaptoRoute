import asyncio
import time
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import BaseStorage
from synaptoroute.models import Route

class DummyStorage(BaseStorage):
    def load_all_routes(self):
        return [], {}
    def save_route(self, route, embeddings=None):
        pass
    def add_utterance(self, route_name, utterance, embedding=None):
        pass
    def get_route(self, route_name):
        return None
    def update_threshold(self, route_name, threshold):
        pass
    def close(self):
        pass

async def main():
    try:
        encoder = Encoder(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception:
        encoder = Encoder()

    storage = DummyStorage()
    router = AdaptiveRouter(encoder, storage)

    router.add_route(Route(name="dummy", utterances=["hello world", "test message"]))

    await router.start()

    queries = ["This is a test query number {}".format(i) for i in range(1000)]

    print("Starting benchmark with 1000 concurrent requests...")
    start_time = time.perf_counter()

    tasks = [router.aquery(q) for q in queries]
    await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"Total time for 1000 queries: {total_time:.4f} seconds")
    print(f"Amortized latency per query (P50 equivalent): {(total_time / 1000) * 1000:.4f} ms")

    await router.stop()

if __name__ == "__main__":
    asyncio.run(main())
