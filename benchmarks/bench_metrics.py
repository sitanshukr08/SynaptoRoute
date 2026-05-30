import asyncio
import time
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import BaseStorage
from synaptoroute.models import Route
from synaptoroute.profile import get_profile, ProfileType

class DummyStorage(BaseStorage):
    def load_all_routes(self): return [], {}
    def save_route(self, route, embeddings=None): pass
    def add_utterance(self, route_name, utterance, embedding=None): pass
    def get_route(self, route_name): return None
    def update_threshold(self, route_name, threshold): pass
    def delete_route(self, route_name): pass
    def close(self): pass

async def main():
    try:
        encoder = Encoder(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception:
        encoder = Encoder()
    
    storage = DummyStorage()
    profile = get_profile(ProfileType.THROUGHPUT)
    router = AdaptiveRouter(encoder, storage, profile=profile)

    router.add_route(Route(name="dummy", utterances=["hello world"]))

    await router.start()

    queries = ["This is a test query number {}".format(i) for i in range(100)]
    
    tasks = [router.aquery(q) for q in queries]
    await asyncio.gather(*tasks)

    await router.stop()
    
    print("\n--- METRICS OUTPUT ---")
    print(router.metrics.export_metrics())
    print("----------------------")

if __name__ == "__main__":
    asyncio.run(main())
