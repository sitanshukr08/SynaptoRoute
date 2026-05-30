import asyncio
import time
import numpy as np
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import OpenAIEncoder
from synaptoroute.models import Route
from synaptoroute.profile import get_profile, ProfileType

class MockOpenAIClient:
    def __init__(self):
        self.call_count = 0
        self.total_texts_processed = 0
        
        class Embeddings:
            def __init__(self, parent):
                self.parent = parent
                
            def create(self, input, model):
                self.parent.call_count += 1
                
                # If a single string is passed, wrap it in a list to count properly
                if isinstance(input, str):
                    texts = [input]
                else:
                    texts = input
                    
                self.parent.total_texts_processed += len(texts)
                
                # Simulate 150ms network round-trip time for hitting the OpenAI API
                time.sleep(0.150)
                
                class DummyData:
                    def __init__(self):
                        self.embedding = np.random.rand(1536).tolist()
                        
                class DummyResponse:
                    def __init__(self, count):
                        self.data = [DummyData() for _ in range(count)]
                        
                return DummyResponse(len(texts))
                
        self.embeddings = Embeddings(self)

class DummyStorage:
    def __init__(self): pass
    def save_route(self, route, embeddings=None): pass
    def add_utterance(self, route_name, utterance, embedding): pass
    def delete_route(self, route_name): pass
    def load_all_routes(self): return [], []
    def update_threshold(self, route_name, threshold): pass

async def inject_utterances(router):
    # This will prove the thread is NOT blocked by the 150ms network calls
    added_count = 0
    for i in range(100):
        # add_utterance is synchronous, so we will wrap it to prove it returns instantly
        start = time.time()
        await asyncio.to_thread(router.add_utterance, "route_0", f"new injection {i}")
        duration = time.time() - start
        
        if duration > 0.200:
            print(f"[ERROR] LOCK CONTENTION! add_utterance took {duration:.3f}s")
        else:
            added_count += 1
            
        await asyncio.sleep(0.02) # Try to add one every 20ms
        
    print(f"[SUCCESS] Successfully injected {added_count}/100 utterances without lock contention!")

async def main():
    print("Initializing Mocked OpenAI Encoder...")
    mock_client = MockOpenAIClient()
    
    # We use THROUGHPUT profile to trigger the batching window
    profile = get_profile(ProfileType.THROUGHPUT)
    encoder = OpenAIEncoder(client=mock_client)
    router = AdaptiveRouter(encoder, DummyStorage(), profile=profile, max_capacity=10000)
    
    print("Loading 5 dummy routes...")
    for i in range(5):
        route = Route(name=f"route_{i}", utterances=[f"this is utterance {i}"])
        router.add_route(route)
        
    await router.start()
    
    print("\n=========================================")
    print("DEEP CONCURRENCY STRESS TEST (2000 Requests + Live Injections)")
    print("=========================================\n")
    
    mock_client.call_count = 0
    mock_client.total_texts_processed = 0
    
    start_time = time.time()
    
    # Blast 2000 parallel async queries
    num_requests = 2000
    
    async def run_queries():
        tasks = [router.aquery(f"dummy api query {i}") for i in range(num_requests)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
    await asyncio.gather(
        run_queries(),
        inject_utterances(router)
    )
    
    duration = time.time() - start_time
    await router.stop()
    
    print(f"\nTotal Asyncio Wall-Clock Time: {duration:.2f} seconds")
    print(f"Total API Calls Executed:      {mock_client.call_count}")
    print(f"Total Texts Processed:         {mock_client.total_texts_processed}")
    
    # If we executed 2000 sequential API calls at 150ms each, it would take 300 seconds
    theoretical_sequential_time = num_requests * 0.150
    print(f"Theoretical Sequential Time:   {theoretical_sequential_time:.2f} seconds")
    print(f"Speedup Multiplier:            {theoretical_sequential_time / duration:.2f}x")

if __name__ == "__main__":
    asyncio.run(main())
