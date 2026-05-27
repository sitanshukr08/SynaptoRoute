import asyncio
import time
import string
import random
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.models import Route

async def main():
    print("--- Input Poisoning Test ---")
    try:
        encoder = Encoder(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except:
        encoder = Encoder()
        
    storage = SQLiteStorage("data/test_poison.sqlite")
    router = AdaptiveRouter(encoder, storage)
    router.add_route(Route(name="dummy", utterances=["dummy test utterance"]))
    
    await router.start()
    
    tests = [
        ("Empty String", ""),
        ("Whitespace", "    \n\t  "),
        ("Huge String (1MB)", "A" * 1024 * 1024),
        ("Raw Noise", "".join(random.choices(string.printable, k=5000))),
        ("Unicode/Emojis", "🔥💀" * 500)
    ]
    
    for name, payload in tests:
        start = time.perf_counter()
        try:
            res = await router.aquery(payload)
            print(f"[{name}] SUCCESS in {(time.perf_counter() - start)*1000:.2f}ms. Matched: {res.name if res else 'None'}")
        except Exception as e:
            print(f"[{name}] CRASHED: {e}")
            
    await router.stop()

if __name__ == "__main__":
    asyncio.run(main())
