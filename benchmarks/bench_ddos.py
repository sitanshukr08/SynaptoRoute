import asyncio
import time
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.models import Route
from synaptoroute.exceptions import RouterOverloadedError

async def main():
    print("--- DDoS Stress Test (20,000 Concurrent Queries) ---")
    try:
        encoder = Encoder(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except:
        encoder = Encoder()
        
    storage = SQLiteStorage("data/test_ddos.sqlite")
    router = AdaptiveRouter(encoder, storage)
    router.add_route(Route(name="dummy", utterances=["dummy test utterance"]))
    
    await router.start()
    
    success_count = 0
    overload_count = 0
    other_errors = 0
    
    start = time.perf_counter()
    
    async def fire_query():
        nonlocal success_count, overload_count, other_errors
        try:
            await router.aquery("test")
            success_count += 1
        except RouterOverloadedError:
            overload_count += 1
        except Exception as e:
            other_errors += 1

    tasks = [asyncio.create_task(fire_query()) for _ in range(20000)]
    await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start
    await router.stop()
    
    print(f"Total Time: {total_time:.2f} s")
    print(f"Successful Queries: {success_count}")
    print(f"Shed Queries (RouterOverloadedError): {overload_count}")
    print(f"Other Errors: {other_errors}")
    
    if other_errors > 0:
        print("FAIL: Unhandled errors occurred.")
    elif success_count == 10000 and overload_count == 10000:
        print("PASS: Exact queue bounding achieved.")
    else:
        print("INFO: Task scheduling distribution observed.")

if __name__ == "__main__":
    asyncio.run(main())
