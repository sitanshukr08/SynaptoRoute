# SynaptoRoute

SynaptoRoute is a high-throughput, local semantic routing engine designed to replace slow, costly Large Language Model (LLM) routing chains. By leveraging local INT8 quantized embeddings and asynchronous dynamic batching, SynaptoRoute achieves zero-token intent classification in under 3 milliseconds on standard cloud hardware.

## Architecture

Traditional semantic routers suffer from O(N) memory degradation during live updates because they execute a deep memory copy of their entire vector space on every addition. SynaptoRoute solves this via a lazy-compilation strategy, deferring vector reallocation until strictly necessary.

Furthermore, SynaptoRoute implements a dynamic asynchronous batching queue. Rather than evaluating queries sequentially, the background worker intercepts parallel HTTP requests, groups them within a 5-millisecond window, and processes the batch as a single matrix multiplication operation.

## Performance Benchmarks

The following metrics were captured via automated GitHub Actions CI/CD running on a standard `ubuntu-latest` 2-core cloud CPU.

| Metric | Cloud CPU Latency | Context |
| :--- | :--- | :--- |
| **Inference P99** | 3.94 ms | Single sequential query latency. |
| **Amortized P50** | 2.69 ms | Per-query latency when processing 1,000 concurrent requests via dynamic batching. |
| **Hot-Reload** | 5.04 ms | Time required to dynamically inject a new utterance into memory without dropping active API requests. |

## Deployment & Usage

### Method 1: Docker REST API (Recommended)

SynaptoRoute includes a fully asynchronous FastAPI wrapper designed for immediate microservice deployment.

```bash
# Build the image
docker build -t synaptoroute .

# Run the container
docker run -p 8000:8000 synaptoroute
```

Once running, you can interface with the router via standard HTTP requests:

```bash
curl -X POST http://localhost:8000/route \
     -H "Content-Type: application/json" \
     -d '{"query": "I need help resetting my password"}'
```

### Method 2: Python Library

To embed SynaptoRoute directly into your existing Python applications:

```bash
pip install git+https://github.com/sitanshukr08/SynaptoRoute.git
```

```python
import asyncio
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.models import Route

async def main():
    encoder = Encoder()
    storage = SQLiteStorage("data/memory.sqlite")
    router = AdaptiveRouter(encoder, storage)
    
    # Add an intent
    router.add_route(Route(name="billing", utterances=["I need a refund"]))
    
    # Start the background batching worker
    await router.start()
    
    # Query the router
    result = await router.aquery("How do I get my money back?")
    print(result.name) # Output: billing
    
    await router.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## System Limitations

SynaptoRoute utilizes local SQLite and in-memory NumPy matrices to achieve microsecond latency. As such, it is structurally bound to a single node. Deploying this system across multiple Kubernetes pods without a distributed event bus (such as Redis Pub/Sub) will result in cache incoherency (split-brain routing) during hot-reloads.
