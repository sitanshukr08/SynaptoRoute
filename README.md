# SynaptoRoute

[![PyPI version](https://badge.fury.io/py/synaptoroute.svg)](https://pypi.org/project/synaptoroute/)
[![CI/CD Pipeline](https://github.com/sitanshukr08/SynaptoRoute/actions/workflows/ci.yml/badge.svg)](https://github.com/sitanshukr08/SynaptoRoute/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)

SynaptoRoute is a high-throughput, local semantic routing engine built for production Python microservices. Designed as an efficient alternative to LLM routing chains, it executes intent classification entirely locally using ONNX-accelerated vector embeddings.

The `v0.2.0` architecture transitions the engine from a sequential low-latency prototype into a **highly concurrent, ACID-compliant async batching router** capable of safely absorbing asynchronous FastAPI server loads.

## Table of Contents
- [Why SynaptoRoute?](#why-synaptoroute)
- [v0.2.0 Architecture & Optimizations](#v020-architecture--optimizations)
- [Head-to-Head Benchmark](#head-to-head-benchmark)
- [Installation & Deployment](#installation--deployment)
- [Quick Start Guide](#quick-start-guide)
- [System Limitations](#system-limitations)
- [Community & Contributing](#community--contributing)

---

## Why SynaptoRoute?

In modern agentic systems, relying on an external API (like OpenAI or Anthropic) to make simple routing decisions introduces unacceptable latency and token costs. 

SynaptoRoute solves this by executing intent classification locally. It is engineered to solve two specific problems that plague naive semantic routers: **$O(N)$ memory degradation during live updates** and **thread-blocking under asynchronous web server concurrency**.

## v0.2.0 Architecture & Optimizations

### 1. Amortized $O(1)$ Lazy Memory Slicing
Traditional routers suffer from severe performance degradation during live updates. When a new route is added, they execute an immediate `numpy.vstack`, copying the entire vector array in memory ($O(N)$ complexity). 

SynaptoRoute `v0.2.0` pre-allocates a static tensor buffer at initialization. When routes are added dynamically, the router slots the embedding directly into a reserved `float32` memory slice via list assignment. This bounds memory growth strictly to $O(1)$ and prevents server RAM exhaustion, even when handling 50,000 dense vectors.

### 2. Dynamic Asynchronous Batching
Hardware accelerators (GPUs, AVX CPUs) are optimized for large matrix multiplications. Processing single web requests sequentially wastes hardware potential and blocks Python's `asyncio` event loop. 

SynaptoRoute introduces a background `_batch_worker` queue. It traps parallel HTTP requests, waits for a configurable window (e.g., 5 milliseconds), groups them into a dense batch, and processes them in a single mathematical hardware cycle. This architecture safely doubles throughput under heavy concurrent load.

### 3. SQLite Thread-Local Pooling & BLOB Caching
Routing logic is only useful if it's durable. SynaptoRoute serializes vectors into a local SQLite database. 
- **Concurrency:** Thread-local connection pooling ensures 100% data integrity even when 2,000 overlapping web requests attempt to modify routes simultaneously. 
- **Cold Booting:** The `v0.2.1` hotfix introduced binary `float32` BLOB caching to the schema. Bypassing CPU re-encoding allows a 50,000-vector routing table to boot in **0.45 seconds**, completely eliminating cold-start bottlenecks.

---

## Head-to-Head Benchmark

SynaptoRoute is architecturally optimized for async concurrent deployments. We evaluated it against `semantic-router` (using their default `FastEmbedEncoder`) to measure architectural scaling.

| Metric | `semantic-router` | `SynaptoRoute` (`v0.2.0`) |
|--------|-------------------|---------------------------|
| **Hot-Reload Degradation** (500 Routes) | +6.46 ms | **+0.74 ms** |
| **Concurrent Async Load** | `Index is not ready` (Thread Blocked) | **Successfully Batched** (38+ QPS) |

> **The Architectural Difference:** The +0.74ms vs +6.46ms hot-reload degradation is a direct consequence of $O(1)$ lazy memory slicing vs $O(N)$ index recompilation. Under `asyncio` concurrent load, `semantic-router`'s sync-first design produced `Index is not ready` failures; SynaptoRoute's `_batch_worker` queue absorbed all requests without dropping a query.

> **View Full Benchmarks:** For detailed statistical analysis, including GPU physics scaling, P50 vs P99 tradeoffs, and our roadmap for fixing distributed system limitations, see our official [BENCHMARKS.md](BENCHMARKS.md).

---

## Installation & Deployment

### Method 1: Docker REST API (Recommended)

SynaptoRoute ships with a fully asynchronous FastAPI wrapper, designed for immediate drop-in deployment as a scalable microservice.

```bash
# Build the Docker image
docker build -t synaptoroute .

# Run the container
docker run -p 8000:8000 synaptoroute
```

### Method 2: Standard Python Package

To embed SynaptoRoute natively into your existing Python pipelines:

```bash
pip install synaptoroute
```

---

## Quick Start Guide

```python
import asyncio
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.models import Route

async def main():
    # 1. Initialize Components
    encoder = Encoder(providers=["CPUExecutionProvider"])
    storage = SQLiteStorage("data/memory.sqlite")
    router = AdaptiveRouter(encoder, storage)
    
    # 2. Define Routes
    billing_route = Route(
        name="billing", 
        utterances=["I need a refund", "Where is my receipt?", "Cancel my subscription"]
    )
    router.add_route(billing_route)
    
    # 3. Start the Background Batching Worker
    await router.start()
    
    # 4. Execute Async Queries
    result = await router.aquery("How do I get my money back?")
    print(f"Matched Intent: {result.name}") # Output: billing
    
    # 5. Graceful Shutdown
    await router.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## System Limitations 

**Horizontal Scaling (Kubernetes Split-Brain)**  
SynaptoRoute relies on a highly optimized, local in-memory NumPy tensor to achieve its routing speed. As such, it is structurally bound to a single node. If deployed across multiple load-balanced Kubernetes pods, a hot-reload request hitting Pod A will update Pod A's local memory, but Pod B will remain unaware. 

Evaluating distributed embedding synchronization (e.g., Redis Pub/Sub or shared-memory) to unblock horizontal scaling is a core research focus for `v0.3.0`.

---

## Community & Contributing

We welcome contributions of all sizes from the open-source community! 

- **Contributing:** Please read our [Contributing Guidelines](CONTRIBUTING.md) to learn how to set up your development environment.
- **Issues:** If you discover a bug or have a feature request, please [open an issue](https://github.com/sitanshukr08/SynaptoRoute/issues).
