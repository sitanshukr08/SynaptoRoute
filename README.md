# SynaptoRoute

[![PyPI version](https://badge.fury.io/py/synaptoroute.svg)](https://pypi.org/project/synaptoroute/)
[![CI/CD Pipeline](https://github.com/sitanshukr08/SynaptoRoute/actions/workflows/ci.yml/badge.svg)](https://github.com/sitanshukr08/SynaptoRoute/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/sitanshukr08/SynaptoRoute/blob/main/CONTRIBUTING.md)

SynaptoRoute is a high-throughput, local semantic routing engine built for production Python microservices. Designed as a mathematically optimal alternative to Large Language Model (LLM) routing chains and slower local routers, it provides zero-token intent classification in under 3 milliseconds on standard cloud hardware.

## Table of Contents
- [Why SynaptoRoute?](#why-synaptoroute)
- [Architecture & Optimizations](#architecture--optimizations)
- [Performance Benchmarks](#performance-benchmarks)
- [Installation & Deployment](#installation--deployment)
- [Quick Start Guide](#quick-start-guide)
- [System Limitations](#system-limitations)
- [Community & Contributing](#community--contributing)

---

## Why SynaptoRoute?

In modern agentic systems, relying on an external API (like OpenAI or Anthropic) to make simple routing decisions—such as determining if a user wants to reset their password or check their balance—introduces unacceptable latency (300ms+) and high token costs.

SynaptoRoute solves this by executing intent classification entirely locally using INT8 quantized vector embeddings. 

SynaptoRoute was engineered specifically to solve the $O(N)$ memory degradation problem during live hot-reloading and to maximize hardware utilization via asynchronous dynamic batching.

## Architecture & Optimizations

### 1. Lazy Memory Compilation
Traditional routers suffer from severe performance degradation during live updates. When a new route is added, they execute an immediate `numpy.vstack`, copying the entire vector array in memory ($O(N)$ complexity). SynaptoRoute defers this reallocation, appending new vectors to a lightweight list ($O(1)$) and only executing the heavy compilation precisely when the next query arrives, preventing server freezes.

### 2. Dynamic Asynchronous Batching
Hardware accelerators (GPUs, AVX512 CPUs) are optimized for large matrix multiplications. Sending single queries sequentially incurs massive transfer overhead. SynaptoRoute utilizes a background `asyncio.Queue` worker that traps parallel HTTP requests, waits 5 milliseconds, groups them into a batch, and processes them in a single hardware cycle.

### 3. INT8 Quantization
By default, SynaptoRoute leverages the `BAAI/bge-small-en-v1.5` model quantized to 8-bit integers via the ONNX runtime, slashing memory bandwidth requirements by 4x and maximizing CPU cache utilization.

---

## Performance Benchmarks

The following metrics were captured via automated GitHub Actions CI/CD running on a standard, unaccelerated `ubuntu-latest` 2-core cloud CPU.

| Metric | Cloud CPU Latency | Context |
| :--- | :--- | :--- |
| **Inference P99** | 3.94 ms | Single sequential query latency. |
| **Amortized P50** | 2.69 ms | Per-query latency when processing 1,000 concurrent requests via dynamic batching. |
| **Hot-Reload** | 5.04 ms | Time required to dynamically inject a new utterance into memory without dropping active API requests. |

> **📊 View Full Benchmarks:** For detailed analysis including Memory Leak Endurance, GPU Scaling, Classification F1-Scores, and Input Poisoning Survival Metrics, see our official [BENCHMARKS.md](BENCHMARKS.md).

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

You can interface with the router immediately:
```bash
curl -X POST http://localhost:8000/route \
     -H "Content-Type: application/json" \
     -d '{"query": "I need help resetting my password"}'
```

### Method 2: Standard Python Package

To embed SynaptoRoute natively into your existing Python pipelines, install directly from pip (or via git if testing the latest main branch):

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
    encoder = Encoder()
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
SynaptoRoute relies on a highly optimized, local in-memory NumPy matrix to achieve its microsecond latency. As such, it is structurally bound to a single node. If deployed across multiple load-balanced Kubernetes pods, a hot-reload request hitting Pod A will update Pod A's local memory, but Pod B will remain unaware. Scaling horizontally requires implementing an external event bus (e.g., Redis Pub/Sub) to broadcast memory invalidation events across the cluster.

---

## Community & Contributing

We welcome contributions of all sizes from the open-source community! 

- **Contributing:** Please read our [Contributing Guidelines](CONTRIBUTING.md) to learn how to set up your development environment, run the test suite, and submit Pull Requests.
- **Code of Conduct:** We are committed to fostering a welcoming environment. Please review our [Code of Conduct](CODE_OF_CONDUCT.md).
- **Issues:** If you discover a bug or have a feature request, please [open an issue](https://github.com/sitanshukr08/SynaptoRoute/issues).
