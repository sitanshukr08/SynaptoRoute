# SynaptoRoute

[![PyPI version](https://badge.fury.io/py/synaptoroute.svg)](https://pypi.org/project/synaptoroute/)
[![CI/CD Pipeline](https://github.com/sitanshukr08/SynaptoRoute/actions/workflows/ci.yml/badge.svg)](https://github.com/sitanshukr08/SynaptoRoute/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python >=3.9](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://www.python.org/)

SynaptoRoute is a high-performance semantic routing engine designed for production Python microservices. It executes intent classification locally using hardware-accelerated vector embeddings to route natural language to deterministic software logic.

It is specifically designed for:
- **Tool Routing** (Mapping prompts to function signatures)
- **Agent Routing** (Handing off state to specialized subagents)
- **Workflow Orchestration** (Triggering RAG chains or DB queries)
- **Large-Scale Intent Classification** (Supporting massive, dense domain definitions)

## Key Features

✓ **Async Batching Queue:** Prevents event loop blocking and absorbs massive concurrent loads without hardware lockup.
✓ **Fast Route Mutations:** Hot-swap intents in near-constant-time memory without rebuilding the search index.
✓ **50,000+ Route Support:** Backed by an optional `Faiss` index to maintain interactivity at massive scale.
✓ **Pluggable Architecture:** Seamlessly swap embedding providers (Local ONNX, OpenAI, etc.).
✓ **Distributed Sync:** Redis-backed pub/sub to keep Kubernetes replicas aligned.

---

## Benchmark Highlights

SynaptoRoute has been rigorously benchmarked against both procedural stressors and non-synthetic canonical datasets.

- **50,000 routes tested** interactively.
- **~49ms retrieval latency** at 250,000 embeddings.
- **CLINC150:** 74.20% Top-1 Accuracy
- **Banking77:** 91.81% Top-1 Accuracy (Zero-shot mapping on 77 highly overlapping intents)
- **~302 QPS** on concurrent batched benchmark workloads.

*For full statistical breakdowns, methodology, and comparisons, see [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [docs/COMPARISON.md](docs/COMPARISON.md).*

---

## When to Use SynaptoRoute

**Use SynaptoRoute if:**
- You need local, edge-deployed routing without API dependencies.
- You need high concurrency capable of surviving asynchronous spikes.
- You expect massive routing tables (1,000 to 50,000+ routes).
- You want highly predictable query latency regardless of scale.

**Consider alternatives if:**
- You need logical reasoning or downstream multi-step planning.
- You need complex multi-intent decomposition.
- You require strict Out-Of-Distribution detection without manual calibration.

---

## Architecture & Design

```mermaid
graph TD
    Client([Client / Microservice]) -->|aquery()| AR[AdaptiveRouter]
    
    subgraph Routing Engine
        AR -->|Queue| Worker[Batch Worker Task]
        Worker -->|process_batch| Encoder[Encoder<br>FastEmbed / OpenAI]
        Encoder -->|Vectors| Index[(Vector Index<br>Numpy / Faiss)]
        Index -->|Top-K Match| AR
    end
    
    subgraph State Management
        AR -->|Save/Load| SQL[(SQLiteStorage)]
        SQL -.->|Hydrate| Index
        AR <-->|Pub/Sub| Sync[RedisSyncManager]
        Sync <-->|synaptoroute:sync| Cluster([Other Kubernetes Nodes])
    end
```

In modern microservice architectures, relying on external APIs for classification routing introduces high latency, cost, and rate limits. SynaptoRoute executes intent classification locally, avoiding two structural bottlenecks common in semantic routing:

1. **Sequential Compute Starvation:** Processing single semantic requests sequentially creates a bottleneck for parallel API calls, eventually forcing thermal throttling or thread exhaustion on local hardware. SynaptoRoute captures concurrent requests in a background `_batch_worker` queue, groups them (e.g., batch size 32), and executes them in a single optimized pass through the inference engine.
2. **Index Rebuilding Penalty:** Standard routers execute an $O(N)$ reallocation of the entire memory space when routes change. SynaptoRoute utilizes lazy slicing and memory-mapped tombstoning to allow instant insertions and deletions.

---

### 1. Installation

```bash
pip install synaptoroute

# Optional Extras
pip install synaptoroute[api]          # For FastAPI integration
pip install synaptoroute[openai]       # For using OpenAI embeddings
pip install synaptoroute[metrics]      # For telemetry and evaluation
pip install synaptoroute[redis]        # For distributed deployment syncing
pip install synaptoroute[faiss]        # For massive route scaling (50,000+)
pip install synaptoroute[langchain]    # For LangChain ecosystem integration
pip install synaptoroute[llamaindex]   # For LlamaIndex ecosystem integration
pip install synaptoroute[all]          # Installs all optional dependencies
```

---

## Quick Start Guide

### Basic Example

```python
import asyncio
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import FastEmbedEncoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.models import Route

async def main():
    # 1. Initialize Components
    encoder = FastEmbedEncoder(model_name="BAAI/bge-small-en-v1.5")
    storage = SQLiteStorage("data/memory.sqlite")
    router = AdaptiveRouter(encoder=encoder, storage=storage)
    
    # 2. Define Routes
    billing_route = Route(
        name="billing", 
        utterances=["I need a refund", "Where is my receipt?", "Cancel my subscription"],
        threshold=0.60
    )
    router.add_route(billing_route)
    
    # 3. Start the Background Batching Worker
    await router.start()
    
    # 4. Execute Async Queries
    result = await router.aquery("How do I get my money back?")
    if result:
        print(f"Matched Intent: {result.name}") # Output: billing
    
    # 5. Graceful Shutdown
    await router.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Advanced Configuration

### Optimization Profiles
SynaptoRoute allows you to load strict optimization profiles depending on your infrastructure constraints:
```python
from synaptoroute import OptimizationProfile, AdaptiveRouter

# THROUGHPUT: Maximizes QPS for heavy concurrent loads
router = AdaptiveRouter(profile=OptimizationProfile.THROUGHPUT)

# LATENCY: Minimizes response time for sequential or low-concurrency systems
router = AdaptiveRouter(profile=OptimizationProfile.LATENCY)
```
*Caveat: `profile.threads` must be passed explicitly to `FastEmbedEncoder`. The router does not propagate thread count automatically.*

### Distributed Deployment
For multi-pod Kubernetes or horizontal scaling, SynaptoRoute uses `RedisSyncManager` to synchronize SQLite route databases across nodes:
```python
from synaptoroute.sync import RedisSyncManager

sync_manager = RedisSyncManager(redis_url="redis://localhost:6379")
router = AdaptiveRouter(sync_manager=sync_manager)
```
*Caveat: The current `RedisSyncManager` implementation does not retry on Redis disconnects.*

## Roadmap

- **v0.4.0 (Dynamic Boundaries):** Automatic docstring extraction and LLM-assisted synthetic utterance generation to seed intents with zero manual configuration. LangGraph native `ToolNode` injection.
- **v0.5.0 (Multi-Modal):** CLIP/ImageBind integration to accept `PIL.Image` objects and route visual data directly to specialized subsystems.
- **v0.6.0 (Advanced Network Distribution):** Packaging SynaptoRoute as a standalone gRPC microservice for federated remote cluster routing.

For the detailed strategic vision, see [docs/ROADMAP.md](docs/ROADMAP.md).

## Known Limitations

1. **Directional Semantics:** Vector similarity cannot distinguish between "flight book" and "cancel flight".
2. **Deep Logical Negation:** Modifiers like "don't", "never", and "not" are inherently problematic for dense embeddings.
3. **Threshold Calibration:** Defining a global threshold across highly disparate intents requires manual tuning.
4. **Mixed Intent Parsing:** Cannot natively split multi-action sentences into discrete routes.
5. **Context Amnesia:** Evaluates single utterances strictly without conversation history.
6. **Cross-Language Drift:** Cosine boundary margins differ significantly when evaluating multiple languages simultaneously.
7. **Adversarial Resilience:** Keyword traps will natively bypass standard embeddings unless explicitly trained out.

For a detailed analysis of these failure modes and how to implement recommended fallback mechanisms (like LLM verification), please read our [limitations documentation](docs/limitations.md).

---

## Community & Contributing

We welcome professional contributions to expand the framework.

- **Contributing:** Review the [Contributing Guidelines](CONTRIBUTING.md) for architectural enforcement policies.
- **Issues:** Report bugs or request features via the [Issue Tracker](https://github.com/sitanshukr08/SynaptoRoute/issues).
