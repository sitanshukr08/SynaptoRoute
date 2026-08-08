# SynaptoRoute

Local, persistent semantic routing with explicit abstention, mutation, and
durability behavior.

[![PyPI](https://img.shields.io/pypi/v/synaptoroute.svg)](https://pypi.org/project/synaptoroute/)
[![CI](https://github.com/sitanshukr08/SynaptoRoute/actions/workflows/ci.yml/badge.svg)](https://github.com/sitanshukr08/SynaptoRoute/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

SynaptoRoute maps natural-language queries to predefined tool handlers, APIs,
or workflow chains. It is not an LLM, an authorization boundary, or a prompt
injection defense. Downstream systems must still validate inputs and authorize
actions.

## Capabilities

* local embedding-based retrieval with NumPy exact search and optional FAISS;
* route thresholds, score margins, explicit abstention, and ranked results;
* bounded asynchronous query batching and overload signaling;
* versioned SQLite WAL persistence with explicit mutation receipts;
* online add, replace, utterance-add, threshold-update, and delete operations;
* optional LangChain, LlamaIndex, Redis, reranking, and metrics integrations.

Adaptive-memory and Redis synchronization features are experimental, disabled
by default, and excluded from the primary research artifact.

No performance or semantic-quality number in this repository is publication
evidence unless its manifest is independently reproduced and marked
`verified`. See [Current Evidence Status](docs/CURRENT_EVIDENCE_STATUS.md).

## Installation

```bash
pip install synaptoroute
```

For local development:

```bash
git clone https://github.com/sitanshukr08/SynaptoRoute.git
cd SynaptoRoute
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev,test]"
python -m pytest tests -q
```

## Quickstart

```python
from synaptoroute import AdaptiveRouter, Route, SQLiteStorage

storage = SQLiteStorage("routes.sqlite3", synchronous="FULL")
router = AdaptiveRouter(storage=storage)

receipt = router.add_route(
    Route(
        name="billing_support",
        utterances=[
            "I need a refund",
            "Where is my invoice?",
            "Cancel my subscription",
        ],
        threshold=0.75,
    )
)

# The mutation is already visible in this process. Wait explicitly when the
# caller needs process-crash durability before continuing.
receipt.wait_durable(timeout=10.0)

result = router.match("Where can I download my invoice?")
if result.matched:
    print(result.route_name, result.score, result.decision_reason)
else:
    print("abstained", result.decision_reason)

router.close()
```

`MutationReceipt` reports the route, resulting route version, queue state,
durable latency, and failure detail. A memory-visible mutation is not described
as durable until `wait_durable()` succeeds.

## Async Queries

```python
import asyncio

from synaptoroute import AdaptiveRouter, Route


async def main():
    router = AdaptiveRouter(
        max_queue_size=1000,
        max_in_flight_batches=4,
        max_storage_queue_size=1000,
    )
    router.add_route(
        Route(name="password_reset", utterances=["forgot password", "reset login"])
    )
    await router.start()
    try:
        result = await router.amatch("How do I reset my password?")
        print(result.route_name, result.decision_reason)
    finally:
        await router.stop()


asyncio.run(main())
```

## Evidence Workflow

Run the offline structural smoke:

```bash
python benchmarks/run_ci_smoke_benchmark.py --output-dir benchmark_results/ci_smoke
```

The output is always `unverified` and is not semantic-quality or paper
evidence. Long-running candidate experiments use the unified runner:

```bash
python benchmarks/run_all_benchmarks.py \
  --benchmarks banking77_multiseed clinc150_multiseed \
  --output-dir benchmark_results/candidate
```

Evidence promotion is a separate operation requiring a clean original run, a
clean reproduction from another machine, reviewer attestation, and an
immutable archive digest. See [Artifact Evaluation](paper/ARTIFACT_EVALUATION.md).

## Research Direction

The primary research question is whether a mutable local semantic router can
provide explicit process-crash durability and predictable overload behavior
while routes change concurrently. Calibration remains an ablation; the project
does not claim superior intent-classification accuracy.

Working paper:

> SynaptoRoute: Durability and Backpressure Semantics for Mutable Local
> Semantic Routers

The frozen questions, baselines, metrics, experiment matrix, and evidence gates
are documented in:

* [Research Protocol](docs/RESEARCH_PROTOCOL.md)
* [Research Roadmap](ROADMAP.md)
* [Mutation And Durability Contract](docs/DURABILITY_CONTRACT.md)
* [Benchmark Registry](docs/BENCHMARK_REGISTRY.md)
* [Paper Skeleton](paper/PAPER.md)

## Project Status

PyPI `0.4.1` is the latest published release. The research artifact line is
developed as `0.5.0.dev0`; `v0.5.0` will be released only after its source,
documentation, tests, and archived evidence agree.

SynaptoRoute is licensed under the [MIT License](LICENSE).
