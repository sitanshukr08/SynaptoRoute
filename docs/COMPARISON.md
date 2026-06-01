# SynaptoRoute vs Semantic Router

## Executive Summary

Both SynaptoRoute and Semantic Router are semantic classification engines built for routing LLM requests, AI agents, and tools. However, they approach the problem with different architectural priorities. 

**Semantic Router** is designed as a straightforward, highly accessible routing layer that works excellently out of the box for simple use cases. **SynaptoRoute** is designed as a high-concurrency, stateful routing engine built explicitly to scale to massive route counts (50,000+) without blocking the event loop or causing significant throughput degradation during burst loads.

This document provides a brutally honest, benchmark-backed comparison between the two systems.

---

## Methodology

All benchmarks were run on identical hardware (CPU) using the `BAAI/bge-small-en-v1.5` embedding model via `FastEmbed`.
To prevent test leakage, datasets with a strict separation between `train` (used to define the routes) and `test` (used to evaluate queries) were utilized. 

*Note: During real-world testing (5,500 queries), Semantic Router exhibited significant throughput degradation and became impractical to benchmark under the tested sequential workload.*

---

## Clean Intent Benchmarks

These benchmarks measure performance on well-structured, distinct canonical intents without adversarial manipulation.

### CLINC150 (150 Intents)
*SynaptoRoute demonstrates competitive accuracy on broad-domain intent classification.*
- **SynaptoRoute Top-1:** 74.20%
- **SynaptoRoute Top-5:** 78.33%
- **Semantic Router Top-1:** 73.35%

### Banking77 (77 Dense Intents)
*A notoriously difficult dataset containing 77 highly overlapping intents in a single banking domain.*
- **SynaptoRoute Top-1:** 91.81%
- **SynaptoRoute Top-5:** 96.16%
- **Semantic Router Top-1:** 91.29%

---

## Adversarial Benchmarks

These benchmarks stress the fundamental limits of dense vector similarity by using language designed to trick keyword matchers and cosine similarity algorithms. Both systems were evaluated using the identical `FastEmbedEncoder(model_name="BAAI/bge-small-en-v1.5")` configuration for a strict apples-to-apples comparison.

| Adversarial Category | SynaptoRoute | Semantic Router |
|----------------------|--------------|-----------------|
| **Hard Negatives** | Poor | Good |
| **Keyword Traps** | 58.8% | 85.0% |
| **Lexical Overlap** | 25.0% | 40.0% |
| **Negation** | 0% | 15% |

*Analysis: Semantic Router has superior default thresholding logic for handling hard negatives out-of-the-box using pure embeddings. SynaptoRoute plans to address these lexical traps in v0.6.0 via a Cross-Encoder reranking pipeline.*

---

## Scalability Benchmarks

This is the primary architectural divergence between the two systems.

| Metric | SynaptoRoute | Semantic Router |
|--------|-------------|-----------------|
| **Route Count** | Tested up to 50,000 routes | Not measured (degrades significantly) |
| **Concurrency** | Async Batch Worker Queue | Blocking Sequential Execution |
| **Throughput (QPS)** | ~302 QPS (batched) | ~30 QPS (estimated unbatched) |
| **Route Mutation** | Near-constant-time Hot-Swapping via SQLite | Complete Index Rebuild Required |

---

## Architecture Comparison

### SynaptoRoute
Uses an **asynchronous batch worker** decoupled from the HTTP/API event loop. When 100 concurrent queries arrive, they are placed in an internal queue, grouped into batches (e.g., size 32), and passed to the ONNX runtime exactly once. The underlying index is a `Faiss` memory-mapped flat index backed by `SQLite`, allowing instant route additions, edits, and deletions without rebuilding the index.

### Semantic Router
Uses a **synchronous routing layer**. When a query arrives, it immediately requests an embedding. If 100 queries arrive, the system initializes 100 individual compute graphs, causing massive CPU overhead, significant throughput degradation, and blocking execution. Route mutations require redefining the `RouteLayer` entirely.

---

## Where SynaptoRoute Wins

✓ **Throughput & Concurrency:** The async batch queue prevents hardware lockup under load.
✓ **Hot Reloading:** `add_route` and `update_route` apply instantly in memory and SQLite.
✓ **Large Route Counts:** Proven to scale to 50,000+ routes with ~49ms query latency.
✓ **Route Mutation Performance:** Near-constant-time tombstone memory management.

## Where Semantic Router Wins

✓ **Out-of-box adversarial performance:** Better default logic for trick queries.
✓ **Simpler architecture:** Less complex to instantiate if concurrency doesn't matter.
✓ **Better default threshold behavior:** Requires less manual margin calibration for simple OOD rejection.

## Where Both Fail

Dense embeddings natively struggle with semantic directionality. Both systems universally fail on:
✗ **Double negation** (*"I do not want to not cancel this"*).
✗ **Mixed intent** (*"Book a flight but also delete my account"*).
✗ **Logical contradiction handling**.

---

## Reproducibility

The metrics in this document were generated using the `benchmarks/` suite in the SynaptoRoute repository. 
You can reproduce the real-world metrics (without Kaggle APIs) using:
```bash
python benchmarks/bench_realworld.py
```
*(Note: Attempting to reproduce the Semantic Router real-world run on consumer hardware results in impractical execution times under the tested workload due to sequential ONNX compute locking).*
