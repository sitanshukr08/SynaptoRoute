# SynaptoRoute Roadmap

This roadmap documents the architectural and research trajectory for SynaptoRoute.

## Completed (Verified in v0.4.0)

* **Phase 3 Baseline Validation:** ~91.16% Banking77 accuracy and 3.0ms large-scale latency mathematically proven.
* **Critical Remediation Audit:** Concurrency thread crashes, FAISS overwriting leaks, and queue exhaustion limits successfully remediated.
* **Automated Regression Suite:** Introduced `run_regression_suite.py` to prevent cyclic synchronization failures.
* **SQLite WAL Persistence:** Replaced brittle JSON snapshotting with robust Write-Ahead-Logging local storage.

---

## Research Roadmap

### Goal: Out-of-Distribution (OOD) Rejection Improvement
* **Why It Matters:** Currently, SynaptoRoute forces matches onto dense boundaries, leading to false positives when a user query does not belong to any defined route.
* **Success Criteria:**
  * Area Under the Receiver Operating Characteristic (AUROC) > 0.90
  * False Positive Rate at 95% True Positive Rate (FPR@95) < 10%
  * The benchmark is fully reproducible in `scratch/`.

### Goal: Cross-Encoder Fallback Baseline
* **Why It Matters:** Bi-encoders (like `FastEmbedEncoder`) are incredibly fast but lack deep contextual matching logic. Cross-encoders are slower but highly accurate. A fallback strategy is required for ambiguous routing calls.
* **Success Criteria:**
  * Implementation of a two-stage retrieval pipeline.
  * Measured end-to-end latency remains under 50ms on the fallback path.
  * Verified accuracy bump > 3.0% over pure bi-encoder baselines on adversarial OOD sets.

---

## Engineering Roadmap

### Goal: Distributed Bootstrapping Overhaul
* **Why It Matters:** The current `request_full_sync` Redis PubSub implementation requires O(N×M) full-state broadcasting when a new node boots. This causes immediate network saturation and OOM faults above 100,000 routes.
* **Success Criteria:**
  * Integration of a central durable ledger (e.g., Litestream-backed SQLite or external PostgreSQL).
  * Redis PubSub is exclusively restricted to incremental deltas (`add_route`, `delete_route`).
  * Bootstrapping a 1M route node completes in under 5 seconds without triggering a broadcast storm on existing nodes.

### Goal: Encoder Throughput Batching
* **Why It Matters:** Local FAISS execution handles ~1,000,000 queries per second, but the internal `FastEmbedEncoder` on a single CPU core caps throughput at ~130 requests per second.
* **Success Criteria:**
  * Implement dynamic micro-batching across incoming HTTP/asyncio requests.
  * Throughput (P99) under load exceeds 1,000 RPS on a standard 4-core instance.
  * Memory footprint remains under 1GB during burst ingestion.
