# SynaptoRoute Roadmap

This document serves as the architectural and feature trajectory for SynaptoRoute. It outlines what has been accomplished in recent releases and details our target objectives up to version 0.6.0.

---

## Completed (v0.4.1 and prior)

* **SQLite WAL Persistence:** Replaced brittle JSON snapshotting with a robust embedded SQLite database using Write-Ahead Logging (WAL) and `IMMEDIATE` transaction isolation.
* **Asynchronous Micro-Batching:** Re-architected the `AdaptiveRouter` to use an `asyncio.Queue` worker that dynamically micro-batches incoming queries, lifting throughput limits significantly and successfully satisfying the "Encoder Throughput Batching" goal.
* **Two-Stage Retrieval (Cross-Encoder Fallback):** Implemented `CrossEncoderReranker` to allow intelligent deferral to dense reasoning models when initial bi-encoder matches fall within uncertain semantic boundaries.
* **Phase 3 Baseline Validation:** Mathematical telemetry proven on 1M vector indices, achieving ~91.16% Banking77 accuracy and 3.0ms large-scale P95 latency.

---

## v0.5.0 (Upcoming: Enterprise Stability & OOD Rejection)

### Goal: Automated Distributed Regression Suite
* **Why It Matters:** While standard unit tests ensure single-node validity, multi-node deployments are prone to race conditions and eventual consistency failures over long uptimes.
* **Success Criteria:**
  * Implement an exhaustive `run_regression_suite.py` specifically simulating long-running cyclic synchronization failures.
  * Ensure FAISS memory leak and Redis broadcast loop regressions are algorithmically prevented in CI/CD.

### Goal: Out-of-Distribution (OOD) Rejection Improvement
* **Why It Matters:** Currently, SynaptoRoute can sometimes force matches onto dense boundaries, leading to false positives when a user query does not belong to any defined route.
* **Success Criteria:**
  * Implement advanced outlier detection or thresholding algorithms to confidently reject unrelated queries.
  * Area Under the Receiver Operating Characteristic (AUROC) > 0.90.
  * False Positive Rate at 95% True Positive Rate (FPR@95) < 10%.
  * The benchmark is fully reproducible in the `scratch/` directory.

---

## v0.6.0 (Upcoming: Distributed Scaling)

### Goal: Distributed Bootstrapping Overhaul
* **Why It Matters:** The current `request_full_sync` Redis PubSub implementation requires `O(N×M)` full-state broadcasting when a new node boots. This causes immediate network saturation and OOM faults above 100,000 routes in enterprise-scale environments.
* **Success Criteria:**
  * Integration of a central durable ledger (e.g., Litestream-backed SQLite or an external PostgreSQL adapter).
  * Redis PubSub is exclusively restricted to broadcasting lightweight incremental deltas (`add_route`, `delete_route`).
  * Bootstrapping a 1M route node completes in under 5 seconds without triggering a broadcast storm on existing nodes.

### Goal: Multi-tenant Semantic Indexing
* **Why It Matters:** Enterprise SaaS users require logical isolation of their vector databases within a single running instance to prevent cross-tenant data leakage.
* **Success Criteria:**
  * Enable the dynamic creation and routing of isolated `FaissIndex` instances on a per-tenant API key basis.
  * Negligible overhead impact on concurrent memory utilization.
