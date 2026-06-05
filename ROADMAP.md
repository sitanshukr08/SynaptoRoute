# SynaptoRoute Roadmap

This document serves as the historical and future architectural trajectory for SynaptoRoute. It outlines our foundational milestones, recent accomplishments, and target objectives through version 0.6.0.

---

## Historical Milestones

### v0.1.0 (Proof of Concept)
* **Core Engine Design:** Established the initial `AdaptiveRouter` architecture for intercepting natural language queries.
* **Dense Vector Routing:** Prototyped routing mechanics using basic local embeddings and exact mathematical distance thresholds.
* **In-Memory State:** Implemented the baseline dynamic mutation API allowing routes to be added or removed without restarting the Python process.

### v0.2.0 (High-Performance Vectorization)
* **FAISS Integration:** Replaced brute-force distance calculations with Facebook's FAISS (Hierarchical Navigable Small World graphs) to achieve sub-linear search latency.
* **Tombstone Architecture:** Developed an O(1) instantaneous deletion array to mask dead vectors, bypassing the severe latency penalties of standard FAISS deletions.
* **FastEmbed Optimization:** Transitioned to ONNX-based `FastEmbedEncoder` for zero-overhead, sub-millisecond local vector generation.

### v0.3.0 (Distributed Synchronization & Tuning)
* **Redis PubSub Topology:** Introduced `RedisSyncManager` to enable live multi-pod state synchronization and horizontal scaling via event broadcasting.
* **Pluggable Encoders:** Abstracted the encoder layer to support external remote execution endpoints (e.g., `OpenAIEncoder`).
* **Automated Tuning:** Deployed `SyntheticTuner` for automated out-of-domain margin boundary analysis using LLMs to generate adversarial data.
* **Telemetry & Observability:** Formalized hardware `OptimizationProfile` presets and established initial Banking77 accuracy baselines.

---

## Completed (v0.4.1)

* **SQLite WAL Persistence:** Replaced brittle JSON snapshotting with a robust embedded SQLite database using Write-Ahead Logging (WAL) and `IMMEDIATE` transaction isolation.
* **Asynchronous Micro-Batching:** Re-architected the `AdaptiveRouter` inference pipeline to use an `asyncio.Queue` worker that dynamically micro-batches incoming queries, drastically lifting throughput limits.
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
