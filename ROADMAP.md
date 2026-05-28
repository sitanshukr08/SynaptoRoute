# SynaptoRoute: Future Development Roadmap

This document outlines the strategic vision and upcoming features for `SynaptoRoute`. Our goal is to evolve from a high-speed text semantic router into an industry-standard, multi-modal, highly scalable routing framework for all agentic architectures.

---

## 🛠️ v0.2.0 - Technical Debt & Polish
**Status:** Planned

The immediate next release focuses on fixing known edge cases discovered during extreme stress testing and optimizing database I/O bottlenecks. **This is the highest priority and must ship before any new features.**

* **SQLite I/O Optimization:** Implement `update_threshold(route, threshold)` to update single records, avoiding a full $O(N)$ database rewrite during `fit_thresholds()`.
* **Graceful Exceptions:** Patch the raw `KeyError` race condition in `add_utterance` to return a clean API-friendly `RouteNotFoundError`.
* **Dynamic Dimensionality:** Remove the hardcoded `384` dimension fallback in `encoder.py`. This is critical to unblock future encoder portability.

---

## 👁️ v0.3.0 - Multi-Modal Routing
**Status:** Conceptual

Expand semantic routing beyond text to handle images and visual data, enabling complex routing in Vision-Agent workflows.

* **CLIP Integration:** Allow the `AdaptiveRouter` to accept `PIL.Image`. 
* **Modality Alignment:** Since CLIP and `bge-small` exist in different semantic spaces, this will require either maintaining separate per-modality indexes or migrating to a unified multimodal model (e.g., ImageBind).
* **Resource Budgeting:** Explicitly define latency and memory budgets before committing, as models like CLIP introduce a significantly heavier footprint (~600MB) compared to our current INT8 architecture.

---

## 🚀 v0.4.0 - Massive Scale (FAISS / HNSW)
**Status:** Conceptual

Introduce approximate nearest-neighbour (ANN) indexes as an *optional pluggable backend* for million-route scale. (Note: For our current sweet spot of <100k vectors, brute-force NumPy will remain the default as it outperforms ANN overhead).

* **Vector Graph Backend:** Integrate Meta's **FAISS** or an **HNSW** graph. ANN search trades exact cosine matching for ~99% recall at $O(\log N)$ query cost.
* **Sub-Linear Inserts:** Shift from our current $O(1)$ NumPy lazy-append trick to an amortized sub-linear $O(\log N)$ insert cost, utilizing shadow indexes merged in the background to prevent blocking during HNSW hot-reloads.

---

## ⚙️ v0.5.0 - Zero-Shot Tool Routing
**Status:** Conceptual

Eliminate the need for manual "utterances" by generating semantic boundaries directly from function signatures. This is a highly differentiated feature designed to drastically improve developer UX.

* **Docstring Extractor:** Parse Python function `docstrings` and `Pydantic` schemas.
* **Synthetic Utterance Generation:** To solve the problem of terse docstrings producing poor embeddings, we will introduce an optional LLM call at *registration time* (a one-time cost, zero latency at inference). The LLM will expand the docstring into 5-10 diverse synthetic utterances to seed a robust semantic boundary.

---

## ⚖️ v0.6.0 - Cross-Encoder Tie-Breakers
**Status:** Conceptual

Maximize classification accuracy in highly ambiguous routing scenarios using a classic two-stage retrieval pattern (bi-encoder shortlist → cross-encoder verdict).

* **Reranking Layer:** If the top two routes score within a configurable margin (e.g., ±0.05) of each other, trigger a lightweight CrossEncoder (e.g., MS MARCO) to perform a surgical tie-break.
* **Accuracy-vs-Latency Mode:** This will be explicitly documented as an optional mode, as the cross-encoder latency penalty (~20-80ms) violates our strict sub-5ms SLA. It preserves the fast-path performance for the common case while ensuring accuracy for edge cases.

---

## 🌐 v0.7.0 - Distributed State Sync (K8s Horizontal Scaling)
**Status:** Conceptual

Resolve the Kubernetes split-brain issue for multi-pod deployments, making `SynaptoRoute` horizontally scalable.

* **Pub/Sub Invalidation:** Implement a Redis Pub/Sub layer to broadcast route updates across multiple worker nodes.
* **Cluster Consistency:** When a new utterance is added to the SQLite backend on Node A, Node B will instantly receive a cache invalidation event, allowing local in-memory NumPy matrices to stay perfectly synced across a distributed enterprise fleet.
* **High-Write Scaling (PostgreSQL):** Acknowledge that while SQLite is excellent for read-heavy workloads, its single-writer lock model will bottleneck under heavy concurrent `add_utterance` calls across pods. Introduce an optional PostgreSQL storage adapter for true high-write enterprise deployments.
