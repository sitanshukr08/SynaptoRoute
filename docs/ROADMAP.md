# SynaptoRoute: Future Development Roadmap

This document outlines the strategic vision and upcoming features for `SynaptoRoute`. Our goal is to evolve from a high-speed text semantic router into a multi-modal, highly scalable routing framework for distributed microservice architectures.

---

## v0.3.0 - Scalability & Distributed Systems
**Status:** Completed

The v0.3.0 release addressed critical architectural gaps required for multi-node deployments, robust index validation, and formal observability.
* **Distributed State Sync (K8s Horizontal Scaling):** Implemented `RedisSyncManager` to broadcast cache invalidation and sync events across instances.
* **Pluggable Encoder Backends:** Abstracted the hard dependency on `FastEmbed` via the `Encoder` interface. Added native support for `OpenAIEncoder`.
* **Observability & Metrics Endpoint:** Introduced `MetricsRegistry` for Prometheus to formally track P50/P99 latency, queue depth, and throughput in production.
* **Cross-Encoder Tie-Breakers (Reranking):** Added native support for a `reranker` layer to surgical tie-break ambiguous top routes.
* **Synthetic Threshold Tuning:** Introduced `SyntheticTuner` to automatically sample out-of-domain edge cases and validate decision boundaries.

---

## v0.3.1 - Stability & Data Integrity
**Status:** Completed

This critical patch hardened the core engine against race conditions and concurrency failures for enterprise deployments.
* **Copy-On-Write GC:** Rebuilt background vector indices off-thread to achieve zero event loop stalling.
* **Rollback Snapshots:** Guaranteed 100% data integrity during memory overflows by snapshotting route metadata and embeddings before mutations.
* **Strict Thread Safety:** Bounded Redis dispatch queues and `BEGIN IMMEDIATE` SQLite transaction isolation.
* **Independent Hardware Validation:** Formally verified semantic routing reproducibility across heterogeneous consumer laptops without GPU acceleration.

---

## v0.4.0 - Dynamic Boundary Generation
**Status:** Planned

Eliminate the need for manual "utterances" by generating semantic boundaries directly from function signatures, enabling true zero-shot agent tool routing.
* **Docstring Extractor:** Parse Python function `docstrings` and `Pydantic` schemas automatically.
* **Synthetic Utterance Generation:** Utilize automated remote completion endpoints at registration to expand terse docstrings into 5-10 diverse synthetic utterances, seeding a robust semantic boundary with zero inference latency penalty.
* **LangGraph Native Tools:** Provide native `ToolNode` injection so developers can pass LangGraph agents directly into SynaptoRoute.

---

## v0.5.0 - Multi-Modal Routing
**Status:** Conceptual

Expand semantic routing beyond text to handle images and visual data, enabling complex routing in multi-modal pipelines.
* **CLIP Integration:** Allow the `AdaptiveRouter` to accept `PIL.Image`. 
* **Modality Alignment:** Migrate to unified multimodal embeddings (e.g., ImageBind) to align text and visual inputs in the same semantic space.
* **Resource Budgeting:** Explicitly define latency and memory constraints for heavy visual architectures.

---

## v0.6.0 - Advanced Network Distribution
**Status:** Conceptual

Transition from a purely embedded library into an optionally standalone microservice.
* **gRPC / REST Server:** Package SynaptoRoute as a standalone Docker container with a high-throughput gRPC API for network-level routing.
* **Federated Routing:** Allow edge nodes to sync partial routing tables from a central SynaptoRoute cluster based on local caching rules.
