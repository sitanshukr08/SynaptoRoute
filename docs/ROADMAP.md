# SynaptoRoute: Future Development Roadmap

This document outlines the strategic vision and upcoming features for `SynaptoRoute`. Our goal is to evolve from a high-speed text semantic router into a multi-modal, highly scalable routing framework for distributed microservice architectures.

---

## v0.2.0 - Production Readiness & Concurrency
**Status:** Completed

The v0.2.0 release transitioned the engine from a sequential prototype into an ACID-compliant, async batching router capable of safely absorbing asynchronous FastAPI server loads.
* **SQLite Thread-Local Pooling:** Guaranteed 100% data integrity under heavy multithreaded writes.
* **Amortized $O(1)$ Lazy Memory Slicing:** Eliminated the $O(N)$ reallocation cascade during live hot-reloading.
* **Dynamic Asynchronous Batching:** Doubled hardware throughput via an internal `_batch_worker` queue.
* **BLOB Caching (v0.2.1 Hotfix):** Slashed the 50,000-vector boot time from 20 minutes to 0.45 seconds.

---

## v0.3.0 - Scalability & Distributed Systems
**Status:** Completed

The v0.3.0 release addresses the critical architectural gaps required for multi-node deployments and robust index validation.
* **Distributed State Sync (K8s Horizontal Scaling):** Resolved the multi-node replica drift issue by implementing the `RedisSyncManager` to broadcast cache invalidation and sync events across instances.
* **Pluggable Encoder Backends:** Abstracted the hard dependency on `FastEmbed` via the `Encoder` interface. Added native support for `OpenAIEncoder`.
* **Safe Optimization Profiles:** Abstracted runtime engine tuning dials (batch size, timeouts, core utilization) into explicit Profiles (e.g., `LATENCY` vs `THROUGHPUT`).
* **Synthetic Threshold Tuning:** Introduced `SyntheticTuner` to automatically sample out-of-domain edge cases and validate decision boundaries.

---

## v0.4.0 - Scientific Benchmarks & Observability
**Status:** Planned

The immediate next phase focuses on formal statistical rigor and production observability.
* **Observability & Metrics Endpoint:** Introduce a `/metrics` integration for Prometheus and OpenTelemetry to formally track P50/P99 latency, queue depth, and throughput in production.
* **Comparative Baselines & Statistical Rigor:** Build comprehensive testing suites utilizing `pytest-benchmark` against competing ecosystem routers. Implement repeated trial averages, standard deviation analysis, and formal queue scheduling tradeoff matrices.
* **Semantic Robustness Evaluation:** Introduce multilingual benchmarking and adversarial semantic drift tests against real conversational traffic.
* **Energy & Cost Efficiency Analytics:** Define watt-per-inference and requests-per-dollar metrics to formally calculate the cost offset over remote classification endpoints.

---

## v0.5.0 - Dynamic Boundary Generation
**Status:** Conceptual

Eliminate the need for manual "utterances" by generating semantic boundaries directly from function signatures.
* **Docstring Extractor:** Parse Python function `docstrings` and `Pydantic` schemas.
* **Synthetic Utterance Generation:** Utilize automated remote completion endpoints at registration to expand terse docstrings into 5-10 diverse synthetic utterances, seeding a robust semantic boundary with zero inference latency penalty.

---

## v0.6.0 - Cross-Encoder Tie-Breakers
**Status:** Conceptual

Maximize classification accuracy in highly ambiguous routing scenarios using a two-stage retrieval pattern.
* **Reranking Layer:** If the top two routes score within a configurable margin (e.g., ±0.05), trigger a lightweight CrossEncoder (e.g., MS MARCO) to perform a surgical tie-break evaluation.
* **Accuracy-vs-Latency Mode:** Preserves the fast-path performance for the common case while ensuring accuracy for edge cases.

---

## v0.7.0 - Multi-Modal Routing
**Status:** Conceptual

Expand semantic routing beyond text to handle images and visual data, enabling complex routing in multi-modal pipelines.
* **CLIP Integration:** Allow the `AdaptiveRouter` to accept `PIL.Image`. 
* **Modality Alignment:** Migrate to unified multimodal embeddings (e.g., ImageBind) to align text and visual inputs in the same semantic space.
* **Resource Budgeting:** Explicitly define latency and memory constraints for heavy visual architectures.
