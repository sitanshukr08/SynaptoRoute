# SynaptoRoute: Future Development Roadmap

This document outlines the strategic vision and upcoming features for `SynaptoRoute`. Our goal is to evolve from a high-speed text semantic router into an industry-standard, multi-modal, highly scalable routing framework for all agentic architectures.

---

## ✅ v0.2.0 - Production Readiness & Concurrency
**Status:** Completed

The v0.2.0 release transitioned the engine from a sequential low-latency prototype into an ACID-compliant, async batching router capable of safely absorbing asynchronous FastAPI server loads.
* **SQLite Thread-Local Pooling:** Guaranteed 100% data integrity under heavy multithreaded writes.
* **Amortized $O(1)$ Lazy Memory Slicing:** Eliminated the $O(N)$ reallocation cascade during live hot-reloading.
* **Dynamic Asynchronous Batching:** Doubled hardware throughput via an internal `_batch_worker` queue.
* **BLOB Caching (v0.2.1 Hotfix):** Slashed the 50,000-vector boot time from 20 minutes to 0.45 seconds.

---

## 🔬 v0.3.0 - Scalability, Distributed Systems & Scientific Benchmarks
**Status:** Planned

The immediate next phase focuses on addressing critical research gaps identified during the v0.2.0 rollout. To validate SynaptoRoute for enterprise adoption, we must break the single-node dense matrix barrier and introduce rigorous statistical controls.

* **Distributed State Sync (K8s Horizontal Scaling):** Resolve the Kubernetes split-brain issue by evaluating an external event bus (e.g., Redis Pub/Sub) or shared-memory architecture to broadcast cache invalidation events across multiple pods.
* **Sub-Linear Scalability (FAISS / HNSW):** Integrate approximate nearest-neighbor (ANN) indexes as an optional backend to unblock routing beyond the current 50k dense vector memory bottleneck.
* **Comparative Baselines & Statistical Rigor:** Build comprehensive testing suites against `LangChain` and `LlamaIndex` routers. Implement repeated trial averages, standard deviation analysis, and formal queue scheduling tradeoff matrices.
* **Semantic Robustness Evaluation:** Introduce multilingual benchmarking and adversarial semantic drift tests against real conversational traffic.
* **Energy & Cost Efficiency Analytics:** Define watt-per-inference and requests-per-dollar metrics to prove viability over cloud LLMs.

---

## ⚙️ v0.4.0 - Zero-Shot Tool Routing
**Status:** Conceptual

Eliminate the need for manual "utterances" by generating semantic boundaries directly from function signatures.
* **Docstring Extractor:** Parse Python function `docstrings` and `Pydantic` schemas.
* **Synthetic Utterance Generation:** Utilize a one-time LLM call at registration to expand terse docstrings into 5-10 diverse synthetic utterances, seeding a robust semantic boundary with zero inference latency penalty.

---

## ⚖️ v0.5.0 - Cross-Encoder Tie-Breakers
**Status:** Conceptual

Maximize classification accuracy in highly ambiguous routing scenarios using a classic two-stage retrieval pattern.
* **Reranking Layer:** If the top two routes score within a configurable margin (e.g., ±0.05), trigger a lightweight CrossEncoder (e.g., MS MARCO) to perform a surgical tie-break.
* **Accuracy-vs-Latency Mode:** Preserves the fast-path performance for the common case while ensuring accuracy for edge cases.

---

## 👁️ v0.6.0 - Multi-Modal Routing
**Status:** Conceptual

Expand semantic routing beyond text to handle images and visual data, enabling complex routing in Vision-Agent workflows.
* **CLIP Integration:** Allow the `AdaptiveRouter` to accept `PIL.Image`. 
* **Modality Alignment:** Migrate to unified multimodal embeddings (e.g., ImageBind) to align text and visual inputs in the same semantic space.
* **Resource Budgeting:** Explicitly define latency and memory constraints for heavy visual architectures.
