# Competitor Comparison Matrix

This document provides a realistic, evidence-backed assessment of how SynaptoRoute compares to alternative routing frameworks in the industry.

To prevent accidental overclaiming, every comparison is explicitly tagged with the constraints under which it was evaluated, and classified as `[MEASURED]`, `[ESTIMATED]`, or `[UNKNOWN]`.

---

## 1. Semantic Router (Aurelio AI)

Semantic Router is the closest direct competitor to SynaptoRoute, functioning as a high-speed semantic classification layer.

### Comparison Constraints
* **Same Encoder?** Yes (`FastEmbed` / `bge-small-en-v1.5`).
* **Same Dataset?** Yes (Banking77, CLINC150).
* **Direct Measurement?** Yes, both libraries were instantiated and executed inside `scratch/benchmarks_semantic_router.md`.
* **Hardware:** Identical single-node CPU instance.

### Performance

* **Accuracy:** `[MEASURED]` SynaptoRoute (~91.16%) is structurally tied with Semantic Router (~91.50%) on Banking77.
* **Latency:** `[MEASURED]` SynaptoRoute's FAISS IndexFlatIP matches Semantic Router's vector math execution. Both operate in the low-millisecond range (1-3ms) for route resolution.

### Architecture

* **Dynamism:** `[MEASURED]` SynaptoRoute possesses a massive advantage in state mutation. SynaptoRoute can add, update, and delete routes dynamically at runtime without dropping traffic. Semantic Router typically requires a full re-initialization of its route layer.
* **Persistence:** `[MEASURED]` SynaptoRoute is backed by instantaneous SQLite WAL persistence. Semantic Router requires explicit serialization pipelines.

---

## 2. LLM-Based Routing (e.g. OpenAI function calling, LangChain LLMRouterChain)

LLM routing delegates the decision boundary to a generative model.

### Comparison Constraints
* **Direct Measurement?** No. 
* **Literature Reference?** Standard industry latency/cost models for GPT-4 / Claude-3.

### Performance

* **Accuracy:** `[ESTIMATED]` LLMs possess vastly superior contextual understanding, easily outperforming bi-encoders in highly complex, multi-turn, or deeply ambiguous routing decisions.
* **Latency:** `[ESTIMATED]` SynaptoRoute (~3ms) is several orders of magnitude faster than an LLM API call (500ms - 2000ms).
* **Cost:** `[ESTIMATED]` SynaptoRoute is free to execute locally. LLMs incur per-token API costs.

### Architecture

* **Scale:** `[ESTIMATED]` SynaptoRoute scales to 100,000 routes. Injecting 100,000 route descriptions into an LLM context window is mathematically and financially infeasible.

---

## 3. Vector Database Routing (e.g. Pinecone, Qdrant)

Using a standalone Vector DB to store route embeddings and querying them via REST/gRPC.

### Comparison Constraints
* **Direct Measurement?** No.

### Performance

* **Latency:** `[ESTIMATED]` SynaptoRoute uses in-memory FAISS, bypassing the network loop. An external Vector DB requires an HTTP/gRPC roundtrip per query, inherently adding 10ms - 50ms of network overhead.

### Architecture

* **Complexity:** `[UNKNOWN]` Vector DBs scale infinitely horizontally, whereas SynaptoRoute currently hits OOM limits at massive distributed scales due to Redis PubSub bottlenecks.
