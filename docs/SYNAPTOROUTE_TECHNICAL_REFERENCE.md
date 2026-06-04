# SynaptoRoute Technical Reference

> **Authoritative Documentation – SynaptoRoute v0.4.0**

## 1. Executive Summary
SynaptoRoute is a high-performance, single-node semantic intent router designed to classify conversational inputs into predefined functions or topics using local embeddings. 
* **What it solves:** It provides ultra-low latency semantic routing (~10ms) without API calls, relying on lightweight ONNX embeddings and FAISS for O(1) similarity search. It allows dynamic thresholds per route.
* **What it does NOT solve:** It is not an absolute security boundary against prompt injection or adversarial jailbreaks (FPR is high at usable thresholds). It is currently not designed for massively scalable distributed writes without a dedicated synchronization layer.

---

## 2. Project Evolution

* **v0.1:** Initial rigid regex/string matching layer. Focus was on basic routing.
* **v0.2:** Introduction of embedding-based routing. Swapped rigid rules for dense vector cosine similarity.
* **v0.3:** Integration of Redis and FAISS to scale routing beyond simple lists. Introduced dynamic route-specific thresholds.
* **v0.3.1:** Hotfixes for memory leaks discovered during prolonged router uptime.
* **v0.4.0 (Current):** Major asynchronous rewrite. Replaced the heavy Redis dependency with a robust SQLite storage layer. Stripped bloated metadata from FAISS to fix severe memory leaks and OOM crashes. Centralized `aquery` logic.

---

## 3. Architecture

* **Router (`AdaptiveRouter`):** The orchestrator. Coordinates the Encoder, Storage, and Index. Exposes the asynchronous `aquery` API for non-blocking routing.
* **Encoder (`FastEmbedEncoder`):** Wraps the FastEmbed library. Converts text strings into 384-dimensional dense vectors locally.
* **FAISS (`FaissIndex`):** An in-memory vector index (IndexFlatIP) used exclusively for sub-millisecond similarity search.
* **SQLite (`SQLiteStorage`):** The durable source of truth. Stores route definitions, thresholds, and all mapped utterances. 
* **Redis:** Deprecated as the primary storage layer in v0.4.0; retained strictly as a Pub/Sub mechanism for eventual distributed consistency.
* **Dynamic Updates:** Adding a route writes to SQLite first (durability), then reconstructs or updates the FAISS index (performance). 
* **Synchronization:** Utilizes Python `asyncio.Lock` and threading locks to prevent index corruption during dynamic rebuilds.

---

## 4. Architectural Decisions

* **Why FAISS?** CPU-based exact inner-product search (IndexFlatIP) provides 3.0ms P95 latency even at 1,000,000 routes.
* **Why SQLite?** Eliminates the need for an external Redis/Postgres server cluster for single-node deployments, reducing operational overhead to zero.
* **Why ONNX / BGE-Small?** Provides a massive speedup by running quantized embeddings locally without network latency or OpenAI API costs.
* **Why Dynamic Thresholds?** Semantic similarity is not uniform. Some intents (e.g., "Transfer Money") require strict 0.85+ similarity, while casual intents ("Hello") can tolerate 0.60.

---

## 5. Benchmark Methodology

Benchmarks are executed via isolated Python scripts in the `scratch/` directory.
* **Hardware Assumptions:** Tested primarily on standard x86 CPUs. GPU benchmarks utilize DirectML for vendor-agnostic hardware acceleration.
* **Timing:** Exclusively uses `time.perf_counter()` for microsecond-precision latency tracking.
* **Data:** Benchmarks use a mix of random normalized vectors for pure stress testing, and standard datasets (`Banking77`, `CLINC150`) for semantic accuracy.
* **Output:** Manifests are generated in `.json` format tracking metrics like P50, P95, AUROC, and memory footprint.

---

## 6. Verified Benchmark Results

* **Banking77 Accuracy:** 91.16%
* **CLINC150 Accuracy:** 92.0% (Top-1)
* **OOD Handling:** AUROC 0.908 | AUPRC 0.898 | FPR@95 = 36.5%
* **Retrieval Latency (Warm):** FAISS search (~0.09ms) + SQLite fetch (~0.08ms) + Encoder (~7.60ms) = **~7.80ms total inference time.**
* **Retrieval Scale:** 1,000,000 vectors load into ~2GB of RAM. FAISS search time at 1M routes is **3.0ms**.

---

## 7. Audit History

* **Memory Leaks & OOM Crashes:** 
  * *Problem:* Passing full string metadata into FAISS and `dict` indices bloated memory exponentially.
  * *Fix:* FAISS was stripped down to handle purely numeric indices mapping back to SQLite integer Primary Keys.
* **Tombstones & Index Deletion:**
  * *Problem:* FAISS `IndexFlatIP` does not support `remove_ids()`. Deleting a route caused memory to remain allocated.
  * *Fix:* System was rebuilt to fully reconstruct the FAISS index from SQLite upon deletion.
* **Unit Conversion Bug:**
  * *Problem:* Phase 3 telemetry reported `0.003ms` latency for 1M routes.
  * *Fix:* Audit discovered `time.perf_counter()` returns seconds. The true latency is `0.003 seconds` (3.0ms). The 0.003ms claim was retracted.
* **Encoder Bottleneck:**
  * *Problem:* Assumed FAISS was the bottleneck.
  * *Fix:* Bottleneck attribution proved the Encoder takes 97% of inference time (~7.6ms). FAISS takes <3%.

---

## 8. Known Limitations

* **Mutation Lock at Scale:** Rebuilding the FAISS index for 1,000,000 vectors takes ~290 seconds. SynaptoRoute cannot handle high-frequency dynamic writes at massive scale.
* **Distributed Synchronization:** While Redis Pub/Sub exists, guaranteed consistency across multiple SynaptoRoute nodes is not natively guaranteed.
* **OOD Vulnerability:** A 36.5% False Positive Rate means adversarial "junk" text will frequently match a valid route if thresholds are tuned for 95% TPR.
* **GPU Utilization:** DirectML provides a 1.4x speedup on batch encoding (N=300), but single-user queries (N=1) do not experience significant acceleration due to tensor transfer overheads.

---

## 9. Future Work

**Engineering Roadmap:**
* Externalize the embedding layer to an asynchronous queue to batch single-user queries automatically.
* Implement a background mutation thread so adding routes doesn't block the async event loop during index rebuilds.

**Research Roadmap:**
* Compare Bi-Encoder architecture latency/accuracy against an equivalent Cross-Encoder.
* Execute sequence-length ablation studies (10 tokens vs 500 tokens).

---

## 10. Research Gaps

SynaptoRoute is an incredibly robust piece of open-source engineering, but it is **not yet ready** for rigorous academic publication (e.g., a formal whitepaper or thesis). This prevents future readers from confusing "well engineered" with "research validated".

**Missing Evidence Required for Publication:**
* **Ablation Studies:** Isolating how much the threshold fitting algorithm impacts end-to-end accuracy.
* **Statistical Significance Testing:** Lack of p-value bounds or variance across multiple random seed initializations.
* **Cross-Encoder Baselines:** Missing upper-bound performance tests against cross-encoder re-ranking.
* **Calibration Analysis:** Expected Calibration Error (ECE) and reliability diagrams for the OOD rejection logic.
* **Multilingual Evaluation:** Accuracy is only verified on English datasets.
