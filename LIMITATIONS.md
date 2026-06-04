# Verified and Unknown Boundaries

To maintain engineering trust, SynaptoRoute explicitly separates verified system limits from theoretical or unknown boundaries. This document outlines the absolute edge of our confidence in the current v0.4.0 architecture.

---

## 1. Verified Boundaries

These limitations have been empirically observed, tested, and confirmed.

### Scale Limitations (Verified)
* **Single-Core Encoding Bottleneck:** While the FAISS router resolves queries in ~3ms, the upstream `FastEmbedEncoder` processing string inputs caps throughput at **~130 RPS (Requests Per Second)** per CPU core. Exceeding this rate will trigger the `QueueFull` fail-fast backpressure mechanism.
* **100,000 Route Capacity:** The system has been validated to hold 100,000 routes in memory safely using ~530MB of RAM.

### Distributed Limitations (Verified)
* **O(N×M) Redis Bootstrapping:** When a new node joins the cluster, `request_full_sync` triggers the existing nodes to broadcast their entire route map across the PubSub channel. For 100,000 routes and 4 nodes, this floods the Redis client buffer with 400,000 JSON payloads. This architectural limitation effectively prohibits massive-scale cluster bootstrapping via Redis.

### Benchmark Limitations (Verified)
* **Retracted Latency Metrics:** The legacy v0.3.0 claim of 0.003ms latency was mathematically false due to a unit conversion bug in the telemetry script (`seconds` reported directly as `ms`). The true, verified latency is **3.0ms**.

---

## 2. Unknown Boundaries

These scenarios have never been empirically tested. Any claims regarding performance in these areas are speculative.

* **Multi-Million Route Deployments:** It is unknown at what scale FAISS `IndexFlatIP` search times become unacceptably slow without quantization or HNSW hierarchies.
* **Multi-Region Deployments:** The impact of intercontinental network latency on the Redis Sync Manager's locking and drift heuristics is unknown.
* **GPU Cluster Deployments:** While ONNX DirectML execution was mocked for batching during Phase 4, sustained high-throughput GPU inference for the encoder remains unverified in a production web server environment.
* **Multilingual Benchmarks:** Accuracy is only verified on English datasets (Banking77, CLINC150). Non-English semantic matching remains unproven.

---

## 3. Research Gaps

SynaptoRoute is a well-engineered software tool, but it currently lacks the academic validation required for "Research Readiness." Future readers must not confuse engineering reliability with empirical AI validation.

**Missing Validations:**
* **Ablation Studies:** There is no documented proof isolating how much the threshold fitting algorithm impacts end-to-end accuracy compared to a flat threshold.
* **Statistical Significance Testing:** 91.16% accuracy on Banking77 lacks p-value bounds or variance intervals across multiple random seed initializations.
* **Cross-Encoder Baselines:** There is no benchmark establishing the maximum possible accuracy ceiling a cross-encoder could achieve on these datasets to serve as an upper-bound comparison.
* **Calibration Analysis:** It is unknown if the confidence scores emitted by the `FastEmbedEncoder` are well-calibrated (i.e., does a 0.90 similarity score genuinely reflect a 90% probability of being correct?).
* **Out-of-Distribution (OOD) Rejection:** Currently, we lack any reproducible benchmarks (AUROC, FPR@95) proving the system can reliably reject queries that do not belong to any defined route.
