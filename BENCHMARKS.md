# SynaptoRoute Benchmarks

This file summarizes the benchmark performance of SynaptoRoute. All claims are backed by empirical telemetry found in the [BENCHMARK_REGISTRY.md](docs/BENCHMARK_REGISTRY.md).

## Metric Definitions

To ensure transparency, all benchmarks are evaluated against the following strict definitions:

### Accuracy Metrics
* **Top-1 Accuracy:** The percentage of queries where the highest-scoring FAISS route correctly matches the labeled ground truth.
* **Top-3 Accuracy:** The percentage of queries where the correct route is within the top 3 highest-scoring routes.
* **Precision:** True Positives / (True Positives + False Positives).
* **Recall:** True Positives / (True Positives + False Negatives).
* **F1:** The harmonic mean of Precision and Recall.

### OOD Metrics (Out-Of-Distribution)
* **AUROC (Area Under the Receiver Operating Characteristic):** Measures the classifier's ability to distinguish between in-distribution queries and out-of-distribution (fallback) queries across all threshold values.
* **AUPRC (Area Under the Precision-Recall Curve):** Evaluates the tradeoff between precision and recall for rare/OOD data.
* **FPR@95:** The False Positive Rate calculated when the True Positive Rate is strictly anchored at 95%.

### Performance Metrics
* **P50:** The median latency; 50% of requests complete faster than this.
* **P95 / P99:** The tail latency bounds; 95% or 99% of requests complete faster than this.
* **Throughput:** Total queries fully processed (including encoding and FAISS retrieval) per second (RPS).
* **Startup Time:** The wall-clock time required to reconstruct the FAISS index from SQLite WAL storage.
* **Memory Usage:** The peak RAM allocated to the python process during sustained loads.

---

## 1. Accuracy (Verified)

* **Date:** 2026-05-31
* **Version:** v0.4.0
* **Dataset:** Banking77 (Test Split)
* **Hardware:** Standard CPU Pipeline
* **Methodology:** Verified via `benchmarks/run_banking77.py`. Embeddings processed by `FastEmbedEncoder` using `bge-small-en-v1.5`.

**Results:**
* **Top-1 Accuracy:** ~91.16%
* **Top-3 Accuracy:** ~98.40%
* **Status:** `[VERIFIED]`

* **Date:** 2026-05-31
* **Version:** v0.4.0
* **Dataset:** CLINC150 (Test Split)

**Results:**
* **Top-1 Accuracy:** ~92.00%
* **Status:** `[VERIFIED]`

---

## 2. Large Scale Latency (Verified)

* **Date:** 2026-06-04
* **Version:** v0.4.0
* **Dataset:** 100,000 Mock Routes (Random uniform vectors anchored to hypersphere)
* **Hardware:** Local CPU / Memory limits enabled
* **Methodology:** Verified via `scratch/bench_large_scale_retrieval.py`. Encoder bypassed to test FAISS + Router lookup overhead.

**Results:**
* **P50:** 3.01 ms
* **P95:** 3.42 ms
* **P99:** 3.81 ms
* **Status:** `[VERIFIED]`

---

## Historical Retractions

SynaptoRoute values engineering honesty. The following historical claims were determined to be mathematically invalid and are preserved here for transparency.

### > [!WARNING]
### <RETRACTED> 0.003ms Ultra-Low Latency Claim

* **Original Claim:** In v0.3.0, documentation widely claimed P50 latency of 0.003ms for 100,000 routes.
* **Why it was wrong:** During Phase 4 architecture audits, a severe unit conversion bug was discovered in `bench_large_scale_retrieval.py`. The `time.perf_counter()` output was measured in seconds, but was improperly labeled as milliseconds without the necessary `* 1000` conversion.
* **Corrected Evidence:** The true latency is 3.01ms (1000x higher than claimed). This has been confirmed and corrected via the v0.4.0 Critical Regression suite.
