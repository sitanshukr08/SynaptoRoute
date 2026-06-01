# Benchmarks

**Benchmark Version:** v1.0  
**SynaptoRoute Version:** v0.3.0  
**Date:** 2026-06-01  
**Embedding Model:** `BAAI/bge-small-en-v1.5`  

This document serves as the canonical record of SynaptoRoute's performance metrics, gathered through rigorous testing against both synthetic stressors and real-world datasets.

## Hardware

- **CPU:** 13th Gen Intel i5-13450HX personal laptop
- **RAM:** 16GB
- **Execution:** CPU-only (ONNX Runtime, non-GPU)

## Embedding Models

- **Primary:** `BAAI/bge-small-en-v1.5`
- **Provider:** `FastEmbed` (Local, unauthenticated)

## Datasets

### Internal
- **Tool Routing:** Synthetic JSON-style API action classification.
- **Agent Routing:** Multi-agent semantic handover intent definitions.

### Public (HuggingFace Parity)
- **CLINC150:** (`clinc/clinc_oos`) 150 diverse intents across 10 domains.
- **Banking77:** (`mteb/banking77`) 77 highly overlapping intents in the banking domain.

---

## Accuracy Results

Evaluated on the **Test** split of external canonical datasets (Routes constructed exclusively from the **Train** split).

| Dataset | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | F1 Score |
|---------|----------------|----------------|----------------|----------|
| **CLINC150** | 74.20% | 77.53% | 78.33% | 81.41% |
| **Banking77** | 91.81% | 95.16% | 96.16% | 91.81% |

## OOD Results

Out-Of-Distribution (OOD) rejection measures the router's ability to say "I don't know" when queried with completely unrelated text.

| Configuration | Dataset | True OOD Rejection | False OOD (Accidental Rejection) |
|---------------|---------|--------------------|----------------------------------|
| **Base Threshold (0.60)** | CLINC150 | 4.00% | (Included in base accuracy) |
| **Base Threshold (0.60) + Margin (0.15)**| Chaos/Synthetic | 94.3% | 59.0% (Massive accuracy drop) |

*Note: Pure cosine thresholding struggles with OOD rejection because dense vector spaces inherently assign some semantic similarity to unrelated queries. Aggressive margin gating destroys in-distribution accuracy. For strict OOD rejection, use the Cross-Encoder pipeline.*

## Adversarial Results

Evaluated using a handcrafted NLI dataset designed to trick cosine similarity algorithms.

| Trap Type | Base SynaptoRoute |
|-----------|-------------------|
| **Keyword Trap** *(e.g., "flight" mapping to "book_flight" instead of "cancel_flight")* | 58.8% |
| **Lexical Overlap** *(e.g., identical vocabulary, rearranged)* | 25.0% |
| **Negation** *(e.g., "I don't want to book")* | 0.0% |

## Optimization Profiles

The following measures sequential execution (Latency Profile) vs batch-queue concurrent handling (Throughput Profile).

| Profile | Sequential Latency (1 worker) | Batched Throughput (Concurrent Load) |
|---------|-------------------------------|--------------------------------------|
| **LATENCY** | **~5.11ms** | 361 QPS |
| **THROUGHPUT** | ~8.37ms | **438 QPS** |

## Architectural Tradeoffs (v0.1 vs v0.2)

To enable thread-safe asynchronous concurrency, the architecture was intentionally transitioned from single-threaded raw execution to a multi-threaded batch worker queue. 
This introduced a deliberate baseline regression in absolute latency to guarantee stability under load:
- **v0.1.0 Latency (Unsafe):** 5.3ms
- **v0.2.0 P50 Latency (Safe):** 31.8ms (reflects minimum batching wait windows)

## Scalability Results

Performance metrics measured using the `FaissIndex` backend. (Note: Sequential throughput metrics here reflect internal testing bounds, real-world batching achieves higher P50s).

| Route Count | Vector Count | QPS | Query Latency | Storage Payload (SQLite) |
|-------------|--------------|-----|---------------|--------------------------|
| **10** | 50 | 302 | ~3.3ms | 114 KB |
| **100** | 500 | 289 | ~3.4ms | 2.5 MB |
| **1,000** | 5,000 | 185 | ~5.4ms | 25 MB |
| **10,000** | 50,000 | 62 | ~16ms | 240 MB |

## 50k Route Results

Stress testing the absolute upper boundaries of the system architecture in an interactive environment using the `FaissIndex` (which scales dynamically, unlike `NumpyIndex` which allocates memory statically).

| Metric | Result |
|--------|--------|
| **Total Routes** | 50,000 |
| **Total Vectors**| 250,000 |
| **Index Build Time** | 682 seconds |
| **Query Latency (Avg)**| 49ms |
| **Query Latency (P99)**| 58ms |
| **RAM Footprint**| ~1.2 GB |

*Takeaway: Latency growth remained sublinear under the tested workloads. 50,000 routes with 250,000 vectors remains easily interactive at ~49ms latency.*

---

## Methodology

1. **Isolation:** Testing strictly uses separated `train` and `test` splits to prevent benchmark inflation through vector memorization.
2. **Pre-Warming:** A "burn-in" query is fired before timing loops to ensure model initialization artifacts (e.g. disk I/O, JIT) do not corrupt inference latency measurements.
3. **Hardware Parity:** All metrics reflect local CPU inference (no network APIs).

## Reproducing Results

You can independently verify these metrics by running the benchmark harness available in the repository:

```bash
python benchmarks/bench_realworld.py
python benchmarks/bench_extreme_scale_v2.py
```
