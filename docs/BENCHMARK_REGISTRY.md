# SynaptoRoute Benchmark Registry

This document serves as the authoritative ledger for all SynaptoRoute performance, accuracy, and scaling benchmarks.

## Verification Tiers
To ensure strict evidence traceability, every metric is classified under one of the following statuses:
* **VERIFIED**: Benchmark script, raw output manifest, and documentation all exist and align.
* **REPRODUCIBLE**: Benchmark script exists and can generate the metric, but raw historical output manifests were not archived to disk.
* **PARTIALLY VERIFIED**: Metric exists but evidence is incomplete or missing.
* **RETRACTED**: Known invalid.

---

## Accuracy & Semantic Capabilities

### 1. Banking77 Accuracy Baseline
* **Dataset:** Banking77 (Test Split)
* **Result:** 91.16%
* **Status:** **[VERIFIED]**

### 2. CLINC150 Accuracy Baseline
* **Dataset:** CLINC150 (Test Split)
* **Result:** 92.0% (Top-1)
* **Status:** **[VERIFIED]**

### 3. Out-Of-Distribution (OOD) Rejection
* **Dataset:** CLINC150 Out-Of-Scope
* **Result:** AUROC 0.908 | AUPRC 0.898 | FPR@95 36.5%
* **Status:** **[VERIFIED]**

---

## Systems & Performance

### 4. Bottleneck Attribution (End-to-End Latency)
* **Hardware:** Standard CPU
* **Result:** Total 7.80ms | Encoder 7.60ms (97%) | FAISS 0.09ms | SQLite 0.08ms
* **Status:** **[VERIFIED]**

### 5. Routing Latency (1M Vector Search)
* **Dataset:** 1,000,000 random/Banking77 vectors
* **Result:** P95 Latency **3.0ms**
* **Status:** **[VERIFIED]**

### 6. Throughput Limits
* **Encoder Throughput:** ~130 RPS (Hard CPU bottleneck per core)
* **Routing Throughput (FAISS-only):** >100,000 QPS (Memory-bound)
* **HTTP Throughput (End-to-End API):** Unknown (Depends heavily on batching and network layer)
* **Status:** **[VERIFIED]**

### 7. Scale Capacity Limits
* **Route Capacity:** Verified up to 100,000 routes (~530MB Memory footprint).
* **Index Capacity:** Verified up to 1,000,000 vectors (~2GB Memory footprint, 290s Boot Load Time).
* **Status:** **[VERIFIED]**

---

## Historical Retractions

### 8. Routing Latency (Unit Bug)
* **Result Claimed:** P95 Latency 0.003ms for 1,000,000 vectors
* **Status:** **[RETRACTED]** (Audit discovered a unit conversion bug; `time.perf_counter()` returns seconds. True value is 0.003s / 3.0ms).

---
*Note: Legacy benchmarks from v0.2 and v0.3 involving Redis overhead have been superseded by the v0.4.0 SQLite architecture and are intentionally omitted from this registry.*
