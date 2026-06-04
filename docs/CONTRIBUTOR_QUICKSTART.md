# Contributor Quickstart

Welcome to the SynaptoRoute project! This guide will get you oriented with the v0.4.0 architecture so you can run benchmarks, reproduce results, and contribute effectively.

## 1. Architecture Overview
SynaptoRoute is an incredibly fast, local embedding-based intent router. 
When a user calls `await router.aquery("some text")`:
1. **Encoder:** The text is embedded into a 384-dimensional vector using `onnxruntime` and `FastEmbed`.
2. **Index:** `FAISS` calculates the inner product (cosine similarity) against all known routes in memory instantly.
3. **Storage:** `SQLite` is queried to fetch the exact Route object and its configured confidence threshold.

## 2. Installation
To develop and run the router locally:
```bash
# Install with development dependencies
pip install -e ".[dev]"
```

## 3. Running Benchmarks
All authoritative benchmarks are located in the `scratch/` directory. To reproduce the metrics documented in the `BENCHMARK_REGISTRY.md`:

```bash
# Test OOD Rejection (AUROC, FPR)
python scratch/bench_ood_metrics.py

# Test CPU vs GPU Acceleration
python scratch/bench_gpu_acceleration.py

# Test 1,000,000 Vector Scaling
python scratch/bench_large_scale_retrieval.py

# Test System Bottlenecks
python scratch/bench_bottleneck_analysis.py
```
*Note: Benchmarks output raw JSON manifests to the `scratch/` folder upon completion.*

## 4. How to Contribute
1. Read the `SYNAPTOROUTE_TECHNICAL_REFERENCE.md` to understand the current architectural limits.
2. Ensure you do not re-introduce bloated metadata into the `FaissIndex` class (this causes known memory leaks).
3. Ensure any new benchmarking scripts utilize `time.perf_counter()` and correctly label units as seconds vs milliseconds.
4. Submit your PR with a benchmark manifest attached if you modify the Encoder, Storage, or FAISS routing layers.
