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
Benchmarks must run through the repo benchmark runner so raw logs and environment metadata are captured:

```bash
# Write a schema-valid run manifest without executing benchmark scripts
python benchmarks/run_all_benchmarks.py --benchmarks latency --dry-run

# Execute selected benchmarks and store raw logs under benchmark_results/
python benchmarks/run_all_benchmarks.py --benchmarks accuracy latency
```

Historical manifests in `benchmarks/manifests/` are audit records. They are not release-grade until their status is `verified` and they pass `benchmarks.manifest_schema.validate_manifest`.

## 4. How to Contribute
1. Read the `SYNAPTOROUTE_TECHNICAL_REFERENCE.md` to understand the current architectural limits.
2. Ensure you do not re-introduce bloated metadata into the `FaissIndex` class (this causes known memory leaks).
3. Ensure any new benchmarking scripts utilize `time.perf_counter()` and correctly label units as seconds vs milliseconds.
4. Submit your PR with a schema-valid benchmark manifest and raw logs if you modify the Encoder, Storage, or FAISS routing layers.
