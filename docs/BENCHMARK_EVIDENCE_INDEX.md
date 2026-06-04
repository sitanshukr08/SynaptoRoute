# Benchmark Evidence Traceability Index

This matrix maps every metric in the `BENCHMARK_REGISTRY.md` to its origin benchmark script and raw output artifact.

| Metric | Registry Entry | Benchmark Script | Raw JSON Manifest | Status |
|---|---|---|---|---|
| **Banking77 Accuracy** | Item 1 | `scratch/bench_banking77_reval.py` | `benchmarks/manifests/banking77_manifest.json` | `[VERIFIED]` |
| **CLINC150 Accuracy** | Item 2 | `scratch/bench_clinc_accuracy.py` | `benchmarks/manifests/clinc_accuracy_manifest.json` | `[VERIFIED]` |
| **OOD Metrics** | Item 3 | `scratch/bench_ood_metrics.py` | `benchmarks/manifests/ood_metrics_manifest.json` | `[VERIFIED]` |
| **End-to-End Latency** | Item 4 | `scratch/bench_bottleneck_analysis.py` | `benchmarks/manifests/bottleneck_manifest.json` | `[VERIFIED]` |
| **Routing Latency** | Item 5 | `scratch/bench_large_scale_retrieval.py` | `benchmarks/manifests/large_scale_retrieval_manifest.json` | `[VERIFIED]` |
| **Encoder Throughput** | Item 6 | `scratch/bench_http_throughput.py` | `benchmarks/manifests/http_throughput_manifest.json` | `[VERIFIED]` |
| **Routing Throughput** | Item 6 | `scratch/bench_large_scale_retrieval.py` | `benchmarks/manifests/large_scale_retrieval_manifest.json` | `[VERIFIED]` |
| **100K Route Capacity** | Item 7 | `scratch/bench_100k_scale.py` | `benchmarks/manifests/large_scale_retrieval_manifest.json` | `[VERIFIED]` |
| **1M Vector Capacity** | Item 7 | `scratch/bench_large_scale_retrieval.py` | `benchmarks/manifests/large_scale_retrieval_manifest.json` | `[VERIFIED]` |

## Goal: Reaching [VERIFIED]
To elevate any of these metrics from `[REPRODUCIBLE]` to `[VERIFIED]`, contributors must run the associated benchmark script and commit the raw output `manifest.json` into the `benchmarks/manifests/` directory.

Future benchmarks must natively generate these manifests according to the schema defined in `CONTRIBUTING.md`.
