# Benchmark Evidence Traceability Index

Every row below points to the current audit manifest for a historical claim. These entries are not release-grade until the status is `verified`.

| Claim | Manifest | Script | Raw Output | Status |
|---|---|---|---|---|
| Banking77 accuracy | `benchmarks/manifests/banking77_manifest.json` | `benchmarks/bench_realworld.py` | missing | unverified |
| CLINC150 accuracy | `benchmarks/manifests/clinc_accuracy_manifest.json` | missing | missing | unverified |
| OOD rejection | `benchmarks/manifests/ood_metrics_manifest.json` | missing | missing | unverified |
| Bottleneck attribution | `benchmarks/manifests/bottleneck_manifest.json` | missing | missing | unverified |
| GPU acceleration | `benchmarks/manifests/gpu_acceleration_manifest.json` | missing | missing | unverified |
| HTTP throughput | `benchmarks/manifests/http_throughput_manifest.json` | missing | missing | unverified |
| 1M vector latency | `benchmarks/manifests/large_scale_retrieval_manifest.json` | missing | missing | retracted |

## Promotion Criteria

To promote any row to `verified`, rerun the benchmark through a committed script, archive the raw output, capture the exact command/environment/dataset metadata, and update the manifest so the schema validator passes.
