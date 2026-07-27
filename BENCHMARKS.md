# SynaptoRoute Benchmarks

This file summarizes benchmark policy and current evidence status. Historical numbers are audit targets, not verified release claims.

## Metric Definitions

* **Top-1 Accuracy:** percentage of queries where the highest-scoring route matches the labeled target.
* **Top-3 Accuracy:** percentage of queries where the target route is in the top three.
* **AUROC / AUPRC / FPR@95:** OOD rejection metrics that must be computed from a documented in-distribution/OOD split.
* **P50 / P95 / P99:** latency percentiles with explicit timing units.
* **Throughput:** completed requests per second for a documented workload.

## Current Status

| Area | Historical Result | Current Status |
|---|---:|---|
| Banking77 accuracy | 91.16% | unverified |
| CLINC150 top-1 accuracy | 92.0% | unverified |
| OOD rejection | AUROC 0.908, FPR@95 36.5% | unverified |
| Warm bottleneck attribution | total 7.80ms | unverified |
| 1M vector latency | historical `0.003ms` | retracted |

The `0.003ms` latency claim was caused by treating `time.perf_counter()` seconds as milliseconds. The corrected interpretation is roughly `3ms`, but the result still needs a clean rerun before it can be cited.

## Reproducing Benchmarks

Use the benchmark runner so commands, environment metadata, and raw log paths are captured:

```bash
python benchmarks/run_all_benchmarks.py --benchmarks local_smoke
```

The local smoke benchmark uses deterministic synthetic vectors to validate
index correctness and the evidence pipeline. It is not eligible for semantic
accuracy claims. Install `.[benchmark]` before running external datasets or
the Semantic Router comparison:

```bash
pip install -e ".[benchmark]"
python benchmarks/run_all_benchmarks.py --benchmarks accuracy latency
```

Pinned external development pilots use validation-only policy fitting and
write per-example prediction records without raw query text:

```bash
python benchmarks/run_all_benchmarks.py --benchmarks banking77_pilot --output-dir benchmark_results/banking77-pilot
python benchmarks/run_all_benchmarks.py --benchmarks clinc150_pilot --output-dir benchmark_results/clinc150-pilot
```

These pilot commands evaluate 500 stratified test examples. They are intended
to find protocol and integration defects before multi-seed runs; their output
is always marked `unverified` and `paper_evidence_eligible=false`.

The fixed five-seed quality studies use the full official test split and the
seeds declared in the research protocol:

```bash
python benchmarks/run_all_benchmarks.py --benchmarks banking77_multiseed --output-dir benchmark_results/banking77-multiseed
python benchmarks/run_all_benchmarks.py --benchmarks clinc150_multiseed --output-dir benchmark_results/clinc150-multiseed
```

These are long-running experiments. The study summary aggregates quality
metrics only; latency requires the separate counterbalanced systems protocol.

Structural durability and mixed-load smoke runs are available through the same
evidence runner:

```bash
python benchmarks/run_all_benchmarks.py --benchmarks durability_smoke crash_recovery_smoke dynamic_workload_smoke backpressure_smoke
```

They use a deterministic non-semantic encoder and always emit unverified
diagnostic results. The crash smoke terminates child processes with
`os._exit`; it never terminates the benchmark coordinator.

The generated run manifest remains `unverified` until the raw logs are reviewed and promoted into a claim manifest that passes `benchmarks.manifest_schema.validate_manifest` with status `verified`.

Experimental definitions and statistical requirements are fixed in
[`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md).
The latest explicitly unverified diagnostic results and their interpretation
are in [`docs/DEVELOPMENT_PILOT_RESULTS.md`](docs/DEVELOPMENT_PILOT_RESULTS.md).
The clean-commit replication commands, invariant outcomes, and artifact
digests are in
[`docs/CLEAN_REPLICATION_RESULTS.md`](docs/CLEAN_REPLICATION_RESULTS.md).
