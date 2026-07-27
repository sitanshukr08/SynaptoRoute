# SynaptoRoute Benchmark Registry

This registry is an audit ledger, not a marketing page. A metric is publishable only when its manifest is schema-valid and its raw output can be reproduced from a clean checkout.

## Verification Tiers

* **verified**: Runnable script, exact command, raw output, environment metadata, dataset/split details, and manifest all exist and align.
* **unverified**: Historical result or partial run record. Useful for investigation, not for release claims.
* **retracted**: Known-invalid historical claim.

## Current Claim Status

| Claim | Historical Result | Status | Reason |
|---|---:|---|---|
| Banking77 accuracy | 91.16% | unverified | Raw output, seed, split details, and complete environment metadata are missing. |
| CLINC150 top-1 accuracy | 92.0% | unverified | The stored manifest has useful metadata, but no runnable script or raw output is archived. |
| OOD rejection | AUROC 0.908, AUPRC 0.899, FPR@95 36.5% | unverified | The docs previously contradicted whether this was validated; rerun before publishing. |
| Bottleneck attribution | Warm total 7.80ms, encoder 7.62ms | unverified | Raw output and runnable script are missing. |
| GPU acceleration | 1.4x batch encoding improvement | unverified | Device details, command, script, and raw output are missing. |
| HTTP throughput | Worker/scenario RPS table | unverified | Server configuration, command, script, and raw output are missing. |
| 1M vector latency | Historical `0.003ms` P95 | retracted | Values were recorded from `time.perf_counter()` seconds and mislabeled as milliseconds. The historical value corresponds to about 3.1ms, but must be rerun. |

## Release Rule

No benchmark result should be described as verified in a release, README, paper draft, or comparison document unless it passes `benchmarks.manifest_schema.validate_manifest` with status `verified`.

Run-level manifests produced by `benchmarks/run_all_benchmarks.py` are intentionally marked `unverified`; they capture raw logs and environment metadata, then require human review before a metric is promoted.
