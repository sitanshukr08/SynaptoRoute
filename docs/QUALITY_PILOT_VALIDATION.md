# Clean Quality Pilot Validation

Date: 2026-08-09

Status: clean local pilot; independently integrity-checked; not paper evidence

## Purpose

This run validates the frozen quality pipeline after adding independent
per-example artifact inspection. It is a bounded, single-seed pilot on an
uncontrolled Windows workstation. It does not satisfy the confirmatory
five-seed, second-machine, immutable-archive, or reviewer-attestation gates.

## Provenance

Source commit:
`c275e3fc4dc1b6172c0e58a42cb54fbd5c19fd12`

Run ID: `268b338d-9680-4c7f-8ab3-f1dbb38ab19b`

Command:

```powershell
.\.venv310\Scripts\python.exe benchmarks\run_all_benchmarks.py `
  --benchmarks banking77_pilot clinc150_pilot `
  --model BAAI/bge-small-en-v1.5 `
  --output-dir benchmark_results\quality-pilot-c275e3f
```

The schema-v2 run manifest records Python 3.10.11, a clean working tree,
exit status zero, the full source commit, and SHA-256 digests for both raw
logs. Strict artifact preflight passed immediately before execution.

## Independent Checks

Both experiment directories passed `paper/verify_quality_artifacts.py`.

| Dataset | Systems | Policy records per calibrated system | Probability-fit records per system | Test records per system | Total records checked | Summary SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| Banking77 | 6 | 385 | 385 | 500 | 7,235 | `4beff0de3ae65161e82299cca82674ce738272510ba8a0159126bf7dad31f5b5` |
| CLINC150/OOS | 6 | 1,550 | 1,550 | 500 | 20,050 | `5fc8e1eb7d254cbbeadd00ab313f32df364111daeda87383531a2885759e8847` |

For every system, the verifier checked:

* declared artifact filenames and SHA-256 bindings;
* absence of raw query text in prediction records;
* policy, probability-fit, and test split disjointness;
* identical per-phase cohorts across compared systems;
* serialized probability-model application for every test prediction;
* recomputed classification, selective, OOD, ECE, MCE, Brier, and reliability-bin metrics;
* explicit `unverified` and paper-ineligible status throughout the artifact chain.

The previous local pilot passed every recomputation but failed the new policy
artifact status rule. The producer now emits that status explicitly. A direct
comparison found no change in fitted policies, fit counts, or non-latency test
metrics between the old and replacement pilots.

## Diagnostic Metrics

These values diagnose the pipeline only. Latency is omitted because the run
was sequential on uncontrolled hardware and was not counterbalanced.

### Banking77

| System | Accuracy | Macro-F1 | Known coverage | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| Exact cosine | 0.8720 | 0.8734 | 0.9660 | 0.0736 | 0.0833 |
| Exact string | 0.0000 | 0.0000 | 0.0000 | 0.0026 | 0.0000 |
| Logistic regression | 0.8620 | 0.8472 | 0.9880 | 0.1023 | 0.0984 |
| Semantic Router | 0.8680 | 0.8734 | 0.9560 | 0.0404 | 0.0711 |
| SynaptoRoute, global | 0.8680 | 0.8734 | 0.9560 | 0.0404 | 0.0711 |
| SynaptoRoute, per-route | 0.7920 | 0.8344 | 0.8540 | 0.0528 | 0.0535 |

### CLINC150/OOS

| System | Accuracy | Macro-F1 | Known coverage | OOD recall | OOD AUROC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact cosine | 0.7900 | 0.8245 | 0.9927 | 0.3846 | 0.9125 | 0.1263 | 0.1465 |
| Exact string | 0.1820 | 0.0020 | 0.0000 | 1.0000 | 0.5000 | 0.1497 | 0.1713 |
| Logistic regression | 0.8500 | 0.8795 | 0.9829 | 0.5604 | 0.9608 | 0.1273 | 0.1276 |
| Semantic Router | 0.7900 | 0.8245 | 0.9927 | 0.3846 | 0.9244 | 0.0828 | 0.1297 |
| SynaptoRoute, global | 0.7900 | 0.8245 | 0.9927 | 0.3846 | 0.9244 | 0.0828 | 0.1297 |
| SynaptoRoute, per-route | 0.8160 | 0.8318 | 0.8729 | 0.8681 | 0.9484 | 0.1722 | 0.1867 |

Exact-string OOD recall is not useful in isolation: it rejected every known
query and had zero known coverage. The pilot again indicates that per-route
calibration improves CLINC open-set rejection while reducing known coverage,
and that logistic regression remains the strongest overall CLINC comparator.
Banking77 does not support a per-route quality-improvement claim.

## Remaining Gate

The ignored local output directory and raw logs must not be cited as results.
The next quality evidence requires the full five-seed frozen matrix on a
controlled Linux machine, independent reproduction on another machine, a
content-inventoried immutable archive, and reviewer attestation. Because the
verifier contract changed after `paper-artifact-v0.5.0-rc1`, a successor
artifact candidate must be frozen after the paper container passes on Linux.
