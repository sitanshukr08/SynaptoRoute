# Clean Replication Results

Date: 2026-07-13

Source commit: `df94df3de6c8e9f559158d021b13ef3ad00d12e7`

Status: clean candidate evidence; unverified and not eligible for paper claims

This record captures the first full quality replication and systems smoke run
performed from a clean SynaptoRoute working tree. It is a provenance note for
reviewers, not an evidence-promotion decision. The generated artifacts remain
local and ignored by Git; they must be archived and independently reviewed
before any result can be marked `verified`.

## Environment

* Python: `3.10.11`
* Platform: `Windows-10-10.0.19045-SP0`
* CPU: `Intel64 Family 6 Model 165 Stepping 5, GenuineIntel`
* Encoder: `BAAI/bge-small-en-v1.5`
* Seeds: `13`, `29`, `42`, `71`, and `101`
* Working tree at launch: clean

## Commands

```bash
python benchmarks/run_all_benchmarks.py --benchmarks banking77_multiseed --model BAAI/bge-small-en-v1.5 --output-dir benchmark_results/clean-candidate/banking77-run
python benchmarks/run_all_benchmarks.py --benchmarks clinc150_multiseed --model BAAI/bge-small-en-v1.5 --output-dir benchmark_results/clean-candidate/clinc150-run
python benchmarks/run_all_benchmarks.py --benchmarks durability_smoke crash_recovery_smoke dynamic_workload_smoke backpressure_smoke --output-dir benchmark_results/clean-candidate/systems-smokes
```

All selected benchmark subprocesses returned exit code `0`. Each run-level
manifest records `working_tree_dirty=false` and the source commit above.

## Static Quality Replication

The Banking77 and CLINC150/OOS clean runs reproduced the earlier diagnostic
study exactly for all of these structured result sections:

* quality aggregates;
* hierarchical paired-bootstrap effects;
* matched known-coverage curves;
* paired matched-coverage effects.

The values and interpretation remain those reported in
[`MULTISEED_DIAGNOSTIC_RESULTS.md`](MULTISEED_DIAGNOSTIC_RESULTS.md). Exact
replication removes dirty-tree nondeterminism as an explanation for those
results. It does not establish external reproducibility or justify a broad
superiority claim.

## Systems Smoke Outcomes

| Experiment | Clean-run outcome |
|---|---|
| Durability | Restart state matched; an injected write failure reached the receipt and a later resynchronization succeeded. |
| Abrupt exit | `0/3` memory acknowledgements and `3/3` durable acknowledgements survived restart; every child exited with the expected code. |
| Mixed workload | No query errors, mutation errors, or visibility failures; restart state matched. |
| Async overload | At offered concurrency `32` with queue capacity `8` and one in-flight batch, `8` requests completed, `24` received the explicit overload result, and no unexpected errors occurred. |

These are short structural smokes using a deterministic non-semantic encoder.
They validate instrumentation and invariants, not production capacity or
semantic quality.

## Artifact Digests

The paths below are relative to `benchmark_results/clean-candidate/` and are
intentionally ignored by Git. SHA-256 digests allow a reviewer to detect local
artifact changes before the archive step.

| Artifact | SHA-256 |
|---|---|
| `banking77-run/benchmark_manifest.json` | `e86c85f8e1a5a69d97232ee507f0035cf24552d00f4a842ff22970d066fe2112` |
| `banking77-run/banking77_multiseed.log` | `3a5179870907339701f91510889f7c2c7cee687df80e1321ff6cc737afb8f71c` |
| `banking77-run/banking77_multiseed/multiseed_summary.json` | `09770f3c893ebff5e7c6764d4e5877b5dfcfc4b00d00f890b3df9fd49463accf` |
| `banking77-run/banking77_multiseed/statistical_analysis.json` | `d865a05469a5a94413dc2331061e34c4367120a0c54321d1923b5db0817fbc60` |
| `clinc150-run/benchmark_manifest.json` | `5c95da144c322d1c0a0f3f63ba2fe5695b5c79b012dd73952c51e0a64a7a0d46` |
| `clinc150-run/clinc150_multiseed.log` | `f4d30a754e02dd80d5f1fb9f945bf6bd6705134b0e4452a3b975c1a9b2dc8813` |
| `clinc150-run/clinc150_oos_small_multiseed/multiseed_summary.json` | `0805956478d226fdbf59ce55ac7f9f07fb582942924f26e0cd2448db46a07529` |
| `clinc150-run/clinc150_oos_small_multiseed/statistical_analysis.json` | `27bebdc1001fe4163a77f0c0cb0433aef6a08669186c6c1c454de90b9f0384c1` |
| `systems-smokes/benchmark_manifest.json` | `75bf0157542510e7c0a64e40b984a603759aecb6e00c98c9959c1a985eb98c9a` |
| `systems-smokes/durability_smoke.log` | `93db3cda0c6a11bee3f61ae9d34a5e2f2d418689142184c1ca28c7c433096245` |
| `systems-smokes/crash_recovery_smoke.log` | `fcd804c8a60da608a44f422f8674ef3288bbd2addcb94df87fa664a592677337` |
| `systems-smokes/dynamic_workload_smoke.log` | `4b0c8832641b85de602b19ac184ecf02e89c686855c33ac642ac0c127bc17708` |
| `systems-smokes/backpressure_smoke.log` | `d2a08ae3e4adeb09b0b3444e1fcc5c3ba29399c5f93aa9e173d4177b55809e1f` |

## Promotion Work Still Required

1. Archive the raw logs, summaries, predictions, calibration records, and run
   manifests in immutable storage.
2. Recompute and check every digest after archival.
3. Have a contributor who did not implement the benchmark reproduce the runs
   from a fresh environment.
4. Review dataset licenses, exclusions, split integrity, and statistical code.
5. Create claim-specific manifests and mark them `verified` only after those
   checks pass.
