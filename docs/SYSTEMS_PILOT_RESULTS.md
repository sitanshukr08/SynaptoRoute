# Systems Pilot Results

Status: unverified corroborating development evidence. These values are not
eligible for a paper table or release claim.

## Crash-Recovery Pilot

## Candidate And Run

| Field | Value |
|---|---|
| Source commit | `78e96ec392b448f84f3b677d2d3af859153961d0` |
| Run ID | `c32772bc-6ad7-455f-badb-0d32ef44c8a5` |
| Date completed | 2026-08-10 |
| Host | Uncontrolled Windows 10 development workstation |
| Python | 3.10.11 |
| Family | Frozen `crash_recovery` matrix |
| Cells | 16 of 16 completed |
| Trials | 3,200 |
| Recorded completed-cell time | 4,247.55 seconds |
| Resume count | 1 |

The desktop host terminated the original parent after nine cells. The runner
resumed from the same clean commit, command-plan hash, matrix hash, and output
directory. It skipped the nine hash-matched successful cells and completed the
remaining seven. The interruption and resume are retained in `run_state.json`.

## Frozen Coverage

The run crossed:

* `add_route`, `add_utterance`, `update_threshold`, and `delete_route`;
* SQLite `FULL` and `NORMAL` synchronous modes;
* 10 ms and 100 ms injected pre-commit delays;
* memory and durable acknowledgement modes;
* 100 trials per acknowledgement mode in each cell.

## Aggregate Outcomes

| Outcome | Count |
|---|---:|
| Acknowledged trials | 3,200 / 3,200 |
| Clean child exits | 3,200 / 3,200 |
| Memory acknowledgements surviving restart | 0 / 1,600 |
| Durable acknowledgements surviving restart | 1,600 / 1,600 |
| Declared restart-contract violations | 0 |
| SQLite evidence files | 3,200 |
| Acknowledgement marker files | 3,200 |

The result matches the declared injected-delay process-crash contract. It does
not establish behavior under power loss, kernel panic, filesystem corruption,
or storage-device failure.

## Verification

The standalone matrix verifier checked:

* candidate SHA, frozen command plan, matrix hash, and dependency-lock hash;
* contiguous state and raw-result ledgers;
* all 16 command-log hashes and summary-to-log bindings;
* every trial identity, acknowledgement marker, SQLite path, file size, and
  SHA-256;
* per-mode counts and rates recomputed from all 3,200 trial records.

It returned `valid_unverified_matrix_run`, 16 verified command logs, 3,200
verified trial records, zero integrity errors, and zero outcome observations.

## Why It Is Not Publication Evidence

The run lacks a controlled Linux host, a captured environment-evidence bundle,
an immutable archive, independent execution by another contributor, and
reviewer attestation. The local output remains under ignored
`benchmark_results/` storage. It may guide the next controlled run but cannot
promote claims C2 or C6 in `paper/CLAIM_LEDGER.md`.

## Backpressure Pilot

### Candidate And Run

| Field | Value |
|---|---|
| Source commit | `52968b6fc72350f4e6b2e4adba030d045931b39a` |
| Run ID | `fe41ffb0-7d8c-48d5-86d4-778790e26a71` |
| Date completed | 2026-08-11 |
| Host | Uncontrolled Windows 10 development workstation |
| Python | 3.10.11 |
| Family | Frozen `backpressure` matrix |
| Cells | 15 of 15 completed |
| Scenarios | 60: 3 profiles x 5 repetitions x 4 offered loads |
| Recorded completed-cell time | 3,803.63 seconds |
| Resume count | 0 |

Every repetition separately calibrated saturation for its queue and batch
profile, then offered 0.5x, 1.0x, 1.5x, and 2.0x that measured rate for 60
seconds per scenario. The open-loop harness retains successful, explicitly
shed, and failed requests in the denominator.

### Aggregate Outcomes

| Outcome | Count |
|---|---:|
| Offered requests | 3,961,344 |
| Successful requests | 2,714,164 |
| Explicitly shed requests | 1,247,180 |
| Request errors | 0 |
| Correct successful routes | 2,714,164 / 2,714,164 |
| Incorrect successful routes | 0 |

The repetition-level analysis uses a deterministic 10,000-resample percentile
bootstrap. Intervals below summarize five separately calibrated repetitions;
shedding rates pool all offered requests.

| Profile | Load | Success QPS, mean [95% CI] | Shed | P95 success latency, mean [95% CI] |
|---|---:|---:|---:|---:|
| Low latency | 0.5x | 32.24 [32.17, 32.33] | 0.00% | 32.66 [32.63, 32.68] ms |
| Low latency | 1.0x | 64.29 [64.16, 64.44] | 0.13% | 102.42 [68.09, 136.59] ms |
| Low latency | 1.5x | 64.47 [64.40, 64.55] | 33.20% | 155.73 [155.37, 156.00] ms |
| Low latency | 2.0x | 64.45 [64.28, 64.58] | 49.87% | 156.37 [156.01, 156.73] ms |
| Balanced | 0.5x | 257.19 [256.30, 258.07] | 0.00% | 46.26 [46.12, 46.44] ms |
| Balanced | 1.0x | 511.46 [510.28, 512.95] | 0.34% | 80.51 [80.05, 80.86] ms |
| Balanced | 1.5x | 509.89 [508.98, 510.88] | 33.67% | 94.26 [94.02, 94.49] ms |
| Balanced | 2.0x | 508.94 [508.40, 509.48] | 50.22% | 94.39 [94.26, 94.54] ms |
| Throughput | 0.5x | 1,024.41 [1,022.74, 1,025.92] | 0.00% | 50.02 [49.92, 50.13] ms |
| Throughput | 1.0x | 1,987.53 [1,986.16, 1,988.99] | 2.46% | 93.80 [93.41, 94.02] ms |
| Throughput | 1.5x | 1,964.20 [1,957.69, 1,969.07] | 35.38% | 95.06 [94.95, 95.21] ms |
| Throughput | 2.0x | 1,943.50 [1,941.27, 1,945.85] | 51.83% | 95.34 [95.29, 95.40] ms |

The 1.0x low-latency P95 varied from 46.82 ms to 141.51 ms across
repetitions. That variance must remain visible and prevents a broad
low-latency claim from this pilot. The stable result is narrower: the harness
accounted for every request, exposed overload explicitly, preserved correctness
for completed requests, and did not record worker errors under this synthetic
delayed-encoder workload.

### Verification And Analysis

The standalone verifier returned `valid_unverified_matrix_run`, checked all 15
command logs, recomputed the family invariants, and retained 43
`requests_shed` outcome observations. Shedding is an expected experimental
outcome rather than an integrity failure.

`paper/analyze_backpressure.py` binds its output to the run ID, candidate SHA,
manifest hash, run-state hash, and all 15 source-summary hashes. It emits JSON,
CSV, and Markdown views. Pooled rates use request counts; confidence intervals
resample repetition-level metrics; per-run percentiles are not incorrectly
pooled as raw latency samples.

This pilot does not promote claim C5. It lacks controlled Linux hardware,
captured environment evidence, an immutable archive, independent execution,
and reviewer attestation. Its synthetic delayed encoder isolates queue
behavior and does not establish end-to-end performance with a production
embedding provider.
