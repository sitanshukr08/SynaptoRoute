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

## Scale Pilot

### Candidate And Run

| Field | Value |
|---|---|
| Source commit | `039971adbbea48ab00b3f18e603f9a1f55fee243` |
| Run ID | `85dc2982-b9f7-42c3-82a9-dd655fd55f42` |
| Date completed | 2026-08-11 |
| Host | Uncontrolled Windows 10 development workstation |
| Python | 3.10.11 |
| Family | Frozen `scale` matrix |
| Cells | 40 of 40 completed |
| Queries | 400,000: 2 engines x 4 sizes x 5 repetitions x 10,000 |
| Recorded completed-cell time | 2,622.89 seconds |
| Resume count | 0 |

The structural workload used normalized 64-dimensional synthetic vectors and
queried vectors already present in each index. It crossed NumPy exact and
FAISS HNSW indexes at 1,000, 10,000, 50,000, and 100,000 vectors with matched
seeds 42 through 46. Identity accuracy is an index-retrieval invariant, not
semantic-routing accuracy.

### Aggregate Outcomes

| Outcome | Count |
|---|---:|
| Scale cells | 40 / 40 |
| Query attempts | 400,000 |
| Correct identity lookups | 399,495 |
| Incorrect identity lookups | 505 |
| NumPy exact misses | 0 / 200,000 |
| FAISS HNSW misses | 505 / 200,000 |
| Command failures | 0 |

The repetition-level analysis used a deterministic 10,000-resample percentile
bootstrap over five cells per engine and route count.

| Engine | Routes | Accuracy | Misses | Build, mean [95% CI] | QPS, mean [95% CI] | P95, mean [95% CI] | RSS delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| NumPy | 1,000 | 100.000% | 0 | 0.01 [0.01, 0.01] s | 9,612.55 [8,680.66, 10,112.90] | 0.133 [0.122, 0.156] ms | 0.22 MB |
| NumPy | 10,000 | 100.000% | 0 | 0.13 [0.13, 0.13] s | 7,079.37 [7,048.35, 7,112.08] | 0.172 [0.171, 0.174] ms | 4.88 MB |
| NumPy | 50,000 | 100.000% | 0 | 0.68 [0.68, 0.69] s | 1,676.47 [1,636.59, 1,707.22] | 0.745 [0.716, 0.783] ms | 26.62 MB |
| NumPy | 100,000 | 100.000% | 0 | 1.37 [1.35, 1.38] s | 529.34 [511.63, 540.85] | 2.206 [2.060, 2.457] ms | 52.32 MB |
| FAISS | 1,000 | 100.000% | 0 | 0.08 [0.08, 0.08] s | 6,642.44 [6,172.06, 6,893.17] | 0.180 [0.167, 0.205] ms | 1.25 MB |
| FAISS | 10,000 | 100.000% | 0 | 3.99 [3.97, 4.01] s | 3,975.90 [3,907.98, 4,015.56] | 0.288 [0.283, 0.297] ms | 10.02 MB |
| FAISS | 50,000 | 99.806% | 97 | 96.44 [94.80, 97.78] s | 3,269.45 [3,246.76, 3,284.76] | 0.367 [0.363, 0.370] ms | 45.59 MB |
| FAISS | 100,000 | 99.184% | 408 | 372.21 [355.54, 384.28] s | 3,101.83 [2,892.59, 3,226.51] | 0.403 [0.375, 0.454] ms | 88.70 MB |

Matched-seed comparisons show a crossover rather than one universally better
index. At 1,000 and 10,000 vectors, FAISS throughput was 0.70x and 0.56x
NumPy throughput and its P95 latency was 1.38x and 1.67x NumPy P95. At 50,000
and 100,000 vectors, FAISS throughput rose to 1.95x and 5.87x NumPy throughput,
while its P95 fell to 0.49x and 0.19x NumPy P95. Those large-size gains came
with mean identity-accuracy changes of -0.194 and -0.816 percentage points,
build-time ratios of 140.97x and 272.26x, and larger observed RSS deltas.

### Verification And Analysis

The standalone verifier returned `valid_unverified_matrix_run`, checked all 40
command logs and count/timing invariants, and retained ten
`identity_retrieval_misses` observations: one for every 50,000- and
100,000-vector FAISS repetition. These misses are experimental outcomes, not
evidence-integrity failures.

`paper/analyze_scale.py` binds the analysis to the run ID, candidate SHA,
manifest hash, run-state hash, and all 40 source-summary hashes. Pooled
accuracy retains every query. Performance intervals resample repetition-level
metrics, and exact-versus-HNSW effects pair the same route count and generated
vector seed.

This pilot does not promote claim C9. It lacks controlled Linux hardware,
captured environment evidence, independent execution, immutable archival, and
reviewer attestation. The synthetic identity workload does not measure
semantic quality. The sealed summaries also do not capture FAISS version,
HNSW construction/search parameters, thread count, or the benchmark's
one-vector-at-a-time construction policy as first-class fields. Those fields,
a bulk-construction comparison, and an HNSW parameter sweep are required before
the controlled scale experiment.
