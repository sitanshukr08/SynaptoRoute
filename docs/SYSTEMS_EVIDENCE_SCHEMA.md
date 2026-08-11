# Systems Evidence Schema

The dynamic, scale, crash-recovery, and backpressure harnesses emit version 2
summaries. This document defines the fields that support independent arithmetic
and artifact checks. It does not define thresholds for accepting a research
hypothesis.

## Shared Contract

Every summary contains:

* `schema_version=2`;
* a family-specific `benchmark` identifier;
* `status=unverified` and `paper_evidence_eligible=false`;
* the workload or configuration and execution environment;
* raw outcome counts alongside derived metrics.

`paper/verify_matrix_run.py` binds each summary to its hashed command log and
recomputes its count, rate, and throughput relationships. It produces two
different classes of result:

* **integrity error**: missing evidence, a changed hash, incomplete records, or
  arithmetic that cannot be reproduced. Verification stops.
* **outcome observation**: a valid but unfavorable result, such as an incorrect
  route or a durability-contract violation. Verification succeeds and retains
  the observation for analysis.

## Dynamic Workload

Query accounting follows:

```text
query_attempts = completed_queries + query_errors
completed_queries = query_correct + query_incorrect
```

Mutation accounting follows:

```text
mutation_attempts = mutation_successes + mutation_errors
operation_failures = query_errors + mutation_errors
```

`correctness_violations` counts incorrect query results, visibility failures,
deletion visibility failures, and failed state comparisons. Operational
failures are reported separately to avoid conflating availability with a wrong
answer. Throughput uses `measurement_wall_seconds`, which excludes the durable
barrier. The final SQLite database path, size, and SHA-256 are recorded.

## Scale Matrix

The structural identity workload records `query_count`, `correct_count`, and
`incorrect_count`. Accuracy and throughput are derived from those counts and
`query_seconds`. Identity accuracy is an index-structure check; it is not
semantic-routing quality evidence.

`paper/analyze_scale.py` aggregates counts and repetition-level performance by
engine and route count, then pairs NumPy and FAISS cells by generated-vector
seed. Accuracy effects are FAISS minus NumPy; build-time, throughput, and P95
effects are FAISS divided by NumPy. RSS is reported separately because process
allocator behavior can make deltas noisy. Approximate-retrieval misses remain
in both the pooled denominator and verifier outcome observations.

New scale cells also record construction-call policy, metric, implementation,
FAISS version, OpenMP thread count, HNSW `M`, `efConstruction`, `efSearch`, and
the search-candidate floor. The verifier validates these fields when present;
older unverified pilots remain readable but must not be promoted without them.

## Crash Recovery

Each trial records acknowledgement, child exit, restart survival, and the path,
size, and SHA-256 of its SQLite database. A present acknowledgement marker is
also path- and hash-bound. Per-mode counts and rates are recomputed from the
trial ledger.

The declared contract expects memory acknowledgement not to survive the
injected pre-commit exit and durable acknowledgement to survive. A contract
violation is a research outcome, not evidence corruption.

## Sustained Backpressure

Calibration retains success, overload, and error counts. Every offered-load
scenario follows:

```text
offered_count = successful_count + overloaded_count + error_count
successful_count = successful_correct_count + successful_incorrect_count
```

Success, shedding, and error rates retain all offered requests in the
denominator. `offering_wall_seconds` covers request creation,
`drain_seconds` covers completion after offering stops, and
`scenario_wall_seconds` covers both. Offered, successful, and resolved
throughputs state which count and timing window they use.

`paper/analyze_backpressure.py` accepts only a complete verifier-accepted
backpressure family. It treats each separately calibrated repetition as the
experimental unit for bootstrap intervals, while separately pooling raw counts
for success, shedding, error, and correctness rates. Reported latency intervals
summarize per-repetition percentiles; they are not percentiles over unavailable
pooled request-level samples. The analysis remains unverified until its source
runs pass the independent reproduction and evidence-promotion gates.

## Compatibility

Version 1 systems summaries are development artifacts and are not accepted by
the version 2 matrix verifier. Regenerate affected cells from a clean candidate
commit; do not edit old summaries in place.
