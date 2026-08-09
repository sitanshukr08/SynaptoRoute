# Systems Pilot Results

Status: unverified corroborating development evidence. These values are not
eligible for a paper table or release claim.

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
