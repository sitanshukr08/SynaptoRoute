# Paper Claim Ledger

This ledger prevents engineering behavior, pilot observations, and
publication evidence from being conflated.

| ID | Candidate statement | Current class | Required promotion evidence |
|---|---|---|---|
| C1 | Mutations expose route/version/acknowledgement/final-state receipts. | Tested engineering fact | API tests, wheel smoke, source commit |
| C2 | A completed durable receipt survives the defined process-crash model. | Unverified pilot | Controlled-hardware run, second machine, archive, reviewer |
| C3 | Memory acknowledgement is not described as durable. | Contract definition | Contract review and API documentation |
| C4 | Runtime and SQLite route state remain equal across the dynamic workload and restart. | Unverified pilot | All dynamic cells, zero violations, reproduction |
| C5 | Bounded queues expose overload and preserve correctness for completed requests. | Unverified pilot | Sustained-load matrix, denominator audit, reproduction |
| C6 | Durability mode trades commit latency for a stronger process-crash guarantee. | Planned systems claim | FULL/NORMAL crash cells, confidence intervals, archive |
| C7 | Per-route policy calibration changes selective risk at matched coverage. | Planned quality claim | Five-seed datasets, paired analysis, archive |
| C8 | The correctness-probability layer has measurable calibration behavior. | Planned quality claim | Held-out ECE/Brier/reliability outputs and reproduction |
| C9 | Exact and HNSW indexes have different scale/latency/memory regimes. | Planned systems claim | Frozen scale matrix and hardware metadata |

No numerical statement enters the abstract, README, release notes, or results
section until its claim-specific manifest is verified. Negative or null results
remain reportable and are not removed from the ledger.

## Pilot Records

The non-Docker GitHub-hosted crash-recovery pilot for
`paper-artifact-v0.5.0-rc1` completed on 2026-08-08 in Actions run
[`31263822435`](https://github.com/sitanshukr08/SynaptoRoute/actions/runs/31263822435).
All 16 cells completed and the extracted bundle passed matrix,
state, raw-output, command-log, and environment hash checks. Across 3,200 child
processes, every acknowledged memory-mode mutation was absent after restart and
every completed durable mutation survived for add-route, add-utterance,
threshold-update, and delete-route operations under SQLite `FULL` and `NORMAL`
with 10ms and 100ms injected commit delays.

This is an unverified pilot observation, not a paper result. GitHub-hosted
hardware is uncontrolled, the Actions artifact expires, and no independent
reproduction or reviewer attestation exists.

The clean local quality pilot at commit
`c275e3fc4dc1b6172c0e58a42cb54fbd5c19fd12` completed bounded Banking77 and
CLINC150/OOS runs on 2026-08-09. Independent artifact inspection recomputed
classification, selective, OOD, and calibration metrics across 27,285 records
without finding a discrepancy. The run reproduces the existing diagnostic
direction: per-route calibration helps CLINC open-set rejection but harms
Banking77, while logistic regression remains stronger on overall CLINC
quality. This is still an unverified single-seed, single-machine observation;
the exact provenance and limitations are in
`docs/QUALITY_PILOT_VALIDATION.md`.
