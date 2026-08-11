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

A second clean local pilot on 2026-08-10 exercised the revised version 2
systems summaries at commit
`78e96ec392b448f84f3b677d2d3af859153961d0`. It reproduced the same aggregate
crash-recovery outcome across 3,200 trials and passed the matrix verifier with
zero outcome observations. This is corroborating development evidence, not the
independent reproduction required to promote C2 or C6; the host was
uncontrolled and no environment bundle, immutable archive, or reviewer
attestation was produced.

A clean local backpressure pilot on 2026-08-11 completed all 15 frozen cells
at `52968b6fc72350f4e6b2e4adba030d045931b39a`. Across 60 offered-load
scenarios, every one of 3,961,344 requests was accounted for as successful or
explicitly shed, all 2,714,164 successful routes were correct, and no request
errors occurred. The verifier retained 43 shedding observations and the
repetition-level analysis exposed material P95 variance in the low-latency
profile at 1.0x calibrated load. This supports continued investigation of C5
but does not promote it: the run lacks controlled hardware, independent
reproduction, immutable archival, and reviewer attestation.
