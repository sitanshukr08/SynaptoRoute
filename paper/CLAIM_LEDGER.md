# Paper Claim Ledger

This ledger prevents engineering behavior, pilot observations, and
publication evidence from being conflated.

| ID | Candidate statement | Current class | Required promotion evidence |
|---|---|---|---|
| C1 | Mutations expose route/version/acknowledgement/final-state receipts. | Tested engineering fact | API tests, wheel smoke, source commit |
| C2 | A completed durable receipt survives the defined process-crash model. | Unverified pilot | Full crash matrix, second machine, archive, reviewer |
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
