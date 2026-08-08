# SynaptoRoute Next Steps

Status: artifact foundation and held-out probability-calibration outputs
implemented on `codex/paper-artifact-foundation`; controlled pilots pending.

## Current Position

Implemented:

* versioned SQLite persistence and explicit mutation durability receipts;
* bounded query and storage queues, generation-based rebuilds, and deterministic shutdown;
* schema-v2 unverified run manifests and independently reviewed evidence promotion;
* deterministic dynamic, crash-recovery, scale, and sustained-load harnesses;
* the frozen 208-command paper matrix and pinned CPU-only Python 3.11 environment;
* network-free unit tests and a separately marked real-model integration test;
* disjoint policy/probability calibration, ECE/Brier output, reliability-bin
  JSON, and generated SVG reliability diagrams;
* a repeatable five-family bounded protocol smoke with machine-readable
  invariants and a schema-v2 unverified manifest.

Not yet complete:

* the paper container has not been rebuilt on a running Linux Docker engine;
* the bounded native protocol smoke passes, but the Linux paper-container run
  remains blocked until a Docker daemon is available;
* controlled matrix runs, second-machine reproduction, archival DOI, and paper results remain pending.

## Execution Order

### 1. Freeze The Foundation (Completed)

1. Review and commit the current branch without adding experimental hybrid, session, permission, or Redis work.
2. Run `python paper/preflight.py` from the clean commit.
3. Require the full CI matrix, static checks, wheel smoke, and FastEmbed model job to pass.
4. Tag the resulting commit as an artifact-candidate identifier, not as `v0.5.0`.

### 2. Finish Quality Artifact Outputs (Implementation Completed)

1. Freeze a held-out probability-calibration method and split. Raw cosine scores
   and margins must not be relabeled or clamped as probabilities.
2. Emit ECE, Brier score, reliability-bin data, and SVG reliability diagrams
   only for systems that produce probabilities under that frozen method.
3. Add exact-string results to paired analysis and preserve matched-coverage comparisons.
4. Generate tables and figures only from archived machine-readable outputs.
5. Run small protocol pilots before freezing a new candidate. This is the
   remaining step in this section.

### 3. Validate The Systems Harness

1. Build `Dockerfile.paper` and record its image digest and resolved package inventory.
2. Execute one short smoke cell from each experiment family. The native runner
   is implemented and passes; repeat it inside the paper container.
3. Confirm zero correctness violations, complete raw logs, stable units, and
   schema-valid unverified manifests. Native smoke currently passes this gate.
4. Freeze a new candidate if any correctness code changes.

### 4. Run Confirmatory Experiments

Execute the frozen matrix on controlled hardware. Do not edit outputs manually and do not promote results from a dirty tree. Any correctness change invalidates affected runs.

### 5. Reproduce And Archive

1. Reproduce the candidate on a separate Linux machine.
2. Compare invariants and analyze performance with the frozen statistical protocol.
3. Archive source, manifests, raw logs, predictions, figures, and analysis outputs in a versioned bundle.
4. Promote claim-specific manifests only with reviewer attestation and immutable archive metadata.

### 6. Complete The Paper And Release

Write results, threats to validity, ethics, limitations, and related work around verified artifacts. Release `v0.5.0` only when source commit, package version, documentation, manifests, and archive agree.
