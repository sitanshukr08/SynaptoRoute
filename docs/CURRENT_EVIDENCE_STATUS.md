# Current Evidence Status

SynaptoRoute has a coherent engineering direction, but its historical benchmark claims are not currently research-grade evidence.

## Engineering Facts

* The package implements a local embedding-based router with SQLite persistence and an in-memory vector index.
* Redis sync exists as an optional Pub/Sub mechanism and should be treated as experimental until bootstrap/replay behavior is validated.
* Dynamic mutation is supported, but large indexes still require careful tombstone cleanup and rebuild testing.
* The benchmark runner now records commands, environment metadata, and raw log paths in a schema-validated manifest.
* The benchmark methodology now omits unsupported Top-K baseline metrics and
  executes each timed query once under the same bounded-concurrency harness.
* A deterministic NumPy index smoke benchmark validates structural behavior
  and the evidence pipeline. Its timing is diagnostic, not paper evidence.
* Pinned Banking77 and CLINC150/OOS loaders enforce disjoint route,
  calibration, and test text while recording source-dataset exclusions.
* Development pilots emit hashed calibration artifacts and privacy-conscious
  per-example predictions. Both pilots completed locally, but their dirty-tree
  manifests keep them ineligible for paper claims.
* Full-test five-seed diagnostic studies and paired bootstrap analyses now
  exist for Banking77 and CLINC150/OOS. Their interpretation is recorded in
  `MULTISEED_DIAGNOSTIC_RESULTS.md`; the runs remain unverified.

## Research Claims

* Banking77, CLINC150, OOD, latency, throughput, GPU, and scale numbers are historical audit records only.
* The old `0.003ms` 1M-vector latency claim is retracted. The historical value appears to be about 3.1ms after unit correction, but it still needs a clean rerun.
* OOD metrics must be rerun before publication because existing docs previously contradicted whether OOD validation existed.
* The CLINC development pilot indicates that OOD rejection is currently the
  main quality gap. That observation guides engineering only; the value must
  be established by clean multi-seed experiments with confidence intervals.
* The diagnostic multi-seed result rejects a general static-accuracy claim:
  per-route calibration helps CLINC open-set behavior but harms Banking77, and
  logistic regression remains stronger on overall CLINC quality.

## Publication Gate

A result can be used in a paper or release only after it has:

* a committed, non-empty benchmark script;
* an exact command;
* raw output logs;
* a schema-valid manifest with status `verified`;
* dataset/split details, seed, environment, encoder, route count, query count, and explicit timing units.
* a clean working tree and a SHA-256 digest matching the archived raw log.
