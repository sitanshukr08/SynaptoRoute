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
* The former CI "verified" manifest was reclassified as unverified because it
  recorded the placeholder `ci_commit_build` instead of an immutable commit.
  CI runners now emit unverified schema-v2 evidence and cannot promote it.
* Pinned Banking77 and CLINC150/OOS loaders enforce disjoint route,
  calibration, and test text while recording source-dataset exclusions.
* Development pilots emit hashed calibration artifacts and privacy-conscious
  per-example predictions. Both pilots completed locally, but their dirty-tree
  manifests keep them ineligible for paper claims.
* A clean, bounded Banking77 and CLINC150/OOS validation run at commit
  `c275e3fc4dc1b6172c0e58a42cb54fbd5c19fd12` passed independent inspection of
  27,285 policy, probability-fit, and test records. It remains an unverified
  single-machine pilot; details and artifact hashes are recorded in
  `QUALITY_PILOT_VALIDATION.md`.
* Full-test five-seed diagnostic studies and paired bootstrap analyses now
  exist for Banking77 and CLINC150/OOS. Their interpretation is recorded in
  `MULTISEED_DIAGNOSTIC_RESULTS.md`.
* `CLEAN_REPLICATION_RESULTS.md` records a historical local rerun. It remains
  an unverified candidate record, not an independent reproduction under the
  schema-v2 protocol.
* A full GitHub-hosted crash-recovery pilot completed all 16 frozen cells and
  3,200 child-process trials for artifact candidate
  `paper-artifact-v0.5.0-rc1`. Its extracted manifest, state, raw-output, and
  command-log hashes pass the standalone matrix-run verifier. The observation
  remains unverified because the hardware is uncontrolled, the Actions bundle
  is temporary, and there is no second-machine reproduction or reviewer
  attestation.

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
* Clean replication strengthens confidence in that negative result but does
  not change its publication status.
* The clean bounded quality pilot reproduced the prior non-latency metrics
  exactly after the artifact-contract fix. This validates the measurement
  path, not the hypotheses or publication claims.

## Publication Gate

A result can be used in a paper or release only after it has:

* a committed, non-empty benchmark script;
* an exact command;
* raw output logs;
* a schema-valid manifest with status `verified`;
* dataset/split details, seed, environment, encoder, route count, query count, and explicit timing units.
* a clean working tree and a SHA-256 digest matching the archived raw log.
* an independent reproduction from a different machine and reviewer
  attestation;
* an immutable archive URI and bundle digest.
