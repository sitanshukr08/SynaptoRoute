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
* A clean local Windows pilot at `78e96ec392b448f84f3b677d2d3af859153961d0`
  separately exercised the revised version 2 systems ledger. All 16 frozen
  crash-recovery cells and 3,200 trials completed, and the matrix verifier
  checked all command logs plus 3,200 SQLite and 3,200 acknowledgement-marker
  hashes. It reported zero integrity errors and zero outcome observations. The
  run is corroborating development evidence only; it has no controlled-host
  environment bundle, immutable archive, independent reviewer, or attestation.
  Details are in `SYSTEMS_PILOT_RESULTS.md`.
* A second clean local Windows run at
  `52968b6fc72350f4e6b2e4adba030d045931b39a` completed the 15-cell frozen
  sustained-backpressure matrix: 60 scenarios and 3,961,344 offered requests.
  All outcomes were accounted for, all 2,714,164 successful routes were
  correct, 1,247,180 requests were explicitly shed, and no request errors were
  recorded. Repetition-level bootstrap analysis also exposed substantial P95
  variance in the low-latency profile at calibrated saturation. This is
  diagnostic evidence only because the host was uncontrolled and the run has
  no environment bundle, independent reproduction, archive, or attestation.
  Details are in `SYSTEMS_PILOT_RESULTS.md`.
* A clean local scale run at
  `039971adbbea48ab00b3f18e603f9a1f55fee243` completed all 40 frozen cells
  and 400,000 structural identity queries. NumPy exact returned all 200,000
  identities. FAISS HNSW returned 199,495 of 200,000, with misses confined to
  the 50,000- and 100,000-vector cells. The paired analysis found a query
  performance crossover at larger sizes alongside lower identity accuracy,
  much higher one-at-a-time build cost, and higher observed RSS. This is an
  unverified diagnostic result, not a release or paper claim. The uncontrolled
  run also omitted first-class FAISS/HNSW configuration fields needed for
  confirmation. Details are in `SYSTEMS_PILOT_RESULTS.md`.

## Research Claims

* Historical Banking77, CLINC150, OOD, latency, throughput, GPU, and 1M-scale
  numbers remain audit records only. The clean local 100k scale pilot is a new
  unverified diagnostic and does not supersede that publication restriction.
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
