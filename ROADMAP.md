# SynaptoRoute Research Roadmap

This roadmap converts SynaptoRoute from an engineering prototype into a
reproducible systems research artifact. It is deliberately ordered so that no
accuracy or latency result is promoted before the benchmark that produced it
is reviewable.

The detailed experimental contract is in
[`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md). Competitive context
and artifact execution instructions are in
[`paper/ARTIFACT_EVALUATION.md`](paper/ARTIFACT_EVALUATION.md).

## Target Contribution

Working thesis:

> A local semantic router can provide calibrated selective routing with
> predictable tail latency and explicit durability behavior while routes are
> updated concurrently.

This is a systems claim, not a claim that embeddings, HNSW, SQLite, or
thresholding are individually novel.

Working paper title:

> SynaptoRoute: Durability and Backpressure Semantics for Mutable Local
> Semantic Routers

The primary contribution is the observable mutation/durability contract and
its fault-injection evaluation. Calibration is an ablation, not a general
accuracy-superiority claim.

## Current State

Implemented engineering capabilities:

* local embedding-based route retrieval;
* NumPy and FAISS HNSW indexes;
* per-route thresholds, margin gating, and optional reranking;
* asynchronous batching and bounded request queues;
* SQLite WAL persistence with queued writes;
* experimental Redis Pub/Sub synchronization.

Evidence limitations:

* historical accuracy, OOD, throughput, GPU, and scale results are unverified;
* the former `0.003ms` one-million-vector result is retracted;
* bundled semantic datasets are development fixtures, not a publishable
  dataset contribution;
* external dataset pilots are diagnostic and remain unverified until rerun
  from a clean commit;
* Redis synchronization has no durable replay or consistency guarantee.

## Phase 0: Evidence Integrity

**Status:** schema-v2 foundation and candidate preflight implemented; clean
candidate freeze, final runs, immutable archive, independent reproduction, and
evidence promotion pending

Deliverables:

* schema-validated benchmark manifests and archived raw logs;
* a dirty-worktree gate for verified evidence;
* one reproducible development and benchmark installation command;
* corrected Top-K, latency, split, seed, and timing-unit definitions;
* a deterministic local structural benchmark suitable for CI;
* documentation that maps every public claim to evidence.
* strict candidate preflight for source, lock, matrix, manifests, and package version.
* a five-family bounded protocol smoke with explicit correctness invariants.
* resumable matrix execution with atomic per-command state and hashed logs.
* deterministic archive construction with a streamed content inventory.

Exit criteria:

* `pytest` discovers only the supported test suite and passes from a clean
  editable installation;
* every historical manifest is valid and remains `unverified` or `retracted`;
* the local smoke benchmark runs through the benchmark runner and emits raw
  output plus a valid run manifest;
* no public document calls historical numbers verified.

## Phase 1: Observable Decisions And Calibration

**Status:** implementation complete; bounded pilot and independent review pending

Deliverables:

* `RouterResult` containing route, score, margin, ranked candidates, and
  decision reason;
* identical decision semantics in synchronous and asynchronous paths;
* validation-only fitting of global and per-route thresholds;
* calibrated abstention policy with held-out calibration data;
* reliability, risk-coverage, and failure-bucket reporting.

Exit criteria:

* existing `Optional[Route]` APIs remain backward compatible;
* no test or calibration example is used as a route utterance;
* calibration behavior is covered by deterministic unit tests;
* the chosen policy is fixed before final test-set evaluation.

Implemented so far:

* backward-compatible `RouterResult` output for sync and async routing;
* ranked candidates, score, margin, and explicit decision reasons;
* validation-only global threshold and margin fitting with hashed artifacts;
* validation-only per-route threshold fitting with a shared margin;
* OOD ranking and selective risk-coverage metrics;
* disjoint held-out correctness-probability calibration with an explicit
  one-class fallback;
* ECE, Brier score, reliability-bin JSON, and generated SVG diagrams;
* unit and integration tests that prevent fitting on the test split.

Still required: bounded quality pilots and independent review of the multi-seed
confidence intervals, probability calibration, and matched-coverage implementation.

The ordered execution checklist is maintained in
[`paper/NEXT_STEPS.md`](paper/NEXT_STEPS.md).

## Phase 2: Static Quality Baselines

**Status:** in progress

Datasets:

* Banking77 official train/test split;
* CLINC150/OOS official train, validation, and test splits;
* selected BOLT open-set text-classification tasks;
* optional MASSIVE multilingual subset after the English protocol is stable.

Required baselines:

* exact string/rule matching;
* logistic regression over the same embeddings;
* exact cosine retrieval over the same embeddings;
* SynaptoRoute NumPy and FAISS HNSW indexes;
* Aurelio Semantic Router `>=0.1.15` with the same encoder and route examples;
* at least one published open-intent baseline when licenses and code permit.

Exit criteria:

* five predetermined seeds where sampling is involved;
* top-1/macro-F1, AUROC, AUPRC, FPR@95, coverage, and selective-risk results;
* paired confidence intervals and effect sizes;
* all raw predictions, configurations, and manifests archived.

Implemented so far:

* immutable-revision Banking77 and CLINC150/OOS loaders;
* deterministic stratified pilot subsets and explicit cross-split
  decontamination counts;
* exact string, exact cosine, logistic-regression, and Semantic Router
  baselines;
* 500-query development pilots for both primary datasets.
* a fixed five-seed driver that retains per-seed artifacts and aggregates only
  quality metrics.

The pilots are intentionally marked `unverified` and are not paper evidence.
Diagnostic five-seed runs and paired statistics are implemented. Historical
local reruns remain unverified under schema v2. Artifact archival, independent
reproduction, and evidence promotion remain open.

Pilot interpretation is recorded in
[`docs/DEVELOPMENT_PILOT_RESULTS.md`](docs/DEVELOPMENT_PILOT_RESULTS.md). The
external comparator and OOD ranking metrics are now implemented; multi-seed
statistics are diagnostic only, and clean evidence promotion remains open.

## Phase 3: Dynamic Routing And Durability

**Status:** in progress

Deliverables:

* a written acknowledgement and durability contract;
* route versions and deterministic mutation ordering;
* read/write workload generator with controlled mutation rates;
* crash, restart, failed-write, and queue-backpressure fault injection;
* mutation visibility, durable-commit latency, recovery time, and lost-update
  measurements.

Exit criteria:

* every acknowledged durability level has a testable definition;
* recovery tests distinguish acknowledged, queued, and committed mutations;
* no correctness claim depends only on a successful happy-path run.

Implemented so far:

* process-local mutation receipts with explicit queued, durable, and failed
  states;
* a failure-reporting durable barrier while preserving nonblocking mutation
  calls;
* normal-restart and injected storage-failure tests;
* abrupt child-process exit experiments at memory and durable acknowledgement
  boundaries;
* ordered migrations, route versions, and stale-write rejection;
* a controlled concurrent read/write workload with queue depth, memory,
  visibility, durable latency, query latency, correctness, and restart checks;
* a bounded in-flight async batch executor and measured-saturation offered-load
  shedding sweep;
* four-mutation process-crash trials across acknowledgement and SQLite modes.

Still required: frozen controlled-hardware runs, independent reproduction,
archive publication, and broader cross-process semantics if multi-writer
operation is later claimed.

## Phase 4: Scale And Ablation Study

**Target:** weeks 9-14

Experiments:

* route and utterance scale sweeps;
* query-only and mixed read/write loads;
* exact versus HNSW retrieval;
* encoder, threshold, margin, reranker, batch size, queue size, persistence, and
  HNSW-parameter ablations;
* at least two CPU hardware profiles and an optional GPU profile;
* p50/p95/p99 latency, throughput, memory, cold-start, and recovery metrics.

Exit criteria:

* workloads are identical across systems where comparison is claimed;
* warmup, concurrency, repetitions, and timing boundaries are documented;
* conclusions include negative results and practical operating limits.

## Phase 5: Artifact And Paper

**Target:** weeks 15-18

Deliverables:

* clean tagged release and immutable commit;
* lockfile or containerized reproduction environment;
* archived artifact and DOI;
* paper tables generated from validated result files;
* limitations, ethics, and threat-to-validity sections;
* independent reproduction by a contributor who did not write the benchmark.

Submission direction is intentionally venue-neutral until final results exist.
An applied systems venue is the primary target; JOSS remains the software-paper
fallback. A top-tier measurement claim is considered only if the controlled
results establish a contribution beyond straightforward component assembly.

## First 90 Days

### Days 1-30

* complete Phase 0;
* freeze research questions, hypotheses, datasets, metrics, and exclusions;
* reconcile public documentation with the evidence registry;
* implement scored decision results without breaking compatibility.

### Days 31-60

* finish calibration and abstention policies;
* implement exact, logistic-regression, and Semantic Router baselines;
* run pilots on Banking77 and CLINC150;
* finalize dynamic workload and fault-injection specifications.

### Days 61-90

* run the fixed multi-seed static experiments;
* run initial mutation, durability, and recovery experiments;
* produce the first statistical tables and figures;
* complete a paper skeleton containing only verified results.

## Decision Gates

At day 60, stop the algorithmic paper direction if calibrated routing does not
beat simple baselines at matched coverage. Continue only with a systems or
negative-results framing.

At day 90, stop the full conference-paper plan if the team cannot reproduce
the experiment artifact from a clean machine. A technical report or JOSS
software paper remains a valid fallback.

## Product Release Context

The research phases govern evidence quality; release versions govern
user-facing features. The current package release is v0.4.1. Distributed
bootstrapping, multi-tenant indexes, and multimodal encoders remain product
backlog items and must not bypass the research validation gates above.
