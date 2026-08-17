# SynaptoRoute: Durability and Backpressure Semantics for Mutable Local Semantic Routers

Status: methods draft; numerical results may be added only from verified manifests

## Abstract

Local semantic routers are commonly evaluated as static classifiers even when
deployed inside mutable, concurrent applications. This paper studies the
systems semantics of such a router: when a mutation becomes visible, when it
is durable, how overload is exposed, and whether runtime, index, and SQLite
state remain consistent across failures and restart. The final abstract will
summarize only results promoted from independently reproduced artifacts.

## 1. Introduction

SynaptoRoute is framed as a mutable local systems component, not as a novel
embedding model or a replacement for authorization. Its research question is
whether explicit acknowledgement, versioning, bounded queues, and recovery
semantics make online routing behavior measurable and reproducible under mixed
queries and mutations.

The intended contributions are:

1. an observable mutation contract separating memory visibility from durable
   completion;
2. a fault and dynamic-workload harness covering concurrency, overload,
   process crashes, and restart;
3. an empirical characterization of quality and systems tradeoffs, including
   negative and null results.

## 2. System Model

`RouterResult` exposes the selected route, score, top-two margin, ranked
candidates, and decision reason. A mutation receipt identifies the route,
persisted route version, acknowledgement mode, final state, and failure.
`queued` means the mutation is memory-visible and awaiting storage completion;
`durable` means its SQLite transaction completed and the mutation survives the
defined process-crash/restart model. Power loss, kernel failure, filesystem
corruption, and distributed consensus are outside this definition.

Queues are bounded. Saturation is observable as shedding or a failed receipt,
not hidden as unbounded memory growth. Route versions prevent an older failed
write from rolling back a newer mutation. Shutdown stops admission, drains or
fails outstanding work, stops workers, and closes storage deterministically.

## 3. Implementation

The implementation provides exact NumPy and optional FAISS HNSW indexes over a
shared encoder contract. SQLite uses ordered migrations, WAL mode, explicit
read transactions, `BEGIN IMMEDIATE` writes, and configurable `FULL` or
`NORMAL` synchronous behavior. Complete route state and deterministic
utterance order are persisted with monotonically increasing versions.

Index rebuilds use generations: a rebuild superseded by a mutation retries
with bounded backoff and replays intervening mutations. Query batching and
storage writes use separately bounded queues so their overload and latency can
be measured independently. Redis synchronization and adaptive memory are
experimental and excluded from the primary artifact.

## 4. Methodology

The candidate protocol is defined in `docs/RESEARCH_PROTOCOL.md`,
`paper/QUALITY_PROTOCOL.md`, and `paper/experiment_matrix.json`. Every run
records a full commit SHA, dirty-tree state, command plan, dependency-lock hash,
environment, hardware, seed, units, raw-output hashes, and exit status.

Quality experiments use five fixed seeds and disjoint route, policy-fit,
probability-fit, and test examples. Systems share the same encoder and route
examples where applicable. Comparators include exact string, logistic
regression, exact cosine, Semantic Router, and SynaptoRoute policy variants.
Paired bootstrap analysis operates on aligned per-example predictions.

Systems experiments vary routes, query workers, mutation rates, index engine,
durability mode, SQLite synchronous mode, injected commit delay, queue profile,
and offered load. Warmup, measurement duration, repetitions, and all matrix
cells are fixed before confirmatory execution. Dynamic-workload cells request
FAISS explicitly; development-only `auto` resolution is excluded because
optional dependency availability would otherwise alter the experiment.

## 5. Results

This section remains intentionally empty until claim manifests are promoted.
Tables and figures are generated from archived machine-readable outputs.
Correctness violations, failed requests, shedding, abstentions, and
high-confidence errors remain in their declared denominators. Negative results
are retained.

## 6. Discussion And Limitations

The evaluation is encoder-dependent and cannot establish universal semantic
quality. OOD detection remains distribution-dependent. Structural workloads
isolate queue/index behavior but do not reproduce every production workload.
The durability model covers a single local node and process crashes, not power
loss or distributed replication. The held-out correctness model estimates a
final decision's correctness under the evaluated distribution; it is not an
authorization or safety probability.

## 7. Related Work

The final paper will compare semantic-routing libraries, open-intent and
selective classification, mutable approximate-nearest-neighbor indexes,
durable local storage, and backpressure systems. Component techniques are not
claimed as individually novel; the contribution is their explicit contract
and combined fault/dynamic-workload evaluation.

## 8. Reproducibility

The final section will link an immutable source tag, container digest, artifact
DOI, content inventory, original and reproduction manifests, raw logs,
predictions, generated tables/figures, and reviewer attestation. The executable
procedure is `paper/AFK_RUNBOOK.md`; current claim state is
`paper/CLAIM_LEDGER.md`.
