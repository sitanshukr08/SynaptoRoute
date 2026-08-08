# SynaptoRoute: Durability and Backpressure Semantics for Mutable Local Semantic Routers

Status: paper skeleton; results may be added only from verified manifests

## Abstract

Describe the problem, explicit mutation contract, controlled evaluation, and
observed tradeoffs. Do not insert numerical results until evidence promotion.

## 1. Introduction

Frame local semantic routing as a mutable systems component. State that routing
is not authorization and that semantic quality is not the primary novelty.

## 2. System Model

Define observable decisions, in-memory acknowledgement, durable receipts,
versioned mutations, queue backpressure, index rebuilds, and process-crash
scope.

## 3. Implementation

Describe exact and HNSW indexes, bounded query and mutation queues, SQLite WAL
transactions, and restart recovery.

## 4. Methodology

Reference `docs/RESEARCH_PROTOCOL.md`, the frozen artifact commit, dependency
lock, experiment matrix, baselines, hardware, seeds, and statistics.

## 5. Results

Generated tables and figures only. Preserve negative calibration results and
report correctness violations in the denominator.

## 6. Discussion And Limitations

Cover encoder dependence, OOD risk, process-crash versus power-loss scope,
single-node scope, synthetic structural workloads, and external validity.

## 7. Related Work

Compare semantic routers, open-intent classification, mutable indexes, and
durability/backpressure mechanisms without claiming component novelty.

## 8. Reproducibility

Link the immutable source tag, artifact DOI, manifests, raw logs, generated
tables, and independent reproduction record.
