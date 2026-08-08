# SynaptoRoute Technical Reference

Status: development reference for `0.5.0.dev0`.

SynaptoRoute is a mutable, single-process semantic router. It maps text to a
named route using a caller-supplied or local embedding encoder and an in-memory
vector index. SQLite stores route definitions for restart recovery.

This document describes implemented behavior. Historical benchmark values are
audit records only; see [Current Evidence Status](CURRENT_EVIDENCE_STATUS.md).

## Components

### `AdaptiveRouter`

The router owns the route map, vector index, query batching queues, mutation
queue, and lifecycle. Its stable routing interfaces are:

* `match(text)` for synchronous decisions;
* `amatch(text)` for asynchronous decisions after `start()`;
* callable compatibility, equivalent to the legacy route-returning query path.

`match()` and `amatch()` return `RouterResult`, which records the selected
route, score, candidate margin, unique route candidates, and decision reason.

### Encoders

`FastEmbedEncoder` is the default local encoder. `BaseEncoder` permits a
deterministic or application-specific implementation to be injected. Unit
tests use a deterministic network-free encoder; downloading a FastEmbed model
belongs in the integration test job and paper image.

### Indexes

The built-in NumPy index performs exact cosine retrieval over normalized
vectors. FAISS is optional and can be selected explicitly for scale
experiments. No latency or complexity claim is implied by selecting either
engine. Deleted vectors become tombstones until a generation-based rebuild
successfully swaps in a stable index snapshot.

### SQLite Storage

SQLite is the restart source of truth for local route state. The schema uses
ordered migrations recorded in `schema_migrations`. Route state includes:

* route name, threshold, metadata, and monotonically increasing version;
* utterances in deterministic order;
* optional persisted utterance embeddings.

Connections run in autocommit mode so transaction boundaries are explicit.
Snapshot reads use `BEGIN`; writes use `BEGIN IMMEDIATE`. Research runs default
to `PRAGMA synchronous=FULL`; `NORMAL` is an explicit experimental setting.

## Mutation Contract

Route mutations update memory and the index, then submit a versioned write to
a bounded storage queue. An accepted mutation returns `MutationReceipt` with:

* route name and route version;
* acknowledgement mode;
* `queued`, `durable`, or `failed` state;
* durable commit latency or failure details.

Memory visibility is not durability. In this project, `durable` means the
SQLite commit completed and the mutation survives a subsequent process crash.
It does not claim survival from power loss, storage-controller loss, or broken
filesystem guarantees.

If a queued write fails, the router reconciles only that route from SQLite and
only when the failed version is still the newest in-memory version. SQLite
also rejects stale expected versions, preventing an older replacement from
overwriting a newer stored route.

Missing-route deletion and duplicate-utterance insertion remain compatibility
no-ops and return no receipt because no mutation occurred.

## Overload And Shutdown

Query and storage queues are bounded. Queue saturation raises
`RouterOverloadedError` before applying a mutation that cannot be queued.
Shutdown stops accepting new mutations, drains queued storage work or reports
a timeout, stops workers, waits for rebuild work, and closes executor pools.
Callers that require confirmation of every write should retain receipts or call
`durable_barrier()` before shutdown.

## Experimental Features

The following are excluded from the primary paper configuration:

* adaptive memory and learned weighting;
* hybrid lexical routing;
* sessions and slots;
* permission-based route filtering;
* Redis synchronization.

Redis Pub/Sub does not currently establish complete bootstrap, replay, or
missed-window semantics. Permission metadata is route filtering, not an
authorization boundary.

## Evidence Policy

Benchmark scripts produce schema-v2 `unverified` manifests. Scripts cannot
verify their own outputs. A claim becomes `verified` only through the separate
promotion command after a clean original run, independent reproduction on a
different machine, reviewer attestation, and immutable archive metadata.

The working paper is framed as a systems study of durability, online mutation,
overload, crash recovery, and selective routing. It does not claim a novel
classifier or a generally superior routing algorithm.
