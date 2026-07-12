# SynaptoRoute System Architecture

This document maps the core subsystems of SynaptoRoute v0.4.0, defining strict ownership, dependencies, and known failure modes to facilitate safe architectural reasoning for contributors.

## Subsystem 1: Router (`AdaptiveRouter`)
The central orchestrator of the system, handling API ingestion, workload dispatching, and lookup execution.

* **Owner:** Memory Dictionary (`_route_map`), Async Event Loop (`_loop`), Batch Queues (`_batch_queue`).
* **Depends On:** 
  * `Encoder` (for text vectorization)
  * `FaissIndex` (for distance searches)
  * `SQLiteStorage` (for WAL persistence)
  * `RedisSyncManager` (for cluster broadcasting)
* **Failure Modes:** 
  * **Queue Saturation:** If the `Encoder` processes slower than incoming HTTP traffic, `_batch_queue` will hit its `maxsize`, rejecting traffic with a fast-fail Exception.
  * **Deadlocks:** Improper acquisition of `_route_map_lock` and `rwlock` across `fit_thresholds` and `_dispatch` can freeze the API.

The async worker bounds both queued requests and in-flight batches. Executor
work is awaited before an in-flight permit is released, so draining the queue
cannot create an unbounded hidden executor backlog.

## Subsystem 2: Storage (`SQLiteStorage`)
The single-process durable state boundary. Runtime mutations are visible before
they are committed; callers use mutation receipts or a durable barrier when
SQLite acknowledgement is required. See
[`DURABILITY_CONTRACT.md`](DURABILITY_CONTRACT.md).

* **Owner:** SQLite Disk I/O, Write-Ahead-Log (WAL) mode connections, Database Schemas (`routes`, `utterances`).
* **Depends On:** Local Filesystem.
* **Failure Modes:**
  * **Concurrency Locked:** If thread-pool executors exceed SQLite's internal isolation level limits, writes will queue and timeout.
  * **Disk Full:** Aborts application state tracking on write failures.
  * **Pre-Commit Exit:** Mutations acknowledged only in memory can be lost if
    the process exits before the storage worker commits them.

## Subsystem 3: Index (`FaissIndex`)
The mathematical vector execution plane.

* **Owner:** `IndexFlatIP` (Inner Product / Cosine Similarity C++ vectors).
* **Depends On:** `numpy` ndarrays fed from the `AdaptiveRouter`.
* **Failure Modes:**
  * **Memory Leaks:** If vectors are appended without explicit tombstoning (`index.delete()`) during route updates, RAM grows unboundedly and top-K calculations skew.

## Subsystem 4: Sync (`RedisSyncManager`)
Redis sync is optional and experimental in the current release line. It is suitable for Pub/Sub experiments, not for claims of guaranteed distributed consistency.
The distributed consistency fabric for multi-node deployments.

* **Owner:** Redis PubSub Channels (`synaptoroute:sync`), Background Worker Threads (`sync_worker_count`).
* **Depends On:** External Redis Cluster, `AdaptiveRouter` (for local injection via `add_route(_broadcast=False)`).
* **Failure Modes:**
  * **Broadcast Storms (O(N×M)):** Bootstrapping a new node via `request_full_sync` triggers the entire cluster to shout their full route map over the network, causing buffer overflows above 100,000 routes.
  * **Cache Divergence:** Naive caching (`_synced_routes`) can permanently ignore valid route recreations if not relying purely on `_broadcast=False` logic.

## Subsystem 5: Encoder (`FastEmbedEncoder`)
The NLP boundary that translates text to hypersphere geometry.

* **Owner:** ONNX Runtime session, Tokenizer vocabulary.
* **Depends On:** CPU/Memory hardware layer.
* **Failure Modes:**
  * **Throughput Bottlenecks:** Hard-capped to ~130 requests per second per core.
  * **OOM (Out of Memory):** Loading multiple massive bi-encoder models concurrently without sharing memory structures will exhaust RAM.
