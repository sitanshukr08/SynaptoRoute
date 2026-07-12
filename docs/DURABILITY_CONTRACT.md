# Mutation And Durability Contract

Status: implemented single-process contract  
Scope: `AdaptiveRouter` with `SQLiteStorage`

## Acknowledgement Levels

SynaptoRoute exposes two distinct mutation milestones.

### In-Memory Acknowledgement

`add_route`, `add_utterance`, `delete_route`, and `update_threshold` update the
runtime route map and index before returning. A successful return means the
mutation is visible to subsequent queries in the current process and has been
enqueued for storage. It does not mean SQLite has committed the mutation.

Each non-no-op mutation returns a `MutationReceipt`. The receipt starts in the
`queued` state and carries a process-local monotonically increasing sequence.

### Durable Acknowledgement

`receipt.wait_durable(timeout=...)` waits for the corresponding SQLite method
to commit. It returns durable-commit latency in milliseconds and changes the
receipt state to `durable`. A storage failure raises `StorageMutationError`.

`router.durable_barrier(timeout=...)` waits for all mutations queued before the
barrier and raises `StorageFlushError` if any queued mutation failed. The
existing `flush_storage` method remains a queue-drain primitive for backward
compatibility and does not report historical failures.

## Ordering

The process-local storage queue is FIFO. Receipt sequence numbers describe
enqueue order, and one storage worker applies mutations in that order. This is
not a distributed ordering guarantee and sequence numbers reset after restart.

## Failure Behavior

If a durable write fails:

1. the failed receipt is marked `failed`;
2. the error is retained for the next durable barrier;
3. runtime route and index state are rebuilt from SQLite;
4. callers that only used in-memory acknowledgement may observe the mutation
   disappear after resynchronization.

Successful writes in the same batch remain durable. A failed write is never
reported as durable.

## Restart Guarantee

A mutation whose receipt reached `durable` must be present after a normal
process restart using the same SQLite database, unless a later durable mutation
overwrote or deleted it. Crash consistency under abrupt process termination is
measured separately by the fault-injection benchmark.

## Exclusions

This contract does not cover Redis Pub/Sub delivery, multi-process writes to a
shared database, filesystem or hardware failure beyond SQLite's guarantees,
or authorization to execute a routed action.
