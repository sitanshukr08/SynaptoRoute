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
`queued` state and carries a process-local monotonically increasing sequence,
route name, resulting route version, and acknowledgement mode.

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
3. the affected route is reconciled from SQLite only if the failed version is
   still the newest in-memory version;
4. callers that only used in-memory acknowledgement may observe the mutation
   disappear after resynchronization.

Successful writes in the same batch remain durable. A failed write is never
reported as durable.

## Restart Guarantee

A mutation whose receipt reached `durable` must survive a process crash and
restart using the same SQLite database, unless a later durable mutation
overwrote or deleted it. This is a process-crash contract. It is not a
power-loss or storage-hardware guarantee.

## Exclusions

This contract does not cover Redis Pub/Sub delivery, multi-process writes to a
shared database, power loss beyond the configured SQLite synchronization mode,
storage-hardware failure, or authorization to execute a routed action.
