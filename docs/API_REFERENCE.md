# SynaptoRoute Public API Reference

## Primary Classes

### `AdaptiveRouter`

The core local semantic router component.

```python
from synaptoroute import AdaptiveRouter, Route, SQLiteStorage

router = AdaptiveRouter(
    encoder=None,
    storage=None,
    sync_manager=None,
    profile=None,
    max_capacity=50000,
    max_queue_size=1000,
    max_storage_queue_size=1000,
    max_in_flight_batches=4,
    margin=0.0
)
```

#### Synchronous Methods

* **`add_route(route: Route) -> MutationReceipt`**: Registers a new route or updates an existing route definition. Enqueues an asynchronous write mutation to storage.
* **`add_utterance(route_name: str, utterance: str) -> MutationReceipt`**: Dynamically appends a new example utterance to an existing route.
* **`delete_route(route_name: str) -> MutationReceipt`**: Removes a route and all its associated vector embeddings.
* **`update_threshold(route_name: str, threshold: float) -> MutationReceipt`**: Updates the minimum confidence threshold for a route.
* **`match(query: str) -> RouterResult`**: Synchronously evaluates a query string and returns a `RouterResult` object containing the decision reason, top candidate, score, and candidate pool.
* **`durable_barrier(timeout: float = 10.0)`**: Blocks until all pending storage mutations are committed to SQLite disk WAL.
* **`fit_thresholds(samples: list[str], labels: list[str])`**: Calibrates per-route thresholds using validation samples and ground-truth intent labels.
* **`close()`**: Stops accepting mutations, flushes pending writes, and deterministically stops router-owned workers.

#### Asynchronous Methods

* **`async start()`**: Starts background batch worker tasks.
* **`async amatch(query: str) -> RouterResult`**: Asynchronously enqueues a query into the microbatching pipeline and returns a `RouterResult`.
* **`async stop()`**: Stops async workers and flushes queued tasks.

---

### `Route`

Data model defining a single semantic intent route.

```python
from synaptoroute import Route

route = Route(
    name="billing",
    utterances=["payment failed", "cancel subscription", "refund request"],
    threshold=0.75,
    version=1,
    metadata={"department": "finance"}
)
```

**Attributes:**
* `name` (`str`): Unique identifier (letters, numbers, underscores, hyphens).
* `utterances` (`list[str]`): Non-empty list of example queries. Automatically deduplicated on initialization.
* `threshold` (`float`): Similarity score threshold in range `[-1.0, 1.0]`. Default: `0.5`.
* `version` (`int`): Schema/mutation version integer. Default: `1`.
* `metadata` (`dict`, optional): Optional JSON-serializable dictionary.

---

### `RouterResult`

Immutable result object returned by `match()` and `amatch()`.

**Properties:**
* `matched` (`bool`): `True` if a candidate passed threshold and margin criteria.
* `route` (`Route`, optional): Matched `Route` object, or `None` if rejected.
* `route_name` (`str`, optional): Name of matched route, or `None`.
* `score` (`float`, optional): Cosine similarity score of top candidate.
* `margin` (`float`, optional): Difference between top candidate score and second-best candidate score.
* `candidates` (`list[RouteCandidate]`): Ranked list of top candidate matches.
* `decision_reason` (`DecisionReason`): Decision classification (`MATCHED`, `BELOW_THRESHOLD`, `AMBIGUOUS_MARGIN`, `EMPTY_INDEX`).

---

### `MutationReceipt`

Receipt object returned by mutation operations (`add_route`, `add_utterance`, `delete_route`).

**Attributes:**
* `sequence`: Process-local FIFO mutation sequence.
* `route_name`: Affected route.
* `route_version`: Resulting route version.
* `acknowledgement_mode`: `memory` for the immediate return milestone.
* `state`: `queued`, `durable`, or `failed`.
* `error_detail`: Failure detail when state is `failed`.

**Methods:**
* **`wait_durable(timeout: float = 10.0) -> float`**: Blocks until the mutation is committed to disk storage and returns disk commit latency in milliseconds.

---

### `SQLiteStorage`

Local SQLite storage engine providing persistent WAL storage for routes and BLOB-cached float32 embeddings.

```python
from synaptoroute import SQLiteStorage

storage = SQLiteStorage(db_path="routes.sqlite3", synchronous="FULL")
```

`FULL` is the default research mode. `NORMAL` is available for explicitly
configured performance experiments. Both use WAL; neither is presented as a
hardware power-loss guarantee.
