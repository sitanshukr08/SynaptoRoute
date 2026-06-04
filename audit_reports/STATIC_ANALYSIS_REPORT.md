# Static Analysis Report

## 1. Logic Bug: Duplicate Vector Insertion in NumpyIndex
- **Severity**: High
- **File**: `src/synaptoroute/index.py`
- **Line numbers**: 210-234
- **Code evidence**: 
  ```python
  def add(self, embeddings: np.ndarray, route_name: str):
      if True:
          self._add_unlocked(embeddings, route_name)
          num_embs = embeddings.shape[0]
          if self._next_id + num_embs > self.max_capacity:
              raise ValueError("Capacity exceeded")
          ...
          self.embeddings[self._next_id:self._next_id + num_embs] = embeddings
          ...
          self._next_id += num_embs
  ```
- **Explanation**: The `add` method first calls `self._add_unlocked(embeddings, route_name)` which handles the actual insertion and increments `self._next_id`. However, immediately following this call, the exact same insertion logic is duplicated. This results in every embedding being inserted twice, prematurely draining index capacity by 2x and corrupting the state of internal trackers.
- **Reproduction path**: Initialize a `NumpyIndex` and call `add()` once with a single embedding. Observe that `index.total_vectors` and `self._next_id` incorrectly increment by 2.
- **Recommended fix**: Remove the duplicated insertion code from `add()`, leaving only the call to `self._add_unlocked()`.

## 2. Race Condition: Async Future InvalidStateError
- **Severity**: Medium
- **File**: `src/synaptoroute/router.py`
- **Line numbers**: 1241-1243, 1245-1247
- **Code evidence**:
  ```python
  for f, r in zip(futures, res):
      if not f.done():
          event_loop.call_soon_threadsafe(f.set_result, r)
  ```
- **Explanation**: `f.done()` is evaluated synchronously in the worker thread before queueing `f.set_result` onto the async event loop. If the future `f` is cancelled by the event loop in the microsecond gap between the queue operation and execution, `f.set_result(r)` will be called on a completed future. This raises an unhandled `asyncio.exceptions.InvalidStateError`, which will crash the event loop task processor.
- **Reproduction path**: Bombard the router with `aquery` calls and randomly cancel futures while the background `_dispatch_batch` executor thread is yielding results back to the event loop.
- **Recommended fix**: Wrap the result assignment in a thread-safe callback that performs the `f.done()` check precisely at execution time. 
  ```python
  def _safe_set(f, r):
      if not f.done():
          f.set_result(r)
  event_loop.call_soon_threadsafe(_safe_set, f, r)
  ```

## 3. Null Handling Issue: Missing Attribute Initialization
- **Severity**: Medium
- **File**: `src/synaptoroute/router.py`
- **Line numbers**: 954-956
- **Code evidence**:
  ```python
  if not self._rebuild_pending and len(self.index.tombstones) > 1000 ...:
      self._rebuild_pending = True
      if self._loop and not self._loop.is_closed():
          asyncio.run_coroutine_threadsafe(self._rebuild_index(), self._loop)
  ```
- **Explanation**: The `_flush_storage_batch` method attempts to check `if self._loop...` to initiate garbage collection. However, `self._loop` is only initialized inside the async `start()` method. If a user utilizes `AdaptiveRouter` strictly via the synchronous `__call__` API without booting the async queue, `self._loop` is never defined. When garbage collection thresholds are hit, this triggers an `AttributeError: 'AdaptiveRouter' object has no attribute '_loop'`, breaking the background storage worker continuously.
- **Reproduction path**: Instantiate `AdaptiveRouter` synchronously. Add and delete 1,001 routes synchronously. The background flush worker will crash on the `AttributeError`.
- **Recommended fix**: Define `self._loop = None` in `__init__()`, or safely query the attribute using `getattr(self, '_loop', None)`.
