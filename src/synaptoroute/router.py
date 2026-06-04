import threading
import numpy as np
from sklearn.metrics import f1_score
from typing import Optional
import time
import asyncio
import logging
import concurrent.futures

from synaptoroute.metrics import MetricsRegistry
from synaptoroute.models import Route
from synaptoroute.encoder import Encoder
from synaptoroute.storage import BaseStorage, SQLiteStorage
from synaptoroute.sync import BaseSyncManager
from synaptoroute.exceptions import RouteNotFoundError, RouterOverloadedError, RouterCapacityError
from synaptoroute.profile import OptimizationProfile, get_profile, ProfileType
from synaptoroute.index import get_index

class AdaptiveRouter:
    def __init__(self, encoder: Optional[Encoder] = None, storage: BaseStorage = None, profile: OptimizationProfile = None, max_capacity: int = 50000, max_queue_size: int = 10000, metrics: MetricsRegistry = None, sync_manager: Optional[BaseSyncManager] = None, margin: float = 0.0, reranker=None):
        if profile is None:
            profile = get_profile(ProfileType.THROUGHPUT)
            
        if encoder is None:
            from synaptoroute.encoder import FastEmbedEncoder
            encoder = FastEmbedEncoder(threads=profile.threads)
        self.encoder = encoder
        self.storage = storage or SQLiteStorage(":memory:")
        from synaptoroute.locks import RWLock
        self.rwlock = RWLock()
        self._encoder_lock = threading.Lock()
        self._rebuild_pending = False
        self.metrics = metrics or MetricsRegistry()
        self.sync_manager = sync_manager
        if self.sync_manager:
            self.sync_manager.register(self)
        
        self.max_capacity = max_capacity
        self.max_queue_size = max_queue_size
        self.margin = margin
        self.reranker = reranker
        
        self.index = get_index(dim=self.encoder.dim, max_capacity=self.max_capacity)
        self._route_map = {}
        self._route_map_lock = threading.Lock()
        self._mutation_count = 0
        self._pending_rebuild_mutations = []
        
        import queue
        self._storage_queue = queue.Queue()
        self._storage_worker_thread = threading.Thread(target=self._storage_worker, daemon=True)
        self._storage_worker_thread.start()
        
        self._batch_queue = None
        self._worker_task = None
        self.batch_size = profile.batch_size
        self.batch_timeout = profile.batch_timeout
        
        self._read_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(4, profile.threads),
            thread_name_prefix="router_read"
        )
        self._write_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(2, profile.threads // 2),
            thread_name_prefix="router_write"
        )
        
        self._load_routes()
        
    async def start(self):
        if self.sync_manager:
            await self.sync_manager.start()
        self._loop = asyncio.get_running_loop()
        self._batch_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self.sync_manager:
            await self.sync_manager.stop()
        if hasattr(self, '_read_pool'):
            self._read_pool.shutdown(wait=False)
        if hasattr(self, '_write_pool'):
            self._write_pool.shutdown(wait=False)

    def _load_routes(self):
        routes, embeddings_map = self.storage.load_all_routes()
        
        print(f"Loading {len(routes)} routes from storage...")
        import sys
        for r_idx, route in enumerate(routes):
            if r_idx % 5000 == 0:
                print(f"Processed {r_idx}/{len(routes)} routes for loading...")
                sys.stdout.flush()
            self._route_map[route.name] = route
            if not route.utterances:
                continue
                
            embs_data = embeddings_map.get(route.name, [])
            missing_idx = []
            missing_texts = []
            final_embeddings = np.zeros((len(route.utterances), self.encoder.dim), dtype=np.float32)
            
            expected_bytes = self.encoder.dim * 4  # float32 = 4 bytes
            
            for i, (u, e_bytes) in enumerate(zip(route.utterances, embs_data)):
                if e_bytes is not None and len(e_bytes) == expected_bytes:
                    final_embeddings[i] = np.frombuffer(e_bytes, dtype=np.float32)
                else:
                    if e_bytes is not None:
                        logging.getLogger(__name__).warning(
                            f"Discarding stale embedding for utterance "
                            f"'{u}' in route '{route.name}': expected "
                            f"{expected_bytes} bytes, got {len(e_bytes)}. "
                            f"Will re-encode."
                        )
                    missing_idx.append(i)
                    missing_texts.append(u)
                    
            if missing_texts:
                if r_idx % 5000 == 0:
                    print(f"Route {r_idx}: re-encoding {len(missing_texts)} missing embeddings! e_bytes len was {len(embs_data[0]) if embs_data else 'None'}, expected {expected_bytes}")
                    sys.stdout.flush()
                new_embs = self.encoder.encode_batch(missing_texts)
                for i, new_emb in zip(missing_idx, new_embs):
                    final_embeddings[i] = new_emb
                # Backfill DB so future boots are fast
                self.storage.save_route(route, final_embeddings)
                
            num_embs = len(final_embeddings)
            if self.index.total_vectors + num_embs > self.max_capacity:
                raise RouterCapacityError(f"Maximum capacity ({self.max_capacity}) exceeded.")
            
            self.index.add(final_embeddings, route.name)
            
        self.metrics.capacity_usage.set(self.index.total_vectors)

    def delete_route(self, route_name: str, _broadcast: bool = True):
        with self._route_map_lock:
            if route_name not in self._route_map:
                return
            
            self._route_map.pop(route_name)
            self._mutation_count += 1
            
        self._storage_queue.put(("delete_route", (route_name,)))

        if _broadcast and self.sync_manager:
            self.sync_manager.broadcast("delete_route", {"route_name": route_name})

    def _flush_storage_batch(self):
        import queue
        batch = []
        try:
            item = self._storage_queue.get(timeout=0.1)
            batch.append(item)
            while len(batch) < 250:
                try:
                    batch.append(self._storage_queue.get_nowait())
                except queue.Empty:
                    break
        except queue.Empty:
            pass
        
        if batch:
            # Batch apply index mutations under a single write lock acquisition
            with self.rwlock.write_lock():
                import numpy as np
                for action, args in batch:
                    try:
                        if action == "add_route":
                            route, embeddings = args
                            if route.name in self._route_map:
                                self.index.delete(route.name)
                            if embeddings is not None:
                                self.index.add(embeddings, route.name)
                                if self._rebuild_pending:
                                    self._pending_rebuild_mutations.append(("add_route", route.name, embeddings, route))
                        elif action == "delete_route":
                            route_name = args[0]
                            self.index.delete(route_name)
                            if self._rebuild_pending:
                                self._pending_rebuild_mutations.append(("delete_route", route_name, None, None))
                        elif action == "add_utterance":
                            route_name, utterance, embedding = args
                            self.index.add(np.array([embedding]), route_name)
                            if self._rebuild_pending:
                                self._pending_rebuild_mutations.append(("add_utterance", route_name, np.array([embedding]), utterance))
                    except Exception as e:
                        print(f"FATAL ERROR IN BATCH PROCESSING FOR {action}: {e}")
                
                self.metrics.capacity_usage.set(self.index.total_vectors)
            
            # Flush storage without holding the lock
            try:
                if self.storage:
                    for action, args in batch:
                        if action == "add_route":
                            route, embeddings = args
                            self.storage.save_route(route, embeddings)
                        elif action == "delete_route":
                            self.storage.delete_route(args[0])
                        elif action == "update_threshold":
                            self.storage.update_threshold(*args)
                        elif action == "add_utterance":
                            route_name, utterance, embedding = args
                            self.storage.add_utterance(route_name, utterance, embedding)
            except Exception as e:
                import logging
                print(f"SQLITE BATCH ERROR: {e}")
                logging.getLogger(__name__).error(f"Storage batch worker failed: {e}")
                
            if not self._rebuild_pending and len(self.index.tombstones) > 1000 and len(self.index.tombstones) > self.index.total_vectors * 0.2:
                self._rebuild_pending = True
                if self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(self._rebuild_index(), self._loop)
                
    def _storage_worker(self):
        while True:
            try:
                self._flush_storage_batch()
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"FATAL ERROR IN STORAGE WORKER: {e}")

    def add_route(self, route: Route, _broadcast: bool = True, _precomputed_embeddings=None):
        embeddings = _precomputed_embeddings
        if embeddings is None and route.utterances:
            if getattr(self.encoder, 'requires_lock', True):
                with self._encoder_lock:
                    embeddings = self.encoder.encode_batch(route.utterances)
            else:
                embeddings = self.encoder.encode_batch(route.utterances)
            len(embeddings)

        with self._route_map_lock:
            
            # Check capacity based on vectors
            new_vectors = len(route.utterances)
            current_vectors = sum(len(existing.utterances) for existing in self._route_map.values())
            if route.name in self._route_map:
                current_vectors -= len(self._route_map[route.name].utterances)
            if current_vectors + new_vectors > self.max_capacity:
                raise RouterCapacityError(f"Maximum capacity ({self.max_capacity}) exceeded.")
            
            self._route_map[route.name] = route
            self._mutation_count += 1
            
        self._storage_queue.put(("add_route", (route, embeddings)))

        if _broadcast and self.sync_manager:
            emb_bytes = embeddings.tobytes() if embeddings is not None else None
            self.sync_manager.broadcast("add_route", route.model_dump(mode="json"), embeddings=emb_bytes)

    def add_utterance(self, route_name: str, utterance: str, _broadcast: bool = True, _precomputed_embedding=None):
        if _precomputed_embedding is not None:
            embedding = _precomputed_embedding
        else:
            if getattr(self.encoder, 'requires_lock', True):
                with self._encoder_lock:
                    embedding = self.encoder.encode(utterance)
            else:
                embedding = self.encoder.encode(utterance)
        
        with self._route_map_lock:
            if route_name not in self._route_map:
                raise RouteNotFoundError(f"Route '{route_name}' not found.")
            if utterance in self._route_map[route_name].utterances:
                return
            current_vectors = sum(len(route.utterances) for route in self._route_map.values())
            if current_vectors + 1 > self.max_capacity:
                raise RouterCapacityError(f"Maximum capacity ({self.max_capacity}) exceeded.")

            self._route_map[route_name].utterances.append(utterance)
            self._mutation_count += 1
            
        self._storage_queue.put(("add_utterance", (route_name, utterance, embedding)))

        if _broadcast and self.sync_manager:
            emb_bytes = embedding.tobytes() if embedding is not None else None
            self.sync_manager.broadcast("add_utterance", {"route_name": route_name, "utterance": utterance}, embeddings=emb_bytes)

    async def _rebuild_index(self):
        try:
            routes, embeddings_map = await asyncio.to_thread(self.storage.load_all_routes)
            route_dict = {r.name: r for r in routes}
            new_index = get_index(dim=self.encoder.dim, max_capacity=self.max_capacity)
            await asyncio.to_thread(new_index.rebuild, route_dict, embeddings_map)
            
            with self.rwlock.write_lock():
                self.index = new_index
                if hasattr(self, '_pending_rebuild_mutations'):
                    for action, r_name, data, extra in self._pending_rebuild_mutations:
                        try:
                            if action == "add_route" and data is not None:
                                self.index.delete(r_name)
                                self.index.add(data, r_name)
                                route_dict[r_name] = extra
                            elif action == "add_utterance" and data is not None:
                                utterance = extra
                                if r_name in route_dict:
                                    if utterance not in route_dict[r_name].utterances:
                                        self.index.add(data, r_name)
                                        route_dict[r_name].utterances.append(utterance)
                                else:
                                    self.index.add(data, r_name)
                            elif action == "delete_route":
                                self.index.delete(r_name)
                                route_dict.pop(r_name, None)
                        except Exception as e:
                            logging.getLogger(__name__).warning(f"Failed to replay WAL mutation {action} on {r_name}: {e}")
                    self._pending_rebuild_mutations.clear()
        except Exception as e:
            logging.getLogger(__name__).error(f"Rebuild failed: {e}")
            self.metrics.gc_errors.inc()
        finally:
            self._rebuild_pending = False

    async def aadd_route(self, route: Route, _broadcast: bool = True, _precomputed_embeddings=None):
        return await asyncio.to_thread(
            self.add_route,
            route,
            _broadcast,
            _precomputed_embeddings,
        )

    async def aadd_utterance(self, route_name: str, utterance: str, _broadcast: bool = True, _precomputed_embedding=None):
        return await asyncio.to_thread(
            self.add_utterance,
            route_name,
            utterance,
            _broadcast,
            _precomputed_embedding,
        )

    async def adelete_route(self, route_name: str, _broadcast: bool = True):
        return await asyncio.to_thread(self.delete_route, route_name, _broadcast)

    async def aupdate_threshold(self, route_name: str, threshold: float, _broadcast: bool = True):
        return await asyncio.to_thread(
            self.update_threshold,
            route_name,
            threshold,
            _broadcast,
        )

    def __call__(self, query: str) -> Optional[Route]:
        start_time = time.perf_counter()
        
        def _encode():
            if getattr(self.encoder, 'requires_lock', True):
                with self._encoder_lock:
                    return self.encoder.encode(query)
            return self.encoder.encode(query)
            
        query_embedding = self._read_pool.submit(_encode).result()
        
        with self.rwlock.read_lock():
            if self.index.total_vectors == 0:
                return None
            results = self.index.search(np.array([query_embedding]), top_k=5)
            
            # If we have a reranker, use it
            if self.reranker is not None and len(results[0]) > 0:
                candidates = []
                for score, route_name in results[0]:
                    if route_name in self._route_map:
                        candidates.append((score, self._route_map[route_name]))
                
                if not candidates:
                    return None
                    
                # Evaluate with reranker
                best_route = self.reranker.rerank(query, candidates)
                self.metrics.inference_latency_seconds.observe(time.perf_counter() - start_time)
                return best_route
            
            # Standard embedding routing with margin gating
            best_route = None
            best_score = -1.0
            second_best_score = -1.0
            
            for score, route_name in results[0]:
                if route_name not in self._route_map:
                    continue
                route = self._route_map[route_name]
                if score >= route.threshold:
                    if score > best_score:
                        if best_route is not None and route.name != best_route.name:
                            second_best_score = best_score
                        best_score = score
                        best_route = route
                    elif score > second_best_score and (best_route is None or route.name != best_route.name):
                        second_best_score = score
                        
            if best_route is not None:
                # Apply margin gating
                if second_best_score != -1.0 and (best_score - second_best_score) < self.margin:
                    best_route = None
                    
            self.metrics.inference_latency_seconds.observe(time.perf_counter() - start_time)
            return best_route

    async def aquery(self, query: str) -> Optional[Route]:
        if self._batch_queue is None:
            raise RuntimeError("Router must be started with `await router.start()` before calling aquery.")
        if self._worker_task is None or self._worker_task.done():
            if self._worker_task and self._worker_task.done():
                try:
                    exc = self._worker_task.exception()
                    if exc:
                        raise exc
                except asyncio.CancelledError:
                    pass
            raise RuntimeError("Router worker has crashed or stopped.")
            
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        try:
            self._batch_queue.put_nowait((query, future))
            self.metrics.queue_depth.inc()
        except asyncio.QueueFull:
            raise RouterOverloadedError("Router queue is full (max 10000). Shedding load.")
        except Exception as e:
            future.set_exception(e)
            
        start_time = time.perf_counter()
        try:
            done, _ = await asyncio.wait(
                [future, self._worker_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            if future in done:
                return future.result()
            else:
                raise RuntimeError("Router worker stopped before completing this query.")
        finally:
            self.metrics.inference_latency_seconds.observe(time.perf_counter() - start_time)

    async def _batch_worker(self):
        try:
            while True:
                batch = []
                try:
                    item = await self._batch_queue.get()
                    self.metrics.queue_depth.dec()
                    batch.append(item)
                    
                    while len(batch) < self.batch_size:
                        try:
                            item = self._batch_queue.get_nowait()
                            self.metrics.queue_depth.dec()
                            batch.append(item)
                        except asyncio.QueueEmpty:
                            break
                except asyncio.CancelledError:
                    for _, future in batch:
                        if not future.done():
                            future.set_exception(asyncio.CancelledError())
                    for _ in batch:
                        self._batch_queue.task_done()
                    break
                except Exception as e:
                    for _, future in batch:
                        if not future.done():
                            future.set_exception(e)
                    for _ in batch:
                        self._batch_queue.task_done()
                    continue

                if not batch:
                    continue

                self.metrics.batch_size.observe(len(batch))

                queries = [q for q, _ in batch]
                futures = [f for _, f in batch]

                try:
                    async def _dispatch_batch(batch):
                        qs = [q for q, f in batch]
                        fs = [f for q, f in batch]
                        try:
                            def _resolve_task(query_strings):
                                # Encode
                                if getattr(self.encoder, 'requires_lock', True):
                                    with self._encoder_lock:
                                        query_embeddings = self.encoder.encode_batch(query_strings)
                                else:
                                    query_embeddings = self.encoder.encode_batch(query_strings)
                                    
                                # Resolve
                                with self.rwlock.read_lock():
                                    if self.index.total_vectors == 0:
                                        return [None] * len(query_strings)
                                        
                                    search_results = self.index.search(query_embeddings, top_k=5)
                                    results = []
                                    for i in range(len(query_strings)):
                                        q_text = query_strings[i]
                                        q_results = search_results[i]
                                        
                                        if self.reranker is not None and len(q_results) > 0:
                                            candidates = []
                                            for score, route_name in q_results:
                                                if route_name in self._route_map:
                                                    candidates.append((score, self._route_map[route_name]))
                                            if candidates:
                                                best_route = self.reranker.rerank(q_text, candidates)
                                                results.append(best_route)
                                                continue
                                                
                                        best_route = None
                                        best_score = -1.0
                                        
                                        for score, route_name in q_results:
                                            if route_name not in self._route_map:
                                                continue
                                            route = self._route_map[route_name]
                                            if score >= route.threshold:
                                                if score > best_score:
                                                    best_score = score
                                                    best_route = route
                                        results.append(best_route)
                                return results

                            def _dispatch_and_set(query_strings, futures, event_loop):
                                try:
                                    res = _resolve_task(query_strings)
                                    for f, r in zip(futures, res):
                                        if not f.done():
                                            event_loop.call_soon_threadsafe(f.set_result, r)
                                except Exception as e:
                                    for f in futures:
                                        if not f.done():
                                            event_loop.call_soon_threadsafe(f.set_exception, e)

                            loop = asyncio.get_running_loop()
                            loop.run_in_executor(self._read_pool, _dispatch_and_set, qs, fs, loop)
                            
                        except Exception as e:
                            for f in fs:
                                if not f.done():
                                    f.set_exception(e)
                        finally:
                            for _ in qs:
                                self._batch_queue.task_done()

                    asyncio.create_task(_dispatch_batch(batch))
                except Exception as e:
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
                    for _ in queries:
                        self._batch_queue.task_done()
        finally:
            while not self._batch_queue.empty():
                try:
                    _, future = self._batch_queue.get_nowait()
                    self.metrics.queue_depth.dec()
                    if not future.done():
                        future.set_exception(asyncio.CancelledError("Router worker shutting down."))
                except asyncio.QueueEmpty:
                    break

    def update_threshold(self, route_name: str, threshold: float, _broadcast: bool = True):
        with self._route_map_lock:
            if route_name in self._route_map:
                self._route_map[route_name].threshold = threshold
                self._storage_queue.put(("update_threshold", (route_name, threshold)))

        if _broadcast and self.sync_manager:
            self.sync_manager.broadcast("update_threshold", {"route_name": route_name, "threshold": threshold})

    def fit_thresholds(self, samples: list[str], labels: list[str]):
        if not samples:
            return
        if len(samples) != len(labels):
            raise ValueError("samples and labels lists must have the exact same length.")
            
        query_embeddings = self.encoder.encode_batch(samples)
        
        with self.rwlock.read_lock():
            if self.index.total_vectors == 0:
                return
                
        with self._route_map_lock:
            if not self._route_map:
                return
            route_map_snapshot = dict(self._route_map)

        # Evaluate all samples against the index
        search_results = self.index.search(query_embeddings, top_k=50)
        
        labels_arr = np.array(labels)
        
        thresholds_to_test = np.arange(-1.0, 1.05, 0.05)
        new_thresholds = {}
        
        for route_name, route in route_map_snapshot.items():
            best_f1 = -1.0
            best_t = route.threshold
            
            y_true = (labels_arr == route_name).astype(int)
            
            for t in thresholds_to_test:
                y_pred = []
                for i in range(len(samples)):
                    b_route = ""
                    b_score = -1.0
                    sb_score = -1.0
                    for score, r_name in search_results[i]:
                        if r_name not in route_map_snapshot:
                            continue
                        test_threshold = t if r_name == route_name else route_map_snapshot[r_name].threshold
                        if score >= test_threshold:
                            if score > b_score:
                                if b_route and r_name != b_route:
                                    sb_score = b_score
                                b_score = score
                                b_route = r_name
                            elif score > sb_score and (not b_route or r_name != b_route):
                                sb_score = score
                    if b_route != "":
                        if sb_score != -1.0 and (b_score - sb_score) < self.margin:
                            b_route = ""
                    y_pred.append(1 if b_route == route_name else 0)
                    
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                if f1 >= best_f1:
                    best_f1 = f1
                    best_t = t
            new_thresholds[route_name] = round(float(best_t), 4)
            
        for route_name, t in new_thresholds.items():
            self.update_threshold(route_name, t)
