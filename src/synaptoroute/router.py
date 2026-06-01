import threading
import numpy as np
from sklearn.metrics import f1_score
from typing import Optional
import time
import asyncio
import logging

from synaptoroute.metrics import MetricsRegistry
from synaptoroute.models import Route
from synaptoroute.encoder import Encoder
from synaptoroute.storage import BaseStorage
from synaptoroute.sync import BaseSyncManager
from synaptoroute.exceptions import RouteNotFoundError, RouterOverloadedError, RouterCapacityError, SynaptoRouteError
from synaptoroute.profile import OptimizationProfile, get_profile, ProfileType
from synaptoroute.index import get_index

class AdaptiveRouter:
    def __init__(self, encoder: Optional[Encoder] = None, storage: BaseStorage = None, profile: OptimizationProfile = None, max_capacity: int = 50000, metrics: MetricsRegistry = None, sync_manager: Optional[BaseSyncManager] = None, margin: float = 0.0, reranker=None):
        if profile is None:
            profile = get_profile(ProfileType.THROUGHPUT)
            
        if encoder is None:
            from synaptoroute.encoder import FastEmbedEncoder
            encoder = FastEmbedEncoder(threads=profile.threads)
        self.encoder = encoder
        self.storage = storage
        self.lock = threading.Lock()
        self._encoder_lock = threading.Lock()
        self._rebuild_pending = False
        self.metrics = metrics or MetricsRegistry()
        self.sync_manager = sync_manager
        if self.sync_manager:
            self.sync_manager.register(self)
        
        self.max_capacity = max_capacity
        self.margin = margin
        self.reranker = reranker
        
        # Dynamic Index (Numpy default, FAISS HNSW optional)
        self.index = get_index(dim=self.encoder.dim, max_capacity=self.max_capacity)
        self._route_map = {}
        
        self._batch_queue = None
        self._worker_task = None
        self.batch_size = profile.batch_size
        self.batch_timeout = profile.batch_timeout
        
        self._load_routes()
        
    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._batch_queue = asyncio.Queue(maxsize=10000)
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

    def _load_routes(self):
        routes, embeddings_map = self.storage.load_all_routes()
        
        print(f"Loading {len(routes)} routes from storage...")
        t_load = __import__('time').perf_counter()
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
        with self.lock:
            if route_name not in self._route_map:
                return
            
            self.storage.delete_route(route_name)
            self._route_map.pop(route_name)
            
            self.index.delete(route_name)
            
            self.metrics.capacity_usage.set(self.index.total_vectors)

            live_vectors = max(1, self.index.total_vectors)
            tombstone_ratio = len(self.index.tombstones) / live_vectors
            if tombstone_ratio > 0.1 or len(self.index.tombstones) > 1500:
                if not self._rebuild_pending:
                    self._rebuild_pending = True
                    try:
                        if hasattr(self, '_loop') and self._loop and not self._loop.is_closed():
                            self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._rebuild_index()))
                        else:
                            self._rebuild_pending = False
                    except Exception:
                        self._rebuild_pending = False

        if _broadcast and self.sync_manager:
            self.sync_manager.broadcast("delete_route", {"route_name": route_name})

    def add_route(self, route: Route, _broadcast: bool = True, _precomputed_embeddings=None):
        embeddings = _precomputed_embeddings
        num_embs = 0
        if embeddings is None and route.utterances:
            if getattr(self.encoder, 'requires_lock', True):
                with self._encoder_lock:
                    embeddings = self.encoder.encode_batch(route.utterances)
            else:
                embeddings = self.encoder.encode_batch(route.utterances)
            num_embs = len(embeddings)
            
        with self.lock:
            is_overwrite = route.name in self._route_map
            old_route = self._route_map.get(route.name) if is_overwrite else None
            net_increase = num_embs
            if is_overwrite:
                net_increase -= len(self._route_map[route.name].utterances)

            if self.index.total_vectors + net_increase > self.max_capacity:
                raise RouterCapacityError(f"Maximum capacity ({self.max_capacity}) exceeded.")

            self.storage.save_route(route, embeddings)

            if is_overwrite:
                self.index.delete(route.name)

            self._route_map[route.name] = route

            try:
                if num_embs > 0:
                    self.index.add(embeddings, route.name)
            except Exception as e:
                # Rollback: restore old route if index insertion failed
                if is_overwrite and old_route is not None:
                    self._route_map[route.name] = old_route
                    self.storage.save_route(old_route, None)
                    logging.getLogger(__name__).error(
                        f"Index add failed during overwrite of "
                        f"'{route.name}'. Rolled back to previous route. "
                        f"Error: {e}"
                    )
                else:
                    self._route_map.pop(route.name, None)
                raise
            
            self.metrics.capacity_usage.set(self.index.total_vectors)
            
            live_vectors = max(1, self.index.total_vectors)
            tombstone_ratio = len(self.index.tombstones) / live_vectors
            if is_overwrite and (tombstone_ratio > 0.1 or len(self.index.tombstones) > 1500):
                if not self._rebuild_pending:
                    self._rebuild_pending = True
                    try:
                        if hasattr(self, '_loop') and self._loop and not self._loop.is_closed():
                            self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._rebuild_index()))
                        else:
                            self._rebuild_pending = False
                    except Exception:
                        self._rebuild_pending = False

        if _broadcast and self.sync_manager:
            emb_bytes = embeddings.tobytes() if embeddings is not None else None
            self.sync_manager.broadcast("add_route", route.model_dump(mode="json"), embeddings=emb_bytes)

    def add_utterance(self, route_name: str, utterance: str, _broadcast: bool = True, _precomputed_embedding=None):
        with self.lock:
            if route_name not in self._route_map:
                raise RouteNotFoundError(f"Route '{route_name}' not found.")
            if utterance in self._route_map[route_name].utterances:
                return
            if self.index.total_vectors + 1 > self.max_capacity:
                raise RouterCapacityError(f"Maximum capacity ({self.max_capacity}) exceeded.")
                
        if _precomputed_embedding is not None:
            embedding = _precomputed_embedding
        else:
            if getattr(self.encoder, 'requires_lock', True):
                with self._encoder_lock:
                    embedding = self.encoder.encode(utterance)
            else:
                embedding = self.encoder.encode(utterance)
        
        with self.lock:
            if route_name not in self._route_map:
                raise RouteNotFoundError(f"Route '{route_name}' was deleted during encoding.")
            if self.index.total_vectors + 1 > self.max_capacity:
                raise RouterCapacityError(f"Maximum capacity ({self.max_capacity}) exceeded.")

            self.storage.add_utterance(route_name, utterance, embedding)
            
            try:
                self.index.add(np.array([embedding]), route_name)
            except ValueError as e:
                if str(e) == "ID_OVERFLOW":
                    if not self._rebuild_pending:
                        self._rebuild_pending = True
                        try:
                            if hasattr(self, '_loop') and self._loop and not self._loop.is_closed():
                                self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._rebuild_index()))
                            else:
                                self._rebuild_pending = False
                        except Exception:
                            self._rebuild_pending = False
                    raise SynaptoRouteError("Index exhausted, triggered rebuild. Next query will incorporate DB utterance.") from e
                raise
            
            route = self._route_map[route_name]
            # Reassign to trigger pydantic validation
            route.utterances = route.utterances + [utterance]
            
            self.metrics.capacity_usage.set(self.index.total_vectors)

        if _broadcast and self.sync_manager:
            emb_bytes = embedding.tobytes() if embedding is not None else None
            self.sync_manager.broadcast("add_utterance", {"route_name": route_name, "utterance": utterance}, embeddings=emb_bytes)

    async def _rebuild_index(self):
        try:
            routes, embeddings_map = await asyncio.to_thread(self.storage.load_all_routes)
            route_dict = {r.name: r for r in routes}
            await asyncio.to_thread(self.index.rebuild, route_dict, embeddings_map)
        except Exception as e:
            logging.getLogger(__name__).error(f"Rebuild failed: {e}")
            self.metrics.gc_errors.inc()
        finally:
            self._rebuild_pending = False

    def __call__(self, query: str) -> Optional[Route]:
        start_time = time.perf_counter()
        
        if getattr(self.encoder, 'requires_lock', True):
            with self._encoder_lock:
                query_embedding = self.encoder.encode(query)
        else:
            query_embedding = self.encoder.encode(query)
        
        with self.lock:
            if self.index.total_vectors == 0:
                return None
                
        results = self.index.search(np.array([query_embedding]), top_k=5)
        
        with self.lock:
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
                        # Demote current best to second best
                        second_best_score = best_score
                        best_score = score
                        best_route = route
                    elif score > second_best_score:
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
            raise RuntimeError("Router worker has crashed or stopped.")
            
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        try:
            self._batch_queue.put_nowait((query, future))
            self.metrics.queue_depth.inc()
        except asyncio.QueueFull:
            raise RouterOverloadedError("Router queue is full (max 10000). Shedding load.")
            
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
                    
                    timeout_end = time.monotonic() + self.batch_timeout
                    while len(batch) < self.batch_size:
                        try:
                            time_left = timeout_end - time.monotonic()
                            if time_left <= 0:
                                break
                            item = await asyncio.wait_for(self._batch_queue.get(), timeout=time_left)
                            self.metrics.queue_depth.dec()
                            batch.append(item)
                        except asyncio.TimeoutError:
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
                    def process_batch(qs):
                        if getattr(self.encoder, 'requires_lock', True):
                            with self._encoder_lock:
                                query_embeddings = self.encoder.encode_batch(qs)
                        else:
                            query_embeddings = self.encoder.encode_batch(qs)
                            
                        with self.lock:
                            if self.index.total_vectors == 0:
                                return [None] * len(qs)
                                
                        search_results = self.index.search(query_embeddings, top_k=5)
                        
                        results = []
                        with self.lock:
                            for i in range(len(qs)):
                                q_text = qs[i]
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
                                second_best_score = -1.0
                                
                                for score, route_name in q_results:
                                    if route_name not in self._route_map:
                                        continue
                                    route = self._route_map[route_name]
                                    if score >= route.threshold:
                                        if score > best_score:
                                            second_best_score = best_score
                                            best_score = score
                                            best_route = route
                                        elif score > second_best_score:
                                            second_best_score = score
                                            
                                if best_route is not None:
                                    if second_best_score != -1.0 and (best_score - second_best_score) < self.margin:
                                        best_route = None
                                results.append(best_route)
                            return results

                    results = await asyncio.to_thread(process_batch, queries)
                    
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
                finally:
                    for _, future in batch:
                        if not future.done():
                            future.set_exception(asyncio.CancelledError())
                    for _ in batch:
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
        with self.lock:
            if route_name in self._route_map:
                route = self._route_map[route_name]
                old_t = route.threshold
                try:
                    route.threshold = threshold
                    self.storage.update_threshold(route_name, threshold)
                except Exception as e:
                    route.threshold = old_t
                    raise SynaptoRouteError(f"Failed: {e}") from e

        if _broadcast and self.sync_manager:
            self.sync_manager.broadcast("update_threshold", {"route_name": route_name, "threshold": threshold})

    def fit_thresholds(self, samples: list[str], labels: list[str]):
        if not samples:
            return
        if len(samples) != len(labels):
            raise ValueError("samples and labels lists must have the exact same length.")
            
        query_embeddings = self.encoder.encode_batch(samples)
        
        with self.lock:
            if self.index.total_vectors == 0:
                return
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
                    for score, r_name in search_results[i]:
                        test_threshold = t if r_name == route_name else route_map_snapshot[r_name].threshold
                        if score >= test_threshold and score > b_score:
                            b_score = score
                            b_route = r_name
                    y_pred.append(1 if b_route == route_name else 0)
                    
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                if f1 >= best_f1:
                    best_f1 = f1
                    best_t = t
            new_thresholds[route_name] = float(best_t)
            
        for route_name, t in new_thresholds.items():
            self.update_threshold(route_name, t)
