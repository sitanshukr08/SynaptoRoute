import threading
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from typing import Optional
import asyncio

from synaptoroute.models import Route
from synaptoroute.encoder import Encoder
from synaptoroute.storage import BaseStorage
from synaptoroute.exceptions import RouteNotFoundError, RouterOverloadedError

class AdaptiveRouter:
    def __init__(self, encoder: Encoder, storage: BaseStorage):
        self.encoder = encoder
        self.storage = storage
        self.lock = threading.Lock()
        
        self._vectors = None
        self._uncompiled_vectors = []
        self._meta = []
        self._route_map = {}
        
        self._batch_queue = None
        self._worker_task = None
        self.batch_size = 32
        self.batch_timeout = 0.005
        
        self._load_routes()
        
    async def start(self):
        self._batch_queue = asyncio.Queue(maxsize=10000)
        self._worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def _load_routes(self):
        routes = self.storage.load_all_routes()
        
        for route in routes:
            self._route_map[route.name] = route
            if route.utterances:
                embeddings = self.encoder.encode_batch(route.utterances)
                self._uncompiled_vectors.append(embeddings)
                self._meta.extend([route] * len(route.utterances))

    def _rebuild_memory_locked(self):
        self._vectors = None
        self._uncompiled_vectors = []
        self._meta = []
        self._route_map = {}
        self._load_routes()

    def add_route(self, route: Route):
        with self.lock:
            is_overwrite = route.name in self._route_map
            self.storage.save_route(route)
            
            if is_overwrite:
                # O(1) Memory Replacement: Filter out the old route's vectors without full DB re-encoding
                self._compile_vectors_locked()
                if self._vectors is not None:
                    mask = [r.name != route.name for r in self._meta]
                    if len(mask) > 0:
                        self._vectors = self._vectors[mask]
                        self._meta = [r for r, keep in zip(self._meta, mask) if keep]
            
            self._route_map[route.name] = route
            if route.utterances:
                embeddings = self.encoder.encode_batch(route.utterances)
                self._uncompiled_vectors.append(embeddings)
                self._meta.extend([route] * len(route.utterances))

    def add_utterance(self, route_name: str, utterance: str):
        with self.lock:
            if route_name not in self._route_map:
                raise RouteNotFoundError(f"Route '{route_name}' not found.")
                
        # Reshape to 2D to ensure safe vstack compatibility
        embedding = self.encoder.encode(utterance).reshape(1, -1)
        
        with self.lock:
            if route_name not in self._route_map:
                raise RouteNotFoundError(f"Route '{route_name}' was deleted during encoding.")
            self.storage.add_utterance(route_name, utterance)
            self._uncompiled_vectors.append(embedding)
            route = self._route_map[route_name]
            route.utterances.append(utterance)
            self._meta.append(route)

    def _compile_vectors_locked(self):
        if self._uncompiled_vectors:
            if self._vectors is not None:
                self._vectors = np.vstack([self._vectors] + self._uncompiled_vectors)
            else:
                self._vectors = np.vstack(self._uncompiled_vectors)
            self._uncompiled_vectors = []

    def __call__(self, query: str) -> Optional[Route]:
        query_embedding = self.encoder.encode(query)
        
        with self.lock:
            self._compile_vectors_locked()
            
            if self._vectors is None or len(self._vectors) == 0:
                return None
                
            similarities = cosine_similarity([query_embedding], self._vectors)[0]
            
            best_route = None
            best_score = -1.0
            
            for score, route in zip(similarities, self._meta):
                if score >= route.threshold and score > best_score:
                    best_score = score
                    best_route = route
                    
            return best_route

    async def aquery(self, query: str) -> Optional[Route]:
        if self._batch_queue is None:
            raise RuntimeError("Router must be started with `await router.start()` before calling aquery.")
            
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        try:
            self._batch_queue.put_nowait((query, future))
        except asyncio.QueueFull:
            raise RouterOverloadedError("Router queue is full (max 10000). Shedding load.")
        return await future

    async def _batch_worker(self):
        try:
            while True:
                batch = []
                try:
                    item = await self._batch_queue.get()
                    batch.append(item)
                    
                    while len(batch) < self.batch_size:
                        try:
                            item = await asyncio.wait_for(self._batch_queue.get(), timeout=self.batch_timeout)
                            batch.append(item)
                        except asyncio.TimeoutError:
                            break
                except asyncio.CancelledError:
                    break
                except Exception:
                    continue

                if not batch:
                    continue

                queries = [q for q, _ in batch]
                futures = [f for _, f in batch]

                try:
                    def process_batch(qs):
                        query_embeddings = self.encoder.encode_batch(qs)
                        with self.lock:
                            self._compile_vectors_locked()
                            if self._vectors is None or len(self._vectors) == 0:
                                return [None] * len(qs)
                            similarities = cosine_similarity(query_embeddings, self._vectors)
                            results = []
                            for i in range(len(qs)):
                                best_route = None
                                best_score = -1.0
                                for score, route in zip(similarities[i], self._meta):
                                    if score >= route.threshold and score > best_score:
                                        best_score = score
                                        best_route = route
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
                    # Prevent async deadlocks by marking tasks done
                    for _ in batch:
                        self._batch_queue.task_done()
        finally:
            while not self._batch_queue.empty():
                try:
                    _, future = self._batch_queue.get_nowait()
                    if not future.done():
                        future.set_exception(asyncio.CancelledError("Router worker shutting down."))
                except asyncio.QueueEmpty:
                    break

    def fit_thresholds(self, samples: list[str], labels: list[str]):
        if not samples:
            return
        if len(samples) != len(labels):
            raise ValueError("samples and labels lists must have the exact same length.")
            
        query_embeddings = self.encoder.encode_batch(samples)
        
        with self.lock:
            self._compile_vectors_locked()
            
            if self._vectors is None or len(self._vectors) == 0:
                return
                
            if not self._route_map:
                return

            vectors_snapshot = self._vectors
            meta_snapshot = list(self._meta)
            route_map_snapshot = dict(self._route_map)

        similarities = cosine_similarity(query_embeddings, vectors_snapshot)
        
        best_routes = []
        best_scores = []
        
        for i in range(len(samples)):
            best_idx = np.argmax(similarities[i])
            best_score = similarities[i][best_idx]
            best_route = meta_snapshot[best_idx].name
            best_routes.append(best_route)
            best_scores.append(best_score)
            
        best_routes = np.array(best_routes)
        best_scores = np.array(best_scores)
        labels_arr = np.array(labels)
        
        # Test full cosine similarity range
        thresholds_to_test = np.arange(-1.0, 1.05, 0.05)
        new_thresholds = {}
        
        for route_name, route in route_map_snapshot.items():
            best_f1 = -1.0
            best_t = route.threshold
            
            y_true = (labels_arr == route_name).astype(int)
            
            for t in thresholds_to_test:
                y_pred = ((best_routes == route_name) & (best_scores > t)).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                if f1 >= best_f1:
                    best_f1 = f1
                    best_t = t
            new_thresholds[route_name] = float(best_t)
            
        with self.lock:
            for route_name, t in new_thresholds.items():
                if route_name in self._route_map:
                    route = self._route_map[route_name]
                    old_t = route.threshold
                    try:
                        route.threshold = t
                        self.storage.update_threshold(route_name, t)
                    except Exception as e:
                        route.threshold = old_t
                        raise e

