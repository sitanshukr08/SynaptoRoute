import numpy as np
import threading
from typing import List, Tuple

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class NumpyIndex:
    """
    O(1) Lazy Memory Slicing dense numpy index.
    Used as the default engine if FAISS is not installed.
    """
    def __init__(self, dim: int, max_capacity: int = 50000):
        self.dim = dim
        self.lock = threading.Lock()
        self.embeddings = np.zeros((max_capacity, dim), dtype=np.float32)
        self.tombstones = set()
        self._tombstone_array = np.array([], dtype=int)
        self._id_to_route = {}
        self._route_to_ids = {}
        self._next_id = 0
        self.max_capacity = max_capacity
        self.ntotal = 0
        
    def _add_unlocked(self, embeddings: np.ndarray, route_name: str):
        num_embs = embeddings.shape[0]
        if self._next_id + num_embs > self.max_capacity:
            raise ValueError("ID_OVERFLOW")
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        self.embeddings[self._next_id:self._next_id + num_embs] = embeddings
        ids = list(range(self._next_id, self._next_id + num_embs))
        
        if route_name not in self._route_to_ids:
            self._route_to_ids[route_name] = []
        self._route_to_ids[route_name].extend(ids)
        for i in ids:
            self._id_to_route[i] = route_name
            
        self._next_id += num_embs
        self.ntotal += num_embs

    def add(self, embeddings: np.ndarray, route_name: str):
        with self.lock:
            self._add_unlocked(embeddings, route_name)
            num_embs = embeddings.shape[0]
            if self._next_id + num_embs > self.max_capacity:
                raise ValueError("Capacity exceeded")
            
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
                
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
            
            self.embeddings[self._next_id:self._next_id + num_embs] = embeddings
            ids = list(range(self._next_id, self._next_id + num_embs))
            
            if route_name not in self._route_to_ids:
                self._route_to_ids[route_name] = []
            self._route_to_ids[route_name].extend(ids)
            for i in ids:
                self._id_to_route[i] = route_name
                
            self._next_id += num_embs
            self.ntotal += num_embs

    def delete(self, route_name: str):
        with self.lock:
            if route_name in self._route_to_ids:
                ids = self._route_to_ids[route_name]
                self.tombstones.update(ids)
                self._tombstone_array = np.array(list(self.tombstones), dtype=int)
                for i in ids:
                    self._id_to_route.pop(i, None)
                del self._route_to_ids[route_name]

    def search(self, query_embeddings: np.ndarray, top_k: int = 1) -> List[List[Tuple[float, str]]]:
        with self.lock:
            if self.ntotal == 0 or self._next_id == 0:
                return [[] for _ in range(query_embeddings.shape[0])]
                
            if query_embeddings.dtype != np.float32:
                query_embeddings = query_embeddings.astype(np.float32)
                
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query_embeddings = query_embeddings / norms
            
            valid_mask = np.ones(self._next_id, dtype=bool)
            if self._tombstone_array.size > 0:
                valid_mask[self._tombstone_array] = False
                
            if not np.any(valid_mask):
                return [[] for _ in range(query_embeddings.shape[0])]
                
            scores = np.dot(query_embeddings, self.embeddings[:self._next_id].T)
            
            results = []
            for i in range(scores.shape[0]):
                valid_scores = scores[i][valid_mask]
                valid_indices = np.arange(self._next_id)[valid_mask]
                
                # Sort descending
                num_results = min(top_k, len(valid_scores))
                if num_results == 0:
                    results.append([])
                    continue
                    
                if len(valid_scores) > num_results:
                    sort_idx = np.argpartition(valid_scores, -num_results)[-num_results:]
                    sort_idx = sort_idx[np.argsort(valid_scores[sort_idx])[::-1]]
                else:
                    sort_idx = np.argsort(valid_scores)[::-1]
                
                query_results = []
                for idx in sort_idx:
                    real_idx = valid_indices[idx]
                    route_name = self._id_to_route[real_idx]
                    query_results.append((float(valid_scores[idx]), route_name))
                results.append(query_results)
            return results

    @property
    def total_vectors(self) -> int:
        return self.ntotal - len(self.tombstones)

    def rebuild(self, route_map: dict, embeddings_map: dict):
        with self.lock:
            self.embeddings = np.zeros((self.max_capacity, self.dim), dtype=np.float32)
            self._route_to_ids = {}
            self._id_to_route = {}
            self._next_id = 0
            self.tombstones.clear()
            self._tombstone_array = np.array([], dtype=int)
            self.ntotal = 0
            
            for route_name, route in route_map.items():
                if route_name not in embeddings_map:
                    continue
                embs_data = embeddings_map[route_name]
                if not embs_data:
                    continue
                valid_embs = []
                for e_bytes in embs_data:
                    if e_bytes is not None:
                        valid_embs.append(np.frombuffer(e_bytes, dtype=np.float32))
                if not valid_embs:
                    continue
                embeddings = np.stack(valid_embs)
                self._add_unlocked(embeddings, route_name)

def get_index(dim: int, max_capacity: int = 50000):
    if HAS_FAISS:
        return FaissIndex(dim)
    else:
        return NumpyIndex(dim, max_capacity)

class FaissIndex:
    """
    A FAISS-based vector index utilizing HNSW for sub-linear search latency.
    Employs a Tombstone architecture for O(1) instantaneous deletions.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.lock = threading.Lock()
        
        # Inner Product (Cosine Similarity for normalized embeddings)
        base_index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(base_index)
        
        self.tombstones = set()
        
        # Bidirectional mapping
        self._id_to_route = {}
        self._route_to_ids = {}
        self._next_id = 0

    def add(self, embeddings: np.ndarray, route_name: str):
        with self.lock:
            num_embs = embeddings.shape[0]
            ids = np.arange(self._next_id, self._next_id + num_embs, dtype=np.int64)
            
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
                
            faiss.normalize_L2(embeddings)
            self.index.add_with_ids(embeddings, ids)
            
            if route_name not in self._route_to_ids:
                self._route_to_ids[route_name] = []
                
            self._route_to_ids[route_name].extend(ids.tolist())
            for i in ids:
                self._id_to_route[int(i)] = route_name
                
            self._next_id += num_embs

    def delete(self, route_name: str):
        with self.lock:
            if route_name in self._route_to_ids:
                ids = self._route_to_ids[route_name]
                self.tombstones.update(ids)
                for i in ids:
                    self._id_to_route.pop(i, None)
                del self._route_to_ids[route_name]

    def search(self, query_embeddings: np.ndarray, top_k: int = 1) -> List[List[Tuple[float, str]]]:
        with self.lock:
            # Overfetch to account for tombstones 
            search_k = min(self.index.ntotal, max(top_k + len(self.tombstones) * 2, 2048))
            
            if search_k == 0:
                return [[] for _ in range(query_embeddings.shape[0])]
                
            if query_embeddings.dtype != np.float32:
                query_embeddings = query_embeddings.astype(np.float32)
                
            faiss.normalize_L2(query_embeddings)
            distances, indices = self.index.search(query_embeddings, search_k)
            
            results = []
            for i in range(query_embeddings.shape[0]):
                query_results = []
                for j in range(search_k):
                    idx = int(indices[i][j])
                    if idx != -1 and idx not in self.tombstones:
                        route_name = self._id_to_route[idx]
                        query_results.append((float(distances[i][j]), route_name))
                        if len(query_results) == top_k:
                            break
                results.append(query_results)
                
            return results

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal - len(self.tombstones)

    def rebuild(self, route_map: dict, embeddings_map: dict):
        """Garbage Collection: Completely reconstructs the HNSW index to flush dead vectors."""
        with self.lock:
            # Create a brand new index
            base_index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            new_index = faiss.IndexIDMap(base_index)
            
            new_route_to_ids = {}
            new_id_to_route = {}
            next_id = 0
            
            for route_name, route in route_map.items():
                if route_name not in embeddings_map:
                    continue
                
                embs_data = embeddings_map[route_name]
                if not embs_data:
                    continue
                    
                # Collect embeddings for this route
                valid_embs = []
                for e_bytes in embs_data:
                    if e_bytes is not None:
                        valid_embs.append(np.frombuffer(e_bytes, dtype=np.float32))
                        
                if not valid_embs:
                    continue
                    
                embeddings = np.stack(valid_embs)
                num_embs = embeddings.shape[0]
                ids = np.arange(next_id, next_id + num_embs, dtype=np.int64)
                
                faiss.normalize_L2(embeddings)
                new_index.add_with_ids(embeddings, ids)
                
                new_route_to_ids[route_name] = ids.tolist()
                for i in ids:
                    new_id_to_route[int(i)] = route_name
                    
                next_id += num_embs
                
            # Atomic swap
            self.index = new_index
            self._route_to_ids = new_route_to_ids
            self._id_to_route = new_id_to_route
            self._next_id = next_id
            self.tombstones.clear()
