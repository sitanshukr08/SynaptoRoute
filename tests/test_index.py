import pytest
import numpy as np
from synaptoroute.index import NumpyIndex, FaissIndex, get_index, HAS_FAISS

@pytest.fixture
def dummy_embeddings():
    return np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [-0.1, -0.2, -0.3, -0.4]
    ], dtype=np.float32)

def test_numpy_index_add_inserts_correct_count(dummy_embeddings):
    index = NumpyIndex(dim=4, max_capacity=10)
    index.add(dummy_embeddings, "route_a")
    assert index._next_id == 3
    assert index.ntotal == 3
    assert index.total_vectors == 3

def test_numpy_index_search(dummy_embeddings):
    index = NumpyIndex(dim=4, max_capacity=10)
    index.add(dummy_embeddings, "route_a")
    results = index.search(np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0][1] == "route_a"

def test_numpy_index_delete_tombstones_route(dummy_embeddings):
    index = NumpyIndex(dim=4, max_capacity=10)
    index.add(dummy_embeddings, "route_a")
    index.delete("route_a")
    assert len(index.tombstones) == 3
    assert index.total_vectors == 0
    results = index.search(np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert len(results[0]) == 0

def test_numpy_index_capacity_overflow(dummy_embeddings):
    index = NumpyIndex(dim=4, max_capacity=2)
    with pytest.raises(ValueError, match="Capacity exceeded"):
        index.add(dummy_embeddings, "route_a")

def test_numpy_index_rebuild(dummy_embeddings):
    index = NumpyIndex(dim=4, max_capacity=10)
    index.add(dummy_embeddings, "route_a")
    index.delete("route_a")
    
    route_map = {"route_b": None}
    embeddings_map = {"route_b": [dummy_embeddings[0].tobytes()]}
    
    index.rebuild(route_map, embeddings_map)
    assert len(index.tombstones) == 0
    assert index.total_vectors == 1
    assert index._next_id == 1

@pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed")
def test_faiss_index_add_and_search(dummy_embeddings):
    index = FaissIndex(dim=4, max_capacity=10)
    index.add(dummy_embeddings, "route_a")
    assert index.total_vectors == 3
    results = index.search(np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0][1] == "route_a"

@pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed")
def test_faiss_index_delete(dummy_embeddings):
    index = FaissIndex(dim=4)
    index.add(dummy_embeddings, "route_a")
    index.delete("route_a")
    assert index.total_vectors == 0
    results = index.search(np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert len(results[0]) == 0
