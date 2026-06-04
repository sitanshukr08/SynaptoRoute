import numpy as np

from synaptoroute.index import NumpyIndex


def test_numpy_index_add_inserts_each_vector_once():
    index = NumpyIndex(dim=2, max_capacity=10)

    index.add(np.array([[1.0, 0.0]], dtype=np.float32), "route_a")

    assert index.ntotal == 1
    assert index.total_vectors == 1
    assert index._next_id == 1
    assert index._route_to_ids["route_a"] == [0]


def test_numpy_index_capacity_counts_actual_vectors():
    index = NumpyIndex(dim=2, max_capacity=2)

    index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), "route_a")

    assert index.total_vectors == 2
    assert index.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=2)[0]
