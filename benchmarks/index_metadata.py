"""Describe index settings that materially affect benchmark outcomes."""

from __future__ import annotations

from typing import Any

from synaptoroute.index import FaissIndex, NumpyIndex


def describe_index(index: Any) -> dict[str, Any]:
    if isinstance(index, NumpyIndex):
        return {
            "resolved_engine": "numpy",
            "implementation": "numpy_exact",
            "metric": "normalized_inner_product",
            "max_capacity": index.max_capacity,
        }
    if not isinstance(index, FaissIndex):
        raise TypeError(f"unsupported benchmark index: {type(index).__name__}")

    import faiss

    base_index = faiss.downcast_index(index.index.index)
    hnsw = getattr(base_index, "hnsw")
    return {
        "resolved_engine": "faiss",
        "implementation": "faiss_hnsw",
        "metric": "normalized_inner_product",
        "faiss_version": faiss.__version__,
        "omp_threads": faiss.omp_get_max_threads(),
        "hnsw_m": hnsw.nb_neighbors(1),
        "hnsw_ef_construction": hnsw.efConstruction,
        "hnsw_ef_search": hnsw.efSearch,
        "search_candidate_floor": index.SEARCH_CANDIDATE_FLOOR,
    }
