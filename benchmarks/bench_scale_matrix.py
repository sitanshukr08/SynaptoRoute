"""Measure exact and HNSW index scaling with identical precomputed vectors."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from synaptoroute.index import get_index


def _percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": max(values),
    }


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def _index_parameters(index, engine: str, route_count: int) -> dict:
    common = {
        "construction_add_calls": route_count,
        "vectors_per_add_call": 1,
        "metric": "normalized_inner_product",
    }
    if engine == "numpy":
        return {
            **common,
            "implementation": "numpy_exact",
            "max_capacity": index.max_capacity,
        }

    import faiss

    base_index = faiss.downcast_index(index.index.index)
    hnsw = getattr(base_index, "hnsw")
    return {
        **common,
        "implementation": "faiss_hnsw",
        "faiss_version": faiss.__version__,
        "omp_threads": faiss.omp_get_max_threads(),
        "hnsw_m": hnsw.nb_neighbors(1),
        "hnsw_ef_construction": hnsw.efConstruction,
        "hnsw_ef_search": hnsw.efSearch,
        "search_candidate_floor": index.SEARCH_CANDIDATE_FLOOR,
    }


def run_benchmark(
    *,
    engine: str,
    route_count: int,
    query_count: int,
    dim: int,
    seed: int,
) -> dict:
    if route_count < 2 or query_count < 1 or dim < 2:
        raise ValueError("route_count, query_count, and dim must be positive")
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(route_count, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = get_index(dim=dim, max_capacity=route_count, engine=engine)
    index_parameters = _index_parameters(index, engine, route_count)

    rss_before = _rss_mb()
    build_started = time.perf_counter()
    for route_index, vector in enumerate(vectors):
        index.add(vector.reshape(1, -1), f"route_{route_index}")
    build_seconds = time.perf_counter() - build_started
    rss_after = _rss_mb()

    selected = rng.integers(0, route_count, size=query_count)
    latencies_ms: list[float] = []
    correct = 0
    query_started = time.perf_counter()
    for route_index in selected:
        started_ns = time.perf_counter_ns()
        result = index.search(vectors[route_index].reshape(1, -1), top_k=1)[0]
        latencies_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
        correct += int(bool(result) and result[0][1] == f"route_{route_index}")
    query_seconds = time.perf_counter() - query_started

    return {
        "schema_version": 2,
        "benchmark": "precomputed_vector_scale",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "configuration": {
            "engine": engine,
            "route_count": route_count,
            "query_count": query_count,
            "dimension": dim,
            "seed": seed,
            "index_parameters": index_parameters,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "numpy": np.__version__,
            "faiss": index_parameters.get("faiss_version"),
        },
        "metrics": {
            "query_count": query_count,
            "correct_count": correct,
            "incorrect_count": query_count - correct,
            "top1_identity_accuracy": correct / query_count,
            "build_seconds": build_seconds,
            "query_seconds": query_seconds,
            "throughput_qps": query_count / query_seconds,
            "latency": _percentiles(latencies_ms),
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "rss_delta_mb": (
                rss_after - rss_before
                if rss_before is not None and rss_after is not None
                else None
            ),
        },
        "notes": [
            "Precomputed synthetic vectors isolate index structure from encoder cost.",
            "Identity accuracy is structural and is not semantic-quality evidence.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("numpy", "faiss"), required=True)
    parser.add_argument("--routes", type=int, required=True)
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_benchmark(
        engine=args.engine,
        route_count=args.routes,
        query_count=args.queries,
        dim=args.dim,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
