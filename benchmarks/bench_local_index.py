"""Deterministic structural smoke benchmark for the exact NumPy index.

This benchmark validates the evidence pipeline and basic index behavior. Its
synthetic-vector accuracy is not a semantic quality result and must not be used
as one in documentation or papers.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

from synaptoroute.index import NumpyIndex


def percentile_ms(samples_seconds: list[float], percentile: float) -> float:
    return float(np.percentile(samples_seconds, percentile) * 1000.0)


def run_benchmark(route_count: int, query_count: int, dim: int, seed: int) -> dict:
    if route_count < 2:
        raise ValueError("route_count must be at least 2")
    if query_count < 1:
        raise ValueError("query_count must be positive")
    if dim < 2:
        raise ValueError("dim must be at least 2")

    rng = np.random.default_rng(seed)
    route_vectors = rng.normal(size=(route_count, dim)).astype(np.float32)
    route_vectors /= np.linalg.norm(route_vectors, axis=1, keepdims=True)

    index = NumpyIndex(dim=dim, max_capacity=route_count + 1)
    build_start = time.perf_counter()
    for route_id, vector in enumerate(route_vectors):
        index.add(vector.reshape(1, -1), f"route_{route_id}")
    build_seconds = time.perf_counter() - build_start

    selected_ids = rng.integers(0, route_count, size=query_count)
    query_vectors = route_vectors[selected_ids]

    for vector in query_vectors[: min(20, query_count)]:
        index.search(vector.reshape(1, -1), top_k=1)

    latencies_seconds: list[float] = []
    correct = 0
    wall_start = time.perf_counter()
    for expected_id, vector in zip(selected_ids, query_vectors):
        query_start = time.perf_counter()
        result = index.search(vector.reshape(1, -1), top_k=1)[0]
        latencies_seconds.append(time.perf_counter() - query_start)
        if result and result[0][1] == f"route_{expected_id}":
            correct += 1
    wall_seconds = time.perf_counter() - wall_start

    deleted_route = "route_0"
    index.delete(deleted_route)
    after_delete = index.search(route_vectors[0].reshape(1, -1), top_k=5)[0]
    deletion_visible = all(route_name != deleted_route for _, route_name in after_delete)

    return {
        "benchmark": "local_numpy_index_smoke",
        "status": "structural_only",
        "dataset": {
            "type": "deterministic synthetic normalized vectors",
            "semantic_quality_eligible": False,
            "seed": seed,
            "route_count": route_count,
            "query_count": query_count,
            "dimension": dim,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "numpy": np.__version__,
        },
        "metrics": {
            "top1_identity_accuracy": correct / query_count,
            "build_ms": build_seconds * 1000.0,
            "query_wall_ms": wall_seconds * 1000.0,
            "throughput_qps": query_count / wall_seconds,
            "latency_p50_ms": percentile_ms(latencies_seconds, 50),
            "latency_p95_ms": percentile_ms(latencies_seconds, 95),
            "latency_p99_ms": percentile_ms(latencies_seconds, 99),
            "latency_max_ms": max(latencies_seconds) * 1000.0,
            "deletion_visible": deletion_visible,
        },
        "notes": [
            "The benchmark uses exact self-vector queries and tests structural correctness only.",
            "CI timing is diagnostic and must not be promoted as a paper latency result.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_benchmark(
        route_count=args.routes,
        query_count=args.queries,
        dim=args.dim,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
