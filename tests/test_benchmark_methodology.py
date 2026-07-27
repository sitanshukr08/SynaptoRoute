import asyncio

import numpy as np
import pytest

from benchmarks.bench_local_index import run_benchmark
from benchmarks.eval_accuracy import (
    evaluate_semantic_router_public_api,
    evaluate_top_k_synaptoroute,
    unique_route_names,
)
from benchmarks.eval_latency import measure_latency


class RankingEncoder:
    def encode(self, query):
        return np.array([float(query)], dtype=np.float32)


class RankingIndex:
    total_vectors = 8

    def search(self, query_embeddings, top_k=1):
        query_id = int(query_embeddings[0][0])
        rankings = {
            1: [
                (0.99, "alpha"),
                (0.98, "alpha"),
                (0.97, "beta"),
                (0.96, "gamma"),
                (0.95, "delta"),
                (0.94, "epsilon"),
            ],
            2: [
                (0.99, "gamma"),
                (0.98, "beta"),
                (0.97, "alpha"),
                (0.96, "delta"),
                (0.95, "epsilon"),
            ],
        }
        return [rankings[query_id][:top_k]]


class RankingRouter:
    encoder = RankingEncoder()
    index = RankingIndex()


class Result:
    def __init__(self, name):
        self.name = name


def test_unique_route_names_collapses_utterance_duplicates():
    candidates = [(0.9, "alpha"), (0.8, "alpha"), (0.7, "beta")]
    assert unique_route_names(candidates) == ["alpha", "beta"]


def test_top_k_accuracy_uses_unique_routes_and_excludes_ood():
    metrics = evaluate_top_k_synaptoroute(
        RankingRouter(),
        queries=["1", "2", "1"],
        expected=["beta", "alpha", None],
    )

    assert metrics == {
        "top_1": 0.0,
        "top_3": 1.0,
        "top_5": 1.0,
    }


def test_semantic_router_does_not_fabricate_top_k_metrics():
    def layer(query):
        return Result("alpha" if query == "known" else "beta")

    metrics = evaluate_semantic_router_public_api(
        layer,
        queries=["known", "other", "ood"],
        expected=["alpha", "alpha", None],
    )

    assert metrics["top_1"] == 0.5
    assert metrics["top_3"] is None
    assert metrics["top_5"] is None


@pytest.mark.asyncio
async def test_latency_measurement_executes_each_query_once_and_bounds_concurrency():
    calls = []
    active = 0
    peak_active = 0
    lock = asyncio.Lock()

    async def predict(query):
        nonlocal active, peak_active
        calls.append(query)
        async with lock:
            active += 1
            peak_active = max(peak_active, active)
        await asyncio.sleep(0.001)
        async with lock:
            active -= 1

    queries = [str(index) for index in range(12)]
    measurement = await measure_latency(predict, queries, max_concurrency=3)

    assert calls == queries
    assert peak_active == 3
    assert len(measurement.latencies_seconds) == len(queries)
    assert measurement.wall_seconds > 0
    assert measurement.throughput_qps > 0


def test_local_smoke_benchmark_is_deterministic_and_structural_only():
    first = run_benchmark(route_count=20, query_count=10, dim=8, seed=7)
    second = run_benchmark(route_count=20, query_count=10, dim=8, seed=7)

    assert first["dataset"] == second["dataset"]
    assert first["dataset"]["semantic_quality_eligible"] is False
    assert first["metrics"]["top1_identity_accuracy"] == 1.0
    assert first["metrics"]["deletion_visible"] is True
