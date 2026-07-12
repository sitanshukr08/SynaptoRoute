"""Measure comparable end-to-end routing latency under bounded concurrency."""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LatencyMeasurement:
    latencies_seconds: list[float]
    wall_seconds: float

    @property
    def throughput_qps(self) -> float:
        return len(self.latencies_seconds) / self.wall_seconds if self.wall_seconds > 0 else 0.0


async def measure_latency(
    predict_fn: Callable[[str], Awaitable[Any]],
    queries: Sequence[str],
    *,
    max_concurrency: int,
) -> LatencyMeasurement:
    """Execute every query exactly once and include harness queueing latency."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    if not queries:
        raise ValueError("queries must not be empty")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def timed_query(query: str) -> float:
        start = time.perf_counter()
        async with semaphore:
            await predict_fn(query)
        return time.perf_counter() - start

    wall_start = time.perf_counter()
    latencies = await asyncio.gather(*(timed_query(query) for query in queries))
    wall_seconds = time.perf_counter() - wall_start
    return LatencyMeasurement(latencies_seconds=list(latencies), wall_seconds=wall_seconds)


def percentile_ms(samples_seconds: Sequence[float], percentile: float) -> float:
    return float(np.percentile(samples_seconds, percentile) * 1000.0)


def print_measurement(name: str, measurement: LatencyMeasurement) -> None:
    samples = measurement.latencies_seconds
    print(
        f"[{name}] P50: {percentile_ms(samples, 50):.2f}ms "
        f"| P90: {percentile_ms(samples, 90):.2f}ms "
        f"| P95: {percentile_ms(samples, 95):.2f}ms "
        f"| P99: {percentile_ms(samples, 99):.2f}ms "
        f"| Max: {max(samples) * 1000.0:.2f}ms "
        f"| Wall: {measurement.wall_seconds * 1000.0:.2f}ms "
        f"| Throughput: {measurement.throughput_qps:.2f} qps"
    )


async def run_latency_evaluation(
    model_name: str,
    load_profiles: Sequence[int],
    max_concurrency: int,
    warmup_count: int,
) -> None:
    from stats_utils import calculate_statistics, print_statistics_report
    from utils import init_semantic_router, init_synaptoroute, load_datasets

    print(f"=== Running Latency Evaluation (Model: {model_name}) ===")
    dataset_version, routes_data, test_queries = load_datasets()
    query_texts = [query["query"] for query in test_queries]
    if not query_texts:
        raise RuntimeError("No benchmark queries were loaded")

    print(
        f"Dataset version={dataset_version} routes={len(routes_data)} "
        f"queries={len(query_texts)} max_concurrency={max_concurrency}"
    )
    router = init_synaptoroute(routes_data, model_name)
    layer = init_semantic_router(routes_data, model_name)
    await router.start()

    async def baseline_predict(query: str) -> Any:
        return await asyncio.to_thread(layer, query)

    try:
        print(f"Warming up with {warmup_count} queries per system...")
        warmup_queries = (query_texts * (warmup_count // len(query_texts) + 1))[:warmup_count]
        for query in warmup_queries:
            await router.aquery(query)
            await baseline_predict(query)

        for count in load_profiles:
            if count < 1:
                raise ValueError("load profiles must be positive")
            print(f"\n--- Burst Load: {count} Queries ---")
            workload = (query_texts * (count // len(query_texts) + 1))[:count]
            offered_concurrency = min(count, max_concurrency)

            baseline = await measure_latency(
                baseline_predict,
                workload,
                max_concurrency=offered_concurrency,
            )
            synaptoroute = await measure_latency(
                router.aquery,
                workload,
                max_concurrency=offered_concurrency,
            )

            print_measurement("Semantic Router", baseline)
            print_measurement("SynaptoRoute", synaptoroute)

            statistics = calculate_statistics(
                synaptoroute.latencies_seconds,
                baseline.latencies_seconds,
            )
            print_statistics_report(
                statistics,
                name_a="SynaptoRoute",
                name_b="Semantic Router",
            )
    finally:
        await router.stop()


def parse_load_profiles(value: str) -> list[int]:
    profiles = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not profiles:
        raise argparse.ArgumentTypeError("at least one load profile is required")
    if any(profile < 1 for profile in profiles):
        raise argparse.ArgumentTypeError("load profiles must be positive")
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--loads", type=parse_load_profiles, default=[1, 100, 1000])
    parser.add_argument("--max-concurrency", type=int, default=100)
    parser.add_argument("--warmup-count", type=int, default=20)
    args = parser.parse_args()

    asyncio.run(
        run_latency_evaluation(
            model_name=args.model,
            load_profiles=args.loads,
            max_concurrency=args.max_concurrency,
            warmup_count=args.warmup_count,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
