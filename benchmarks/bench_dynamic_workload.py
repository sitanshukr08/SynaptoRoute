"""Run a controlled concurrent query and mutation workload."""

from __future__ import annotations

import argparse
import json
import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.deterministic_encoder import DeterministicHashEncoder
from synaptoroute import AdaptiveRouter, MutationReceipt, Route, SQLiteStorage


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": max(values),
    }


def _snapshot(storage: SQLiteStorage) -> dict[str, tuple[float, tuple[str, ...]]]:
    routes, _ = storage.load_all_routes()
    return {
        route.name: (route.threshold, tuple(sorted(route.utterances)))
        for route in routes
    }


def run_benchmark(
    *,
    database_path: Path,
    duration_seconds: float,
    route_count: int,
    query_workers: int,
    mutation_rate: float,
    dim: int = 32,
) -> dict[str, Any]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if route_count < 2 or query_workers < 1 or mutation_rate <= 0:
        raise ValueError("route_count, query_workers, and mutation_rate must be positive")
    if database_path.exists():
        raise FileExistsError(f"benchmark database already exists: {database_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = DeterministicHashEncoder(dim=dim)
    storage = SQLiteStorage(str(database_path))
    router = AdaptiveRouter(encoder, storage, max_capacity=route_count * 10)
    base_queries = []
    for index in range(route_count):
        utterance = f"base utterance {index}"
        router.add_route(Route(name=f"base_{index}", utterances=[utterance], threshold=0.8))
        base_queries.append((utterance, f"base_{index}"))
    router.durable_barrier(timeout=30.0)

    stop_event = threading.Event()
    measurements_lock = threading.Lock()
    query_latencies_ms: list[float] = []
    query_errors: list[str] = []
    query_correct = 0

    def query_worker(worker_id: int) -> None:
        nonlocal query_correct
        cursor = worker_id
        while not stop_event.is_set():
            query, expected = base_queries[cursor % len(base_queries)]
            cursor += query_workers
            started = time.perf_counter_ns()
            try:
                result = router.match(query)
                latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                with measurements_lock:
                    query_latencies_ms.append(latency_ms)
                    query_correct += result.route_name == expected
            except Exception as error:
                with measurements_lock:
                    query_errors.append(type(error).__name__)

    mutation_receipts: list[MutationReceipt] = []
    mutation_memory_ack_ms: list[float] = []
    visibility_failures = 0
    deletion_visibility_failures = 0
    mutation_errors: list[str] = []
    mutation_count = 0
    workload_started = time.perf_counter()
    deadline = workload_started + duration_seconds
    interval = 1.0 / mutation_rate
    next_mutation = workload_started

    with ThreadPoolExecutor(max_workers=query_workers, thread_name_prefix="dynamic_query") as executor:
        futures = [executor.submit(query_worker, worker_id) for worker_id in range(query_workers)]
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if now < next_mutation:
                time.sleep(min(next_mutation - now, 0.005))
                continue
            cycle, phase = divmod(mutation_count, 4)
            route_name = f"dynamic_{cycle}"
            started = time.perf_counter_ns()
            try:
                if phase == 0:
                    utterance = f"dynamic utterance {cycle}"
                    receipt = router.add_route(
                        Route(name=route_name, utterances=[utterance], threshold=0.8)
                    )
                    if router.match(utterance).route_name != route_name:
                        visibility_failures += 1
                elif phase == 1:
                    utterance = f"dynamic additional {cycle}"
                    receipt = router.add_utterance(route_name, utterance)
                    if router.match(utterance).route_name != route_name:
                        visibility_failures += 1
                elif phase == 2:
                    receipt = router.update_threshold(route_name, 0.9)
                else:
                    utterance = f"dynamic utterance {cycle}"
                    receipt = router.delete_route(route_name)
                    if router.match(utterance).route_name == route_name:
                        deletion_visibility_failures += 1
                mutation_memory_ack_ms.append(
                    (time.perf_counter_ns() - started) / 1_000_000.0
                )
                if receipt is not None:
                    mutation_receipts.append(receipt)
            except Exception as error:
                mutation_errors.append(type(error).__name__)
            mutation_count += 1
            next_mutation += interval

        stop_event.set()
        for future in futures:
            future.result(timeout=10.0)

    barrier_started = time.perf_counter()
    router.durable_barrier(timeout=30.0)
    barrier_ms = (time.perf_counter() - barrier_started) * 1000.0
    workload_wall_seconds = time.perf_counter() - workload_started
    durable_latencies_ms = [
        receipt.durable_latency_ms
        for receipt in mutation_receipts
        if receipt.durable_latency_ms is not None
    ]
    expected_snapshot = _snapshot(storage)
    router.close()

    restarted_storage = SQLiteStorage(str(database_path))
    restarted = AdaptiveRouter(encoder, restarted_storage, max_capacity=route_count * 10)
    restart_state_equal = _snapshot(restarted_storage) == expected_snapshot
    restarted.close()

    completed_queries = len(query_latencies_ms)
    return {
        "benchmark": "concurrent_dynamic_routing_workload",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "workload": {
            "duration_seconds": duration_seconds,
            "route_count": route_count,
            "query_workers": query_workers,
            "target_mutations_per_second": mutation_rate,
            "encoder": encoder.model_name,
            "dimension": dim,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "metrics": {
            "completed_queries": completed_queries,
            "query_errors": len(query_errors),
            "query_accuracy": query_correct / completed_queries if completed_queries else None,
            "query_throughput_per_second": completed_queries / workload_wall_seconds,
            "query_latency": _percentiles(query_latencies_ms),
            "mutation_attempts": mutation_count,
            "mutation_errors": len(mutation_errors),
            "mutation_memory_ack": _percentiles(mutation_memory_ack_ms),
            "mutation_durable_commit": _percentiles([float(value) for value in durable_latencies_ms]),
            "durable_barrier_ms": barrier_ms,
            "visibility_failures": visibility_failures,
            "deletion_visibility_failures": deletion_visibility_failures,
            "restart_state_equal": restart_state_equal,
        },
        "errors": {
            "query": sorted(query_errors),
            "mutation": sorted(mutation_errors),
        },
        "notes": [
            "The deterministic hash encoder isolates structural concurrency behavior.",
            "This mixed workload uses synchronous query calls from multiple threads.",
            "Results remain unverified until run from a clean commit and repeated on controlled hardware.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--routes", type=int, default=100)
    parser.add_argument("--query-workers", type=int, default=4)
    parser.add_argument("--mutation-rate", type=float, default=20.0)
    parser.add_argument("--dim", type=int, default=32)
    default_dir = Path(os.environ.get("SYNAPTOROUTE_RUN_DIR", "benchmark_results/dynamic"))
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = args.output_dir / f"dynamic-{time.time_ns()}.sqlite3"
    result = run_benchmark(
        database_path=database_path,
        duration_seconds=args.duration,
        route_count=args.routes,
        query_workers=args.query_workers,
        mutation_rate=args.mutation_rate,
        dim=args.dim,
    )
    output_path = args.output_dir / "dynamic_workload_summary.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
