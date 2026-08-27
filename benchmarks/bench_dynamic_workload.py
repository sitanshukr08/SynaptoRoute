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
from benchmarks.index_metadata import describe_index
from benchmarks.manifest_schema import sha256_file
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


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def run_benchmark(
    *,
    database_path: Path,
    duration_seconds: float,
    route_count: int,
    query_workers: int,
    mutation_rate: float,
    index_engine: str = "auto",
    dim: int = 32,
    warmup_seconds: float = 0.0,
) -> dict[str, Any]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if route_count < 2 or query_workers < 1 or mutation_rate < 0:
        raise ValueError("route_count and query_workers must be positive; mutation_rate cannot be negative")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds cannot be negative")
    if index_engine not in {"numpy", "faiss", "auto"}:
        raise ValueError("index_engine must be numpy, faiss, or auto")
    if database_path.exists():
        raise FileExistsError(f"benchmark database already exists: {database_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = DeterministicHashEncoder(dim=dim)
    storage = SQLiteStorage(str(database_path))
    router = AdaptiveRouter(
        encoder,
        storage,
        max_capacity=route_count * 10,
        max_storage_queue_size=max(1000, route_count + 100),
        index_engine=index_engine,
    )
    index_parameters = describe_index(router.index)
    if index_engine != "auto" and index_parameters["resolved_engine"] != index_engine:
        raise RuntimeError("resolved dynamic index differs from the requested engine")
    rss_before_mb = _rss_mb()
    base_queries = []
    for index in range(route_count):
        utterance = f"base utterance {index}"
        router.add_route(Route(name=f"base_{index}", utterances=[utterance], threshold=0.8))
        base_queries.append((utterance, f"base_{index}"))
    router.durable_barrier(timeout=30.0)

    warmup_deadline = time.perf_counter() + warmup_seconds
    warmup_cursor = 0
    while time.perf_counter() < warmup_deadline:
        query, _ = base_queries[warmup_cursor % len(base_queries)]
        router.match(query)
        warmup_cursor += 1

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
    storage_queue_depths: list[int] = []
    mutation_count = 0
    workload_started = time.perf_counter()
    deadline = workload_started + duration_seconds
    interval = 1.0 / mutation_rate if mutation_rate else None
    next_mutation = workload_started

    with ThreadPoolExecutor(max_workers=query_workers, thread_name_prefix="dynamic_query") as executor:
        futures = [executor.submit(query_worker, worker_id) for worker_id in range(query_workers)]
        while time.perf_counter() < deadline:
            if interval is None:
                time.sleep(min(deadline - time.perf_counter(), 0.01))
                continue
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
                storage_queue_depths.append(router._storage_queue.qsize())
            except Exception as error:
                mutation_errors.append(type(error).__name__)
            mutation_count += 1
            next_mutation += interval

        stop_event.set()
        for future in futures:
            future.result(timeout=10.0)

    measurement_wall_seconds = time.perf_counter() - workload_started
    barrier_started = time.perf_counter()
    router.durable_barrier(timeout=30.0)
    barrier_ms = (time.perf_counter() - barrier_started) * 1000.0
    durable_latencies_ms = [
        receipt.durable_latency_ms
        for receipt in mutation_receipts
        if receipt.durable_latency_ms is not None
    ]
    runtime_snapshot = {
        route.name: (route.threshold, tuple(sorted(route.utterances)))
        for route in router._route_map.values()
    }
    persisted_snapshot = _snapshot(storage)
    pre_restart_state_equal = persisted_snapshot == runtime_snapshot
    rss_after_workload_mb = _rss_mb()
    router.close()
    storage.close()

    restart_started = time.perf_counter_ns()
    restarted_storage = SQLiteStorage(str(database_path))
    restarted = AdaptiveRouter(encoder, restarted_storage, max_capacity=route_count * 10)
    restart_recovery_ms = (time.perf_counter_ns() - restart_started) / 1_000_000.0
    restart_state_equal = _snapshot(restarted_storage) == runtime_snapshot
    restarted.close()
    restarted_storage.close()
    database_sha256 = sha256_file(database_path)
    database_bytes = database_path.stat().st_size

    completed_queries = len(query_latencies_ms)
    query_error_count = len(query_errors)
    query_attempts = completed_queries + query_error_count
    query_incorrect = completed_queries - query_correct
    mutation_error_count = len(mutation_errors)
    mutation_successes = mutation_count - mutation_error_count
    correctness_violations = (
        query_incorrect
        + visibility_failures
        + deletion_visibility_failures
        + int(not pre_restart_state_equal)
        + int(not restart_state_equal)
    )
    operation_failures = query_error_count + mutation_error_count
    return {
        "schema_version": 2,
        "benchmark": "concurrent_dynamic_routing_workload",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "workload": {
            "duration_seconds": duration_seconds,
            "warmup_seconds": warmup_seconds,
            "route_count": route_count,
            "query_workers": query_workers,
            "target_mutations_per_second": mutation_rate,
            "encoder": encoder.model_name,
            "dimension": dim,
            "index_engine_requested": index_engine,
            "index_parameters": index_parameters,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "faiss": index_parameters.get("faiss_version"),
        },
        "metrics": {
            "measurement_wall_seconds": measurement_wall_seconds,
            "query_attempts": query_attempts,
            "completed_queries": completed_queries,
            "query_correct": query_correct,
            "query_incorrect": query_incorrect,
            "query_errors": query_error_count,
            "query_accuracy": query_correct / completed_queries if completed_queries else None,
            "query_success_rate": completed_queries / query_attempts if query_attempts else None,
            "query_attempt_throughput_per_second": query_attempts / measurement_wall_seconds,
            "query_success_throughput_per_second": completed_queries / measurement_wall_seconds,
            "query_throughput_per_second": completed_queries / measurement_wall_seconds,
            "query_latency": _percentiles(query_latencies_ms),
            "mutation_attempts": mutation_count,
            "mutation_successes": mutation_successes,
            "mutation_errors": mutation_error_count,
            "mutation_success_rate": (
                mutation_successes / mutation_count if mutation_count else None
            ),
            "mutation_error_rate": (
                mutation_error_count / mutation_count if mutation_count else None
            ),
            "mutation_shedding_count": sum(
                error == "RouterOverloadedError" for error in mutation_errors
            ),
            "mutation_attempt_throughput_per_second": mutation_count
            / measurement_wall_seconds,
            "mutation_success_throughput_per_second": mutation_successes
            / measurement_wall_seconds,
            "mutation_throughput_per_second": mutation_count / measurement_wall_seconds,
            "mutation_memory_ack": _percentiles(mutation_memory_ack_ms),
            "mutation_durable_commit": _percentiles([float(value) for value in durable_latencies_ms]),
            "mutation_receipt_count": len(mutation_receipts),
            "durable_latency_count": len(durable_latencies_ms),
            "durable_receipt_count": sum(
                receipt.state == "durable" for receipt in mutation_receipts
            ),
            "durable_barrier_ms": barrier_ms,
            "storage_queue_depth": {
                "max": max(storage_queue_depths, default=0),
                "samples": len(storage_queue_depths),
            },
            "visibility_failures": visibility_failures,
            "deletion_visibility_failures": deletion_visibility_failures,
            "correctness_violations": correctness_violations,
            "operation_failures": operation_failures,
            "total_adverse_outcomes": correctness_violations + operation_failures,
            "pre_restart_state_equal": pre_restart_state_equal,
            "restart_state_equal": restart_state_equal,
            "restart_recovery_ms": restart_recovery_ms,
            "rss_before_mb": rss_before_mb,
            "rss_after_workload_mb": rss_after_workload_mb,
            "rss_delta_mb": (
                rss_after_workload_mb - rss_before_mb
                if rss_before_mb is not None and rss_after_workload_mb is not None
                else None
            ),
        },
        "errors": {
            "query": sorted(query_errors),
            "mutation": sorted(mutation_errors),
        },
        "evidence": {
            "database_path": database_path.resolve().as_posix(),
            "database_sha256": database_sha256,
            "database_bytes": database_bytes,
        },
        "notes": [
            "The deterministic hash encoder isolates structural concurrency behavior.",
            "This mixed workload uses synchronous query calls from multiple threads.",
            "The requested and resolved index configuration is recorded in the workload.",
            "Throughput denominators use the measured workload window and exclude the durable barrier.",
            "Correctness violations and explicit operational failures are reported separately.",
            "Results remain unverified until run from a clean commit and repeated on controlled hardware.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=0.0)
    parser.add_argument("--routes", type=int, default=100)
    parser.add_argument("--query-workers", type=int, default=4)
    parser.add_argument("--mutation-rate", type=float, default=20.0)
    parser.add_argument("--engine", choices=("numpy", "faiss", "auto"), default="auto")
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
        index_engine=args.engine,
        dim=args.dim,
        warmup_seconds=args.warmup,
    )
    output_path = args.output_dir / "dynamic_workload_summary.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
