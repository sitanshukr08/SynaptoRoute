"""Measure explicit mutation acknowledgement and normal-restart durability."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.deterministic_encoder import DeterministicHashEncoder
from synaptoroute import AdaptiveRouter, Route, SQLiteStorage, StorageFlushError, StorageMutationError


def _percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": max(values),
    }


class FailOnceStorage(SQLiteStorage):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.fail_next_save = True

    def save_route(self, route, embeddings=None, expected_version=None):
        if self.fail_next_save:
            self.fail_next_save = False
            raise RuntimeError("injected save failure")
        return super().save_route(route, embeddings, expected_version)


def _route_snapshot(storage: SQLiteStorage) -> dict[str, dict[str, Any]]:
    routes, _ = storage.load_all_routes()
    return {
        route.name: {
            "threshold": route.threshold,
            "utterances": sorted(route.utterances),
            "metadata": route.metadata,
        }
        for route in routes
    }


def run_benchmark(*, mutation_count: int, database_path: Path, dim: int = 64) -> dict[str, Any]:
    if mutation_count < 10:
        raise ValueError("mutation_count must be at least 10")
    if database_path.exists():
        raise FileExistsError(f"benchmark database already exists: {database_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = DeterministicHashEncoder(dim=dim)
    storage = SQLiteStorage(str(database_path))
    router = AdaptiveRouter(encoder, storage, max_capacity=mutation_count * 3)

    serial_count = max(5, mutation_count // 5)
    serial_memory_ack_ms: list[float] = []
    serial_durable_ms: list[float] = []
    for index in range(serial_count):
        route = Route(name=f"serial_{index}", utterances=[f"serial utterance {index}"])
        started = time.perf_counter_ns()
        receipt = router.add_route(route)
        serial_memory_ack_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        serial_durable_ms.append(receipt.wait_durable(timeout=5.0))

    burst_receipts = []
    burst_memory_ack_ms: list[float] = []
    burst_started = time.perf_counter()
    for index in range(mutation_count):
        route = Route(name=f"burst_{index}", utterances=[f"burst utterance {index}"])
        started = time.perf_counter_ns()
        burst_receipts.append(router.add_route(route))
        burst_memory_ack_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    barrier_started = time.perf_counter()
    router.durable_barrier(timeout=30.0)
    barrier_ms = (time.perf_counter() - barrier_started) * 1000.0
    burst_wall_seconds = time.perf_counter() - burst_started
    burst_durable_ms = [receipt.durable_latency_ms for receipt in burst_receipts]
    if any(value is None for value in burst_durable_ms):
        raise RuntimeError("durable barrier returned before all receipts became durable")

    for index in range(0, mutation_count, 4):
        router.update_threshold(f"burst_{index}", 0.75)
    for index in range(1, mutation_count, 4):
        router.add_utterance(f"burst_{index}", f"additional utterance {index}")
    for index in range(2, mutation_count, 4):
        router.delete_route(f"burst_{index}")
    router.durable_barrier(timeout=30.0)

    expected_snapshot = _route_snapshot(storage)
    router.close()
    restarted_storage = SQLiteStorage(str(database_path))
    restarted = AdaptiveRouter(encoder, restarted_storage, max_capacity=mutation_count * 3)
    restarted_snapshot = _route_snapshot(restarted_storage)
    restart_state_equal = restarted_snapshot == expected_snapshot
    restarted.close()

    failure_database = database_path.with_name(f"{database_path.stem}-failure.sqlite3")
    failing_storage = FailOnceStorage(str(failure_database))
    failing_router = AdaptiveRouter(encoder, failing_storage)
    failed_receipt = failing_router.add_route(
        Route(name="injected_failure", utterances=["must not persist"])
    )
    receipt_failure_observed = False
    barrier_failure_observed = False
    try:
        failed_receipt.wait_durable(timeout=5.0)
    except StorageMutationError:
        receipt_failure_observed = True
    try:
        failing_router.durable_barrier(timeout=5.0)
    except StorageFlushError:
        barrier_failure_observed = True
    failure_resynced = "injected_failure" not in failing_router._route_map
    failing_router.close()

    return {
        "benchmark": "single_process_mutation_durability",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "workload": {
            "mutation_count": mutation_count,
            "serial_add_count": serial_count,
            "burst_add_count": mutation_count,
            "mixed_update_count": len(range(0, mutation_count, 4)),
            "mixed_utterance_count": len(range(1, mutation_count, 4)),
            "mixed_delete_count": len(range(2, mutation_count, 4)),
            "encoder": encoder.model_name,
            "dimension": dim,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "sqlite_version": __import__("sqlite3").sqlite_version,
        },
        "metrics": {
            "serial_memory_ack": _percentiles(serial_memory_ack_ms),
            "serial_durable_commit": _percentiles(serial_durable_ms),
            "burst_memory_ack": _percentiles(burst_memory_ack_ms),
            "burst_durable_commit": _percentiles([float(value) for value in burst_durable_ms]),
            "burst_barrier_ms": barrier_ms,
            "burst_throughput_mutations_per_second": mutation_count / burst_wall_seconds,
            "restart_state_equal": restart_state_equal,
            "receipt_failure_observed": receipt_failure_observed,
            "barrier_failure_observed": barrier_failure_observed,
            "failure_resynced": failure_resynced,
        },
        "notes": [
            "The deterministic hash encoder is structural and cannot support semantic quality claims.",
            "This benchmark covers normal restart, not abrupt process termination.",
            "Results remain unverified until run from a clean commit and independently reviewed.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutations", type=int, default=500)
    parser.add_argument("--dim", type=int, default=64)
    default_dir = Path(os.environ.get("SYNAPTOROUTE_RUN_DIR", "benchmark_results/durability"))
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = args.output_dir / f"durability-{time.time_ns()}.sqlite3"
    result = run_benchmark(
        mutation_count=args.mutations,
        database_path=database_path,
        dim=args.dim,
    )
    output_path = args.output_dir / "durability_summary.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
