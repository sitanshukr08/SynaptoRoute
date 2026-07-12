"""Measure bounded async batch execution and overload shedding."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.deterministic_encoder import DeterministicHashEncoder
from synaptoroute import AdaptiveRouter, Route, SQLiteStorage
from synaptoroute.exceptions import RouterOverloadedError


class DelayedEncoder(DeterministicHashEncoder):
    def __init__(self, *, dim: int, delay_ms: float):
        super().__init__(dim=dim)
        self.delay_seconds = delay_ms / 1000.0

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        time.sleep(self.delay_seconds)
        return super().encode_batch(texts)


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": max(values),
    }


async def run_benchmark(
    *,
    offered_concurrency: list[int],
    queue_size: int,
    max_in_flight_batches: int,
    batch_size: int,
    encoder_delay_ms: float,
) -> dict[str, Any]:
    if not offered_concurrency or any(value < 1 for value in offered_concurrency):
        raise ValueError("offered_concurrency values must be positive")
    if queue_size < 1 or max_in_flight_batches < 1 or batch_size < 1:
        raise ValueError("queue and batch limits must be positive")

    encoder = DelayedEncoder(dim=16, delay_ms=encoder_delay_ms)
    router = AdaptiveRouter(
        encoder,
        SQLiteStorage(":memory:"),
        max_queue_size=queue_size,
        max_in_flight_batches=max_in_flight_batches,
    )
    router.batch_size = batch_size
    router.add_route(Route(name="support", utterances=["help me"], threshold=0.8))
    await router.start()
    scenarios = []
    try:
        for concurrency in offered_concurrency:
            async def request(index: int):
                started = time.perf_counter_ns()
                try:
                    result = await router.amatch("help me")
                    return {
                        "status": "success",
                        "correct": result.route_name == "support",
                        "latency_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
                        "index": index,
                    }
                except RouterOverloadedError:
                    return {
                        "status": "overloaded",
                        "correct": False,
                        "latency_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
                        "index": index,
                    }
                except Exception as error:
                    return {
                        "status": f"error:{type(error).__name__}",
                        "correct": False,
                        "latency_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
                        "index": index,
                    }

            wall_started = time.perf_counter()
            results = await asyncio.gather(*(request(index) for index in range(concurrency)))
            wall_seconds = time.perf_counter() - wall_started
            successful = [result for result in results if result["status"] == "success"]
            overloaded = [result for result in results if result["status"] == "overloaded"]
            errors = [
                result for result in results if result["status"] not in {"success", "overloaded"}
            ]
            scenarios.append(
                {
                    "offered_concurrency": concurrency,
                    "success_count": len(successful),
                    "overloaded_count": len(overloaded),
                    "error_count": len(errors),
                    "success_rate": len(successful) / concurrency,
                    "shedding_rate": len(overloaded) / concurrency,
                    "successful_accuracy": (
                        sum(result["correct"] for result in successful) / len(successful)
                        if successful
                        else None
                    ),
                    "successful_latency": _percentiles(
                        [float(result["latency_ms"]) for result in successful]
                    ),
                    "overload_response_latency": _percentiles(
                        [float(result["latency_ms"]) for result in overloaded]
                    ),
                    "wall_ms": wall_seconds * 1000.0,
                    "completed_throughput_per_second": len(results) / wall_seconds,
                }
            )
    finally:
        await router.stop()

    return {
        "benchmark": "async_backpressure_offered_load_sweep",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "configuration": {
            "queue_size": queue_size,
            "max_in_flight_batches": max_in_flight_batches,
            "batch_size": batch_size,
            "encoder_delay_ms": encoder_delay_ms,
            "offered_concurrency": offered_concurrency,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "scenarios": scenarios,
        "notes": [
            "The delayed deterministic encoder makes queue saturation repeatable.",
            "Overloaded requests remain in the denominator and are not retried.",
            "This benchmark measures the router API boundary, not HTTP transport.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument("--queue-size", type=int, default=32)
    parser.add_argument("--max-in-flight-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-delay-ms", type=float, default=20.0)
    default_dir = Path(os.environ.get("SYNAPTOROUTE_RUN_DIR", "benchmark_results/backpressure"))
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    args = parser.parse_args()

    result = asyncio.run(
        run_benchmark(
            offered_concurrency=args.concurrency,
            queue_size=args.queue_size,
            max_in_flight_batches=args.max_in_flight_batches,
            batch_size=args.batch_size,
            encoder_delay_ms=args.encoder_delay_ms,
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "async_backpressure_summary.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
