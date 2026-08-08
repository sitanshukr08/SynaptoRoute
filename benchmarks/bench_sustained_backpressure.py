"""Run sustained open-loop offered-load experiments against the async router."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from pathlib import Path

import numpy as np

from benchmarks.bench_async_backpressure import DelayedEncoder
from synaptoroute import AdaptiveRouter, Route, SQLiteStorage
from synaptoroute.exceptions import RouterOverloadedError


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
    load_fractions: list[float],
    duration_seconds: float,
    queue_size: int,
    batch_size: int,
    max_in_flight_batches: int,
    encoder_delay_ms: float,
    calibration_seconds: float = 0.5,
) -> dict:
    if duration_seconds <= 0 or encoder_delay_ms <= 0 or calibration_seconds <= 0:
        raise ValueError(
            "duration_seconds, encoder_delay_ms, and calibration_seconds must be positive"
        )
    if not load_fractions or any(fraction <= 0 for fraction in load_fractions):
        raise ValueError("load_fractions must be positive")

    encoder = DelayedEncoder(dim=16, delay_ms=encoder_delay_ms)
    router = AdaptiveRouter(
        encoder,
        SQLiteStorage(":memory:"),
        max_queue_size=queue_size,
        max_in_flight_batches=max_in_flight_batches,
    )
    router.batch_size = batch_size
    setup = router.add_route(Route(name="support", utterances=["help me"], threshold=0.8))
    setup.wait_durable(timeout=5.0)
    await router.start()

    calibration_successes = 0

    async def calibration_worker(deadline: float) -> None:
        nonlocal calibration_successes
        while time.perf_counter() < deadline:
            try:
                await router.amatch("help me")
                calibration_successes += 1
            except RouterOverloadedError:
                await asyncio.sleep(0)

    calibration_started = time.perf_counter()
    calibration_deadline = calibration_started + calibration_seconds
    calibration_concurrency = min(
        512,
        max(4, queue_size + batch_size * max_in_flight_batches),
    )
    await asyncio.gather(
        *(
            calibration_worker(calibration_deadline)
            for _ in range(calibration_concurrency)
        )
    )
    calibration_wall_seconds = time.perf_counter() - calibration_started
    measured_saturation_qps = calibration_successes / calibration_wall_seconds
    if measured_saturation_qps <= 0:
        raise RuntimeError("saturation calibration completed no successful queries")

    scenarios = []
    try:
        for fraction in load_fractions:
            target_qps = measured_saturation_qps * fraction
            interval = 1.0 / target_qps

            async def request() -> dict:
                started_ns = time.perf_counter_ns()
                try:
                    result = await router.amatch("help me")
                    return {
                        "status": "success",
                        "correct": result.route_name == "support",
                        "latency_ms": (time.perf_counter_ns() - started_ns) / 1_000_000.0,
                    }
                except RouterOverloadedError:
                    return {
                        "status": "overloaded",
                        "correct": False,
                        "latency_ms": (time.perf_counter_ns() - started_ns) / 1_000_000.0,
                    }
                except Exception as error:
                    return {
                        "status": f"error:{type(error).__name__}",
                        "correct": False,
                        "latency_ms": (time.perf_counter_ns() - started_ns) / 1_000_000.0,
                    }

            tasks: list[asyncio.Task] = []
            started = time.perf_counter()
            deadline = started + duration_seconds
            next_arrival = started
            while time.perf_counter() < deadline:
                now = time.perf_counter()
                if now < next_arrival:
                    await asyncio.sleep(next_arrival - now)
                    continue
                tasks.append(asyncio.create_task(request()))
                next_arrival += interval
            results = await asyncio.gather(*tasks)
            wall_seconds = time.perf_counter() - started
            successful = [item for item in results if item["status"] == "success"]
            overloaded = [item for item in results if item["status"] == "overloaded"]
            errors = [
                item for item in results if item["status"] not in {"success", "overloaded"}
            ]
            scenarios.append(
                {
                    "load_fraction": fraction,
                    "target_qps": target_qps,
                    "offered_count": len(results),
                    "successful_count": len(successful),
                    "overloaded_count": len(overloaded),
                    "error_count": len(errors),
                    "completed_qps": len(successful) / wall_seconds,
                    "shedding_rate": len(overloaded) / len(results) if results else 0.0,
                    "successful_accuracy": (
                        sum(item["correct"] for item in successful) / len(successful)
                        if successful
                        else None
                    ),
                    "successful_latency": _percentiles(
                        [float(item["latency_ms"]) for item in successful]
                    ),
                    "overload_latency": _percentiles(
                        [float(item["latency_ms"]) for item in overloaded]
                    ),
                }
            )
    finally:
        await router.stop()

    return {
        "benchmark": "sustained_async_backpressure",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "configuration": {
            "load_fractions": load_fractions,
            "duration_seconds": duration_seconds,
            "queue_size": queue_size,
            "batch_size": batch_size,
            "max_in_flight_batches": max_in_flight_batches,
            "encoder_delay_ms": encoder_delay_ms,
            "saturation_calibration_seconds": calibration_wall_seconds,
            "saturation_calibration_successes": calibration_successes,
            "measured_saturation_qps": measured_saturation_qps,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "scenarios": scenarios,
        "notes": [
            "Requests are offered on an open-loop fixed-interval schedule.",
            "Offered load is relative to a measured saturated closed-loop calibration.",
            "Overloaded and failed requests remain in the denominator.",
            "The deterministic delayed encoder isolates queue behavior.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-fractions", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--queue-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-in-flight-batches", type=int, default=2)
    parser.add_argument("--encoder-delay-ms", type=float, default=20.0)
    parser.add_argument("--calibration-duration", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(
        run_benchmark(
            load_fractions=args.load_fractions,
            duration_seconds=args.duration,
            queue_size=args.queue_size,
            batch_size=args.batch_size,
            max_in_flight_batches=args.max_in_flight_batches,
            encoder_delay_ms=args.encoder_delay_ms,
            calibration_seconds=args.calibration_duration,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
