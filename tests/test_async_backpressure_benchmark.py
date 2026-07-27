import pytest

from benchmarks.bench_async_backpressure import run_benchmark


@pytest.mark.asyncio
async def test_backpressure_benchmark_sheds_excess_without_worker_failure():
    result = await run_benchmark(
        offered_concurrency=[2, 20],
        queue_size=2,
        max_in_flight_batches=1,
        batch_size=1,
        encoder_delay_ms=20.0,
    )

    low, high = result["scenarios"]
    assert low["error_count"] == 0
    assert high["error_count"] == 0
    assert high["overloaded_count"] > 0
    assert high["success_count"] > 0
    assert high["successful_accuracy"] == 1.0
