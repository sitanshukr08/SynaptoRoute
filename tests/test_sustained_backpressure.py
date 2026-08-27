import asyncio

from benchmarks.bench_sustained_backpressure import run_benchmark


def test_sustained_backpressure_retains_all_offered_requests():
    result = asyncio.run(
        run_benchmark(
            load_fractions=[0.5, 2.0],
            duration_seconds=0.1,
            queue_size=4,
            batch_size=2,
            max_in_flight_batches=1,
            encoder_delay_ms=10.0,
            calibration_seconds=0.1,
        )
    )

    assert result["status"] == "unverified"
    assert result["configuration"]["measured_saturation_qps"] > 0
    assert result["configuration"]["saturation_calibration_attempts"] == (
        result["configuration"]["saturation_calibration_successes"]
        + result["configuration"]["saturation_calibration_overloaded"]
        + result["configuration"]["saturation_calibration_error_count"]
    )
    for scenario in result["scenarios"]:
        accounted = (
            scenario["successful_count"]
            + scenario["overloaded_count"]
            + scenario["error_count"]
        )
        assert accounted == scenario["offered_count"]
        assert scenario["successful_count"] == (
            scenario["successful_correct_count"] + scenario["successful_incorrect_count"]
        )
        outcome_rate = scenario["success_rate"] + scenario["shedding_rate"] + scenario["error_rate"]
        assert abs(outcome_rate - 1.0) < 1e-12
        assert scenario["scenario_wall_seconds"] >= scenario["offering_wall_seconds"]
