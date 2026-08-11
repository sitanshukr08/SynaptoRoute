import copy

import numpy as np
import pytest

from paper.analyze_backpressure import analyze_records, render_csv, render_markdown, summarize_values


def summary(*, queue_size=8, batch_size=1, saturation=100.0, shed_at_one=0):
    scenarios = []
    for load, offered, successful, overloaded, p95 in (
        (0.5, 50, 50, 0, 10.0),
        (1.0, 100, 100 - shed_at_one, shed_at_one, 20.0),
        (1.5, 150, 100, 50, 25.0),
        (2.0, 200, 100, 100, 30.0),
    ):
        correct = successful
        scenarios.append(
            {
                "load_fraction": load,
                "target_qps": saturation * load,
                "offered_count": offered,
                "successful_count": successful,
                "successful_correct_count": correct,
                "successful_incorrect_count": 0,
                "overloaded_count": overloaded,
                "error_count": 0,
                "success_rate": successful / offered,
                "shedding_rate": overloaded / offered,
                "error_rate": 0.0,
                "offered_qps": float(offered),
                "successful_qps": float(successful),
                "successful_latency": {
                    "p50_ms": p95 / 2,
                    "p95_ms": p95,
                    "p99_ms": p95 + 2,
                    "max_ms": p95 + 4,
                },
                "overload_latency": (
                    {"p50_ms": 0.01, "p95_ms": 0.02, "p99_ms": 0.03, "max_ms": 0.04}
                    if overloaded
                    else None
                ),
            }
        )
    return {
        "schema_version": 2,
        "benchmark": "sustained_async_backpressure",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "configuration": {
            "load_fractions": [0.5, 1.0, 1.5, 2.0],
            "duration_seconds": 60.0,
            "queue_size": queue_size,
            "batch_size": batch_size,
            "max_in_flight_batches": 2,
            "encoder_delay_ms": 20.0,
            "measured_saturation_qps": saturation,
        },
        "scenarios": scenarios,
    }


def test_summary_bootstrap_is_deterministic_and_uses_repetitions():
    first = summarize_values(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        bootstrap_repetitions=1000,
        rng=np.random.default_rng(7),
    )
    second = summarize_values(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        bootstrap_repetitions=1000,
        rng=np.random.default_rng(7),
    )

    assert first == second
    assert first["n"] == 5
    assert first["mean"] == 3.0
    assert first["confidence_interval_95"][0] < first["mean"]
    assert first["confidence_interval_95"][1] > first["mean"]


def test_analysis_retains_all_outcomes_and_reports_capacity_retention():
    records = {
        "low_latency-rep0": summary(saturation=100.0, shed_at_one=0),
        "low_latency-rep1": summary(saturation=100.0, shed_at_one=10),
    }

    result = analyze_records(records, bootstrap_repetitions=1000, random_seed=11)
    scenario = result["profiles"]["low_latency"]["scenarios"]["1"]

    assert scenario["pooled_counts"]["offered_count"] == 200
    assert scenario["pooled_counts"]["successful_count"] == 190
    assert scenario["pooled_counts"]["overloaded_count"] == 10
    assert scenario["pooled_rates"]["shedding_rate"] == pytest.approx(0.05)
    assert scenario["pooled_rates"]["successful_accuracy"] == 1.0
    assert scenario["repetition_statistics"]["capacity_retention"]["mean"] == pytest.approx(0.95)
    assert scenario["repetition_statistics"]["p95_inflation_vs_lowest_load"]["mean"] == 2.0


def test_analysis_rejects_configuration_drift_between_repetitions():
    changed = summary(queue_size=16)
    records = {
        "balanced-rep0": summary(queue_size=8),
        "balanced-rep1": changed,
    }

    with pytest.raises(ValueError, match="changes configuration"):
        analyze_records(records, bootstrap_repetitions=100)


def test_analysis_rejects_missing_repetition_and_changed_load_set():
    records = {
        "throughput-rep0": summary(),
        "throughput-rep2": summary(),
    }
    with pytest.raises(ValueError, match="contiguous"):
        analyze_records(records, bootstrap_repetitions=100)

    changed = copy.deepcopy(summary())
    changed["scenarios"].pop()
    with pytest.raises(ValueError, match="changes load fractions"):
        analyze_records(
            {"throughput-rep0": summary(), "throughput-rep1": changed},
            bootstrap_repetitions=100,
        )


def test_renderers_include_status_denominators_and_intervals():
    aggregate = analyze_records(
        {"low_latency-rep0": summary(), "low_latency-rep1": summary()},
        bootstrap_repetitions=100,
    )
    analysis = {"summary": aggregate}

    csv_text = render_csv(analysis)
    markdown = render_markdown(analysis)

    assert "offered_count,successful_count,overloaded_count,error_count" in csv_text
    assert "low_latency,2.0,2,400,200,200,0" in csv_text
    assert "Status: unverified development evidence" in markdown
    assert "95% CI" in markdown
