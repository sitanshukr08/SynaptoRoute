import pytest

from benchmarks.run_multiseed_intent import aggregate_quality_summaries


def summary(accuracy, auroc):
    return {
        "systems": {
            "router": {
                "test": {
                    "overall_accuracy": accuracy,
                    "ood_auroc": auroc,
                }
            }
        }
    }


def test_multiseed_aggregation_reports_distribution_not_latency():
    aggregates = aggregate_quality_summaries(
        [summary(0.8, 0.9), summary(0.9, 0.95), summary(0.85, None)]
    )

    accuracy = aggregates["router"]["overall_accuracy"]
    auroc = aggregates["router"]["ood_auroc"]
    assert accuracy["n"] == 3
    assert accuracy["mean"] == pytest.approx(0.85)
    assert accuracy["sample_std"] == pytest.approx(0.05)
    assert auroc["n"] == 2
    assert "latency_p95_ms" not in aggregates["router"]


def test_multiseed_aggregation_requires_input():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_quality_summaries([])
