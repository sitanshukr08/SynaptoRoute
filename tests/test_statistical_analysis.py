import pytest

from benchmarks.statistical_analysis import (
    align_prediction_records,
    hierarchical_paired_bootstrap,
    matched_coverage_curve,
    paired_matched_coverage_effects,
)


def record(
    example_id,
    *,
    expected="route",
    predicted="route",
    correct=True,
    confidence=0.5,
    raw_top_correct=True,
):
    return {
        "example_id": str(example_id),
        "query_sha256": str(example_id).zfill(64),
        "expected_route": expected,
        "predicted_route": predicted,
        "correct": correct,
        "metadata": {
            "acceptance_confidence": confidence,
            "raw_top_correct": raw_top_correct,
        },
    }


def test_alignment_rejects_changed_ground_truth():
    first = [record(1)]
    second = [record(1, expected="different")]

    with pytest.raises(ValueError, match="expected_route"):
        align_prediction_records(first, second)


def test_hierarchical_bootstrap_preserves_positive_paired_effect():
    paired_by_seed = {}
    for seed in (13, 29, 42):
        first = [record(f"{seed}-{index}", correct=index < 80) for index in range(100)]
        second = [record(f"{seed}-{index}", correct=index < 60) for index in range(100)]
        paired_by_seed[seed] = align_prediction_records(first, second)

    result = hierarchical_paired_bootstrap(
        paired_by_seed,
        metric="overall_accuracy",
        repetitions=1000,
        random_seed=7,
    )

    assert result is not None
    assert result["point_estimate"] == pytest.approx(0.2)
    assert result["confidence_interval_95"][0] > 0.0
    assert result["probability_effect_positive"] > 0.99


def test_matched_coverage_uses_raw_confidence_and_correctness():
    records = [
        record("known-1", confidence=0.9, raw_top_correct=True),
        record("known-2", confidence=0.8, raw_top_correct=True),
        record("known-3", confidence=0.7, raw_top_correct=False),
        record("known-4", confidence=0.1, raw_top_correct=True),
        record("ood-high", expected=None, predicted=None, confidence=0.6, raw_top_correct=False),
        record("ood-low", expected=None, predicted=None, confidence=0.05, raw_top_correct=False),
    ]

    curve = matched_coverage_curve(records, targets=(0.5, 0.75, 1.0))

    assert curve["0.50"]["actual_known_coverage"] == 0.5
    assert curve["0.50"]["selective_accuracy"] == 1.0
    assert curve["0.75"]["selective_accuracy"] == pytest.approx(2 / 3)
    assert curve["1.00"]["ood_false_acceptance_rate"] == 0.5


def test_paired_matched_coverage_effects_resample_seed_differences():
    first = {}
    second = {}
    for seed in (13, 29, 42):
        first[seed] = [
            record(f"known-{seed}-{index}", confidence=1.0 - index / 10, raw_top_correct=True)
            for index in range(10)
        ] + [
            record(
                f"ood-{seed}",
                expected=None,
                predicted=None,
                confidence=-1.0,
                raw_top_correct=False,
            )
        ]
        second[seed] = [
            record(
                f"known-{seed}-{index}",
                confidence=1.0 - index / 10,
                raw_top_correct=index < 8,
            )
            for index in range(10)
        ] + [
            record(
                f"ood-{seed}",
                expected=None,
                predicted=None,
                confidence=0.95,
                raw_top_correct=False,
            )
        ]

    effects = paired_matched_coverage_effects(
        first,
        second,
        targets=(0.9,),
        repetitions=1000,
        random_seed=11,
    )

    assert effects["0.90"]["selective_accuracy"]["point_estimate"] > 0.0
    assert effects["0.90"]["ood_false_acceptance_rate"]["point_estimate"] < 0.0
