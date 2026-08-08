import math

import pytest

from benchmarks.probability_calibration import (
    FEATURE_NAMES,
    correctness_features,
    fit_correctness_probability,
    split_calibration_examples,
)
from benchmarks.research_datasets import IntentExample
from synaptoroute.models import DecisionReason, Route, RouterResult


def test_held_out_split_is_deterministic_disjoint_and_label_stratified():
    examples = tuple(
        IntentExample(f"{label}-{index}", f"text {label} {index}", label, "validation")
        for label in ("alpha", "beta")
        for index in range(4)
    ) + (IntentExample("ood-0", "unknown", None, "validation"),)

    first_policy, first_probability = split_calibration_examples(examples, seed=42)
    second_policy, second_probability = split_calibration_examples(examples, seed=42)

    assert first_policy == second_policy
    assert first_probability == second_probability
    assert {example.example_id for example in first_policy}.isdisjoint(
        example.example_id for example in first_probability
    )
    assert len(first_policy) + len(first_probability) == len(examples)
    for label in ("alpha", "beta"):
        assert sum(example.label == label for example in first_policy) == 2
        assert sum(example.label == label for example in first_probability) == 2


def test_held_out_split_rejects_single_example():
    with pytest.raises(ValueError, match="at least two"):
        split_calibration_examples(
            (IntentExample("only", "text", "alpha", "validation"),),
            seed=42,
        )


def test_logistic_correctness_calibration_produces_bounded_ordered_probabilities():
    low = [-1.0, 0.0, -1.0, 1.0, 0.0, 0.0]
    high = [1.0, 0.8, 0.7, 1.0, 1.0, 1.0]
    calibrator, metrics = fit_correctness_probability(
        [low, low, high, high],
        [False, False, True, True],
        random_state=42,
    )

    assert calibrator.method == "logistic_correctness_calibration"
    assert 0.0 <= calibrator.predict(low) < calibrator.predict(high) <= 1.0
    assert 0.0 <= metrics.expected_calibration_error <= 1.0
    assert metrics.num_samples == 4


def test_one_class_calibration_uses_declared_laplace_fallback():
    row = [0.5, 0.1, 0.2, 1.0, 1.0, 1.0]
    calibrator, metrics = fit_correctness_probability(
        [row, row, row],
        [True, True, True],
        random_state=42,
    )

    assert calibrator.method == "laplace_constant_one_class"
    assert calibrator.predict(row) == pytest.approx(0.8)
    assert metrics.brier_score == pytest.approx(0.04)


def test_correctness_features_replace_missing_values_with_declared_finite_sentinels():
    raw = RouterResult(decision_reason=DecisionReason.NO_CANDIDATES)
    final = RouterResult(
        route=Route(name="alpha", utterances=["hello"]),
        score=0.8,
        decision_reason=DecisionReason.MATCHED,
    )

    features = correctness_features(raw, final, acceptance_confidence=float("-inf"))

    assert len(features) == len(FEATURE_NAMES)
    assert all(math.isfinite(value) for value in features)
    assert features == (-1.0, 0.0, -2.0, 0.0, 0.0, 1.0)
