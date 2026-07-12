import hashlib
import random

import pytest

from benchmarks.calibration import (
    GlobalPolicy,
    PerRoutePolicy,
    PolicyMetrics,
    ScoredExample,
    apply_global_policy,
    apply_per_route_policy,
    calibration_artifact,
    evaluate_global_policy,
    evaluate_per_route_policy,
    fit_per_route_policy,
    fit_global_policy,
)
from synaptoroute import DecisionReason, Route, RouteCandidate, RouterResult


def scored(example_id, expected_route, *candidates, split="validation"):
    return ScoredExample(
        example_id=example_id,
        split=split,
        expected_route=expected_route,
        candidates=tuple(candidates),
    )


def reference_metrics(examples, threshold_for, margin):
    accepted = []
    correct = []
    for example in examples:
        prediction = None
        if example.candidates:
            top_name, top_score = example.candidates[0]
            margin_passed = len(example.candidates) == 1 or (
                top_score - example.candidates[1][1] >= margin
            )
            if top_score >= threshold_for(top_name) and margin_passed:
                prediction = top_name
        accepted.append(prediction is not None)
        correct.append(prediction == example.expected_route)

    known = [example.expected_route is not None for example in examples]
    known_count = sum(known)
    ood_count = len(examples) - known_count
    accepted_count = sum(accepted)
    accepted_known = sum(is_accepted and is_known for is_accepted, is_known in zip(accepted, known))
    accepted_ood = sum(is_accepted and not is_known for is_accepted, is_known in zip(accepted, known))
    accepted_correct = sum(
        is_accepted and is_correct for is_accepted, is_correct in zip(accepted, correct)
    )
    return PolicyMetrics(
        example_count=len(examples),
        known_count=known_count,
        ood_count=ood_count,
        accepted_count=accepted_count,
        correct_count=sum(correct),
        known_coverage=accepted_known / known_count,
        ood_false_acceptance_rate=accepted_ood / ood_count,
        overall_accuracy=sum(correct) / len(examples),
        selective_accuracy=accepted_correct / accepted_count if accepted_count else 1.0,
    )


def calibration_examples():
    return [
        scored("known-a", "alpha", ("alpha", 0.90), ("beta", 0.10)),
        scored("known-b", "beta", ("beta", 0.80), ("alpha", 0.20)),
        scored("known-hard", "alpha", ("beta", 0.70), ("alpha", 0.60)),
        scored("ood-hard", None, ("alpha", 0.65), ("beta", 0.60)),
        scored("ood-easy", None, ("alpha", 0.30), ("beta", 0.20)),
    ]


def test_policy_metrics_distinguish_coverage_and_ood_acceptance():
    metrics = evaluate_global_policy(calibration_examples(), GlobalPolicy(threshold=0.7, margin=0.0))

    assert metrics.known_coverage == 1.0
    assert metrics.ood_false_acceptance_rate == 0.0
    assert metrics.overall_accuracy == pytest.approx(0.8)
    assert metrics.selective_accuracy == pytest.approx(2 / 3)


def test_calibration_fits_only_policies_that_satisfy_constraints():
    result = fit_global_policy(
        calibration_examples(),
        min_known_coverage=2 / 3,
        max_ood_false_acceptance_rate=0.0,
    )

    assert result.metrics.known_coverage >= 2 / 3
    assert result.metrics.ood_false_acceptance_rate == 0.0
    assert result.metrics.overall_accuracy == pytest.approx(0.8)


def test_calibration_rejects_test_split_leakage():
    examples = calibration_examples()
    examples[0] = scored("leaked-test", "alpha", ("alpha", 0.9), split="test")

    with pytest.raises(ValueError, match="non-calibration splits"):
        fit_global_policy(examples)


def test_apply_policy_rebuilds_observable_decision():
    raw = RouterResult(
        route=Route(name="alpha", utterances=["alpha"], threshold=0.1),
        score=0.7,
        margin=0.05,
        candidates=[
            RouteCandidate(route_name="alpha", score=0.70, threshold=0.1, passed_threshold=True),
            RouteCandidate(route_name="beta", score=0.65, threshold=0.1, passed_threshold=True),
        ],
        decision_reason=DecisionReason.MATCHED,
    )
    routes = {
        "alpha": raw.route,
        "beta": Route(name="beta", utterances=["beta"], threshold=0.1),
    }

    rejected = apply_global_policy(raw, routes=routes, policy=GlobalPolicy(threshold=0.6, margin=0.1))
    accepted = apply_global_policy(raw, routes=routes, policy=GlobalPolicy(threshold=0.6, margin=0.01))

    assert rejected.route is None
    assert rejected.decision_reason is DecisionReason.AMBIGUOUS_MARGIN
    assert accepted.route_name == "alpha"
    assert accepted.candidates[0].threshold == 0.6


def test_scored_example_uses_raw_candidates_even_when_result_rejected():
    result = RouterResult(
        score=0.4,
        candidates=[
            RouteCandidate(route_name="alpha", score=0.4, threshold=0.5, passed_threshold=False),
        ],
        decision_reason=DecisionReason.BELOW_THRESHOLD,
    )

    example = ScoredExample.from_result(
        example_id="validation-1",
        split="validation",
        expected_route="alpha",
        result=result,
    )

    assert example.candidates == (("alpha", 0.4),)


def test_calibration_artifact_links_to_prediction_hash():
    fitted = fit_global_policy(calibration_examples(), min_known_coverage=2 / 3)
    prediction_hash = hashlib.sha256(b"predictions").hexdigest()

    artifact = calibration_artifact(
        fitted,
        dataset={"name": "fixture", "split": "validation"},
        source_predictions_sha256=prediction_hash,
    )

    assert artifact["policy"]["threshold"] == fitted.policy.threshold
    assert artifact["source_predictions_sha256"] == prediction_hash


def test_per_route_thresholds_handle_different_score_ranges():
    examples = [
        scored("a-high", "alpha", ("alpha", 0.9)),
        scored("a-low", "alpha", ("alpha", 0.8)),
        scored("a-ood", None, ("alpha", 0.7)),
        scored("b-high", "beta", ("beta", 0.4)),
        scored("b-low", "beta", ("beta", 0.35)),
        scored("b-ood", None, ("beta", 0.3)),
    ]

    global_result = fit_global_policy(examples, min_known_coverage=1.0)
    per_route_result = fit_per_route_policy(
        examples,
        min_known_coverage=1.0,
        min_top_examples_per_route=1,
    )

    assert per_route_result.metrics.overall_accuracy > global_result.metrics.overall_accuracy
    assert per_route_result.policy.threshold_for("alpha") > per_route_result.policy.threshold_for("beta")


def test_apply_per_route_policy_exposes_selected_threshold():
    examples = [
        scored("a", "alpha", ("alpha", 0.9)),
        scored("a-ood", None, ("alpha", 0.7)),
        scored("b", "beta", ("beta", 0.4)),
        scored("b-ood", None, ("beta", 0.3)),
    ]
    fitted = fit_per_route_policy(
        examples,
        min_known_coverage=1.0,
        min_top_examples_per_route=1,
    )
    routes = {
        "alpha": Route(name="alpha", utterances=["alpha"]),
        "beta": Route(name="beta", utterances=["beta"]),
    }
    raw = RouterResult(
        score=0.75,
        candidates=[
            RouteCandidate(
                route_name="alpha",
                score=0.75,
                threshold=-1.0,
                passed_threshold=True,
            )
        ],
        decision_reason=DecisionReason.MATCHED,
    )

    result = apply_per_route_policy(raw, routes=routes, policy=fitted.policy)

    assert result.route is None
    assert result.candidates[0].threshold == fitted.policy.threshold_for("alpha")


def test_vectorized_policy_metrics_match_reference_implementation():
    rng = random.Random(42)
    route_names = ("alpha", "beta", "gamma")
    examples = []
    for index in range(200):
        expected = rng.choice((*route_names, None))
        candidate_count = rng.randrange(0, 4)
        names = rng.sample(route_names, candidate_count)
        scores = sorted((rng.uniform(-0.2, 1.0) for _ in names), reverse=True)
        examples.append(scored(str(index), expected, *zip(names, scores)))

    global_policy = GlobalPolicy(threshold=0.47, margin=0.08)
    per_route_policy = PerRoutePolicy(
        default_threshold=0.47,
        thresholds={"alpha": 0.62, "beta": 0.31},
        margin=0.08,
    )

    assert evaluate_global_policy(examples, global_policy) == reference_metrics(
        examples,
        lambda _route_name: global_policy.threshold,
        global_policy.margin,
    )
    assert evaluate_per_route_policy(examples, per_route_policy) == reference_metrics(
        examples,
        per_route_policy.threshold_for,
        per_route_policy.margin,
    )
