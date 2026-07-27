import pytest

from benchmarks.baselines import (
    ExactCosineBaseline,
    ExactStringBaseline,
    LogisticRegressionBaseline,
    SemanticRouterBaseline,
)
from synaptoroute import DecisionReason, Route


def test_exact_string_baseline_matches_normalized_text_and_abstains():
    baseline = ExactStringBaseline(
        [
            Route(name="greeting", utterances=["Hello there"], threshold=0.5),
            Route(name="farewell", utterances=["Goodbye"], threshold=0.5),
        ]
    )

    assert baseline("  HELLO   there ").name == "greeting"
    missing = baseline.match("something else")
    assert missing.route is None
    assert missing.decision_reason is DecisionReason.NO_CANDIDATES


def test_exact_string_baseline_abstains_on_cross_route_collision():
    baseline = ExactStringBaseline(
        [
            Route(name="first", utterances=["same text"], threshold=0.5),
            Route(name="second", utterances=["same text"], threshold=0.5),
        ]
    )

    result = baseline.match("same text")
    assert result.route is None
    assert result.margin == 0.0
    assert result.decision_reason is DecisionReason.AMBIGUOUS_MARGIN


def test_exact_cosine_baseline_uses_unique_route_scores(fake_encoder):
    baseline = ExactCosineBaseline(
        fake_encoder,
        [
            Route(name="greeting", utterances=["hello", "hi"], threshold=0.5),
            Route(name="farewell", utterances=["bye", "goodbye"], threshold=0.5),
        ],
    )

    result = baseline.match("hello")
    assert result.route_name == "greeting"
    assert result.score == pytest.approx(1.0)
    assert [candidate.route_name for candidate in result.candidates].count("greeting") == 1


def test_exact_cosine_baseline_applies_margin_abstention(fake_encoder):
    baseline = ExactCosineBaseline(
        fake_encoder,
        [
            Route(name="first", utterances=["hello"], threshold=0.5),
            Route(name="second", utterances=["hi"], threshold=0.5),
        ],
        margin=0.1,
    )

    result = baseline.match("hello")
    assert result.route is None
    assert result.decision_reason is DecisionReason.AMBIGUOUS_MARGIN


def test_logistic_regression_baseline_is_deterministic(fake_encoder):
    routes = [
        Route(name="greeting", utterances=["hello", "hi"], threshold=0.5),
        Route(name="farewell", utterances=["bye", "goodbye"], threshold=0.5),
    ]
    first = LogisticRegressionBaseline(fake_encoder, routes, random_state=17)
    second = LogisticRegressionBaseline(fake_encoder, routes, random_state=17)

    first_result = first.match("hello")
    second_result = second.match("hello")

    assert first_result.route_name == "greeting"
    assert second_result.route_name == first_result.route_name
    assert second_result.score == pytest.approx(first_result.score)
    assert second_result.candidates == first_result.candidates


def test_logistic_regression_threshold_can_force_abstention(fake_encoder):
    baseline = LogisticRegressionBaseline(
        fake_encoder,
        [
            Route(name="greeting", utterances=["hello", "hi"], threshold=0.5),
            Route(name="farewell", utterances=["bye", "goodbye"], threshold=0.5),
        ],
        threshold=0.99,
    )

    result = baseline.match("hello")
    assert result.route is None
    assert result.decision_reason is DecisionReason.BELOW_THRESHOLD


def test_semantic_router_adapter_uses_shared_encoder_when_extra_is_installed(fake_encoder):
    pytest.importorskip("semantic_router")
    baseline = SemanticRouterBaseline(
        fake_encoder,
        [
            Route(name="greeting", utterances=["hello", "hi"], threshold=-1.0),
            Route(name="farewell", utterances=["bye", "goodbye"], threshold=-1.0),
        ],
    )

    result = baseline.match("hello")

    assert result.route_name == "greeting"
    assert result.candidates
