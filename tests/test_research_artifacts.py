import random

import pytest

from benchmarks.prediction_io import (
    prediction_record,
    query_digest,
    read_prediction_jsonl,
    write_prediction_jsonl,
)
from benchmarks.bench_local_pilot import run_pilot
from benchmarks.research_datasets import IntentExample, prepare_routing_dataset
from synaptoroute import DecisionReason, RouterResult


def make_training_examples():
    return [
        IntentExample(f"a-{index}", f"alpha example {index}", "alpha", "train")
        for index in range(6)
    ] + [
        IntentExample(f"b-{index}", f"beta example {index}", "beta", "train")
        for index in range(6)
    ]


def make_evaluation_examples():
    return [
        IntentExample("test-a", "new alpha request", "alpha", "test"),
        IntentExample("test-b", "new beta request", "beta", "test"),
        IntentExample("test-ood", "unsupported request", None, "test"),
    ]


def test_dataset_preparation_is_deterministic_and_order_independent():
    training = make_training_examples()
    shuffled = training.copy()
    random.Random(99).shuffle(shuffled)

    first = prepare_routing_dataset(
        name="fixture",
        version="1",
        license="test-only",
        training_examples=training,
        evaluation_examples=make_evaluation_examples(),
        examples_per_route=3,
        seed=42,
    )
    second = prepare_routing_dataset(
        name="fixture",
        version="1",
        license="test-only",
        training_examples=shuffled,
        evaluation_examples=list(reversed(make_evaluation_examples())),
        examples_per_route=3,
        seed=42,
    )

    assert first.route_examples == second.route_examples
    assert first.evaluation_examples == second.evaluation_examples
    assert [route.name for route in first.to_routes()] == ["alpha", "beta"]
    assert first.manifest_metadata()["route_count"] == 2
    assert first.manifest_metadata()["query_count"] == 3


def test_dataset_preparation_rejects_split_leakage():
    evaluation = make_evaluation_examples()
    evaluation[0] = IntentExample("test-a", "alpha example 0", "alpha", "test")

    with pytest.raises(ValueError, match="overlap evaluation text"):
        prepare_routing_dataset(
            name="fixture",
            version="1",
            license="test-only",
            training_examples=make_training_examples(),
            evaluation_examples=evaluation,
            examples_per_route=6,
            seed=42,
        )


def test_dataset_preparation_rejects_ambiguous_training_text():
    training = make_training_examples()
    training.append(IntentExample("ambiguous", "alpha example 0", "beta", "train"))

    with pytest.raises(ValueError, match="multiple routes"):
        prepare_routing_dataset(
            name="fixture",
            version="1",
            license="test-only",
            training_examples=training,
            evaluation_examples=make_evaluation_examples(),
            examples_per_route=3,
            seed=42,
        )


def test_prediction_artifact_round_trip_omits_query_by_default(tmp_path):
    result = RouterResult(decision_reason=DecisionReason.BELOW_THRESHOLD, score=0.4)
    record = prediction_record(
        example_id="example-1",
        query="private query",
        expected_route=None,
        result=result,
        latency_seconds=0.012,
        metadata={"split": "test"},
    )

    assert "query" not in record
    assert record["query_sha256"] == query_digest("private query")
    assert record["correct"] is True
    assert record["latency_ms"] == pytest.approx(12.0)

    output_path = write_prediction_jsonl(tmp_path / "predictions.jsonl", [record])
    loaded = read_prediction_jsonl(output_path)
    assert loaded == [record]


def test_prediction_artifact_rejects_invalid_record(tmp_path):
    with pytest.raises(ValueError, match="invalid prediction record"):
        write_prediction_jsonl(tmp_path / "predictions.jsonl", [{"example_id": "broken"}])


def test_local_pilot_writes_complete_ineligible_artifact(tmp_path):
    summary = run_pilot(tmp_path, seed=42, threshold=0.2, margin=0.05)

    assert summary["paper_evidence_eligible"] is False
    assert summary["dataset"]["route_count"] == 17
    assert summary["dataset"]["query_count"] == 241
    assert set(summary["systems"]) == {
        "synaptoroute",
        "exact_string",
        "exact_cosine",
        "logistic_regression",
    }
    assert (tmp_path / "local_pilot_summary.json").is_file()
    for system_name in summary["systems"]:
        predictions = read_prediction_jsonl(tmp_path / f"predictions_{system_name}.jsonl")
        assert len(predictions) == 241
