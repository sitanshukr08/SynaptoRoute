import json

import pytest

from benchmarks.hf_intent_datasets import ExternalDatasetBundle, HFDatasetSpec
from benchmarks.research_datasets import IntentExample, PreparedRoutingDataset
from benchmarks.run_intent_experiment import run_bundle_experiment
from paper.verify_quality_artifacts import (
    QualityArtifactVerificationError,
    verify_quality_artifacts,
)


def _make_quality_run(tmp_path, fake_encoder):
    spec = HFDatasetSpec(
        name="fixture",
        dataset_id="owner/fixture",
        revision="a" * 40,
        version="1",
        license="test-only",
        train_split="train",
        validation_split="validation",
        test_split="test",
        text_field="text",
        label_field="label",
    )
    prepared = PreparedRoutingDataset(
        name="fixture",
        version="1@" + "a" * 40,
        license="test-only",
        seed=42,
        training_split="train",
        evaluation_split="test",
        route_examples={
            "greeting": ("hello", "hi"),
            "farewell": ("bye", "goodbye"),
        },
        evaluation_examples=(
            IntentExample("test-hello", "hello there", "greeting", "test"),
            IntentExample("test-bye", "bye now", "farewell", "test"),
            IntentExample("test-ood", "unknown request", None, "test"),
        ),
        exact_text_overlap_count=0,
    )
    bundle = ExternalDatasetBundle(
        spec=spec,
        prepared=prepared,
        calibration_examples=(
            IntentExample("val-hello", "hello", "greeting", "validation"),
            IntentExample("val-hi", "hi", "greeting", "validation"),
            IntentExample("val-bye", "bye", "farewell", "validation"),
            IntentExample("val-goodbye", "goodbye", "farewell", "validation"),
            IntentExample("val-ood", "different unknown", None, "validation"),
        ),
        label_mapping={"greeting": "greeting", "farewell": "farewell"},
        split_fingerprints={"train": "train", "validation": "validation", "test": "test"},
        evaluation_limit=None,
    )
    run_bundle_experiment(
        bundle=bundle,
        encoder=fake_encoder,
        output_dir=tmp_path,
        min_known_coverage=0.5,
        max_ood_false_acceptance_rate=None,
    )
    return tmp_path / "experiment_summary.json"


def _rewrite_first_record(path, update):
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    update(record)
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_quality_verifier_recomputes_complete_run(tmp_path, fake_encoder):
    summary_path = _make_quality_run(tmp_path, fake_encoder)

    report = verify_quality_artifacts(summary_path)

    assert report["verification_status"] == "valid_unverified_quality_run"
    assert report["paper_evidence_eligible"] is False
    assert report["system_count"] == 5
    assert report["record_counts"] == {
        "policy_per_calibrated_system": 3,
        "probability_per_system": 2,
        "test_per_system": 3,
        "total_records_checked": 37,
    }


def test_quality_verifier_rejects_probability_not_produced_by_model(tmp_path, fake_encoder):
    summary_path = _make_quality_run(tmp_path, fake_encoder)
    path = tmp_path / "test_predictions_synaptoroute.jsonl"

    def change_probability(record):
        record["metadata"]["correctness_probability"] = 0.123456

    _rewrite_first_record(path, change_probability)

    with pytest.raises(QualityArtifactVerificationError, match="probability differs from model"):
        verify_quality_artifacts(summary_path)


def test_quality_verifier_rejects_cross_split_leakage(tmp_path, fake_encoder):
    summary_path = _make_quality_run(tmp_path, fake_encoder)
    policy_record = json.loads(
        (tmp_path / "calibration_predictions_synaptoroute.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    path = tmp_path / "probability_calibration_predictions_synaptoroute.jsonl"

    def leak_policy_identity(record):
        record["example_id"] = policy_record["example_id"]
        record["query_sha256"] = policy_record["query_sha256"]

    _rewrite_first_record(path, leak_policy_identity)

    with pytest.raises(QualityArtifactVerificationError, match="policy/probability example IDs overlap"):
        verify_quality_artifacts(summary_path)


def test_quality_verifier_rejects_raw_query_text(tmp_path, fake_encoder):
    summary_path = _make_quality_run(tmp_path, fake_encoder)
    path = tmp_path / "test_predictions_exact_cosine.jsonl"

    def expose_query(record):
        record["query"] = "raw private request"

    _rewrite_first_record(path, expose_query)

    with pytest.raises(QualityArtifactVerificationError, match="contains raw query text"):
        verify_quality_artifacts(summary_path)
