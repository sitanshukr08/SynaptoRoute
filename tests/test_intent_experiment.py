from benchmarks.hf_intent_datasets import ExternalDatasetBundle, HFDatasetSpec
from benchmarks.prediction_io import read_prediction_jsonl
from benchmarks.research_datasets import IntentExample, PreparedRoutingDataset
from benchmarks.run_intent_experiment import run_bundle_experiment


def test_bundle_experiment_calibrates_before_test_and_writes_artifacts(tmp_path, fake_encoder):
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
            IntentExample("test-hello", "hello", "greeting", "test"),
            IntentExample("test-bye", "bye", "farewell", "test"),
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
            IntentExample("val-ood", "unknown request", None, "validation"),
        ),
        label_mapping={"greeting": "greeting", "farewell": "farewell"},
        split_fingerprints={"train": "train", "validation": "validation", "test": "test"},
        evaluation_limit=None,
    )

    summary = run_bundle_experiment(
        bundle=bundle,
        encoder=fake_encoder,
        output_dir=tmp_path,
        min_known_coverage=0.5,
        max_ood_false_acceptance_rate=None,
    )

    assert summary["paper_evidence_eligible"] is False
    assert summary["systems"]["synaptoroute"]["calibration"] is not None
    assert summary["systems"]["exact_string"]["calibration"] is None
    synaptoroute_metrics = summary["systems"]["synaptoroute"]["test"]
    assert synaptoroute_metrics["ood_auroc"] is not None
    assert synaptoroute_metrics["ood_auprc"] is not None
    assert synaptoroute_metrics["ood_fpr_at_95_tpr"] is not None
    assert 0.0 <= synaptoroute_metrics["selective_risk_coverage_auc"] <= 1.0
    assert 0.0 <= synaptoroute_metrics["known_coverage"] <= 1.0
    assert (tmp_path / "calibration_synaptoroute.json").is_file()
    assert (tmp_path / "experiment_summary.json").is_file()
    for system_name in summary["systems"]:
        predictions = read_prediction_jsonl(tmp_path / f"test_predictions_{system_name}.jsonl")
        assert len(predictions) == 3
