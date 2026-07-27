import sys
from types import SimpleNamespace

import pytest

from benchmarks.hf_intent_datasets import (
    HFDatasetSpec,
    canonical_route_name,
    deterministic_evaluation_limit,
    load_external_intent_dataset,
    partition_training_for_calibration,
)
from benchmarks.research_datasets import IntentExample


class FakeClassLabel:
    def __init__(self, names):
        self.names = names

    def int2str(self, value):
        return self.names[value]


class FakeValue:
    pass


class FakeSplit(list):
    def __init__(self, rows, feature, fingerprint):
        super().__init__(rows)
        self.features = {"label": feature, "label_text": feature}
        self._fingerprint = fingerprint


def test_dataset_spec_requires_full_revision():
    with pytest.raises(ValueError, match="full 40-character"):
        HFDatasetSpec(
            name="fixture",
            dataset_id="owner/fixture",
            revision="main",
            version="1",
            license="test",
            train_split="train",
            test_split="test",
            text_field="text",
            label_field="label",
        )


def test_route_name_canonicalization_is_explicit():
    assert canonical_route_name("refund status?") == "refund_status"
    with pytest.raises(ValueError):
        canonical_route_name("???")


def test_training_partition_is_deterministic_and_disjoint():
    examples = [
        IntentExample(f"{label}-{index}", f"{label} text {index}", label, "train")
        for label in ("alpha", "beta")
        for index in range(6)
    ]

    route_first, calibration_first = partition_training_for_calibration(
        examples,
        examples_per_route=3,
        calibration_per_route=2,
        seed=42,
    )
    route_second, calibration_second = partition_training_for_calibration(
        list(reversed(examples)),
        examples_per_route=3,
        calibration_per_route=2,
        seed=42,
    )

    assert {example.example_id for example in route_first} == {
        example.example_id for example in route_second
    }
    assert calibration_first == calibration_second
    assert {example.example_id for example in route_first}.isdisjoint(
        example.example_id for example in calibration_first
    )


def test_training_partition_samples_unique_normalized_text():
    examples = [
        IntentExample("alpha-1", "Alpha example", "alpha", "train"),
        IntentExample("alpha-duplicate", "  alpha   example ", "alpha", "train"),
        IntentExample("alpha-2", "Alpha second", "alpha", "train"),
        IntentExample("alpha-3", "Alpha third", "alpha", "train"),
    ]

    route_pool, calibration = partition_training_for_calibration(
        examples,
        examples_per_route=2,
        calibration_per_route=1,
        seed=42,
    )

    selected_text = {" ".join(example.text.casefold().split()) for example in route_pool + calibration}
    assert len(selected_text) == 3


def test_evaluation_limit_is_deterministic_and_stratified():
    examples = [
        IntentExample(f"alpha-{index}", f"alpha {index}", "alpha", "test")
        for index in range(20)
    ] + [
        IntentExample(f"{label}-{index}", f"{label} {index}", label, "test")
        for label in ("beta", None)
        for index in range(5)
    ]

    first = deterministic_evaluation_limit(examples, limit=12, seed=42)
    second = deterministic_evaluation_limit(list(reversed(examples)), limit=12, seed=42)
    counts = {label: sum(example.label == label for example in first) for label in ("alpha", "beta", None)}

    assert first == second
    assert len(first) == 12
    assert counts == {"alpha": 8, "beta": 2, None: 2}


def test_loader_pins_data_and_creates_train_calibration_test_boundaries(monkeypatch, tmp_path):
    labels = FakeClassLabel(["alpha", "beta"])
    dataset = {
        "train": FakeSplit(
            [
                {"text": f"{label} train {index}", "label": label_id}
                for label_id, label in enumerate(("alpha", "beta"))
                for index in range(5)
            ],
            labels,
            "train-fingerprint",
        ),
        "test": FakeSplit(
            [
                {"text": "alpha test", "label": 0},
                {"text": "beta test", "label": 1},
            ],
            labels,
            "test-fingerprint",
        ),
    }
    calls = []

    def load_dataset(**kwargs):
        calls.append(kwargs)
        return dataset

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    spec = HFDatasetSpec(
        name="fixture",
        dataset_id="owner/fixture",
        revision="a" * 40,
        version="1",
        license="test-only",
        train_split="train",
        test_split="test",
        text_field="text",
        label_field="label",
    )

    bundle = load_external_intent_dataset(
        spec,
        examples_per_route=2,
        calibration_per_route=2,
        seed=42,
        cache_dir=tmp_path,
    )

    assert calls == [
        {
            "path": "owner/fixture",
            "revision": "a" * 40,
            "cache_dir": str(tmp_path),
        }
    ]
    assert len(bundle.prepared.route_examples) == 2
    assert all(len(examples) == 2 for examples in bundle.prepared.route_examples.values())
    assert len(bundle.calibration_examples) == 4
    assert len(bundle.prepared.evaluation_examples) == 2
    assert bundle.split_fingerprints == {
        "train": "train-fingerprint",
        "test": "test-fingerprint",
    }


def test_loader_maps_official_oos_label_to_abstention_target(monkeypatch):
    feature = FakeValue()
    dataset = {
        "train": FakeSplit(
            [
                {"text": "alpha train one", "label_text": "alpha"},
                {"text": "alpha train two", "label_text": "alpha"},
                {"text": "alpha validation", "label_text": "alpha"},
                {"text": "beta train one", "label_text": "beta"},
                {"text": "beta train two", "label_text": "beta"},
            ],
            feature,
            "train",
        ),
        "validation": FakeSplit(
            [
                {"text": "alpha validation", "label_text": "alpha"},
                {"text": "shared heldout text", "label_text": "alpha"},
                {"text": "unsupported validation", "label_text": "oos"},
            ],
            feature,
            "validation",
        ),
        "test": FakeSplit(
            [
                {"text": "shared heldout text", "label_text": "alpha"},
                {"text": "unsupported test", "label_text": "oos"},
            ],
            feature,
            "test",
        ),
    }
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda **kwargs: dataset))
    spec = HFDatasetSpec(
        name="fixture_oos",
        dataset_id="owner/fixture-oos",
        revision="b" * 40,
        version="1",
        license="test-only",
        subset="small",
        train_split="train",
        validation_split="validation",
        test_split="test",
        text_field="text",
        label_field="label_text",
        ood_labels=frozenset({"oos"}),
    )

    bundle = load_external_intent_dataset(spec, examples_per_route=2, seed=42)

    assert [example.label for example in bundle.calibration_examples] == ["alpha", None]
    assert [example.label for example in bundle.prepared.evaluation_examples] == ["alpha", None]
    assert "oos" not in bundle.label_mapping
    assert bundle.overlap_exclusions == {
        "calibration_excluded_test_overlap": 1,
        "training_excluded_heldout_overlap": 1,
    }
