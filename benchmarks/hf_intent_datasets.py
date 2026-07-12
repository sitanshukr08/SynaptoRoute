"""Version-pinned Hugging Face loaders for intent-routing experiments."""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from benchmarks.research_datasets import (
    IntentExample,
    PreparedRoutingDataset,
    normalized_text,
    prepare_routing_dataset,
)


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class HFDatasetSpec:
    name: str
    dataset_id: str
    revision: str
    version: str
    license: str
    train_split: str
    test_split: str
    text_field: str
    label_field: str
    subset: str | None = None
    validation_split: str | None = None
    ood_labels: frozenset[str] = frozenset()

    def __post_init__(self):
        if not COMMIT_PATTERN.fullmatch(self.revision):
            raise ValueError("dataset revision must be a full 40-character commit hash")
        for field_name in (
            "name",
            "dataset_id",
            "version",
            "license",
            "train_split",
            "test_split",
            "text_field",
            "label_field",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")


BANKING77_SPEC = HFDatasetSpec(
    name="banking77",
    dataset_id="PolyAI/banking77",
    revision="796a4623935746f71378f0ebd435635a8ce08e50",
    version="1.1.0",
    license="CC-BY-4.0",
    train_split="train",
    test_split="test",
    text_field="text",
    label_field="label",
)


CLINC150_SMALL_SPEC = HFDatasetSpec(
    name="clinc150_oos_small",
    dataset_id="DeepPavlov/clinc_oos",
    revision="9b995dc4a780cfabf0c7bb044bab7f48a3762c8d",
    version="small",
    license="CC-BY-3.0",
    subset="small",
    train_split="train",
    validation_split="validation",
    test_split="test",
    text_field="text",
    label_field="label_text",
    ood_labels=frozenset({"oos"}),
)


@dataclass(frozen=True)
class ExternalDatasetBundle:
    spec: HFDatasetSpec
    prepared: PreparedRoutingDataset
    calibration_examples: tuple[IntentExample, ...]
    label_mapping: dict[str, str]
    split_fingerprints: dict[str, str]
    evaluation_limit: int | None
    overlap_exclusions: dict[str, int] = field(default_factory=dict)

    def manifest_metadata(self) -> dict[str, Any]:
        metadata = self.prepared.manifest_metadata()
        metadata.update(
            {
                "dataset_id": self.spec.dataset_id,
                "revision": self.spec.revision,
                "subset": self.spec.subset,
                "calibration_count": len(self.calibration_examples),
                "evaluation_limit": self.evaluation_limit,
                "label_mapping": dict(sorted(self.label_mapping.items())),
                "split_fingerprints": dict(sorted(self.split_fingerprints.items())),
                "overlap_exclusions": dict(sorted(self.overlap_exclusions.items())),
            }
        )
        return metadata


def canonical_route_name(label: str) -> str:
    canonical = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip())
    canonical = re.sub(r"_+", "_", canonical).strip("_")
    if not canonical:
        raise ValueError(f"label cannot be converted to a route name: {label!r}")
    return canonical


def _label_decoder(split: Any, label_field: str):
    feature = split.features[label_field]
    if hasattr(feature, "int2str"):
        return lambda value: feature.int2str(int(value))
    return lambda value: str(value)


def _rows_to_examples(
    split: Any,
    *,
    split_name: str,
    text_field: str,
    label_field: str,
    ood_labels: frozenset[str],
    label_mapping: dict[str, str],
) -> list[IntentExample]:
    decode_label = _label_decoder(split, label_field)
    examples: list[IntentExample] = []
    reverse_mapping: dict[str, str] = {canonical: raw for raw, canonical in label_mapping.items()}
    for index, row in enumerate(split):
        raw_label = decode_label(row[label_field])
        if raw_label.casefold() in {label.casefold() for label in ood_labels}:
            label = None
        else:
            canonical = canonical_route_name(raw_label)
            previous_raw = reverse_mapping.get(canonical)
            if previous_raw is not None and previous_raw != raw_label:
                raise ValueError(
                    f"route-name collision after canonicalization: {previous_raw!r}, {raw_label!r}"
                )
            label_mapping[raw_label] = canonical
            reverse_mapping[canonical] = raw_label
            label = canonical
        examples.append(
            IntentExample(
                example_id=f"{split_name}:{index}",
                text=str(row[text_field]),
                label=label,
                split=split_name,
            )
        )
    return examples


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\0partition\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def partition_training_for_calibration(
    examples: Sequence[IntentExample],
    *,
    examples_per_route: int,
    calibration_per_route: int,
    seed: int,
) -> tuple[list[IntentExample], list[IntentExample]]:
    if examples_per_route < 1 or calibration_per_route < 1:
        raise ValueError("route and calibration counts must be positive")
    grouped: dict[str, dict[str, IntentExample]] = defaultdict(dict)
    text_labels: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        if example.label is not None:
            normalized = normalized_text(example.text)
            text_labels[normalized].add(example.label)
            existing = grouped[example.label].get(normalized)
            if existing is None or example.example_id < existing.example_id:
                grouped[example.label][normalized] = example

    ambiguous = sorted(text for text, labels in text_labels.items() if len(labels) > 1)
    if ambiguous:
        raise ValueError(f"training texts map to multiple routes: {ambiguous[:5]}")

    route_pool: list[IntentExample] = []
    calibration: list[IntentExample] = []
    required = examples_per_route + calibration_per_route
    for label in sorted(grouped):
        available = sorted(grouped[label].values(), key=lambda example: example.example_id)
        if len(available) < required:
            raise ValueError(f"route {label!r} has {len(available)} examples; {required} required")
        sampler = random.Random(_stable_seed(seed, label))
        selected = sampler.sample(available, required)
        route_pool.extend(selected[:examples_per_route])
        calibration.extend(
            IntentExample(
                example_id=example.example_id,
                text=example.text,
                label=example.label,
                split="calibration",
            )
            for example in selected[examples_per_route:]
        )
    return route_pool, sorted(calibration, key=lambda example: example.example_id)


def deterministic_evaluation_limit(
    examples: Sequence[IntentExample],
    *,
    limit: int | None,
    seed: int,
) -> list[IntentExample]:
    if limit is None or limit >= len(examples):
        return list(examples)
    if limit < 1:
        raise ValueError("evaluation limit must be positive")

    grouped: dict[str | None, list[IntentExample]] = defaultdict(list)
    for example in examples:
        grouped[example.label].append(example)

    labels = sorted(grouped, key=lambda label: "\0OOD" if label is None else label)
    if limit < len(labels):
        label_sampler = random.Random(_stable_seed(seed, "evaluation-labels"))
        selected_labels = label_sampler.sample(labels, limit)
        allocations = {label: 1 for label in selected_labels}
    else:
        exact_allocations = {
            label: limit * len(grouped[label]) / len(examples)
            for label in labels
        }
        allocations = {label: max(1, int(exact_allocations[label])) for label in labels}
        while sum(allocations.values()) > limit:
            label = min(
                (candidate for candidate in labels if allocations[candidate] > 1),
                key=lambda candidate: (
                    exact_allocations[candidate] - allocations[candidate],
                    "\0OOD" if candidate is None else candidate,
                ),
            )
            allocations[label] -= 1
        while sum(allocations.values()) < limit:
            label = max(
                (candidate for candidate in labels if allocations[candidate] < len(grouped[candidate])),
                key=lambda candidate: (
                    exact_allocations[candidate] - allocations[candidate],
                    "\0OOD" if candidate is None else candidate,
                ),
            )
            allocations[label] += 1

    selected: list[IntentExample] = []
    for label, count in allocations.items():
        available = sorted(grouped[label], key=lambda example: example.example_id)
        sampler = random.Random(_stable_seed(seed, f"evaluation:{label!r}"))
        selected.extend(sampler.sample(available, count))
    return sorted(selected, key=lambda example: example.example_id)


def _validate_calibration_disjoint(
    prepared: PreparedRoutingDataset,
    calibration: Sequence[IntentExample],
) -> None:
    route_texts = {
        normalized_text(text)
        for utterances in prepared.route_examples.values()
        for text in utterances
    }
    calibration_texts = {normalized_text(example.text) for example in calibration}
    test_texts = {normalized_text(example.text) for example in prepared.evaluation_examples}
    route_calibration_overlap = route_texts & calibration_texts
    calibration_test_overlap = calibration_texts & test_texts
    if route_calibration_overlap:
        raise ValueError(
            "route examples overlap calibration text: "
            f"{sorted(route_calibration_overlap)[:5]}"
        )
    if calibration_test_overlap:
        raise ValueError(
            "calibration examples overlap test text: "
            f"{sorted(calibration_test_overlap)[:5]}"
        )


def _exclude_text_overlaps(
    examples: Sequence[IntentExample],
    blocked_texts: set[str],
) -> tuple[list[IntentExample], int]:
    retained = [
        example
        for example in examples
        if normalized_text(example.text) not in blocked_texts
    ]
    return retained, len(examples) - len(retained)


def load_external_intent_dataset(
    spec: HFDatasetSpec,
    *,
    examples_per_route: int,
    seed: int,
    calibration_per_route: int = 10,
    evaluation_limit: int | None = None,
    cache_dir: Path | str | None = None,
) -> ExternalDatasetBundle:
    """Load a pinned dataset and keep route, calibration, and test data disjoint."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Install synaptoroute[benchmark] to load external datasets") from error

    load_kwargs: dict[str, Any] = {
        "path": spec.dataset_id,
        "revision": spec.revision,
    }
    if spec.subset is not None:
        load_kwargs["name"] = spec.subset
    if cache_dir is not None:
        load_kwargs["cache_dir"] = str(cache_dir)
    dataset = load_dataset(**load_kwargs)

    label_mapping: dict[str, str] = {}
    training = _rows_to_examples(
        dataset[spec.train_split],
        split_name=spec.train_split,
        text_field=spec.text_field,
        label_field=spec.label_field,
        ood_labels=spec.ood_labels,
        label_mapping=label_mapping,
    )
    full_test = _rows_to_examples(
        dataset[spec.test_split],
        split_name=spec.test_split,
        text_field=spec.text_field,
        label_field=spec.label_field,
        ood_labels=spec.ood_labels,
        label_mapping=label_mapping,
    )
    test_texts = {normalized_text(example.text) for example in full_test}
    overlap_exclusions: dict[str, int] = {}

    if spec.validation_split is not None:
        calibration = _rows_to_examples(
            dataset[spec.validation_split],
            split_name="validation",
            text_field=spec.text_field,
            label_field=spec.label_field,
            ood_labels=spec.ood_labels,
            label_mapping=label_mapping,
        )
        calibration, calibration_excluded = _exclude_text_overlaps(calibration, test_texts)
        overlap_exclusions["calibration_excluded_test_overlap"] = calibration_excluded
        heldout_texts = test_texts | {
            normalized_text(example.text) for example in calibration
        }
        route_pool, training_excluded = _exclude_text_overlaps(training, heldout_texts)
        route_pool = [example for example in route_pool if example.label is not None]
    else:
        training, training_excluded = _exclude_text_overlaps(training, test_texts)
        route_pool, calibration = partition_training_for_calibration(
            training,
            examples_per_route=examples_per_route,
            calibration_per_route=calibration_per_route,
            seed=seed,
        )
    overlap_exclusions["training_excluded_heldout_overlap"] = training_excluded
    test = deterministic_evaluation_limit(full_test, limit=evaluation_limit, seed=seed)

    prepared = prepare_routing_dataset(
        name=spec.name,
        version=f"{spec.version}@{spec.revision}",
        license=spec.license,
        training_examples=route_pool,
        evaluation_examples=test,
        examples_per_route=examples_per_route,
        seed=seed,
    )
    _validate_calibration_disjoint(prepared, calibration)
    split_fingerprints = {
        split_name: str(getattr(dataset[split_name], "_fingerprint", "unknown"))
        for split_name in {spec.train_split, spec.test_split, spec.validation_split}
        if split_name is not None
    }
    return ExternalDatasetBundle(
        spec=spec,
        prepared=prepared,
        calibration_examples=tuple(sorted(calibration, key=lambda example: example.example_id)),
        label_mapping=label_mapping,
        split_fingerprints=split_fingerprints,
        evaluation_limit=evaluation_limit,
        overlap_exclusions=overlap_exclusions,
    )
