"""Deterministic dataset preparation with explicit leakage checks."""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from synaptoroute.models import Route


ROUTE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


@dataclass(frozen=True)
class IntentExample:
    example_id: str
    text: str
    label: str | None
    split: str

    def __post_init__(self):
        if not self.example_id:
            raise ValueError("example_id must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if not self.split:
            raise ValueError("split must not be empty")
        if self.label is not None and not ROUTE_NAME_PATTERN.fullmatch(self.label):
            raise ValueError(f"label is not a valid route name: {self.label!r}")


@dataclass(frozen=True)
class PreparedRoutingDataset:
    name: str
    version: str
    license: str
    seed: int
    training_split: str
    evaluation_split: str
    route_examples: dict[str, tuple[str, ...]]
    evaluation_examples: tuple[IntentExample, ...]
    exact_text_overlap_count: int

    def to_routes(self, threshold: float = 0.5) -> list[Route]:
        return [
            Route(name=route_name, utterances=list(utterances), threshold=threshold)
            for route_name, utterances in sorted(self.route_examples.items())
        ]

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "split": f"{self.training_split}->{self.evaluation_split}",
            "seed": self.seed,
            "route_count": len(self.route_examples),
            "query_count": len(self.evaluation_examples),
            "license": self.license,
            "examples_per_route": {
                route_name: len(examples)
                for route_name, examples in sorted(self.route_examples.items())
            },
            "exact_text_overlap_count": self.exact_text_overlap_count,
        }


def _stable_label_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _validate_unique_ids(examples: list[IntentExample], collection_name: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for example in examples:
        if example.example_id in seen:
            duplicates.add(example.example_id)
        seen.add(example.example_id)
    if duplicates:
        raise ValueError(f"duplicate {collection_name} example IDs: {sorted(duplicates)}")


def prepare_routing_dataset(
    *,
    name: str,
    version: str,
    license: str,
    training_examples: list[IntentExample],
    evaluation_examples: list[IntentExample],
    examples_per_route: int,
    seed: int,
    reject_exact_text_overlap: bool = True,
) -> PreparedRoutingDataset:
    """Create deterministic routes from training data and preserve evaluation data."""
    if not name or not version or not license:
        raise ValueError("name, version, and license must be non-empty")
    if examples_per_route < 1:
        raise ValueError("examples_per_route must be positive")
    if not training_examples or not evaluation_examples:
        raise ValueError("training and evaluation examples must not be empty")

    _validate_unique_ids(training_examples, "training")
    _validate_unique_ids(evaluation_examples, "evaluation")
    shared_ids = {example.example_id for example in training_examples} & {
        example.example_id for example in evaluation_examples
    }
    if shared_ids:
        raise ValueError(f"example IDs appear in both splits: {sorted(shared_ids)}")

    grouped: dict[str, list[str]] = defaultdict(list)
    text_labels: dict[str, set[str]] = defaultdict(set)
    for example in training_examples:
        if example.label is None:
            raise ValueError("training examples must have route labels")
        normalized = normalized_text(example.text)
        grouped[example.label].append(example.text.strip())
        text_labels[normalized].add(example.label)

    ambiguous_training_texts = sorted(text for text, labels in text_labels.items() if len(labels) > 1)
    if ambiguous_training_texts:
        raise ValueError(
            "training texts map to multiple routes: "
            f"{ambiguous_training_texts[:5]}"
        )

    route_examples: dict[str, tuple[str, ...]] = {}
    for label in sorted(grouped):
        unique_by_normalized: dict[str, str] = {}
        for text in sorted(grouped[label]):
            unique_by_normalized.setdefault(normalized_text(text), text)
        available = sorted(unique_by_normalized.values())
        if len(available) < examples_per_route:
            raise ValueError(
                f"route {label!r} has {len(available)} unique examples; "
                f"{examples_per_route} required"
            )
        sampler = random.Random(_stable_label_seed(seed, label))
        selected = tuple(sorted(sampler.sample(available, examples_per_route)))
        route_examples[label] = selected

    selected_training_texts = {
        normalized_text(text)
        for utterances in route_examples.values()
        for text in utterances
    }
    evaluation_texts = {normalized_text(example.text) for example in evaluation_examples}
    exact_overlap = selected_training_texts & evaluation_texts
    if exact_overlap and reject_exact_text_overlap:
        raise ValueError(
            "selected route examples overlap evaluation text: "
            f"{sorted(exact_overlap)[:5]}"
        )

    training_splits = sorted({example.split for example in training_examples})
    evaluation_splits = sorted({example.split for example in evaluation_examples})
    return PreparedRoutingDataset(
        name=name,
        version=version,
        license=license,
        seed=seed,
        training_split="+".join(training_splits),
        evaluation_split="+".join(evaluation_splits),
        route_examples=route_examples,
        evaluation_examples=tuple(sorted(evaluation_examples, key=lambda example: example.example_id)),
        exact_text_overlap_count=len(exact_overlap),
    )
