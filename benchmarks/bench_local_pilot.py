"""End-to-end research-pipeline pilot over repository development fixtures.

The bundled data is synthetic/generated and cannot support paper claims. This
pilot exists to validate deterministic preparation, baselines, prediction
artifacts, result summaries, and evidence capture before external experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from benchmarks.baselines import ExactCosineBaseline, ExactStringBaseline, LogisticRegressionBaseline
from benchmarks.prediction_io import prediction_record, write_prediction_jsonl
from benchmarks.research_datasets import IntentExample, PreparedRoutingDataset, prepare_routing_dataset
from synaptoroute import AdaptiveRouter, RouterResult
from synaptoroute.encoder import BaseEncoder
from synaptoroute.storage import SQLiteStorage


class TfidfEncoder(BaseEncoder):
    """Deterministic local encoder used only for the fixture pilot."""

    model_name = "tfidf-word-bigram-development-pilot"

    def __init__(self, training_texts: list[str]):
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self.vectorizer.fit(training_texts)

    @property
    def requires_lock(self) -> bool:
        return False

    @property
    def dim(self) -> int:
        return len(self.vectorizer.vocabulary_)

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype(np.float32)


def load_repository_fixture(dataset_dir: Path, seed: int, threshold: float) -> PreparedRoutingDataset:
    training: list[IntentExample] = []
    evaluation: list[IntentExample] = []
    for dataset_path in sorted((dataset_dir / "standard").glob("*.json")):
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        for route in payload.get("routes", []):
            for index, utterance in enumerate(route["utterances"]):
                training.append(
                    IntentExample(
                        example_id=f"{dataset_path.stem}:train:{route['name']}:{index}",
                        text=utterance,
                        label=route["name"],
                        split="fixture_train",
                    )
                )
        for index, query in enumerate(payload.get("test_queries", [])):
            evaluation.append(
                IntentExample(
                    example_id=f"{dataset_path.stem}:test:{index}",
                    text=query["query"],
                    label=query.get("expected_route"),
                    split="fixture_test",
                )
            )

    version_path = dataset_dir / "VERSION"
    version = version_path.read_text(encoding="utf-8").strip() if version_path.exists() else "unknown"
    prepared = prepare_routing_dataset(
        name="synaptoroute_repository_fixture",
        version=version,
        license="repository development fixture; not a standalone research dataset",
        training_examples=training,
        evaluation_examples=evaluation,
        examples_per_route=5,
        seed=seed,
    )
    # Rebuild routes once here so invalid thresholds fail before any timed work.
    prepared.to_routes(threshold=threshold)
    return prepared


def evaluate_system(
    *,
    name: str,
    matcher: Any,
    dataset: PreparedRoutingDataset,
    output_dir: Path,
) -> dict[str, Any]:
    records = []
    expected_labels: list[str] = []
    predicted_labels: list[str] = []
    latencies_seconds: list[float] = []
    reasons: Counter[str] = Counter()

    for example in dataset.evaluation_examples:
        start = time.perf_counter()
        result: RouterResult = matcher.match(example.text)
        latency = time.perf_counter() - start
        latencies_seconds.append(latency)
        reasons[result.decision_reason.value] += 1

        expected = example.label if example.label is not None else "OOD"
        predicted = result.route_name if result.route_name is not None else "OOD"
        expected_labels.append(expected)
        predicted_labels.append(predicted)
        records.append(
            prediction_record(
                example_id=example.example_id,
                query=example.text,
                expected_route=example.label,
                result=result,
                latency_seconds=latency,
                metadata={"dataset": dataset.name, "split": example.split, "system": name},
            )
        )

    write_prediction_jsonl(output_dir / f"predictions_{name}.jsonl", records)

    in_domain_indices = [index for index, example in enumerate(dataset.evaluation_examples) if example.label is not None]
    ood_indices = [index for index, example in enumerate(dataset.evaluation_examples) if example.label is None]
    in_domain_accuracy = (
        accuracy_score(
            [expected_labels[index] for index in in_domain_indices],
            [predicted_labels[index] for index in in_domain_indices],
        )
        if in_domain_indices
        else None
    )
    ood_recall = (
        sum(predicted_labels[index] == "OOD" for index in ood_indices) / len(ood_indices)
        if ood_indices
        else None
    )
    return {
        "query_count": len(records),
        "overall_accuracy": float(accuracy_score(expected_labels, predicted_labels)),
        "macro_f1": float(f1_score(expected_labels, predicted_labels, average="macro", zero_division=0)),
        "in_domain_accuracy": float(in_domain_accuracy) if in_domain_accuracy is not None else None,
        "ood_recall": float(ood_recall) if ood_recall is not None else None,
        "latency_p50_ms": float(np.percentile(latencies_seconds, 50) * 1000.0),
        "latency_p95_ms": float(np.percentile(latencies_seconds, 95) * 1000.0),
        "decision_reasons": dict(sorted(reasons.items())),
    }


def run_pilot(output_dir: Path, seed: int, threshold: float, margin: float) -> dict[str, Any]:
    dataset_dir = Path(__file__).resolve().parent / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_repository_fixture(dataset_dir, seed=seed, threshold=threshold)
    routes = dataset.to_routes(threshold=threshold)
    training_texts = [utterance for route in routes for utterance in route.utterances]
    encoder = TfidfEncoder(training_texts)

    router = AdaptiveRouter(
        encoder=encoder,
        storage=SQLiteStorage(":memory:"),
        margin=margin,
    )
    for route in routes:
        router.add_route(route)

    systems = {
        "synaptoroute": router,
        "exact_string": ExactStringBaseline(routes),
        "exact_cosine": ExactCosineBaseline(encoder, routes, margin=margin),
        "logistic_regression": LogisticRegressionBaseline(
            encoder,
            routes,
            threshold=threshold,
            margin=margin,
            random_state=seed,
        ),
    }
    try:
        metrics = {
            name: evaluate_system(
                name=name,
                matcher=matcher,
                dataset=dataset,
                output_dir=output_dir,
            )
            for name, matcher in systems.items()
        }
    finally:
        router.close()

    summary = {
        "benchmark": "local_research_pipeline_pilot",
        "status": "development_fixture_only",
        "paper_evidence_eligible": False,
        "dataset": dataset.manifest_metadata(),
        "configuration": {
            "seed": seed,
            "threshold": threshold,
            "margin": margin,
            "encoder": encoder.model_name,
            "synaptoroute_index": type(router.index).__name__,
        },
        "systems": metrics,
        "notes": [
            "The repository fixture is synthetic/generated and not independently validated.",
            "Thresholds are pilot constants rather than validation-calibrated values.",
            "Comparative values test the pipeline and must not appear in a paper results table.",
        ],
    }
    (output_dir / "local_pilot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = os.environ.get("SYNAPTOROUTE_RUN_DIR", "benchmark_results/local_pilot")
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.05)
    args = parser.parse_args()

    summary = run_pilot(
        output_dir=Path(args.output_dir),
        seed=args.seed,
        threshold=args.threshold,
        margin=args.margin,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
