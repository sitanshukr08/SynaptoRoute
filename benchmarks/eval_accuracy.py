"""Compare routing quality without inventing unsupported baseline metrics."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def unique_route_names(candidates: Iterable[tuple[float, str]]) -> list[str]:
    """Collapse utterance-level candidates into a route ranking."""
    names: list[str] = []
    seen: set[str] = set()
    for _, route_name in candidates:
        if route_name not in seen:
            seen.add(route_name)
            names.append(route_name)
    return names


def _rank_synaptoroute(router: Any, query: str, required_routes: int) -> list[str]:
    embedding = router.encoder.encode(query)
    total_vectors = router.index.total_vectors
    if total_vectors == 0:
        return []

    search_k = min(total_vectors, max(required_routes, 8))
    while True:
        candidates = router.index.search(np.array([embedding]), top_k=search_k)[0]
        names = unique_route_names(candidates)
        if len(names) >= required_routes or search_k == total_vectors:
            return names
        search_k = min(total_vectors, search_k * 2)


def evaluate_top_k_synaptoroute(
    router: Any,
    queries: Sequence[str],
    expected: Sequence[str | None],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Evaluate retrieval Top-K on in-domain examples only.

    OOD behavior is a selective-decision metric and is evaluated through the
    router's thresholded predictions, not raw retrieval candidates.
    """
    if len(queries) != len(expected):
        raise ValueError("queries and expected must have equal length")
    if not k_values or min(k_values) < 1:
        raise ValueError("k_values must contain positive integers")

    in_domain = [(query, label) for query, label in zip(queries, expected) if label is not None]
    if not in_domain:
        raise ValueError("Top-K accuracy requires at least one in-domain example")

    correct = {k: 0 for k in k_values}
    required_routes = max(k_values)
    for query, label in in_domain:
        ranked_names = _rank_synaptoroute(router, query, required_routes)
        for k in k_values:
            if label in ranked_names[:k]:
                correct[k] += 1

    return {f"top_{k}": correct[k] / len(in_domain) for k in k_values}


def evaluate_semantic_router_public_api(
    layer: Any,
    queries: Sequence[str],
    expected: Sequence[str | None],
) -> dict[str, float | None]:
    """Report only metrics exposed by the baseline's stable public API."""
    if len(queries) != len(expected):
        raise ValueError("queries and expected must have equal length")

    in_domain = [(query, label) for query, label in zip(queries, expected) if label is not None]
    if not in_domain:
        raise ValueError("Top-1 accuracy requires at least one in-domain example")

    correct = 0
    for query, label in in_domain:
        result = layer(query)
        if result is not None and result.name == label:
            correct += 1

    return {
        "top_1": correct / len(in_domain),
        "top_3": None,
        "top_5": None,
    }


def get_predictions(predict_fn: Any, queries: Sequence[str]) -> list[str | None]:
    predictions: list[str | None] = []
    for query in queries:
        result = predict_fn(query)
        predictions.append(result.name if result else None)
    return predictions


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def run_accuracy_evaluation(model_name: str = "BAAI/bge-small-en-v1.5") -> None:
    from utils import init_semantic_router, init_synaptoroute, load_datasets

    print(f"=== Running Accuracy Evaluation (Model: {model_name}) ===")

    dataset_version, routes_data, test_queries = load_datasets()
    expected = [query["expected_route"] for query in test_queries]
    queries = [query["query"] for query in test_queries]

    print(
        f"Dataset version={dataset_version} routes={len(routes_data)} "
        f"queries={len(queries)} in_domain={sum(label is not None for label in expected)}"
    )

    router = init_synaptoroute(routes_data, model_name)
    layer = init_semantic_router(routes_data, model_name)

    try:
        y_true = [label if label is not None else "OOD" for label in expected]
        syn_predictions = get_predictions(router, queries)
        baseline_predictions = get_predictions(layer, queries)

        syn_clean = [prediction if prediction is not None else "OOD" for prediction in syn_predictions]
        baseline_clean = [prediction if prediction is not None else "OOD" for prediction in baseline_predictions]

        syn_precision = precision_score(y_true, syn_clean, average="weighted", zero_division=0)
        syn_recall = recall_score(y_true, syn_clean, average="weighted", zero_division=0)
        syn_f1 = f1_score(y_true, syn_clean, average="weighted", zero_division=0)

        baseline_precision = precision_score(y_true, baseline_clean, average="weighted", zero_division=0)
        baseline_recall = recall_score(y_true, baseline_clean, average="weighted", zero_division=0)
        baseline_f1 = f1_score(y_true, baseline_clean, average="weighted", zero_division=0)

        print("\n--- Thresholded Routing Quality ---")
        print(
            f"[SynaptoRoute]    F1: {syn_f1:.4f} | Precision: {syn_precision:.4f} "
            f"| Recall: {syn_recall:.4f}"
        )
        print(
            f"[Semantic Router] F1: {baseline_f1:.4f} | Precision: {baseline_precision:.4f} "
            f"| Recall: {baseline_recall:.4f}"
        )

        print("\n--- In-Domain Retrieval Accuracy ---")
        syn_top_k = evaluate_top_k_synaptoroute(router, queries, expected)
        baseline_top_k = evaluate_semantic_router_public_api(layer, queries, expected)
        print(
            f"[SynaptoRoute]    Top-1: {format_metric(syn_top_k['top_1'])} "
            f"| Top-3: {format_metric(syn_top_k['top_3'])} "
            f"| Top-5: {format_metric(syn_top_k['top_5'])}"
        )
        print(
            f"[Semantic Router] Top-1: {format_metric(baseline_top_k['top_1'])} "
            f"| Top-3: {format_metric(baseline_top_k['top_3'])} "
            f"| Top-5: {format_metric(baseline_top_k['top_5'])}"
        )

        print("\n--- Failure Analysis ---")
        for query, label, syn_prediction, baseline_prediction in zip(
            queries,
            expected,
            syn_predictions,
            baseline_predictions,
        ):
            if syn_prediction != label and baseline_prediction != label:
                print(
                    f"- [BOTH FAIL] Query: {query!r} | Expected: {label} "
                    f"| Synapto: {syn_prediction} | Baseline: {baseline_prediction}"
                )
            elif syn_prediction == label and baseline_prediction != label:
                print(f"+ [SynaptoRoute] Query: {query!r} | Expected: {label}")
            elif syn_prediction != label and baseline_prediction == label:
                print(f"+ [Semantic Router] Query: {query!r} | Expected: {label}")
    finally:
        router.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()
    run_accuracy_evaluation(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
