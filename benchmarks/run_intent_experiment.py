"""Run a calibrated intent-routing experiment on a pinned external dataset."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from benchmarks.baselines import (
    ExactCosineBaseline,
    ExactStringBaseline,
    LogisticRegressionBaseline,
    SemanticRouterBaseline,
)
from benchmarks.calibration import (
    CalibrationResult,
    PerRouteCalibrationResult,
    ScoredExample,
    apply_global_policy,
    apply_per_route_policy,
    calibration_artifact,
    fit_global_policy,
    fit_per_route_policy,
    per_route_calibration_artifact,
    write_calibration_artifact,
)
from benchmarks.hf_intent_datasets import (
    BANKING77_SPEC,
    CLINC150_SMALL_SPEC,
    ExternalDatasetBundle,
    deterministic_evaluation_limit,
    load_external_intent_dataset,
)
from benchmarks.manifest_schema import sha256_file
from benchmarks.prediction_io import prediction_record, write_prediction_jsonl
from synaptoroute import AdaptiveRouter, Route, RouterResult
from synaptoroute.encoder import BaseEncoder, FastEmbedEncoder
from synaptoroute.storage import SQLiteStorage


DATASETS = {
    "banking77": BANKING77_SPEC,
    "clinc150": CLINC150_SMALL_SPEC,
}
PolicyResult = CalibrationResult | PerRouteCalibrationResult


def _collect_raw_predictions(
    matcher: Any,
    examples: tuple,
    *,
    system_name: str,
    output_path: Path,
) -> tuple[list[ScoredExample], Path]:
    scored_examples: list[ScoredExample] = []
    records = []
    for example in examples:
        start = time.perf_counter()
        result: RouterResult = matcher.match(example.text)
        latency = time.perf_counter() - start
        scored_examples.append(
            ScoredExample.from_result(
                example_id=example.example_id,
                split=example.split,
                expected_route=example.label,
                result=result,
            )
        )
        records.append(
            prediction_record(
                example_id=example.example_id,
                query=example.text,
                expected_route=example.label,
                result=result,
                latency_seconds=latency,
                metadata={"phase": "calibration", "system": system_name},
            )
        )
    return scored_examples, write_prediction_jsonl(output_path, records)


def _summary_metrics(
    expected: list[str],
    predicted: list[str],
    latencies_seconds: list[float],
    reasons: Counter[str],
    acceptance_confidences: list[float],
    raw_correct: list[bool],
) -> dict[str, Any]:
    known_indices = [index for index, label in enumerate(expected) if label != "OOD"]
    ood_indices = [index for index, label in enumerate(expected) if label == "OOD"]
    accepted_indices = [index for index, label in enumerate(predicted) if label != "OOD"]
    accepted_known_indices = [index for index in known_indices if predicted[index] != "OOD"]

    known_accuracy = (
        accuracy_score(
            [expected[index] for index in known_indices],
            [predicted[index] for index in known_indices],
        )
        if known_indices
        else None
    )
    ood_recall = (
        sum(predicted[index] == "OOD" for index in ood_indices) / len(ood_indices)
        if ood_indices
        else None
    )
    selective_accuracy = (
        sum(predicted[index] == expected[index] for index in accepted_indices) / len(accepted_indices)
        if accepted_indices
        else 1.0
    )
    finite_confidences = [value for value in acceptance_confidences if np.isfinite(value)]
    confidence_floor = min(finite_confidences, default=0.0) - 1.0
    confidences = np.asarray(
        [value if np.isfinite(value) else confidence_floor for value in acceptance_confidences],
        dtype=np.float64,
    )
    ranked_indices = sorted(range(len(confidences)), key=lambda index: (-confidences[index], index))
    cumulative_errors = 0
    selective_risks: list[float] = []
    for rank, index in enumerate(ranked_indices, start=1):
        cumulative_errors += not raw_correct[index]
        selective_risks.append(cumulative_errors / rank)

    ood_targets = np.asarray([label == "OOD" for label in expected], dtype=np.int8)
    if len(set(ood_targets.tolist())) == 2:
        ood_scores = -confidences
        false_positive_rates, true_positive_rates, _ = roc_curve(ood_targets, ood_scores)
        fpr_at_95_candidates = false_positive_rates[true_positive_rates >= 0.95]
        ood_metrics = {
            "ood_auroc": float(roc_auc_score(ood_targets, ood_scores)),
            "ood_auprc": float(average_precision_score(ood_targets, ood_scores)),
            "ood_fpr_at_95_tpr": (
                float(np.min(fpr_at_95_candidates)) if len(fpr_at_95_candidates) else None
            ),
        }
    else:
        ood_metrics = {
            "ood_auroc": None,
            "ood_auprc": None,
            "ood_fpr_at_95_tpr": None,
        }

    return {
        "query_count": len(expected),
        "overall_accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "known_accuracy": float(known_accuracy) if known_accuracy is not None else None,
        "ood_recall": float(ood_recall) if ood_recall is not None else None,
        "coverage": len(accepted_indices) / len(expected),
        "known_coverage": len(accepted_known_indices) / len(known_indices) if known_indices else None,
        "selective_accuracy": selective_accuracy,
        "selective_risk_coverage_auc": float(np.mean(selective_risks)),
        "latency_p50_ms": float(np.percentile(latencies_seconds, 50) * 1000.0),
        "latency_p95_ms": float(np.percentile(latencies_seconds, 95) * 1000.0),
        "latency_p99_ms": float(np.percentile(latencies_seconds, 99) * 1000.0),
        "decision_reasons": dict(sorted(reasons.items())),
        **ood_metrics,
    }


def _acceptance_confidence(
    result: RouterResult,
    policy_result: PolicyResult | None,
) -> float:
    if result.score is None:
        return float("-inf")
    if policy_result is None:
        return result.score
    if isinstance(policy_result, PerRouteCalibrationResult):
        top_route = result.candidates[0].route_name if result.candidates else ""
        threshold = policy_result.policy.threshold_for(top_route)
    else:
        threshold = policy_result.policy.threshold
    score_confidence = result.score - threshold
    if result.margin is None:
        return score_confidence
    return min(score_confidence, result.margin - policy_result.policy.margin)


def _evaluate_test(
    matcher: Any,
    examples: tuple,
    *,
    routes: Mapping[str, Route],
    policy_result: PolicyResult | None,
    system_name: str,
    output_path: Path,
) -> dict[str, Any]:
    expected: list[str] = []
    predicted: list[str] = []
    latencies: list[float] = []
    reasons: Counter[str] = Counter()
    acceptance_confidences: list[float] = []
    raw_correct: list[bool] = []
    records = []

    for example in examples:
        start = time.perf_counter()
        raw_result: RouterResult = matcher.match(example.text)
        if policy_result is None:
            result = raw_result
        elif isinstance(policy_result, PerRouteCalibrationResult):
            result = apply_per_route_policy(raw_result, routes=routes, policy=policy_result.policy)
        else:
            result = apply_global_policy(raw_result, routes=routes, policy=policy_result.policy)
        latency = time.perf_counter() - start

        expected_label = example.label if example.label is not None else "OOD"
        predicted_label = result.route_name if result.route_name is not None else "OOD"
        expected.append(expected_label)
        predicted.append(predicted_label)
        latencies.append(latency)
        acceptance_confidence = _acceptance_confidence(raw_result, policy_result)
        acceptance_confidences.append(acceptance_confidence)
        raw_route = raw_result.candidates[0].route_name if raw_result.candidates else None
        raw_top_correct = example.label is not None and raw_route == example.label
        raw_correct.append(raw_top_correct)
        reasons[result.decision_reason.value] += 1
        records.append(
            prediction_record(
                example_id=example.example_id,
                query=example.text,
                expected_route=example.label,
                result=result,
                latency_seconds=latency,
                metadata={
                    "phase": "test",
                    "system": system_name,
                    "acceptance_confidence": (
                        acceptance_confidence if np.isfinite(acceptance_confidence) else None
                    ),
                    "raw_top_correct": raw_top_correct,
                },
            )
        )

    write_prediction_jsonl(output_path, records)
    return _summary_metrics(
        expected,
        predicted,
        latencies,
        reasons,
        acceptance_confidences,
        raw_correct,
    )


def run_bundle_experiment(
    *,
    bundle: ExternalDatasetBundle,
    encoder: BaseEncoder,
    output_dir: Path,
    min_known_coverage: float,
    max_ood_false_acceptance_rate: float | None,
    calibration_limit: int | None = None,
    include_semantic_router: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    routes = bundle.prepared.to_routes(threshold=-1.0)
    route_lookup = {route.name: route for route in routes}
    calibration_examples = tuple(
        deterministic_evaluation_limit(
            bundle.calibration_examples,
            limit=calibration_limit,
            seed=bundle.prepared.seed,
        )
    )

    router = AdaptiveRouter(encoder=encoder, storage=SQLiteStorage(":memory:"), margin=0.0)
    for route in routes:
        router.add_route(route)
    systems = {
        "synaptoroute": (router, "global"),
        "synaptoroute_per_route": (router, "per_route"),
        "exact_string": (ExactStringBaseline(routes), "none"),
        "exact_cosine": (ExactCosineBaseline(encoder, routes, margin=0.0), "global"),
        "logistic_regression": (
            LogisticRegressionBaseline(
                encoder,
                routes,
                threshold=0.0,
                margin=0.0,
                random_state=bundle.prepared.seed,
            ),
            "global",
        ),
    }
    if include_semantic_router:
        systems["semantic_router"] = (
            SemanticRouterBaseline(encoder, routes, margin=0.0),
            "global",
        )

    system_summaries: dict[str, Any] = {}
    try:
        for system_name, (matcher, calibration_method) in systems.items():
            calibration_result: PolicyResult | None = None
            if calibration_method != "none":
                scored_examples, calibration_predictions_path = _collect_raw_predictions(
                    matcher,
                    calibration_examples,
                    system_name=system_name,
                    output_path=output_dir / f"calibration_predictions_{system_name}.jsonl",
                )
                fit_dataset = {
                    **bundle.manifest_metadata(),
                    "fit_split": sorted({example.split for example in calibration_examples}),
                    "fit_count": len(calibration_examples),
                }
                if calibration_method == "per_route":
                    calibration_result = fit_per_route_policy(
                        scored_examples,
                        min_known_coverage=min_known_coverage,
                        max_ood_false_acceptance_rate=max_ood_false_acceptance_rate,
                    )
                    artifact = per_route_calibration_artifact(
                        calibration_result,
                        dataset=fit_dataset,
                        source_predictions_sha256=sha256_file(calibration_predictions_path),
                    )
                else:
                    calibration_result = fit_global_policy(
                        scored_examples,
                        min_known_coverage=min_known_coverage,
                        max_ood_false_acceptance_rate=max_ood_false_acceptance_rate,
                    )
                    artifact = calibration_artifact(
                        calibration_result,
                        dataset=fit_dataset,
                        source_predictions_sha256=sha256_file(calibration_predictions_path),
                    )
                write_calibration_artifact(
                    output_dir / f"calibration_{system_name}.json",
                    artifact,
                )

            metrics = _evaluate_test(
                matcher,
                bundle.prepared.evaluation_examples,
                routes=route_lookup,
                policy_result=calibration_result,
                system_name=system_name,
                output_path=output_dir / f"test_predictions_{system_name}.jsonl",
            )
            system_summaries[system_name] = {
                "calibration": (
                    {
                        "method": calibration_method,
                        "policy": asdict(calibration_result.policy),
                        "metrics": calibration_result.metrics.__dict__,
                    }
                    if calibration_result is not None
                    else None
                ),
                "test": metrics,
            }
    finally:
        router.close()

    summary = {
        "benchmark": "external_intent_routing_experiment",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "dataset": bundle.manifest_metadata(),
        "configuration": {
            "encoder": getattr(encoder, "model_name", type(encoder).__name__),
            "min_known_coverage": min_known_coverage,
            "max_ood_false_acceptance_rate": max_ood_false_acceptance_rate,
            "calibration_limit": calibration_limit,
            "index": type(router.index).__name__,
            "semantic_router_version": (
                importlib.metadata.version("semantic-router") if include_semantic_router else None
            ),
        },
        "systems": system_summaries,
        "notes": [
            "Policies were fitted only on calibration/validation examples.",
            "Test predictions were generated after policy fitting.",
            "The run remains unverified until executed from a clean commit and independently reviewed.",
        ],
    }
    summary_path = output_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--examples-per-route", type=int, default=20)
    parser.add_argument("--calibration-per-route", type=int, default=10)
    parser.add_argument("--calibration-limit", type=int)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-known-coverage", type=float, default=0.8)
    parser.add_argument("--max-ood-far", type=float)
    parser.add_argument("--skip-semantic-router", action="store_true")
    parser.add_argument("--cache-dir", default="benchmark_results/dataset_cache")
    default_root = Path(os.environ.get("SYNAPTOROUTE_RUN_DIR", "benchmark_results/external"))
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    output_dir = Path(args.output_dir) if args.output_dir else default_root / spec.name
    bundle = load_external_intent_dataset(
        spec,
        examples_per_route=args.examples_per_route,
        seed=args.seed,
        calibration_per_route=args.calibration_per_route,
        evaluation_limit=args.evaluation_limit,
        cache_dir=args.cache_dir,
    )
    encoder = FastEmbedEncoder(model_name=args.model)
    summary = run_bundle_experiment(
        bundle=bundle,
        encoder=encoder,
        output_dir=output_dir,
        min_known_coverage=args.min_known_coverage,
        max_ood_false_acceptance_rate=args.max_ood_far,
        calibration_limit=args.calibration_limit,
        include_semantic_router=not args.skip_semantic_router,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
