"""Run the fixed multi-seed intent-routing quality study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from benchmarks.embedding_cache import MemoizingEncoder
from benchmarks.hf_intent_datasets import load_external_intent_dataset
from benchmarks.manifest_schema import sha256_file
from benchmarks.run_intent_experiment import DATASETS, run_bundle_experiment
from benchmarks.statistical_analysis import analyze_multiseed_study
from synaptoroute.encoder import FastEmbedEncoder


FIXED_SEEDS = (13, 29, 42, 71, 101)
QUALITY_METRICS = (
    "overall_accuracy",
    "macro_f1",
    "known_accuracy",
    "known_coverage",
    "ood_recall",
    "ood_auroc",
    "ood_auprc",
    "ood_fpr_at_95_tpr",
    "coverage",
    "selective_accuracy",
    "selective_risk_coverage_auc",
)


def aggregate_quality_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("at least one experiment summary is required")
    common_systems = set(summaries[0]["systems"])
    for summary in summaries[1:]:
        common_systems &= set(summary["systems"])

    aggregates: dict[str, Any] = {}
    for system_name in sorted(common_systems):
        metric_aggregates: dict[str, Any] = {}
        for metric_name in QUALITY_METRICS:
            values = [
                summary["systems"][system_name]["test"].get(metric_name)
                for summary in summaries
            ]
            numeric_values = [float(value) for value in values if value is not None]
            metric_aggregates[metric_name] = (
                {
                    "n": len(numeric_values),
                    "mean": float(np.mean(numeric_values)),
                    "sample_std": (
                        float(np.std(numeric_values, ddof=1)) if len(numeric_values) > 1 else None
                    ),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                }
                if numeric_values
                else None
            )
        aggregates[system_name] = metric_aggregates
    return aggregates


def run_multiseed_study(
    *,
    dataset_name: str,
    model_name: str,
    seeds: Sequence[int],
    output_dir: Path,
    cache_dir: Path,
    examples_per_route: int,
    calibration_per_route: int,
    evaluation_limit: int | None,
    min_known_coverage: float,
    max_ood_false_acceptance_rate: float | None,
    include_semantic_router: bool,
) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    spec = DATASETS[dataset_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = MemoizingEncoder(FastEmbedEncoder(model_name=model_name))
    summaries: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        seed_output = output_dir / f"seed-{seed}" / spec.name
        bundle = load_external_intent_dataset(
            spec,
            examples_per_route=examples_per_route,
            seed=seed,
            calibration_per_route=calibration_per_route,
            evaluation_limit=evaluation_limit,
            cache_dir=cache_dir,
        )
        summary = run_bundle_experiment(
            bundle=bundle,
            encoder=encoder,
            output_dir=seed_output,
            min_known_coverage=min_known_coverage,
            max_ood_false_acceptance_rate=max_ood_false_acceptance_rate,
            include_semantic_router=include_semantic_router,
        )
        summaries.append(summary)
        summary_path = seed_output / "experiment_summary.json"
        per_seed.append(
            {
                "seed": seed,
                "summary_path": summary_path.as_posix(),
                "summary_sha256": sha256_file(summary_path),
            }
        )

    study_summary = {
        "benchmark": "external_intent_routing_multiseed_study",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "configuration": {
            "dataset": dataset_name,
            "dataset_revision": spec.revision,
            "model": model_name,
            "seeds": list(seeds),
            "examples_per_route": examples_per_route,
            "calibration_per_route": calibration_per_route,
            "evaluation_limit": evaluation_limit,
            "min_known_coverage": min_known_coverage,
            "max_ood_false_acceptance_rate": max_ood_false_acceptance_rate,
            "include_semantic_router": include_semantic_router,
            "embedding_cache": "exact_text_quality_only",
            "embedding_cache_stats": encoder.cache_stats(),
        },
        "per_seed": per_seed,
        "quality_aggregates": aggregate_quality_summaries(summaries),
        "notes": [
            "Only quality metrics are aggregated; latency requires a separate counterbalanced protocol.",
            "Exact-text embedding memoization is enabled and invalidates per-seed latency for performance claims.",
            "Paired confidence intervals are generated from per-example predictions after the study.",
            "The study remains unverified until rerun from a clean commit and independently reviewed.",
        ],
    }
    output_path = output_dir / "multiseed_summary.json"
    output_path.write_text(json.dumps(study_summary, indent=2, sort_keys=True), encoding="utf-8")
    return study_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FIXED_SEEDS))
    parser.add_argument("--examples-per-route", type=int, default=20)
    parser.add_argument("--calibration-per-route", type=int, default=10)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--min-known-coverage", type=float, default=0.8)
    parser.add_argument("--max-ood-far", type=float)
    parser.add_argument("--skip-semantic-router", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--cache-dir", default="benchmark_results/dataset_cache")
    default_root = Path(os.environ.get("SYNAPTOROUTE_RUN_DIR", "benchmark_results/multiseed"))
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_root / f"{DATASETS[args.dataset].name}_multiseed"
    )
    summary = run_multiseed_study(
        dataset_name=args.dataset,
        model_name=args.model,
        seeds=args.seeds,
        output_dir=output_dir,
        cache_dir=Path(args.cache_dir),
        examples_per_route=args.examples_per_route,
        calibration_per_route=args.calibration_per_route,
        evaluation_limit=args.evaluation_limit,
        min_known_coverage=args.min_known_coverage,
        max_ood_false_acceptance_rate=args.max_ood_far,
        include_semantic_router=not args.skip_semantic_router,
    )
    analysis = None
    if not args.skip_analysis:
        comparators = ["synaptoroute", "logistic_regression", "exact_cosine"]
        if not args.skip_semantic_router:
            comparators.append("semantic_router")
        analysis = analyze_multiseed_study(
            output_dir,
            reference_system="synaptoroute_per_route",
            comparator_systems=comparators,
            bootstrap_repetitions=args.bootstrap_repetitions,
        )
    print(json.dumps({"study": summary, "analysis": analysis}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
