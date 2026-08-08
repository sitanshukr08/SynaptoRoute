"""Paired and matched-coverage analysis for multi-seed prediction artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.manifest_schema import sha256_file
from benchmarks.prediction_io import read_prediction_jsonl


PAIRED_METRICS = ("overall_accuracy", "known_accuracy", "known_coverage", "ood_recall")
DEFAULT_COVERAGE_TARGETS = (0.80, 0.90, 0.95)


def _non_null_float(value: float | int | None) -> float:
    if value is None:
        raise ValueError("expected a numeric matched-coverage value")
    return float(value)


def align_prediction_records(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    first_by_id = {str(record["example_id"]): record for record in first}
    second_by_id = {str(record["example_id"]): record for record in second}
    if len(first_by_id) != len(first) or len(second_by_id) != len(second):
        raise ValueError("prediction artifacts contain duplicate example IDs")
    if first_by_id.keys() != second_by_id.keys():
        raise ValueError("prediction artifacts do not contain the same example IDs")

    aligned = []
    for example_id in sorted(first_by_id):
        first_record = first_by_id[example_id]
        second_record = second_by_id[example_id]
        for field in ("query_sha256", "expected_route"):
            if first_record[field] != second_record[field]:
                raise ValueError(f"paired prediction mismatch for {example_id}: {field}")
        aligned.append((first_record, second_record))
    return aligned


def _indicator(record: Mapping[str, Any], metric: str) -> float | None:
    expected = record["expected_route"]
    predicted = record["predicted_route"]
    if metric == "overall_accuracy":
        return float(bool(record["correct"]))
    if metric == "known_accuracy":
        return None if expected is None else float(bool(record["correct"]))
    if metric == "known_coverage":
        return None if expected is None else float(predicted is not None)
    if metric == "ood_recall":
        return None if expected is not None else float(predicted is None)
    raise ValueError(f"unsupported paired metric: {metric}")


def hierarchical_paired_bootstrap(
    paired_by_seed: Mapping[int, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    *,
    metric: str,
    repetitions: int = 5000,
    random_seed: int = 20260713,
) -> dict[str, Any] | None:
    """Estimate first-minus-second effects while resampling seeds and examples."""
    if repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    seed_differences: dict[int, np.ndarray] = {}
    for seed, pairs in sorted(paired_by_seed.items()):
        differences = []
        for first, second in pairs:
            first_value = _indicator(first, metric)
            second_value = _indicator(second, metric)
            if first_value is not None and second_value is not None:
                differences.append(first_value - second_value)
        if differences:
            seed_differences[seed] = np.asarray(differences, dtype=np.float64)
    if not seed_differences:
        return None

    seeds = sorted(seed_differences)
    point_estimate = float(np.mean([np.mean(seed_differences[seed]) for seed in seeds]))
    rng = np.random.default_rng(random_seed)
    bootstrap_effects = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_effects = []
        for seed in sampled_seeds:
            differences = seed_differences[int(seed)]
            sampled = rng.choice(differences, size=len(differences), replace=True)
            sampled_effects.append(float(np.mean(sampled)))
        bootstrap_effects[repetition] = float(np.mean(sampled_effects))

    lower, upper = np.percentile(bootstrap_effects, [2.5, 97.5])
    probability_positive = float(np.mean(bootstrap_effects > 0.0))
    lower_tail_probability = float(np.mean(bootstrap_effects <= 0.0))
    upper_tail_probability = float(np.mean(bootstrap_effects >= 0.0))
    return {
        "metric": metric,
        "effect": "first_minus_second",
        "seed_count": len(seeds),
        "paired_example_count": sum(len(values) for values in seed_differences.values()),
        "point_estimate": point_estimate,
        "bootstrap_repetitions": repetitions,
        "confidence_interval_95": [float(lower), float(upper)],
        "probability_effect_positive": probability_positive,
        "two_sided_tail_probability": min(
            1.0,
            2.0 * min(lower_tail_probability, upper_tail_probability),
        ),
        "per_seed_effects": {
            str(seed): float(np.mean(seed_differences[seed])) for seed in seeds
        },
    }


def _acceptance_confidence(record: Mapping[str, Any]) -> float:
    value = record.get("metadata", {}).get("acceptance_confidence")
    return float(value) if value is not None else float("-inf")


def matched_coverage_curve(
    records: Sequence[Mapping[str, Any]],
    *,
    targets: Sequence[float] = DEFAULT_COVERAGE_TARGETS,
) -> dict[str, dict[str, float | int | None]]:
    if not records:
        raise ValueError("prediction records must not be empty")
    if any(not 0.0 < target <= 1.0 for target in targets):
        raise ValueError("coverage targets must be in (0, 1]")
    known = [record for record in records if record["expected_route"] is not None]
    ood = [record for record in records if record["expected_route"] is None]
    if not known:
        raise ValueError("matched coverage requires known-intent examples")
    if not any(np.isfinite(_acceptance_confidence(record)) for record in known):
        return {}

    known_confidences = sorted(
        (_acceptance_confidence(record) for record in known),
        reverse=True,
    )
    curve: dict[str, dict[str, float | int | None]] = {}
    for target in targets:
        rank = max(1, math.ceil(target * len(known)))
        cutoff = known_confidences[rank - 1]
        accepted = [record for record in records if _acceptance_confidence(record) >= cutoff]
        accepted_known = [record for record in known if _acceptance_confidence(record) >= cutoff]
        accepted_ood = [record for record in ood if _acceptance_confidence(record) >= cutoff]
        raw_correct = sum(bool(record["metadata"]["raw_top_correct"]) for record in accepted)
        curve[f"{target:.2f}"] = {
            "target_known_coverage": target,
            "actual_known_coverage": len(accepted_known) / len(known),
            "acceptance_cutoff": cutoff if np.isfinite(cutoff) else None,
            "accepted_count": len(accepted),
            "selective_accuracy": raw_correct / len(accepted) if accepted else 1.0,
            "selective_risk": 1.0 - (raw_correct / len(accepted) if accepted else 1.0),
            "ood_false_acceptance_rate": len(accepted_ood) / len(ood) if ood else None,
        }
    return curve


def aggregate_matched_coverage(
    records_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    targets: Sequence[float] = DEFAULT_COVERAGE_TARGETS,
) -> dict[str, Any]:
    per_seed = {
        str(seed): matched_coverage_curve(records, targets=targets)
        for seed, records in sorted(records_by_seed.items())
    }
    aggregate: dict[str, Any] = {}
    for target in targets:
        target_key = f"{target:.2f}"
        metric_names = (
            "actual_known_coverage",
            "selective_accuracy",
            "selective_risk",
            "ood_false_acceptance_rate",
        )
        aggregate[target_key] = {}
        for metric_name in metric_names:
            values = [
                _non_null_float(curve[target_key][metric_name])
                for curve in per_seed.values()
                if target_key in curve and curve[target_key][metric_name] is not None
            ]
            aggregate[target_key][metric_name] = (
                {
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "min": min(values),
                    "max": max(values),
                }
                if values
                else None
            )
    return {"aggregate": aggregate, "per_seed": per_seed}


def paired_matched_coverage_effects(
    first_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    second_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    targets: Sequence[float] = DEFAULT_COVERAGE_TARGETS,
    repetitions: int = 5000,
    random_seed: int = 20260713,
) -> dict[str, Any]:
    if first_by_seed.keys() != second_by_seed.keys():
        raise ValueError("matched-coverage systems do not contain the same seeds")
    if repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    first_curves = {
        seed: matched_coverage_curve(records, targets=targets)
        for seed, records in sorted(first_by_seed.items())
    }
    second_curves = {
        seed: matched_coverage_curve(second_by_seed[seed], targets=targets)
        for seed in sorted(second_by_seed)
    }
    metrics = ("selective_accuracy", "selective_risk", "ood_false_acceptance_rate")
    rng = np.random.default_rng(random_seed)
    effects: dict[str, Any] = {}
    for target in targets:
        target_key = f"{target:.2f}"
        effects[target_key] = {}
        for metric in metrics:
            per_seed = {
                seed: _non_null_float(first_curves[seed][target_key][metric])
                - _non_null_float(second_curves[seed][target_key][metric])
                for seed in first_curves
                if target_key in first_curves[seed]
                and target_key in second_curves[seed]
                and first_curves[seed][target_key][metric] is not None
                and second_curves[seed][target_key][metric] is not None
            }
            if not per_seed:
                effects[target_key][metric] = None
                continue
            values = np.asarray(list(per_seed.values()), dtype=np.float64)
            bootstrap = np.empty(repetitions, dtype=np.float64)
            for index in range(repetitions):
                bootstrap[index] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
            lower, upper = np.percentile(bootstrap, [2.5, 97.5])
            lower_tail = float(np.mean(bootstrap <= 0.0))
            upper_tail = float(np.mean(bootstrap >= 0.0))
            effects[target_key][metric] = {
                "effect": "first_minus_second",
                "seed_count": len(per_seed),
                "point_estimate": float(np.mean(values)),
                "confidence_interval_95": [float(lower), float(upper)],
                "two_sided_tail_probability": min(1.0, 2.0 * min(lower_tail, upper_tail)),
                "per_seed_effects": {str(seed): value for seed, value in per_seed.items()},
            }
    return effects


def _resolve_summary_path(study_dir: Path, path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_file():
        return candidate
    relative_to_study = study_dir / candidate
    if relative_to_study.is_file():
        return relative_to_study
    raise FileNotFoundError(f"per-seed summary does not exist: {path_text}")


def analyze_multiseed_study(
    study_dir: Path,
    *,
    reference_system: str,
    comparator_systems: Sequence[str],
    bootstrap_repetitions: int = 5000,
    coverage_targets: Sequence[float] = DEFAULT_COVERAGE_TARGETS,
) -> dict[str, Any]:
    study_summary_path = study_dir / "multiseed_summary.json"
    study_summary = json.loads(study_summary_path.read_text(encoding="utf-8"))
    systems = {reference_system, *comparator_systems}
    records_by_system: dict[str, dict[int, list[dict[str, Any]]]] = {
        system: {} for system in systems
    }
    source_artifacts = []

    for entry in study_summary["per_seed"]:
        seed = int(entry["seed"])
        summary_path = _resolve_summary_path(study_dir, entry["summary_path"])
        if sha256_file(summary_path) != entry["summary_sha256"]:
            raise ValueError(f"per-seed summary hash mismatch: {summary_path}")
        for system in systems:
            prediction_path = summary_path.parent / f"test_predictions_{system}.jsonl"
            records_by_system[system][seed] = read_prediction_jsonl(prediction_path)
            source_artifacts.append(
                {
                    "seed": seed,
                    "system": system,
                    "path": prediction_path.as_posix(),
                    "sha256": sha256_file(prediction_path),
                }
            )

    comparisons: dict[str, Any] = {}
    matched_coverage_effects: dict[str, Any] = {}
    for comparator in comparator_systems:
        paired_by_seed = {
            seed: align_prediction_records(
                records_by_system[reference_system][seed],
                records_by_system[comparator][seed],
            )
            for seed in sorted(records_by_system[reference_system])
        }
        comparisons[comparator] = {
            metric: hierarchical_paired_bootstrap(
                paired_by_seed,
                metric=metric,
                repetitions=bootstrap_repetitions,
            )
            for metric in PAIRED_METRICS
        }
        matched_coverage_effects[comparator] = paired_matched_coverage_effects(
            records_by_system[reference_system],
            records_by_system[comparator],
            targets=coverage_targets,
            repetitions=bootstrap_repetitions,
        )

    analysis = {
        "analysis": "paired_multiseed_and_matched_coverage",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "source_study": {
            "path": study_summary_path.as_posix(),
            "sha256": sha256_file(study_summary_path),
        },
        "configuration": {
            "reference_system": reference_system,
            "comparator_systems": list(comparator_systems),
            "bootstrap_repetitions": bootstrap_repetitions,
            "coverage_targets": list(coverage_targets),
        },
        "paired_effects": comparisons,
        "paired_matched_coverage_effects": matched_coverage_effects,
        "matched_coverage": {
            system: aggregate_matched_coverage(records, targets=coverage_targets)
            for system, records in sorted(records_by_system.items())
        },
        "source_prediction_artifacts": sorted(
            source_artifacts,
            key=lambda item: (item["seed"], item["system"]),
        ),
        "notes": [
            "Paired effects are reference minus comparator.",
            "The hierarchical bootstrap resamples seeds and then examples within seeds.",
            "Matched-coverage curves characterize test behavior and do not refit deployment policies.",
        ],
    }
    output_path = study_dir / "statistical_analysis.json"
    output_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--reference", default="synaptoroute_per_route")
    parser.add_argument(
        "--comparators",
        nargs="+",
        default=[
            "synaptoroute",
            "exact_string",
            "logistic_regression",
            "semantic_router",
            "exact_cosine",
        ],
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--coverage-targets", nargs="+", type=float, default=list(DEFAULT_COVERAGE_TARGETS))
    args = parser.parse_args()

    analysis = analyze_multiseed_study(
        args.study_dir,
        reference_system=args.reference,
        comparator_systems=args.comparators,
        bootstrap_repetitions=args.bootstrap_repetitions,
        coverage_targets=args.coverage_targets,
    )
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
