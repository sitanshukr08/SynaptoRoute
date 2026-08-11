"""Aggregate a verifier-accepted backpressure run without promoting its claims."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.manifest_schema import sha256_file  # noqa: E402
from paper.verify_matrix_run import verify_matrix_run  # noqa: E402


NAME_PATTERN = re.compile(r"^(?P<profile>[a-z0-9_]+)-rep(?P<repetition>[0-9]+)$")
COUNT_FIELDS = (
    "offered_count",
    "successful_count",
    "successful_correct_count",
    "successful_incorrect_count",
    "overloaded_count",
    "error_count",
)


def _finite_float(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _count(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def summarize_values(
    values: Sequence[float],
    *,
    bootstrap_repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Summarize repetition-level values with a percentile bootstrap interval."""

    if not values:
        raise ValueError("cannot summarize an empty sequence")
    if bootstrap_repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("summary values must be finite")

    mean = float(np.mean(array))
    if len(array) == 1 or np.all(array == array[0]):
        lower = upper = mean
    else:
        sampled_indexes = rng.integers(
            0,
            len(array),
            size=(bootstrap_repetitions, len(array)),
        )
        bootstrap_means = np.mean(array[sampled_indexes], axis=1)
        lower, upper = (float(value) for value in np.percentile(bootstrap_means, [2.5, 97.5]))
    return {
        "n": len(array),
        "mean": mean,
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else None,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "confidence_interval_95": [lower, upper],
    }


def _optional_summary(
    values: Sequence[float | None],
    *,
    bootstrap_repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return summarize_values(
        present,
        bootstrap_repetitions=bootstrap_repetitions,
        rng=rng,
    )


def _parse_record_name(name: str) -> tuple[str, int]:
    match = NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid backpressure record name: {name}")
    return match.group("profile"), int(match.group("repetition"))


def _scenario_by_load(summary: Mapping[str, Any], name: str) -> dict[float, Mapping[str, Any]]:
    scenarios = summary.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"{name} must contain scenarios")
    by_load: dict[float, Mapping[str, Any]] = {}
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise ValueError(f"{name} contains a non-object scenario")
        load = _finite_float(scenario.get("load_fraction"), f"{name} load_fraction")
        if load in by_load:
            raise ValueError(f"{name} repeats load fraction {load}")
        by_load[load] = scenario
    return by_load


def analyze_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_repetitions: int = 10_000,
    random_seed: int = 20260811,
) -> dict[str, Any]:
    """Aggregate backpressure summaries, using repetitions as experimental units."""

    if not records:
        raise ValueError("at least one backpressure record is required")
    grouped: dict[str, list[tuple[int, str, Mapping[str, Any]]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for name, summary in sorted(records.items()):
        profile, repetition = _parse_record_name(name)
        identity = (profile, repetition)
        if identity in seen:
            raise ValueError(f"duplicate profile repetition: {profile} rep{repetition}")
        seen.add(identity)
        if summary.get("benchmark") != "sustained_async_backpressure":
            raise ValueError(f"{name} is not a sustained backpressure summary")
        if summary.get("status") != "unverified" or summary.get("paper_evidence_eligible") is not False:
            raise ValueError(f"{name} must remain unverified and paper-ineligible")
        grouped[profile].append((repetition, name, summary))

    rng = np.random.default_rng(random_seed)
    profile_results: dict[str, Any] = {}
    for profile, repetitions in sorted(grouped.items()):
        repetitions.sort(key=lambda item: item[0])
        repetition_ids = [item[0] for item in repetitions]
        if repetition_ids != list(range(len(repetitions))):
            raise ValueError(f"{profile} repetitions must be contiguous from zero")

        raw_configurations = [item[2].get("configuration") for item in repetitions]
        if any(not isinstance(configuration, Mapping) for configuration in raw_configurations):
            raise ValueError(f"{profile} contains an invalid configuration")
        configurations = [
            cast(Mapping[str, Any], configuration) for configuration in raw_configurations
        ]
        stable_fields = (
            "load_fractions",
            "duration_seconds",
            "queue_size",
            "batch_size",
            "max_in_flight_batches",
            "encoder_delay_ms",
        )
        stable_configuration = {
            field: configurations[0].get(field)
            for field in stable_fields
        }
        for configuration in configurations[1:]:
            if any(configuration.get(field) != stable_configuration[field] for field in stable_fields):
                raise ValueError(f"{profile} changes configuration between repetitions")

        scenarios_by_repetition = [
            _scenario_by_load(summary, name) for _, name, summary in repetitions
        ]
        load_fractions = sorted(scenarios_by_repetition[0])
        if any(sorted(scenarios) != load_fractions for scenarios in scenarios_by_repetition[1:]):
            raise ValueError(f"{profile} changes load fractions between repetitions")

        calibration_values = [
            _finite_float(
                configuration.get("measured_saturation_qps"),
                f"{profile} measured_saturation_qps",
            )
            for configuration in configurations
        ]
        scenario_results: dict[str, Any] = {}
        baseline_p95 = [
            _finite_float(
                scenarios[load_fractions[0]]["successful_latency"]["p95_ms"],
                f"{profile} baseline p95",
            )
            for scenarios in scenarios_by_repetition
        ]
        for load in load_fractions:
            scenarios = [by_load[load] for by_load in scenarios_by_repetition]
            totals = {
                field: sum(_count(scenario.get(field), f"{profile} {load} {field}") for scenario in scenarios)
                for field in COUNT_FIELDS
            }
            if totals["successful_count"] != (
                totals["successful_correct_count"] + totals["successful_incorrect_count"]
            ):
                raise ValueError(f"{profile} {load} successful correctness counts do not add up")
            if totals["offered_count"] != (
                totals["successful_count"] + totals["overloaded_count"] + totals["error_count"]
            ):
                raise ValueError(f"{profile} {load} offered outcome counts do not add up")

            successful_qps = [
                _finite_float(scenario.get("successful_qps"), f"{profile} {load} successful_qps")
                for scenario in scenarios
            ]
            offered_qps = [
                _finite_float(scenario.get("offered_qps"), f"{profile} {load} offered_qps")
                for scenario in scenarios
            ]
            shedding_rates = [
                _finite_float(scenario.get("shedding_rate"), f"{profile} {load} shedding_rate")
                for scenario in scenarios
            ]
            p95_values = [
                _finite_float(
                    scenario["successful_latency"]["p95_ms"],
                    f"{profile} {load} successful p95",
                )
                for scenario in scenarios
            ]
            p99_values = [
                _finite_float(
                    scenario["successful_latency"]["p99_ms"],
                    f"{profile} {load} successful p99",
                )
                for scenario in scenarios
            ]
            overload_p95 = [
                None
                if scenario.get("overload_latency") is None
                else _finite_float(
                    scenario["overload_latency"]["p95_ms"],
                    f"{profile} {load} overload p95",
                )
                for scenario in scenarios
            ]
            target_qps = [
                _finite_float(scenario.get("target_qps"), f"{profile} {load} target_qps")
                for scenario in scenarios
            ]
            scenario_results[f"{load:g}"] = {
                "load_fraction": load,
                "pooled_counts": totals,
                "pooled_rates": {
                    "success_rate": totals["successful_count"] / totals["offered_count"],
                    "shedding_rate": totals["overloaded_count"] / totals["offered_count"],
                    "error_rate": totals["error_count"] / totals["offered_count"],
                    "successful_accuracy": (
                        totals["successful_correct_count"] / totals["successful_count"]
                        if totals["successful_count"]
                        else None
                    ),
                },
                "repetition_statistics": {
                    "offered_qps": summarize_values(
                        offered_qps,
                        bootstrap_repetitions=bootstrap_repetitions,
                        rng=rng,
                    ),
                    "successful_qps": summarize_values(
                        successful_qps,
                        bootstrap_repetitions=bootstrap_repetitions,
                        rng=rng,
                    ),
                    "shedding_rate": summarize_values(
                        shedding_rates,
                        bootstrap_repetitions=bootstrap_repetitions,
                        rng=rng,
                    ),
                    "capacity_retention": summarize_values(
                        [value / saturation for value, saturation in zip(successful_qps, calibration_values)],
                        bootstrap_repetitions=bootstrap_repetitions,
                        rng=rng,
                    ),
                    "target_attainment": summarize_values(
                        [value / target for value, target in zip(successful_qps, target_qps)],
                        bootstrap_repetitions=bootstrap_repetitions,
                        rng=rng,
                    ),
                    "successful_p95_ms": summarize_values(
                        p95_values,
                        bootstrap_repetitions=bootstrap_repetitions,
                        rng=rng,
                    ),
                    "successful_p99_ms": summarize_values(
                        p99_values,
                        bootstrap_repetitions=bootstrap_repetitions,
                        rng=rng,
                    ),
                    "p95_inflation_vs_lowest_load": summarize_values(
                        [value / baseline for value, baseline in zip(p95_values, baseline_p95)],
                        bootstrap_repetitions=bootstrap_repetitions,
                        rng=rng,
                    ),
                    "overload_p95_ms": _optional_summary(
                        overload_p95,
                        bootstrap_repetitions=bootstrap_repetitions,
                        rng=rng,
                    ),
                },
            }

        profile_results[profile] = {
            "configuration": stable_configuration,
            "repetition_count": len(repetitions),
            "repetitions": repetition_ids,
            "measured_saturation_qps": summarize_values(
                calibration_values,
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "scenarios": scenario_results,
        }

    return {
        "bootstrap_repetitions": bootstrap_repetitions,
        "random_seed": random_seed,
        "profile_count": len(profile_results),
        "record_count": len(records),
        "profiles": profile_results,
    }


def _concise_verification(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "verification_status": report["verification_status"],
        "command_count": report["command_count"],
        "log_hashes_verified": report["log_hashes_verified"],
        "raw_and_state_hashes_verified": report["raw_and_state_hashes_verified"],
        "invariants_passed": report["invariants"]["passed"],
        "outcome_observation_count": report["outcome_observation_count"],
        "environment_evidence_verified": report["environment_evidence_verified"],
    }


def analyze_run(
    run_dir: Path,
    *,
    expected_commit: str,
    bootstrap_repetitions: int = 10_000,
    random_seed: int = 20260811,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    verification = verify_matrix_run(
        run_dir,
        expected_commit=expected_commit,
        expected_family="backpressure",
    )
    state_path = run_dir / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    records: dict[str, Mapping[str, Any]] = {}
    source_artifacts = []
    for result in state["results"]:
        name = str(result["name"])
        summary_path = run_dir / "backpressure" / f"{name}.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records[name] = summary
        source_artifacts.append(
            {
                "name": name,
                "path": summary_path.relative_to(run_dir).as_posix(),
                "sha256": sha256_file(summary_path),
            }
        )
    if len(records) != verification["command_count"]:
        raise ValueError("backpressure summary count differs from the verified command count")

    aggregation = analyze_records(
        records,
        bootstrap_repetitions=bootstrap_repetitions,
        random_seed=random_seed,
    )
    return {
        "schema_version": 1,
        "analysis": "repetition_level_backpressure_aggregation",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "source_run": {
            "run_id": verification["run_id"],
            "git_commit": verification["git_commit"],
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "run_state_sha256": sha256_file(state_path),
        },
        "verification": _concise_verification(verification),
        "configuration": {
            "bootstrap_repetitions": bootstrap_repetitions,
            "random_seed": random_seed,
            "experimental_unit": "one separately calibrated benchmark repetition",
        },
        "summary": aggregation,
        "source_artifacts": source_artifacts,
        "notes": [
            "Confidence intervals resample repetition-level metrics, not individual requests.",
            "Pooled rates retain every offered request, including explicit overload and error outcomes.",
            "Percentile statistics summarize per-repetition percentiles and do not pool raw latency samples.",
            "This analysis remains unverified because the source run lacks controlled-host evidence and independent reproduction.",
        ],
    }


def render_csv(analysis: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    fields = (
        "profile",
        "load_fraction",
        "repetitions",
        "offered_count",
        "successful_count",
        "overloaded_count",
        "error_count",
        "pooled_success_rate",
        "pooled_shedding_rate",
        "successful_accuracy",
        "successful_qps_mean",
        "successful_qps_ci95_low",
        "successful_qps_ci95_high",
        "successful_p95_ms_mean",
        "successful_p95_ms_ci95_low",
        "successful_p95_ms_ci95_high",
        "capacity_retention_mean",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for profile, profile_result in analysis["summary"]["profiles"].items():
        for scenario in profile_result["scenarios"].values():
            counts = scenario["pooled_counts"]
            rates = scenario["pooled_rates"]
            statistics = scenario["repetition_statistics"]
            successful_qps = statistics["successful_qps"]
            p95 = statistics["successful_p95_ms"]
            writer.writerow(
                {
                    "profile": profile,
                    "load_fraction": scenario["load_fraction"],
                    "repetitions": profile_result["repetition_count"],
                    "offered_count": counts["offered_count"],
                    "successful_count": counts["successful_count"],
                    "overloaded_count": counts["overloaded_count"],
                    "error_count": counts["error_count"],
                    "pooled_success_rate": rates["success_rate"],
                    "pooled_shedding_rate": rates["shedding_rate"],
                    "successful_accuracy": rates["successful_accuracy"],
                    "successful_qps_mean": successful_qps["mean"],
                    "successful_qps_ci95_low": successful_qps["confidence_interval_95"][0],
                    "successful_qps_ci95_high": successful_qps["confidence_interval_95"][1],
                    "successful_p95_ms_mean": p95["mean"],
                    "successful_p95_ms_ci95_low": p95["confidence_interval_95"][0],
                    "successful_p95_ms_ci95_high": p95["confidence_interval_95"][1],
                    "capacity_retention_mean": statistics["capacity_retention"]["mean"],
                }
            )
    return output.getvalue()


def render_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Backpressure Analysis",
        "",
        "Status: unverified development evidence; not eligible for a paper table.",
        "",
        "| Profile | Load | Success QPS, mean [95% CI] | Shed | P95 success latency, mean [95% CI] | Errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for profile, profile_result in analysis["summary"]["profiles"].items():
        for scenario in profile_result["scenarios"].values():
            statistics = scenario["repetition_statistics"]
            qps = statistics["successful_qps"]
            latency = statistics["successful_p95_ms"]
            qps_interval = qps["confidence_interval_95"]
            latency_interval = latency["confidence_interval_95"]
            lines.append(
                "| "
                f"{profile} | {scenario['load_fraction']:.1f}x | "
                f"{qps['mean']:.2f} [{qps_interval[0]:.2f}, {qps_interval[1]:.2f}] | "
                f"{scenario['pooled_rates']['shedding_rate']:.2%} | "
                f"{latency['mean']:.2f} [{latency_interval[0]:.2f}, {latency_interval[1]:.2f}] ms | "
                f"{scenario['pooled_counts']['error_count']} |"
            )
    lines.extend(
        [
            "",
            "Intervals bootstrap benchmark repetitions. Shed rates pool all offered requests.",
            "The source run remains unverified until controlled-host replication and attestation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(analysis: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "backpressure_analysis.json",
        "csv": output_dir / "backpressure_summary.csv",
        "markdown": output_dir / "backpressure_summary.md",
    }
    paths["json"].write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    paths["csv"].write_text(render_csv(analysis), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(analysis), encoding="utf-8")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260811)
    args = parser.parse_args()

    analysis = analyze_run(
        args.run_dir,
        expected_commit=args.expected_commit,
        bootstrap_repetitions=args.bootstrap_repetitions,
        random_seed=args.random_seed,
    )
    output_dir = args.output_dir or args.run_dir / "analysis" / "backpressure"
    paths = write_outputs(analysis, output_dir)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "paper_evidence_eligible": analysis["paper_evidence_eligible"],
                "source_run": analysis["source_run"],
                "verification": analysis["verification"],
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
