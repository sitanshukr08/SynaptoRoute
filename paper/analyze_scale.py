"""Aggregate a verifier-accepted scale run without promoting its claims."""

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
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.manifest_schema import sha256_file  # noqa: E402
from paper.analysis_utils import concise_matrix_verification, summarize_values  # noqa: E402
from paper.verify_matrix_run import verify_matrix_run  # noqa: E402


NAME_PATTERN = re.compile(
    r"^(?P<engine>numpy|faiss)-r(?P<route_count>[0-9]+)-rep(?P<repetition>[0-9]+)$"
)
ENGINE_ORDER = ("numpy", "faiss")
COUNT_FIELDS = ("query_count", "correct_count", "incorrect_count")


def _finite_float(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _positive_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _count(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _parse_name(name: str) -> tuple[str, int, int]:
    match = NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid scale record name: {name}")
    return (
        match.group("engine"),
        int(match.group("route_count")),
        int(match.group("repetition")),
    )


def _statistics(
    values: Sequence[float],
    *,
    bootstrap_repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    return summarize_values(
        values,
        bootstrap_repetitions=bootstrap_repetitions,
        rng=rng,
    )


def _optional_statistics(
    values: Sequence[float | None],
    *,
    bootstrap_repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return _statistics(
        present,
        bootstrap_repetitions=bootstrap_repetitions,
        rng=rng,
    )


def _metric(summary: Mapping[str, Any], field: str, label: str) -> float:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{label} must contain metrics")
    return _finite_float(metrics.get(field), f"{label} {field}")


def _latency(summary: Mapping[str, Any], field: str, label: str) -> float:
    metrics = summary.get("metrics")
    latency = metrics.get("latency") if isinstance(metrics, Mapping) else None
    if not isinstance(latency, Mapping):
        raise ValueError(f"{label} must contain latency metrics")
    return _finite_float(latency.get(field), f"{label} latency {field}")


def analyze_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_repetitions: int = 10_000,
    random_seed: int = 20260811,
) -> dict[str, Any]:
    """Aggregate scale cells and paired exact-versus-HNSW effects."""

    if not records:
        raise ValueError("at least one scale record is required")
    grouped: dict[tuple[str, int], list[tuple[int, str, Mapping[str, Any]]]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for name, summary in sorted(records.items()):
        engine, route_count, repetition = _parse_name(name)
        identity = (engine, route_count, repetition)
        if identity in seen:
            raise ValueError(f"duplicate scale repetition: {name}")
        seen.add(identity)
        if summary.get("benchmark") != "precomputed_vector_scale":
            raise ValueError(f"{name} is not a scale summary")
        if summary.get("status") != "unverified" or summary.get("paper_evidence_eligible") is not False:
            raise ValueError(f"{name} must remain unverified and paper-ineligible")
        configuration = summary.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError(f"{name} must contain a configuration")
        if configuration.get("engine") != engine or configuration.get("route_count") != route_count:
            raise ValueError(f"{name} configuration differs from its identity")
        grouped[(engine, route_count)].append((repetition, name, summary))

    engines = {engine for engine, _ in grouped}
    if engines != set(ENGINE_ORDER):
        raise ValueError(f"scale analysis requires both engines, found {sorted(engines)}")
    route_sets = {
        engine: {route_count for grouped_engine, route_count in grouped if grouped_engine == engine}
        for engine in ENGINE_ORDER
    }
    if route_sets["numpy"] != route_sets["faiss"]:
        raise ValueError("scale engines do not contain the same route counts")

    rng = np.random.default_rng(random_seed)
    engine_results: dict[str, Any] = {engine: {"route_counts": {}} for engine in ENGINE_ORDER}
    ordered_groups: dict[tuple[str, int], list[tuple[int, str, Mapping[str, Any]]]] = {}
    for (engine, route_count), repetitions in sorted(grouped.items()):
        repetitions.sort(key=lambda item: item[0])
        repetition_ids = [item[0] for item in repetitions]
        if repetition_ids != list(range(len(repetitions))):
            raise ValueError(f"{engine} r{route_count} repetitions must be contiguous from zero")
        ordered_groups[(engine, route_count)] = repetitions

        query_counts: set[int] = set()
        dimensions: set[int] = set()
        seeds: list[int] = []
        totals = {field: 0 for field in COUNT_FIELDS}
        for _, name, summary in repetitions:
            configuration = summary["configuration"]
            metrics = summary.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError(f"{name} must contain metrics")
            query_counts.add(_count(configuration.get("query_count"), f"{name} query_count"))
            dimensions.add(_count(configuration.get("dimension"), f"{name} dimension"))
            seeds.append(_count(configuration.get("seed"), f"{name} seed"))
            for field in COUNT_FIELDS:
                totals[field] += _count(metrics.get(field), f"{name} {field}")
        if len(query_counts) != 1 or len(dimensions) != 1:
            raise ValueError(f"{engine} r{route_count} changes configuration between repetitions")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{engine} r{route_count} repeats a seed")
        if totals["query_count"] != totals["correct_count"] + totals["incorrect_count"]:
            raise ValueError(f"{engine} r{route_count} correctness counts do not add up")

        build_seconds = [
            _positive_float(_metric(summary, "build_seconds", name), f"{name} build_seconds")
            for _, name, summary in repetitions
        ]
        query_seconds = [
            _positive_float(_metric(summary, "query_seconds", name), f"{name} query_seconds")
            for _, name, summary in repetitions
        ]
        throughput = [_metric(summary, "throughput_qps", name) for _, name, summary in repetitions]
        accuracies = [_metric(summary, "top1_identity_accuracy", name) for _, name, summary in repetitions]
        rss_delta = [
            None
            if summary["metrics"].get("rss_delta_mb") is None
            else _finite_float(summary["metrics"]["rss_delta_mb"], f"{name} rss_delta_mb")
            for _, name, summary in repetitions
        ]
        statistics = {
            "top1_identity_accuracy": _statistics(
                accuracies,
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "build_seconds": _statistics(
                build_seconds,
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "query_seconds": _statistics(
                query_seconds,
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "throughput_qps": _statistics(
                throughput,
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "latency_p50_ms": _statistics(
                [_latency(summary, "p50_ms", name) for _, name, summary in repetitions],
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "latency_p95_ms": _statistics(
                [_latency(summary, "p95_ms", name) for _, name, summary in repetitions],
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "latency_p99_ms": _statistics(
                [_latency(summary, "p99_ms", name) for _, name, summary in repetitions],
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "rss_delta_mb": _optional_statistics(
                rss_delta,
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
        }
        engine_results[engine]["route_counts"][str(route_count)] = {
            "route_count": route_count,
            "configuration": {
                "query_count_per_repetition": next(iter(query_counts)),
                "dimension": next(iter(dimensions)),
            },
            "repetition_count": len(repetitions),
            "repetitions": repetition_ids,
            "seeds": seeds,
            "pooled_counts": totals,
            "pooled_top1_identity_accuracy": totals["correct_count"] / totals["query_count"],
            "repetition_statistics": statistics,
        }

    paired_comparisons: dict[str, Any] = {}
    for route_count in sorted(route_sets["numpy"]):
        numpy_by_seed = {
            int(summary["configuration"]["seed"]): summary
            for _, _, summary in ordered_groups[("numpy", route_count)]
        }
        faiss_by_seed = {
            int(summary["configuration"]["seed"]): summary
            for _, _, summary in ordered_groups[("faiss", route_count)]
        }
        if numpy_by_seed.keys() != faiss_by_seed.keys():
            raise ValueError(f"r{route_count} engines do not contain the same seeds")
        seeds = sorted(numpy_by_seed)

        def paired_values(function) -> list[float]:
            return [function(faiss_by_seed[seed], numpy_by_seed[seed]) for seed in seeds]

        rss_differences = []
        for seed in seeds:
            faiss_rss = faiss_by_seed[seed]["metrics"].get("rss_delta_mb")
            numpy_rss = numpy_by_seed[seed]["metrics"].get("rss_delta_mb")
            if faiss_rss is not None and numpy_rss is not None:
                rss_differences.append(
                    _finite_float(faiss_rss, "faiss rss")
                    - _finite_float(numpy_rss, "numpy rss")
                )

        paired_comparisons[str(route_count)] = {
            "route_count": route_count,
            "seeds": seeds,
            "faiss_minus_numpy_accuracy": _statistics(
                paired_values(
                    lambda faiss, numpy: _metric(faiss, "top1_identity_accuracy", "faiss")
                    - _metric(numpy, "top1_identity_accuracy", "numpy")
                ),
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "faiss_minus_numpy_incorrect_count": _statistics(
                paired_values(
                    lambda faiss, numpy: float(faiss["metrics"]["incorrect_count"])
                    - float(numpy["metrics"]["incorrect_count"])
                ),
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "faiss_to_numpy_build_time_ratio": _statistics(
                paired_values(
                    lambda faiss, numpy: _metric(faiss, "build_seconds", "faiss")
                    / _positive_float(_metric(numpy, "build_seconds", "numpy"), "numpy build_seconds")
                ),
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "faiss_to_numpy_throughput_ratio": _statistics(
                paired_values(
                    lambda faiss, numpy: _metric(faiss, "throughput_qps", "faiss")
                    / _positive_float(_metric(numpy, "throughput_qps", "numpy"), "numpy throughput_qps")
                ),
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "faiss_to_numpy_p95_latency_ratio": _statistics(
                paired_values(
                    lambda faiss, numpy: _latency(faiss, "p95_ms", "faiss")
                    / _positive_float(_latency(numpy, "p95_ms", "numpy"), "numpy p95_ms")
                ),
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
            "faiss_minus_numpy_rss_delta_mb": _optional_statistics(
                rss_differences,
                bootstrap_repetitions=bootstrap_repetitions,
                rng=rng,
            ),
        }

    return {
        "bootstrap_repetitions": bootstrap_repetitions,
        "random_seed": random_seed,
        "record_count": len(records),
        "route_counts": sorted(route_sets["numpy"]),
        "engines": engine_results,
        "paired_comparisons": paired_comparisons,
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
        expected_family="scale",
    )
    state_path = run_dir / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    records: dict[str, Mapping[str, Any]] = {}
    source_artifacts = []
    for result in state["results"]:
        name = str(result["name"])
        summary_path = run_dir / "scale" / f"{name}.json"
        records[name] = json.loads(summary_path.read_text(encoding="utf-8"))
        source_artifacts.append(
            {
                "name": name,
                "path": summary_path.relative_to(run_dir).as_posix(),
                "sha256": sha256_file(summary_path),
            }
        )
    if len(records) != verification["command_count"]:
        raise ValueError("scale summary count differs from the verified command count")

    return {
        "schema_version": 1,
        "analysis": "paired_repetition_level_scale_aggregation",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "source_run": {
            "run_id": verification["run_id"],
            "git_commit": verification["git_commit"],
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "run_state_sha256": sha256_file(state_path),
        },
        "verification": concise_matrix_verification(verification),
        "configuration": {
            "bootstrap_repetitions": bootstrap_repetitions,
            "random_seed": random_seed,
            "experimental_unit": "one scale cell at one engine, route count, and seed",
            "paired_unit": "the same route count and generated-vector seed across engines",
        },
        "summary": analyze_records(
            records,
            bootstrap_repetitions=bootstrap_repetitions,
            random_seed=random_seed,
        ),
        "source_artifacts": source_artifacts,
        "notes": [
            "Identity accuracy measures index retrieval of a vector already present in the index; it is not semantic accuracy.",
            "Confidence intervals resample five repetition-level values at each engine and route count.",
            "Paired effects compare FAISS with NumPy at the same route count and generated-vector seed.",
            "RSS deltas are process observations and may include allocator effects.",
            "This analysis remains unverified because the source run lacks controlled-host evidence and independent reproduction.",
        ],
    }


def render_csv(analysis: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    fields = (
        "engine",
        "route_count",
        "repetitions",
        "query_count",
        "correct_count",
        "incorrect_count",
        "pooled_accuracy",
        "build_seconds_mean",
        "build_seconds_ci95_low",
        "build_seconds_ci95_high",
        "throughput_qps_mean",
        "throughput_qps_ci95_low",
        "throughput_qps_ci95_high",
        "p95_ms_mean",
        "p95_ms_ci95_low",
        "p95_ms_ci95_high",
        "rss_delta_mb_mean",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for engine in ENGINE_ORDER:
        route_results = analysis["summary"]["engines"][engine]["route_counts"]
        for route_result in route_results.values():
            counts = route_result["pooled_counts"]
            statistics = route_result["repetition_statistics"]
            build = statistics["build_seconds"]
            throughput = statistics["throughput_qps"]
            p95 = statistics["latency_p95_ms"]
            rss = statistics["rss_delta_mb"]
            writer.writerow(
                {
                    "engine": engine,
                    "route_count": route_result["route_count"],
                    "repetitions": route_result["repetition_count"],
                    "query_count": counts["query_count"],
                    "correct_count": counts["correct_count"],
                    "incorrect_count": counts["incorrect_count"],
                    "pooled_accuracy": route_result["pooled_top1_identity_accuracy"],
                    "build_seconds_mean": build["mean"],
                    "build_seconds_ci95_low": build["confidence_interval_95"][0],
                    "build_seconds_ci95_high": build["confidence_interval_95"][1],
                    "throughput_qps_mean": throughput["mean"],
                    "throughput_qps_ci95_low": throughput["confidence_interval_95"][0],
                    "throughput_qps_ci95_high": throughput["confidence_interval_95"][1],
                    "p95_ms_mean": p95["mean"],
                    "p95_ms_ci95_low": p95["confidence_interval_95"][0],
                    "p95_ms_ci95_high": p95["confidence_interval_95"][1],
                    "rss_delta_mb_mean": None if rss is None else rss["mean"],
                }
            )
    return output.getvalue()


def _format_mean_interval(
    statistics: Mapping[str, Any],
    *,
    decimals: int,
    scale: float = 1.0,
    suffix: str = "",
) -> str:
    interval = statistics["confidence_interval_95"]
    return (
        f"{statistics['mean'] * scale:.{decimals}f} "
        f"[{interval[0] * scale:.{decimals}f}, {interval[1] * scale:.{decimals}f}]{suffix}"
    )


def render_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Scale Analysis",
        "",
        "Status: unverified development evidence; not eligible for a paper table.",
        "",
        "| Engine | Routes | Accuracy | Misses | Build, mean [95% CI] | QPS, mean [95% CI] | P95, mean [95% CI] | RSS delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for engine in ENGINE_ORDER:
        for route_result in analysis["summary"]["engines"][engine]["route_counts"].values():
            statistics = route_result["repetition_statistics"]
            build = statistics["build_seconds"]
            throughput = statistics["throughput_qps"]
            p95 = statistics["latency_p95_ms"]
            rss = statistics["rss_delta_mb"]
            rss_text = "n/a" if rss is None else f"{rss['mean']:.2f} MB"
            lines.append(
                "| "
                f"{engine} | {route_result['route_count']:,} | "
                f"{route_result['pooled_top1_identity_accuracy']:.3%} | "
                f"{route_result['pooled_counts']['incorrect_count']:,} | "
                f"{build['mean']:.2f} [{build['confidence_interval_95'][0]:.2f}, {build['confidence_interval_95'][1]:.2f}] s | "
                f"{throughput['mean']:.2f} [{throughput['confidence_interval_95'][0]:.2f}, {throughput['confidence_interval_95'][1]:.2f}] | "
                f"{p95['mean']:.3f} [{p95['confidence_interval_95'][0]:.3f}, {p95['confidence_interval_95'][1]:.3f}] ms | "
                f"{rss_text} |"
            )
    lines.extend(
        [
            "",
            "## Paired FAISS / NumPy Effects",
            "",
            "| Routes | Accuracy delta | Build-time ratio | Throughput ratio | P95 ratio | RSS delta |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in analysis["summary"]["paired_comparisons"].values():
        accuracy_effect = _format_mean_interval(
            comparison["faiss_minus_numpy_accuracy"],
            decimals=3,
            scale=100,
            suffix=" pp",
        )
        build_effect = _format_mean_interval(
            comparison["faiss_to_numpy_build_time_ratio"],
            decimals=2,
            suffix="x",
        )
        throughput_effect = _format_mean_interval(
            comparison["faiss_to_numpy_throughput_ratio"],
            decimals=2,
            suffix="x",
        )
        p95_effect = _format_mean_interval(
            comparison["faiss_to_numpy_p95_latency_ratio"],
            decimals=2,
            suffix="x",
        )
        rss_effect = comparison["faiss_minus_numpy_rss_delta_mb"]
        rss_effect_text = (
            "n/a"
            if rss_effect is None
            else _format_mean_interval(rss_effect, decimals=2, suffix=" MB")
        )
        lines.append(
            "| "
            f"{comparison['route_count']:,} | "
            f"{accuracy_effect} | "
            f"{build_effect} | "
            f"{throughput_effect} | "
            f"{p95_effect} | "
            f"{rss_effect_text} |"
        )
    lines.extend(
        [
            "",
            "Effects are FAISS minus or divided by NumPy within matched seeds.",
            "The source run remains unverified until controlled-host replication and attestation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(analysis: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "scale_analysis.json",
        "csv": output_dir / "scale_summary.csv",
        "markdown": output_dir / "scale_summary.md",
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
    output_dir = args.output_dir or args.run_dir / "analysis" / "scale"
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
