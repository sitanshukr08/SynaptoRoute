"""Verify an extracted frozen-matrix run without promoting its claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.manifest_schema import sha256_file, validate_manifest  # noqa: E402
from benchmarks.run_paper_matrix import build_commands  # noqa: E402


FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
FAMILIES = {"quality", "dynamic", "scale", "crash_recovery", "backpressure"}


class MatrixRunVerificationError(RuntimeError):
    """Raised when an extracted matrix run fails one or more integrity checks."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__("matrix run verification failed:\n- " + "\n- ".join(errors))


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_number(actual: Any, expected: float) -> bool:
    return (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and math.isfinite(float(actual))
        and math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-9)
    )


def _count(value: Any, label: str, errors: list[str]) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(f"{label} must be a non-negative integer")
        return None
    return value


def _nonnegative_number(value: Any, label: str, errors: list[str]) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        errors.append(f"{label} must be a finite non-negative number")
        return None
    return float(value)


def _verify_ratio(
    actual: Any,
    numerator: int,
    denominator: int,
    label: str,
    errors: list[str],
) -> None:
    if denominator == 0:
        if actual is not None:
            errors.append(f"{label} must be null when its denominator is zero")
        return
    expected = numerator / denominator
    if not _same_number(actual, expected):
        errors.append(f"{label} differs from its recorded numerator and denominator")


def _verify_throughput(
    actual: Any,
    count: int,
    wall_seconds: Any,
    label: str,
    errors: list[str],
) -> None:
    if (
        not isinstance(wall_seconds, (int, float))
        or isinstance(wall_seconds, bool)
        or not math.isfinite(float(wall_seconds))
    ):
        errors.append(f"{label} has an invalid wall-clock denominator")
        return
    wall = float(wall_seconds)
    if wall <= 0 or not _same_number(actual, count / wall):
        errors.append(f"{label} differs from its recorded count and wall time")


def _verify_percentiles(
    value: Any,
    *,
    expected_samples: int,
    label: str,
    errors: list[str],
) -> None:
    if expected_samples == 0:
        if value is not None:
            errors.append(f"{label} must be null when no samples were recorded")
        return
    if not isinstance(value, dict):
        errors.append(f"{label} must contain percentile metrics")
        return
    fields = ("p50_ms", "p95_ms", "p99_ms", "max_ms")
    values = [value.get(field) for field in fields]
    valid_values = all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        and item >= 0
        for item in values
    )
    if not valid_values:
        errors.append(f"{label} contains invalid percentile values")
    else:
        numeric_values = [float(cast(int | float, item)) for item in values]
        if numeric_values != sorted(numeric_values):
            errors.append(f"{label} percentiles are not monotonically ordered")


def _observe(
    observations: list[dict[str, Any]],
    *,
    family: str,
    name: str,
    code: str,
    observed: Any,
    expected: Any,
    message: str,
) -> None:
    observations.append(
        {
            "family": family,
            "name": name,
            "code": code,
            "observed": observed,
            "expected": expected,
            "message": message,
        }
    )


def _load_json(path: Path, label: str, errors: list[str]) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        errors.append(f"{label} is missing or empty: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{label} is not readable JSON: {error}")
        return None


def _resolve_reference(run_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if ".." in path.parts:
        return None
    parts = list(path.parts)
    candidates: list[Path] = []
    if not path.is_absolute():
        candidates.append(run_dir.joinpath(*parts))
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == run_dir.name:
            candidates.append(run_dir.joinpath(*parts[index + 1 :]))
            break
    for candidate in candidates:
        try:
            candidate.resolve().relative_to(run_dir.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def _parse_logged_json(path: Path, errors: list[str]) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"cannot read command log {path}: {error}")
        return None
    decoder = json.JSONDecoder()
    offset = 0
    for line in text.splitlines(keepends=True):
        if not line.startswith(("{", "[")):
            offset += len(line)
            continue
        try:
            value, end = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            offset += len(line)
            continue
        if not text[end:].strip():
            return value
        offset += len(line)
    errors.append(f"command log does not end with a JSON payload: {path}")
    return None


def _summary_for_result(run_dir: Path, result: dict[str, Any]) -> Path:
    family = result["family"]
    name = result["name"]
    if family == "quality":
        return run_dir / family / name / "multiseed_summary.json"
    if family == "dynamic":
        return run_dir / family / name / "dynamic_workload_summary.json"
    if family == "crash_recovery":
        return run_dir / family / name / "crash_recovery_summary.json"
    return run_dir / family / f"{name}.json"


def _verify_summary_binding(
    run_dir: Path,
    result: dict[str, Any],
    log_path: Path,
    errors: list[str],
) -> dict[str, Any] | None:
    summary_path = _summary_for_result(run_dir, result)
    summary = _load_json(summary_path, f"{result['family']} summary {result['name']}", errors)
    logged = _parse_logged_json(log_path, errors)
    if not isinstance(summary, dict) or logged is None:
        return None
    logged_summary = logged.get("study") if result["family"] == "quality" and isinstance(logged, dict) else logged
    if logged_summary != summary:
        errors.append(f"{result['family']} summary differs from hashed command log: {result['name']}")
    if summary.get("status") != "unverified" or summary.get("paper_evidence_eligible") is not False:
        errors.append(f"{result['family']} summary is not explicitly ineligible: {result['name']}")
    return summary


def _command_option(command: Any, flag: str) -> str | None:
    if not isinstance(command, list):
        return None
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def _verify_system_configuration(
    result: dict[str, Any],
    summary: dict[str, Any],
    errors: list[str],
) -> None:
    family = result["family"]
    name = result["name"]
    command = result.get("command")
    section_name = "workload" if family in {"dynamic", "crash_recovery"} else "configuration"
    section = summary.get(section_name)
    if not isinstance(section, dict):
        errors.append(f"{family}:{name}: {section_name} must be an object")
        return

    string_fields: tuple[tuple[str, str], ...] = ()
    numeric_fields: tuple[tuple[str, str], ...] = ()
    if family == "dynamic":
        numeric_fields = (
            ("route_count", "--routes"),
            ("query_workers", "--query-workers"),
            ("target_mutations_per_second", "--mutation-rate"),
            ("warmup_seconds", "--warmup"),
            ("duration_seconds", "--duration"),
        )
    elif family == "scale":
        string_fields = (("engine", "--engine"),)
        numeric_fields = (
            ("route_count", "--routes"),
            ("query_count", "--queries"),
            ("seed", "--seed"),
        )
    elif family == "crash_recovery":
        string_fields = (
            ("mutation", "--mutation"),
            ("sqlite_synchronous", "--synchronous"),
        )
        numeric_fields = (
            ("injected_storage_delay_ms", "--delay-ms"),
            ("trials_per_mode", "--trials"),
        )
    elif family == "backpressure":
        numeric_fields = (
            ("duration_seconds", "--duration"),
            ("saturation_calibration_target_seconds", "--calibration-duration"),
            ("queue_size", "--queue-size"),
            ("batch_size", "--batch-size"),
        )

    for field, flag in string_fields:
        expected = _command_option(command, flag)
        actual = section.get(field)
        if field == "sqlite_synchronous" and isinstance(actual, str):
            actual = actual.upper()
        if expected is None or actual != expected:
            errors.append(f"{family}:{name}: {field} differs from frozen command")
    for field, flag in numeric_fields:
        expected = _command_option(command, flag)
        try:
            matches = expected is not None and _same_number(section.get(field), float(expected))
        except ValueError:
            matches = False
        if not matches:
            errors.append(f"{family}:{name}: {field} differs from frozen command")


def _verify_crash_summary(
    run_dir: Path,
    result: dict[str, Any],
    summary: dict[str, Any],
    expected_trials: int,
    errors: list[str],
    observations: list[dict[str, Any]],
) -> int:
    name = result["name"]
    workload = summary.get("workload", {})
    if not isinstance(workload, dict):
        errors.append(f"crash_recovery:{name}: workload must be an object")
        return 0
    mutation = workload.get("mutation")
    synchronous = str(workload.get("sqlite_synchronous", "")).lower()
    trial_records = summary.get("trials")
    if workload.get("trials_per_mode") != expected_trials:
        errors.append(f"crash_recovery:{name}: unexpected trials_per_mode")
    if not isinstance(trial_records, list) or len(trial_records) != expected_trials * 2:
        errors.append(f"crash_recovery:{name}: incomplete trial records")
        return 0

    cell_dir = _summary_for_result(run_dir, result).parent
    seen: set[tuple[str, int]] = set()
    valid_trials: list[dict[str, Any]] = []
    for trial in trial_records:
        if not isinstance(trial, dict):
            errors.append(f"crash_recovery:{name}: trial record must be an object")
            continue
        mode = trial.get("mode")
        number = trial.get("trial")
        identity = (str(mode), number) if isinstance(number, int) else (str(mode), -1)
        if mode not in {"memory", "durable"} or not isinstance(number, int) or identity in seen:
            errors.append(f"crash_recovery:{name}: invalid or duplicate trial identity {identity}")
            continue
        seen.add(identity)
        valid_trials.append(trial)
        if trial.get("mutation") != mutation:
            errors.append(f"crash_recovery:{name}:{mode}:{number}: mutation differs")
        if str(trial.get("sqlite_synchronous", "")).lower() != synchronous:
            errors.append(f"crash_recovery:{name}:{mode}:{number}: synchronous mode differs")
        if not isinstance(trial.get("acknowledged"), bool):
            errors.append(f"crash_recovery:{name}:{mode}:{number}: invalid acknowledgement")
        if not isinstance(trial.get("return_code"), int):
            errors.append(f"crash_recovery:{name}:{mode}:{number}: invalid return code")
        if not isinstance(trial.get("survived_restart"), bool):
            errors.append(f"crash_recovery:{name}:{mode}:{number}: invalid survival result")
        _nonnegative_number(
            trial.get("wall_ms"),
            f"crash_recovery:{name}:{mode}:{number}: wall_ms",
            errors,
        )

        prefix = f"{mutation}-{synchronous}-{mode}-{number}"
        database = cell_dir / f"{prefix}.sqlite3"
        marker = cell_dir / f"{prefix}.ack"
        if not database.is_file() or database.stat().st_size == 0:
            errors.append(f"crash_recovery:{name}:{mode}:{number}: database evidence missing")
        elif _resolve_reference(run_dir, trial.get("database_path")) != database:
            errors.append(f"crash_recovery:{name}:{mode}:{number}: database path differs")
        elif trial.get("database_bytes") != database.stat().st_size:
            errors.append(f"crash_recovery:{name}:{mode}:{number}: database size differs")
        elif trial.get("database_sha256") != sha256_file(database):
            errors.append(f"crash_recovery:{name}:{mode}:{number}: database hash differs")
        if trial.get("acknowledged") is True and (
            not marker.is_file() or marker.stat().st_size == 0
        ):
            errors.append(f"crash_recovery:{name}:{mode}:{number}: acknowledgement marker missing")
        elif marker.is_file() and marker.read_text(encoding="utf-8").strip() != trial.get("marker"):
            errors.append(f"crash_recovery:{name}:{mode}:{number}: marker content differs")
        elif marker.is_file() and _resolve_reference(run_dir, trial.get("marker_path")) != marker:
            errors.append(f"crash_recovery:{name}:{mode}:{number}: marker path differs")
        elif marker.is_file() and trial.get("marker_sha256") != sha256_file(marker):
            errors.append(f"crash_recovery:{name}:{mode}:{number}: marker hash differs")
        elif not marker.is_file() and trial.get("marker_sha256") is not None:
            errors.append(f"crash_recovery:{name}:{mode}:{number}: marker hash has no artifact")

    expected_identities = {
        (mode, trial)
        for mode in ("memory", "durable")
        for trial in range(expected_trials)
    }
    if seen != expected_identities:
        errors.append(f"crash_recovery:{name}: trial identity set differs from workload")

    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        errors.append(f"crash_recovery:{name}: metrics must be an object")
        return len(trial_records)
    for mode, expected_survival in (("memory", False), ("durable", True)):
        mode_trials = [trial for trial in valid_trials if trial.get("mode") == mode]
        acknowledged_count = sum(trial.get("acknowledged") is True for trial in mode_trials)
        survived_count = sum(trial.get("survived_restart") is True for trial in mode_trials)
        clean_exit_count = sum(trial.get("return_code") == 0 for trial in mode_trials)
        contract_violation_count = sum(
            trial.get("survived_restart") is not expected_survival for trial in mode_trials
        )
        mode_metrics = metrics.get(mode, {})
        if not isinstance(mode_metrics, dict):
            errors.append(f"crash_recovery:{name}:{mode}: metrics must be an object")
            continue
        expected_values = {
            "trial_count": len(mode_trials),
            "acknowledged_count": acknowledged_count,
            "survived_count": survived_count,
            "clean_exit_count": clean_exit_count,
            "contract_violation_count": contract_violation_count,
        }
        for field, expected in expected_values.items():
            if mode_metrics.get(field) != expected:
                errors.append(f"crash_recovery:{name}:{mode}: {field} differs from trials")
        _verify_ratio(
            mode_metrics.get("acknowledgement_rate"),
            acknowledged_count,
            len(mode_trials),
            f"crash_recovery:{name}:{mode}: acknowledgement_rate",
            errors,
        )
        _verify_ratio(
            mode_metrics.get("restart_survival_rate"),
            survived_count,
            len(mode_trials),
            f"crash_recovery:{name}:{mode}: restart_survival_rate",
            errors,
        )
        _verify_ratio(
            mode_metrics.get("clean_exit_rate"),
            clean_exit_count,
            len(mode_trials),
            f"crash_recovery:{name}:{mode}: clean_exit_rate",
            errors,
        )
        _verify_ratio(
            mode_metrics.get("contract_success_rate"),
            len(mode_trials) - contract_violation_count,
            len(mode_trials),
            f"crash_recovery:{name}:{mode}: contract_success_rate",
            errors,
        )
        if mode_metrics.get("all_children_exited_cleanly") is not (
            clean_exit_count == len(mode_trials)
        ):
            errors.append(f"crash_recovery:{name}:{mode}: clean-exit summary differs")
        if mode_metrics.get("expected_restart_survival") is not expected_survival:
            errors.append(f"crash_recovery:{name}:{mode}: expected survival contract differs")
        if acknowledged_count != len(mode_trials):
            _observe(
                observations,
                family="crash_recovery",
                name=name,
                code=f"{mode}_acknowledgement_failures",
                observed=len(mode_trials) - acknowledged_count,
                expected=0,
                message=f"{mode} trials without an acknowledgement were retained.",
            )
        if clean_exit_count != len(mode_trials):
            _observe(
                observations,
                family="crash_recovery",
                name=name,
                code=f"{mode}_child_exit_failures",
                observed=len(mode_trials) - clean_exit_count,
                expected=0,
                message=f"{mode} trials with non-zero child exits were retained.",
            )
        if contract_violation_count:
            _observe(
                observations,
                family="crash_recovery",
                name=name,
                code=f"{mode}_restart_contract_violations",
                observed=contract_violation_count,
                expected=0,
                message=f"{mode} restart outcomes that violated the declared contract were retained.",
            )
    return len(trial_records)


def _verify_dynamic_summary(
    run_dir: Path,
    name: str,
    summary: dict[str, Any],
    errors: list[str],
    observations: list[dict[str, Any]],
) -> None:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        errors.append(f"dynamic:{name}: metrics must be an object")
        return
    fields = (
        "query_attempts",
        "completed_queries",
        "query_correct",
        "query_incorrect",
        "query_errors",
        "mutation_attempts",
        "mutation_successes",
        "mutation_errors",
        "mutation_shedding_count",
        "mutation_receipt_count",
        "durable_latency_count",
        "durable_receipt_count",
        "visibility_failures",
        "deletion_visibility_failures",
        "correctness_violations",
        "operation_failures",
        "total_adverse_outcomes",
    )
    counts = {
        field: _count(metrics.get(field), f"dynamic:{name}: {field}", errors)
        for field in fields
    }
    if any(value is None for value in counts.values()):
        return
    values = {field: int(value) for field, value in counts.items() if value is not None}
    if values["query_attempts"] != values["completed_queries"] + values["query_errors"]:
        errors.append(f"dynamic:{name}: query attempt denominator mismatch")
    if values["completed_queries"] != values["query_correct"] + values["query_incorrect"]:
        errors.append(f"dynamic:{name}: completed query denominator mismatch")
    if values["mutation_attempts"] != values["mutation_successes"] + values["mutation_errors"]:
        errors.append(f"dynamic:{name}: mutation attempt denominator mismatch")
    if values["mutation_shedding_count"] > values["mutation_errors"]:
        errors.append(f"dynamic:{name}: shedding exceeds mutation errors")
    if values["mutation_receipt_count"] != values["mutation_successes"]:
        errors.append(f"dynamic:{name}: mutation receipt count differs from successes")
    if values["durable_latency_count"] > values["mutation_receipt_count"]:
        errors.append(f"dynamic:{name}: durable latency count exceeds receipts")
    if values["durable_receipt_count"] > values["mutation_receipt_count"]:
        errors.append(f"dynamic:{name}: durable receipt count exceeds receipts")
    if values["durable_receipt_count"] != values["durable_latency_count"]:
        errors.append(f"dynamic:{name}: durable receipt and latency counts differ")
    queue_depth = metrics.get("storage_queue_depth", {})
    if not isinstance(queue_depth, dict):
        errors.append(f"dynamic:{name}: storage_queue_depth must be an object")
        queue_depth = {}
    _count(queue_depth.get("max"), f"dynamic:{name}: storage queue maximum", errors)
    if queue_depth.get("samples") != values["mutation_successes"]:
        errors.append(f"dynamic:{name}: storage queue samples differ from mutation successes")

    _verify_ratio(
        metrics.get("query_accuracy"),
        values["query_correct"],
        values["completed_queries"],
        f"dynamic:{name}: query_accuracy",
        errors,
    )
    _verify_ratio(
        metrics.get("query_success_rate"),
        values["completed_queries"],
        values["query_attempts"],
        f"dynamic:{name}: query_success_rate",
        errors,
    )
    _verify_ratio(
        metrics.get("mutation_success_rate"),
        values["mutation_successes"],
        values["mutation_attempts"],
        f"dynamic:{name}: mutation_success_rate",
        errors,
    )
    _verify_ratio(
        metrics.get("mutation_error_rate"),
        values["mutation_errors"],
        values["mutation_attempts"],
        f"dynamic:{name}: mutation_error_rate",
        errors,
    )
    wall = metrics.get("measurement_wall_seconds")
    throughputs = (
        ("query_attempt_throughput_per_second", values["query_attempts"]),
        ("query_success_throughput_per_second", values["completed_queries"]),
        ("query_throughput_per_second", values["completed_queries"]),
        ("mutation_attempt_throughput_per_second", values["mutation_attempts"]),
        ("mutation_success_throughput_per_second", values["mutation_successes"]),
        ("mutation_throughput_per_second", values["mutation_attempts"]),
    )
    for field, count in throughputs:
        _verify_throughput(metrics.get(field), count, wall, f"dynamic:{name}: {field}", errors)
    _verify_percentiles(
        metrics.get("query_latency"),
        expected_samples=values["completed_queries"],
        label=f"dynamic:{name}: query_latency",
        errors=errors,
    )
    _verify_percentiles(
        metrics.get("mutation_memory_ack"),
        expected_samples=values["mutation_successes"],
        label=f"dynamic:{name}: mutation_memory_ack",
        errors=errors,
    )
    _verify_percentiles(
        metrics.get("mutation_durable_commit"),
        expected_samples=values["durable_latency_count"],
        label=f"dynamic:{name}: mutation_durable_commit",
        errors=errors,
    )

    state_fields = ("pre_restart_state_equal", "restart_state_equal")
    for field in state_fields:
        if not isinstance(metrics.get(field), bool):
            errors.append(f"dynamic:{name}: {field} must be boolean")
    state_divergences = sum(metrics.get(field) is False for field in state_fields)
    expected_correctness = (
        values["query_incorrect"]
        + values["visibility_failures"]
        + values["deletion_visibility_failures"]
        + state_divergences
    )
    if values["correctness_violations"] != expected_correctness:
        errors.append(f"dynamic:{name}: correctness violation summary differs")
    expected_failures = values["query_errors"] + values["mutation_errors"]
    if values["operation_failures"] != expected_failures:
        errors.append(f"dynamic:{name}: operation failure summary differs")
    if values["total_adverse_outcomes"] != expected_correctness + expected_failures:
        errors.append(f"dynamic:{name}: total adverse outcome summary differs")

    error_evidence = summary.get("errors", {})
    if not isinstance(error_evidence, dict):
        errors.append(f"dynamic:{name}: errors must be an object")
        error_evidence = {}
    query_error_types = error_evidence.get("query")
    mutation_error_types = error_evidence.get("mutation")
    if not isinstance(query_error_types, list) or len(query_error_types) != values["query_errors"]:
        errors.append(f"dynamic:{name}: query error evidence differs from count")
    if (
        not isinstance(mutation_error_types, list)
        or len(mutation_error_types) != values["mutation_errors"]
    ):
        errors.append(f"dynamic:{name}: mutation error evidence differs from count")
    elif sum(item == "RouterOverloadedError" for item in mutation_error_types) != values[
        "mutation_shedding_count"
    ]:
        errors.append(f"dynamic:{name}: mutation shedding evidence differs from count")

    observation_fields = (
        ("query_incorrect", "incorrect_query_results", "Incorrect query results were retained."),
        ("query_errors", "query_operation_failures", "Query operation failures were retained."),
        ("mutation_errors", "mutation_operation_failures", "Mutation failures were retained."),
        ("visibility_failures", "mutation_visibility_failures", "Visibility failures were retained."),
        (
            "deletion_visibility_failures",
            "deletion_visibility_failures",
            "Deletion visibility failures were retained.",
        ),
    )
    for field, code, message in observation_fields:
        if values[field]:
            _observe(
                observations,
                family="dynamic",
                name=name,
                code=code,
                observed=values[field],
                expected=0,
                message=message,
            )
    for field in state_fields:
        if metrics.get(field) is False:
            _observe(
                observations,
                family="dynamic",
                name=name,
                code=field,
                observed=False,
                expected=True,
                message=f"The {field} state comparison failed and was retained.",
            )

    evidence = summary.get("evidence", {})
    if not isinstance(evidence, dict):
        errors.append(f"dynamic:{name}: evidence must be an object")
        return
    database = _resolve_reference(run_dir, evidence.get("database_path"))
    if database is None:
        errors.append(f"dynamic:{name}: database evidence is missing")
    elif evidence.get("database_bytes") != database.stat().st_size:
        errors.append(f"dynamic:{name}: database evidence size differs")
    elif evidence.get("database_sha256") != sha256_file(database):
        errors.append(f"dynamic:{name}: database evidence hash differs")


def _verify_scale_summary(
    name: str,
    summary: dict[str, Any],
    errors: list[str],
    observations: list[dict[str, Any]],
) -> None:
    configuration = summary.get("configuration", {})
    metrics = summary.get("metrics", {})
    if not isinstance(configuration, dict) or not isinstance(metrics, dict):
        errors.append(f"scale:{name}: configuration and metrics must be objects")
        return
    index_parameters = configuration.get("index_parameters")
    if index_parameters is not None:
        if not isinstance(index_parameters, dict):
            errors.append(f"scale:{name}: index_parameters must be an object")
        else:
            route_count = configuration.get("route_count")
            if index_parameters.get("construction_add_calls") != route_count:
                errors.append(f"scale:{name}: construction calls differ from route count")
            if index_parameters.get("vectors_per_add_call") != 1:
                errors.append(f"scale:{name}: vectors_per_add_call must equal one")
            if index_parameters.get("metric") != "normalized_inner_product":
                errors.append(f"scale:{name}: index metric is not normalized inner product")
            engine = configuration.get("engine")
            expected_implementation = {
                "numpy": "numpy_exact",
                "faiss": "faiss_hnsw",
            }.get(engine) if isinstance(engine, str) else None
            if index_parameters.get("implementation") != expected_implementation:
                errors.append(f"scale:{name}: index implementation differs from engine")
            if engine == "faiss":
                version = index_parameters.get("faiss_version")
                if not isinstance(version, str) or not version:
                    errors.append(f"scale:{name}: FAISS version is missing")
                for field in (
                    "omp_threads",
                    "hnsw_m",
                    "hnsw_ef_construction",
                    "hnsw_ef_search",
                    "search_candidate_floor",
                ):
                    value = index_parameters.get(field)
                    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                        errors.append(f"scale:{name}: {field} must be a positive integer")
            elif engine == "numpy" and index_parameters.get("max_capacity") != route_count:
                errors.append(f"scale:{name}: NumPy capacity differs from route count")
    query_count = _count(metrics.get("query_count"), f"scale:{name}: query_count", errors)
    correct = _count(metrics.get("correct_count"), f"scale:{name}: correct_count", errors)
    incorrect = _count(metrics.get("incorrect_count"), f"scale:{name}: incorrect_count", errors)
    if query_count is None or correct is None or incorrect is None:
        return
    if configuration.get("query_count") != query_count:
        errors.append(f"scale:{name}: query count differs from configuration")
    if query_count != correct + incorrect:
        errors.append(f"scale:{name}: query denominator mismatch")
    _verify_ratio(
        metrics.get("top1_identity_accuracy"),
        correct,
        query_count,
        f"scale:{name}: top1_identity_accuracy",
        errors,
    )
    _verify_throughput(
        metrics.get("throughput_qps"),
        query_count,
        metrics.get("query_seconds"),
        f"scale:{name}: throughput_qps",
        errors,
    )
    _verify_percentiles(
        metrics.get("latency"),
        expected_samples=query_count,
        label=f"scale:{name}: latency",
        errors=errors,
    )
    _nonnegative_number(metrics.get("build_seconds"), f"scale:{name}: build_seconds", errors)
    rss_before = metrics.get("rss_before_mb")
    rss_after = metrics.get("rss_after_mb")
    rss_delta = metrics.get("rss_delta_mb")
    if rss_before is None or rss_after is None:
        if rss_delta is not None:
            errors.append(f"scale:{name}: RSS delta requires before and after measurements")
    elif not (
        _same_number(rss_delta, float(rss_after) - float(rss_before))
        if isinstance(rss_before, (int, float))
        and not isinstance(rss_before, bool)
        and isinstance(rss_after, (int, float))
        and not isinstance(rss_after, bool)
        else False
    ):
        errors.append(f"scale:{name}: RSS delta differs from before and after measurements")
    if incorrect:
        _observe(
            observations,
            family="scale",
            name=name,
            code="identity_retrieval_misses",
            observed=incorrect,
            expected=0,
            message="Structural identity-retrieval misses were retained.",
        )


def _verify_backpressure_summary(
    name: str,
    summary: dict[str, Any],
    expected_loads: list[float],
    errors: list[str],
    observations: list[dict[str, Any]],
) -> None:
    configuration = summary.get("configuration", {})
    if not isinstance(configuration, dict):
        errors.append(f"backpressure:{name}: configuration must be an object")
        return
    attempts = _count(
        configuration.get("saturation_calibration_attempts"),
        f"backpressure:{name}: calibration attempts",
        errors,
    )
    successes = _count(
        configuration.get("saturation_calibration_successes"),
        f"backpressure:{name}: calibration successes",
        errors,
    )
    overloaded = _count(
        configuration.get("saturation_calibration_overloaded"),
        f"backpressure:{name}: calibration overloads",
        errors,
    )
    calibration_errors = _count(
        configuration.get("saturation_calibration_error_count"),
        f"backpressure:{name}: calibration errors",
        errors,
    )
    if (
        attempts is not None
        and successes is not None
        and overloaded is not None
        and calibration_errors is not None
    ):
        if attempts != successes + overloaded + calibration_errors:
            errors.append(f"backpressure:{name}: calibration denominator mismatch")
        _verify_throughput(
            configuration.get("measured_saturation_qps"),
            successes,
            configuration.get("saturation_calibration_seconds"),
            f"backpressure:{name}: measured_saturation_qps",
            errors,
        )
        if calibration_errors:
            _observe(
                observations,
                family="backpressure",
                name=name,
                code="calibration_errors",
                observed=calibration_errors,
                expected=0,
                message="Calibration errors were retained.",
            )
        calibration_error_types = configuration.get("saturation_calibration_error_types")
        if (
            not isinstance(calibration_error_types, list)
            or len(calibration_error_types) != calibration_errors
        ):
            errors.append(f"backpressure:{name}: calibration error evidence differs from count")

    scenarios = summary.get("scenarios", [])
    if not isinstance(scenarios, list):
        errors.append(f"backpressure:{name}: scenarios must be a list")
        return
    if [item.get("load_fraction") for item in scenarios if isinstance(item, dict)] != expected_loads:
        errors.append(f"backpressure:{name}: load scenarios differ from matrix")
    measured_saturation = configuration.get("measured_saturation_qps")
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            errors.append(f"backpressure:{name}: scenario {index} must be an object")
            continue
        load = scenario.get("load_fraction")
        label = f"backpressure:{name}:load={load}"
        count_fields = (
            "offered_count",
            "successful_count",
            "successful_correct_count",
            "successful_incorrect_count",
            "overloaded_count",
            "error_count",
        )
        counts = {
            field: _count(scenario.get(field), f"{label}: {field}", errors)
            for field in count_fields
        }
        if any(value is None for value in counts.values()):
            continue
        values = {field: int(value) for field, value in counts.items() if value is not None}
        if values["offered_count"] != (
            values["successful_count"] + values["overloaded_count"] + values["error_count"]
        ):
            errors.append(f"{label}: offered-load denominator mismatch")
        if values["successful_count"] != (
            values["successful_correct_count"] + values["successful_incorrect_count"]
        ):
            errors.append(f"{label}: successful-result denominator mismatch")
        _verify_ratio(
            scenario.get("success_rate"),
            values["successful_count"],
            values["offered_count"],
            f"{label}: success_rate",
            errors,
        )
        _verify_ratio(
            scenario.get("shedding_rate"),
            values["overloaded_count"],
            values["offered_count"],
            f"{label}: shedding_rate",
            errors,
        )
        _verify_ratio(
            scenario.get("error_rate"),
            values["error_count"],
            values["offered_count"],
            f"{label}: error_rate",
            errors,
        )
        _verify_ratio(
            scenario.get("successful_accuracy"),
            values["successful_correct_count"],
            values["successful_count"],
            f"{label}: successful_accuracy",
            errors,
        )
        _verify_throughput(
            scenario.get("offered_qps"),
            values["offered_count"],
            scenario.get("offering_wall_seconds"),
            f"{label}: offered_qps",
            errors,
        )
        for field, count in (
            ("successful_qps", values["successful_count"]),
            ("completed_qps", values["successful_count"]),
            ("resolved_qps", values["offered_count"]),
        ):
            _verify_throughput(
                scenario.get(field),
                count,
                scenario.get("scenario_wall_seconds"),
                f"{label}: {field}",
                errors,
            )
        offering_wall = _nonnegative_number(
            scenario.get("offering_wall_seconds"),
            f"{label}: offering_wall_seconds",
            errors,
        )
        drain = _nonnegative_number(
            scenario.get("drain_seconds"),
            f"{label}: drain_seconds",
            errors,
        )
        scenario_wall = _nonnegative_number(
            scenario.get("scenario_wall_seconds"),
            f"{label}: scenario_wall_seconds",
            errors,
        )
        if (
            offering_wall is not None
            and drain is not None
            and scenario_wall is not None
            and (offering_wall > scenario_wall or drain > scenario_wall)
        ):
            errors.append(f"{label}: timing windows are inconsistent")
        if not (
            isinstance(load, (int, float))
            and not isinstance(load, bool)
            and isinstance(measured_saturation, (int, float))
            and not isinstance(measured_saturation, bool)
            and _same_number(scenario.get("target_qps"), float(load) * float(measured_saturation))
        ):
            errors.append(f"{label}: target_qps differs from calibrated saturation")
        _verify_percentiles(
            scenario.get("successful_latency"),
            expected_samples=values["successful_count"],
            label=f"{label}: successful_latency",
            errors=errors,
        )
        _verify_percentiles(
            scenario.get("overload_latency"),
            expected_samples=values["overloaded_count"],
            label=f"{label}: overload_latency",
            errors=errors,
        )
        error_types = scenario.get("error_types")
        if not isinstance(error_types, list) or len(error_types) != values["error_count"]:
            errors.append(f"{label}: error type evidence differs from error count")
        for field, code, message in (
            ("overloaded_count", "requests_shed", "Explicitly shed requests were retained."),
            ("error_count", "request_errors", "Unexpected request errors were retained."),
            (
                "successful_incorrect_count",
                "incorrect_successes",
                "Incorrect successful routing results were retained.",
            ),
        ):
            if values[field]:
                _observe(
                    observations,
                    family="backpressure",
                    name=name,
                    code=code,
                    observed={"load_fraction": load, "count": values[field]},
                    expected={"count": 0},
                    message=message,
                )


def _verify_family_invariants(
    run_dir: Path,
    results: list[dict[str, Any]],
    matrix: dict[str, Any],
    log_paths: dict[int, Path],
    errors: list[str],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    crash_trials = 0
    observations: list[dict[str, Any]] = []
    expected_benchmarks = {
        "dynamic": "concurrent_dynamic_routing_workload",
        "scale": "precomputed_vector_scale",
        "crash_recovery": "abrupt_process_restart_durability",
        "backpressure": "sustained_async_backpressure",
    }
    for result in results:
        family = result["family"]
        summary = _verify_summary_binding(run_dir, result, log_paths[result["index"]], errors)
        if summary is None:
            continue
        counts[family] = counts.get(family, 0) + 1
        name = result["name"]
        if family in expected_benchmarks:
            if summary.get("schema_version") != 2:
                errors.append(f"{family}:{name}: unsupported summary schema version")
            if summary.get("benchmark") != expected_benchmarks[family]:
                errors.append(f"{family}:{name}: benchmark identity differs")
            _verify_system_configuration(result, summary, errors)
        if family == "crash_recovery":
            crash_trials += _verify_crash_summary(
                run_dir,
                result,
                summary,
                int(matrix[family]["trials_per_cell"]),
                errors,
                observations,
            )
        elif family == "dynamic":
            _verify_dynamic_summary(run_dir, name, summary, errors, observations)
        elif family == "scale":
            _verify_scale_summary(name, summary, errors, observations)
        elif family == "backpressure":
            _verify_backpressure_summary(
                name,
                summary,
                matrix[family]["offered_load_fraction"],
                errors,
                observations,
            )
        elif family == "quality":
            per_seed = summary.get("per_seed", [])
            if not isinstance(per_seed, list) or any(
                not isinstance(item, dict) for item in per_seed
            ):
                errors.append(f"quality:{name}: per_seed must be a list of objects")
                continue
            expected_seeds = matrix[family]["seeds"]
            if [item.get("seed") for item in per_seed] != expected_seeds:
                errors.append(f"quality:{name}: seed outputs differ from matrix")
            for item in per_seed:
                seed_path = _resolve_reference(run_dir, item.get("summary_path"))
                if seed_path is None:
                    errors.append(f"quality:{name}: per-seed summary missing")
                elif sha256_file(seed_path) != item.get("summary_sha256"):
                    errors.append(f"quality:{name}: per-seed summary hash mismatch")
    return {
        "family_command_counts": dict(sorted(counts.items())),
        "crash_trial_record_count": crash_trials,
        "outcome_observation_count": len(observations),
        "outcome_observations": observations,
    }


def _verify_environment(
    environment_dir: Path | None,
    *,
    commit: str,
    python_version: str,
    required: bool,
    errors: list[str],
) -> bool:
    if environment_dir is None or not environment_dir.is_dir():
        if required:
            errors.append("environment evidence directory is required")
        return False
    required_nonempty = (
        "resolved-environment.txt",
        "python-version.txt",
        "uname.txt",
        "lscpu.txt",
        "git-commit.txt",
        "github-runner.txt",
    )
    for name in required_nonempty:
        path = environment_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"environment evidence is missing or empty: {name}")
    commit_path = environment_dir / "git-commit.txt"
    if commit_path.is_file() and commit_path.read_text(encoding="utf-8").strip() != commit:
        errors.append("environment git commit differs from manifest")
    status_path = environment_dir / "git-status.txt"
    if not status_path.is_file():
        errors.append("environment git-status.txt is missing")
    elif status_path.read_text(encoding="utf-8").strip():
        errors.append("environment recorded a dirty candidate checkout")
    version_path = environment_dir / "python-version.txt"
    if version_path.is_file() and version_path.read_text(encoding="utf-8").strip() != f"Python {python_version}":
        errors.append("environment Python version differs from manifest")
    return True


def verify_matrix_run(
    run_dir: Path,
    *,
    source_root: Path = REPO_ROOT,
    expected_commit: str | None = None,
    expected_family: str | None = None,
    environment_dir: Path | None = None,
    require_environment: bool = False,
) -> dict[str, Any]:
    """Verify one extracted matrix output directory and return a concise report."""

    run_dir = run_dir.resolve()
    source_root = source_root.resolve()
    errors: list[str] = []
    manifest_path = run_dir / "manifest.json"
    state_path = run_dir / "run_state.json"
    raw_path = run_dir / "matrix_results.json"
    manifest = _load_json(manifest_path, "matrix manifest", errors)
    state = _load_json(state_path, "matrix run state", errors)
    raw_results = _load_json(raw_path, "matrix raw results", errors)
    if not isinstance(manifest, dict) or not isinstance(state, dict) or not isinstance(raw_results, list):
        raise MatrixRunVerificationError(errors)

    errors.extend(f"manifest: {error}" for error in validate_manifest(manifest, repo_root=source_root))
    commit = str(manifest.get("git_commit", ""))
    if not FULL_SHA.fullmatch(commit):
        errors.append("manifest git_commit is not a full lowercase SHA")
    if expected_commit is not None and commit != expected_commit:
        errors.append(f"candidate commit differs: expected {expected_commit}, found {commit}")
    if manifest.get("status") != "unverified" or manifest.get("paper_evidence_eligible") is not False:
        errors.append("matrix run must remain unverified and paper-ineligible")
    if manifest.get("working_tree_dirty") is not False:
        errors.append("matrix run did not record a clean working tree")
    if manifest.get("exit_status") != 0:
        errors.append("matrix manifest exit_status is not zero")
    if state.get("git_commit") != commit:
        errors.append("run state commit differs from manifest")
    if state.get("run_id") != manifest.get("run_id"):
        errors.append("run state ID differs from manifest")
    if state.get("status") != "completed":
        errors.append("run state is not completed")

    results = state.get("results")
    if not isinstance(results, list):
        errors.append("run state results must be a list")
        results = []
    if not results:
        errors.append("run state contains no command results")
    if any(not isinstance(result, dict) for result in results):
        errors.append("every run state result must be an object")
        raise MatrixRunVerificationError(errors)
    if raw_results != results:
        errors.append("matrix_results.json differs from run_state results")
    expected_indexes = list(range(len(results)))
    if [item.get("index") for item in results] != expected_indexes:
        errors.append("result indexes are not contiguous and ordered")
        raise MatrixRunVerificationError(errors)

    metrics = manifest.get("metrics", {})
    expected_count = state.get("command_count")
    if expected_count != len(results):
        errors.append("run state command count differs from completed results")
    if metrics.get("command_count") != expected_count:
        errors.append("manifest command count differs from run state")
    if metrics.get("completed_command_count") != len(results):
        errors.append("manifest completed count differs from run state")
    if metrics.get("successful_command_count") != len(results) or metrics.get("failed_command_count") != 0:
        errors.append("manifest does not report every command successful")

    evidence = manifest.get("evidence", {})
    resolved_raw = _resolve_reference(run_dir, evidence.get("raw_output_path"))
    resolved_state = _resolve_reference(run_dir, evidence.get("run_state_path"))
    if resolved_raw != raw_path or sha256_file(raw_path) != evidence.get("raw_output_sha256"):
        errors.append("raw results reference or SHA-256 is invalid")
    if resolved_state != state_path or sha256_file(state_path) != evidence.get("run_state_sha256"):
        errors.append("run state reference or SHA-256 is invalid")

    matrix_path = source_root / "paper" / "experiment_matrix.json"
    matrix = _load_json(matrix_path, "source experiment matrix", errors)
    if not isinstance(matrix, dict):
        raise MatrixRunVerificationError(errors)
    if sha256_file(matrix_path) != state.get("matrix_sha256"):
        errors.append("source matrix SHA-256 differs from run state")
    if manifest.get("configuration", {}).get("matrix") != matrix:
        errors.append("manifest matrix differs from source matrix")
    if manifest.get("configuration", {}).get("matrix_sha256") != state.get("matrix_sha256"):
        errors.append("manifest matrix SHA-256 differs from run state")

    lock = manifest.get("dependency_lock", {})
    lock_path = (source_root / str(lock.get("path", ""))).resolve()
    try:
        lock_path.relative_to(source_root)
    except ValueError:
        errors.append("dependency lock path escapes the source checkout")
    if not lock_path.is_file() or sha256_file(lock_path) != lock.get("sha256"):
        errors.append("dependency lock is missing or has a different SHA-256")

    command_plan = [
        {field: result.get(field) for field in ("family", "name", "command")}
        for result in results
    ]
    plan_sha256 = _json_sha256(command_plan)
    if plan_sha256 != state.get("command_plan_sha256"):
        errors.append("result command plan SHA-256 differs from run state")
    if manifest.get("configuration", {}).get("command_plan_sha256") != plan_sha256:
        errors.append("manifest command plan SHA-256 differs from results")

    if expected_commit is not None and not FULL_SHA.fullmatch(expected_commit):
        errors.append("expected candidate commit is not a full lowercase SHA")
    if expected_family is not None and expected_family not in FAMILIES:
        errors.append(f"unsupported expected family: {expected_family}")
    families = sorted({str(result.get("family")) for result in results})
    if any(family not in FAMILIES for family in families):
        errors.append(f"run contains an unsupported family: {families}")
    if expected_family is not None and families != [expected_family]:
        errors.append(f"run families differ: expected {[expected_family]}, found {families}")
    expected_identities = [
        (item["family"], item["name"])
        for item in build_commands(matrix, run_dir, set(families))
    ]
    actual_identities = [(result.get("family"), result.get("name")) for result in results]
    if actual_identities != expected_identities:
        errors.append("result family/name plan differs from the frozen matrix")

    log_paths: dict[int, Path] = {}
    for result in results:
        name = result.get("name", "unknown")
        if result.get("return_code") != 0 or result.get("timed_out") is not False:
            errors.append(f"command did not complete successfully: {name}")
        log_path = _resolve_reference(run_dir, result.get("log_path"))
        if log_path is None or log_path.stat().st_size == 0:
            errors.append(f"command log is missing or empty: {name}")
            continue
        if sha256_file(log_path) != result.get("log_sha256"):
            errors.append(f"command log SHA-256 mismatch: {name}")
        log_paths[int(result["index"])] = log_path

    if len(log_paths) == len(results):
        invariant_report = _verify_family_invariants(run_dir, results, matrix, log_paths, errors)
    else:
        invariant_report = {
            "family_command_counts": {},
            "crash_trial_record_count": 0,
            "outcome_observation_count": 0,
            "outcome_observations": [],
        }

    if environment_dir is None:
        discovered = run_dir.parent / "environment"
        environment_dir = discovered if discovered.is_dir() else None
    environment_verified = _verify_environment(
        environment_dir,
        commit=commit,
        python_version=str(manifest.get("environment", {}).get("python_version", "")),
        required=require_environment,
        errors=errors,
    )

    if errors:
        raise MatrixRunVerificationError(errors)
    observations = invariant_report.pop("outcome_observations")
    observation_count = invariant_report.pop("outcome_observation_count")
    return {
        "schema_version": 1,
        "verification_status": "valid_unverified_matrix_run",
        "paper_evidence_eligible": False,
        "run_id": manifest["run_id"],
        "git_commit": commit,
        "families": families,
        "command_count": len(results),
        "log_hashes_verified": len(log_paths),
        "raw_and_state_hashes_verified": True,
        "environment_evidence_verified": environment_verified,
        "invariants": {"passed": True, **invariant_report},
        "outcome_observation_count": observation_count,
        "outcome_observations": observations,
        "warning": "Independent reproduction, immutable archival, and reviewer attestation are still required.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--expected-commit")
    parser.add_argument("--family", choices=sorted(FAMILIES))
    parser.add_argument("--environment-dir", type=Path)
    parser.add_argument("--require-environment", action="store_true")
    args = parser.parse_args()
    try:
        report = verify_matrix_run(
            args.run_dir,
            source_root=args.source_root,
            expected_commit=args.expected_commit,
            expected_family=args.family,
            environment_dir=args.environment_dir,
            require_environment=args.require_environment,
        )
    except MatrixRunVerificationError as error:
        print(json.dumps({"verified": False, "errors": list(error.errors)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
