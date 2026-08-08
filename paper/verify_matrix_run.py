"""Verify an extracted frozen-matrix run without promoting its claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

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


def _verify_crash_summary(
    run_dir: Path,
    result: dict[str, Any],
    summary: dict[str, Any],
    expected_trials: int,
    errors: list[str],
) -> int:
    name = result["name"]
    workload = summary.get("workload", {})
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
    for trial in trial_records:
        mode = trial.get("mode")
        number = trial.get("trial")
        identity = (str(mode), number) if isinstance(number, int) else (str(mode), -1)
        if mode not in {"memory", "durable"} or not isinstance(number, int) or identity in seen:
            errors.append(f"crash_recovery:{name}: invalid or duplicate trial identity {identity}")
            continue
        seen.add(identity)
        expected_survival = mode == "durable"
        if trial.get("acknowledged") is not True:
            errors.append(f"crash_recovery:{name}:{mode}:{number}: acknowledgement missing")
        if trial.get("return_code") != 0:
            errors.append(f"crash_recovery:{name}:{mode}:{number}: child did not exit cleanly")
        if trial.get("survived_restart") is not expected_survival:
            errors.append(f"crash_recovery:{name}:{mode}:{number}: restart survival violated")

        prefix = f"{mutation}-{synchronous}-{mode}-{number}"
        database = cell_dir / f"{prefix}.sqlite3"
        marker = cell_dir / f"{prefix}.ack"
        if not database.is_file() or database.stat().st_size == 0:
            errors.append(f"crash_recovery:{name}:{mode}:{number}: database evidence missing")
        if not marker.is_file() or marker.stat().st_size == 0:
            errors.append(f"crash_recovery:{name}:{mode}:{number}: acknowledgement marker missing")
        elif marker.read_text(encoding="utf-8").strip() != trial.get("marker"):
            errors.append(f"crash_recovery:{name}:{mode}:{number}: marker content differs")

    metrics = summary.get("metrics", {})
    for mode, expected_survival in (("memory", 0.0), ("durable", 1.0)):
        mode_metrics = metrics.get(mode, {})
        if mode_metrics.get("trial_count") != expected_trials:
            errors.append(f"crash_recovery:{name}:{mode}: trial count differs")
        if mode_metrics.get("acknowledged_count") != expected_trials:
            errors.append(f"crash_recovery:{name}:{mode}: acknowledgement count differs")
        if mode_metrics.get("restart_survival_rate") != expected_survival:
            errors.append(f"crash_recovery:{name}:{mode}: restart survival rate differs")
        if mode_metrics.get("all_children_exited_cleanly") is not True:
            errors.append(f"crash_recovery:{name}:{mode}: child exit invariant failed")
    return len(trial_records)


def _verify_family_invariants(
    run_dir: Path,
    results: list[dict[str, Any]],
    matrix: dict[str, Any],
    log_paths: dict[int, Path],
    errors: list[str],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    crash_trials = 0
    for result in results:
        family = result["family"]
        summary = _verify_summary_binding(run_dir, result, log_paths[result["index"]], errors)
        if summary is None:
            continue
        counts[family] = counts.get(family, 0) + 1
        name = result["name"]
        if family == "crash_recovery":
            crash_trials += _verify_crash_summary(
                run_dir,
                result,
                summary,
                int(matrix[family]["trials_per_cell"]),
                errors,
            )
        elif family == "dynamic":
            metrics = summary.get("metrics", {})
            if metrics.get("correctness_violations") != 0:
                errors.append(f"dynamic:{name}: correctness violations")
            if metrics.get("pre_restart_state_equal") is not True:
                errors.append(f"dynamic:{name}: runtime and persisted state diverged")
            if metrics.get("restart_state_equal") is not True:
                errors.append(f"dynamic:{name}: restart state diverged")
        elif family == "scale":
            if summary.get("metrics", {}).get("top1_identity_accuracy") != 1.0:
                errors.append(f"scale:{name}: identity retrieval failed")
        elif family == "backpressure":
            scenarios = summary.get("scenarios", [])
            expected_loads = matrix[family]["offered_load_fraction"]
            if [item.get("load_fraction") for item in scenarios] != expected_loads:
                errors.append(f"backpressure:{name}: load scenarios differ from matrix")
            for scenario in scenarios:
                offered = scenario.get("offered_count")
                denominator = sum(
                    int(scenario.get(field, 0))
                    for field in ("successful_count", "overloaded_count", "error_count")
                )
                if offered != denominator:
                    errors.append(f"backpressure:{name}: offered-load denominator mismatch")
                if scenario.get("error_count") != 0:
                    errors.append(f"backpressure:{name}: unexpected request errors")
                if scenario.get("successful_count") and scenario.get("successful_accuracy") != 1.0:
                    errors.append(f"backpressure:{name}: incorrect successful routing")
        elif family == "quality":
            per_seed = summary.get("per_seed", [])
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
        invariant_report = {"family_command_counts": {}, "crash_trial_record_count": 0}

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
