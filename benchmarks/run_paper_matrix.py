"""Generate or execute the frozen systems-paper experiment matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.manifest_schema import sha256_file, validate_manifest  # noqa: E402

DEFAULT_MATRIX = REPO_ROOT / "paper" / "experiment_matrix.json"


def build_commands(matrix: dict, output_dir: Path, families: set[str]) -> list[dict]:
    commands: list[dict] = []

    if "quality" in families:
        quality = matrix["quality"]
        for dataset in quality["datasets"]:
            output = output_dir / "quality" / dataset
            commands.append(
                {
                    "family": "quality",
                    "name": dataset,
                    "command": [
                        sys.executable,
                        "benchmarks/run_multiseed_intent.py",
                        "--dataset",
                        dataset,
                        "--model",
                        quality["encoder"],
                        "--seeds",
                        *[str(seed) for seed in quality["seeds"]],
                        "--examples-per-route",
                        str(quality["examples_per_intent"]),
                        "--output-dir",
                        str(output),
                    ],
                }
            )

    if "dynamic" in families:
        dynamic = matrix["dynamic"]
        for route_count in dynamic["route_counts"]:
            for workers in dynamic["query_workers"]:
                for rate in dynamic["mutation_rates_per_second"]:
                    for repetition in range(dynamic["repetitions"]):
                        name = f"r{route_count}-w{workers}-m{rate}-rep{repetition}"
                        output = output_dir / "dynamic" / name
                        commands.append(
                            {
                                "family": "dynamic",
                                "name": name,
                                "command": [
                                    sys.executable,
                                    "benchmarks/bench_dynamic_workload.py",
                                    "--routes",
                                    str(route_count),
                                    "--query-workers",
                                    str(workers),
                                    "--mutation-rate",
                                    str(rate),
                                    "--warmup",
                                    str(dynamic["warmup_seconds"]),
                                    "--duration",
                                    str(dynamic["measurement_seconds"]),
                                    "--output-dir",
                                    str(output),
                                ],
                            }
                        )

    if "scale" in families:
        scale = matrix["scale"]
        for route_count in scale["route_counts"]:
            for configured_engine in scale["indexes"]:
                engine = "numpy" if configured_engine == "numpy_exact" else "faiss"
                for repetition in range(scale["repetitions"]):
                    name = f"{engine}-r{route_count}-rep{repetition}"
                    output = output_dir / "scale" / f"{name}.json"
                    commands.append(
                        {
                            "family": "scale",
                            "name": name,
                            "command": [
                                sys.executable,
                                "benchmarks/bench_scale_matrix.py",
                                "--engine",
                                engine,
                                "--routes",
                                str(route_count),
                                "--queries",
                                str(scale["query_count"]),
                                "--seed",
                                str(42 + repetition),
                                "--output",
                                str(output),
                            ],
                        }
                    )

    if "crash_recovery" in families:
        crash = matrix["crash_recovery"]
        for mutation in crash["mutations"]:
            for synchronous in crash["sqlite_synchronous"]:
                for delay_ms in crash["injected_commit_delay_ms"]:
                    name = f"{mutation}-{synchronous.lower()}-{delay_ms}ms"
                    output = output_dir / "crash_recovery" / name
                    commands.append(
                        {
                            "family": "crash_recovery",
                            "name": name,
                            "command": [
                                sys.executable,
                                "benchmarks/bench_crash_recovery.py",
                                "--mutation",
                                mutation,
                                "--synchronous",
                                synchronous,
                                "--delay-ms",
                                str(delay_ms),
                                "--trials",
                                str(crash["trials_per_cell"]),
                                "--output-dir",
                                str(output),
                            ],
                        }
                    )

    if "backpressure" in families:
        backpressure = matrix["backpressure"]
        for profile_name, profile in backpressure["profiles"].items():
            for repetition in range(backpressure["repetitions"]):
                name = f"{profile_name}-rep{repetition}"
                output = output_dir / "backpressure" / f"{name}.json"
                commands.append(
                    {
                        "family": "backpressure",
                        "name": name,
                        "command": [
                            sys.executable,
                            "benchmarks/bench_sustained_backpressure.py",
                            "--load-fractions",
                            *[str(value) for value in backpressure["offered_load_fraction"]],
                            "--duration",
                            str(backpressure["measurement_seconds"]),
                            "--calibration-duration",
                            str(backpressure["saturation_measurement_seconds"]),
                            "--queue-size",
                            str(profile["queue_size"]),
                            "--batch-size",
                            str(profile["batch_size"]),
                            "--output",
                            str(output),
                        ],
                    }
                )
    return commands


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _machine_id() -> str:
    value = f"{platform.node()}|{platform.platform()}|{platform.processor()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_reference(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_reference(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_resume_state(
    state: dict[str, Any],
    *,
    commit: str,
    matrix_sha256: str,
    command_plan_sha256: str,
    commands: list[dict],
    command_timeout_seconds: float | None,
) -> dict[int, dict[str, Any]]:
    expected = {
        "git_commit": commit,
        "matrix_sha256": matrix_sha256,
        "command_plan_sha256": command_plan_sha256,
        "command_count": len(commands),
        "command_timeout_seconds": command_timeout_seconds,
    }
    mismatches = [
        f"{field}: expected {value!r}, found {state.get(field)!r}"
        for field, value in expected.items()
        if state.get(field) != value
    ]
    if state.get("schema_version") != 1:
        mismatches.append("schema_version must be 1")
    if not isinstance(state.get("run_id"), str) or not state["run_id"]:
        mismatches.append("run_id is missing")
    if mismatches:
        raise RuntimeError("cannot resume a different matrix candidate: " + "; ".join(mismatches))

    results_by_index: dict[int, dict[str, Any]] = {}
    checkpoint_results = state.get("results", [])
    if not isinstance(checkpoint_results, list):
        raise RuntimeError("checkpoint results must be a list")
    for result in checkpoint_results:
        index = result.get("index")
        if not isinstance(index, int) or not 0 <= index < len(commands):
            raise RuntimeError(f"invalid checkpoint result index: {index!r}")
        expected_item = commands[index]
        for field in ("family", "name", "command"):
            if result.get(field) != expected_item[field]:
                raise RuntimeError(f"checkpoint command identity mismatch at index {index}: {field}")
        if index in results_by_index:
            raise RuntimeError(f"duplicate checkpoint result index: {index}")
        if result.get("return_code") == 0:
            log_path = _resolve_reference(str(result.get("log_path", "")))
            if not log_path.is_file() or sha256_file(log_path) != result.get("log_sha256"):
                raise RuntimeError(f"successful checkpoint log is missing or changed: {log_path}")
        results_by_index[index] = result
    return results_by_index


def execute(
    commands: list[dict],
    matrix: dict,
    matrix_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    stop_on_failure: bool = False,
    command_timeout_seconds: float | None = None,
    invocation: list[str] | None = None,
) -> int:
    if _git("status", "--porcelain"):
        raise RuntimeError("Final matrix execution requires a clean working tree")
    if command_timeout_seconds is not None and command_timeout_seconds <= 0:
        raise ValueError("command timeout must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "run_state.json"
    raw_path = output_dir / "matrix_results.json"
    manifest_path = output_dir / "manifest.json"
    if not resume and any(output_dir.iterdir()):
        raise RuntimeError("matrix output directory is not empty; use --resume or a new directory")

    commit = _git("rev-parse", "HEAD")
    matrix_sha256 = sha256_file(matrix_path)
    command_plan_sha256 = _json_sha256(commands)
    if resume:
        if not state_path.is_file():
            raise RuntimeError("cannot resume without run_state.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        results_by_index = _validate_resume_state(
            state,
            commit=commit,
            matrix_sha256=matrix_sha256,
            command_plan_sha256=command_plan_sha256,
            commands=commands,
            command_timeout_seconds=command_timeout_seconds,
        )
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
    else:
        state = {
            "schema_version": 1,
            "run_id": str(uuid.uuid4()),
            "status": "running",
            "started_at_utc": _timestamp(),
            "updated_at_utc": _timestamp(),
            "git_commit": commit,
            "matrix_sha256": matrix_sha256,
            "command_plan_sha256": command_plan_sha256,
            "command_count": len(commands),
            "command_timeout_seconds": command_timeout_seconds,
            "resume_count": 0,
            "invocations": [],
            "results": [],
        }
        results_by_index = {}
        _atomic_write_json(state_path, state)

    state.setdefault("invocations", []).append(
        {
            "timestamp_utc": _timestamp(),
            "command": invocation
            or [sys.executable, "benchmarks/run_paper_matrix.py", "--execute"],
            "resume": resume,
            "stop_on_failure": stop_on_failure,
            "command_timeout_seconds": command_timeout_seconds,
        }
    )
    state["updated_at_utc"] = _timestamp()
    _atomic_write_json(state_path, state)

    skipped_success_count = 0
    try:
        for index, item in enumerate(commands):
            previous = results_by_index.get(index)
            if previous is not None and previous.get("return_code") == 0:
                skipped_success_count += 1
                continue

            log_path = output_dir / "logs" / f"{index:04d}-{item['family']}-{item['name']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            started_at = _timestamp()
            started = time.perf_counter()
            timed_out = False
            with log_path.open("w", encoding="utf-8") as output:
                try:
                    completed = subprocess.run(
                        item["command"],
                        cwd=REPO_ROOT,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env={
                            **os.environ,
                            "PYTHONPATH": os.pathsep.join(
                                (str(REPO_ROOT / "src"), str(REPO_ROOT))
                            ),
                        },
                        timeout=command_timeout_seconds,
                        check=False,
                    )
                    return_code = completed.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    return_code = 124
                    output.write(
                        f"\n[matrix-runner] command timed out after "
                        f"{command_timeout_seconds} seconds\n"
                    )
            result = {
                "index": index,
                **item,
                "started_at_utc": started_at,
                "finished_at_utc": _timestamp(),
                "duration_seconds": time.perf_counter() - started,
                "return_code": return_code,
                "timed_out": timed_out,
                "log_path": _path_reference(log_path),
                "log_sha256": sha256_file(log_path),
            }
            results_by_index[index] = result
            state["results"] = [results_by_index[key] for key in sorted(results_by_index)]
            state["updated_at_utc"] = _timestamp()
            state["status"] = "running"
            _atomic_write_json(state_path, state)
            if return_code != 0 and stop_on_failure:
                state["status"] = "stopped_on_failure"
                break
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["updated_at_utc"] = _timestamp()
        state["results"] = [results_by_index[key] for key in sorted(results_by_index)]
        _atomic_write_json(state_path, state)
        raise

    results = [results_by_index[key] for key in sorted(results_by_index)]
    completed_count = len(results)
    failed_count = sum(item["return_code"] != 0 for item in results)
    all_commands_completed = completed_count == len(commands)
    exit_status = 0 if all_commands_completed and failed_count == 0 else 1
    state["status"] = (
        "completed"
        if exit_status == 0
        else "completed_with_failures"
        if all_commands_completed
        else "stopped_on_failure"
    )
    state["updated_at_utc"] = _timestamp()
    state["results"] = results
    _atomic_write_json(state_path, state)
    raw_path = output_dir / "matrix_results.json"
    _atomic_write_json(raw_path, results)
    lock_path = REPO_ROOT / "paper" / "requirements-linux-py311.lock"
    manifest = {
        "schema_version": 2,
        "run_id": state["run_id"],
        "benchmark": "synaptoroute_frozen_paper_matrix",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "timestamp_utc": _timestamp(),
        "git_commit": commit,
        "working_tree_dirty": False,
        "command": invocation
        or [sys.executable, "benchmarks/run_paper_matrix.py", "--execute"],
        "exit_status": exit_status,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor() or "unknown",
            "gpu": "unknown",
            "machine_id": _machine_id(),
        },
        "dependency_lock": {
            "path": lock_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(lock_path),
        },
        "configuration": {
            "matrix_path": matrix_path.relative_to(REPO_ROOT).as_posix(),
            "matrix_sha256": matrix_sha256,
            "command_plan_sha256": command_plan_sha256,
            "resume_count": state["resume_count"],
            "command_timeout_seconds": command_timeout_seconds,
            "runner_invocations": state["invocations"],
            "matrix": matrix,
        },
        "dataset": "multiple frozen datasets and structural workloads",
        "metrics": {
            "command_count": len(commands),
            "completed_command_count": completed_count,
            "successful_command_count": sum(item["return_code"] == 0 for item in results),
            "failed_command_count": failed_count,
            "skipped_successful_command_count": skipped_success_count,
        },
        "evidence": {
            "script_path": "benchmarks/run_paper_matrix.py",
            "raw_output_path": _path_reference(raw_path),
            "raw_output_sha256": sha256_file(raw_path),
            "timing_unit": "varies by experiment; each output labels units",
            "notes": "Candidate matrix run; promotion requires independent reproduction.",
            "run_state_path": _path_reference(state_path),
            "run_state_sha256": sha256_file(state_path),
        },
        "missing_evidence": [
            "Independent reproduction and reviewer attestation are required.",
            "An immutable artifact archive is required.",
            *([] if all_commands_completed else ["The frozen command plan did not complete."]),
        ],
    }
    errors = validate_manifest(manifest, repo_root=REPO_ROOT)
    if errors:
        raise RuntimeError(f"Invalid matrix manifest: {errors}")
    _atomic_write_json(manifest_path, manifest)
    return exit_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "benchmark_results" / "paper-matrix")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("quality", "dynamic", "scale", "crash_recovery", "backpressure"),
        default=["quality", "dynamic", "scale", "crash_recovery", "backpressure"],
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--command-timeout-seconds", type=float)
    args = parser.parse_args()
    matrix_path = args.matrix.resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    commands = build_commands(matrix, output_dir, set(args.families))
    if not args.execute:
        print(json.dumps({"command_count": len(commands), "commands": commands}, indent=2))
        return 0
    return execute(
        commands,
        matrix,
        matrix_path,
        output_dir,
        resume=args.resume,
        stop_on_failure=args.stop_on_failure,
        command_timeout_seconds=args.command_timeout_seconds,
        invocation=[sys.executable, *sys.argv],
    )


if __name__ == "__main__":
    raise SystemExit(main())
