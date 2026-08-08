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


def execute(commands: list[dict], matrix: dict, matrix_path: Path, output_dir: Path) -> int:
    if _git("status", "--porcelain"):
        raise RuntimeError("Final matrix execution requires a clean working tree")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, item in enumerate(commands):
        log_path = output_dir / "logs" / f"{index:04d}-{item['family']}-{item['name']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                item["command"],
                cwd=REPO_ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONPATH": os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT)))},
                check=False,
            )
        results.append(
            {
                **item,
                "return_code": completed.returncode,
                "log_path": log_path.relative_to(REPO_ROOT).as_posix(),
                "log_sha256": sha256_file(log_path),
            }
        )

    raw_path = output_dir / "matrix_results.json"
    raw_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    exit_status = 0 if all(item["return_code"] == 0 for item in results) else 1
    lock_path = REPO_ROOT / "paper" / "requirements-linux-py311.lock"
    manifest = {
        "schema_version": 2,
        "run_id": str(uuid.uuid4()),
        "benchmark": "synaptoroute_frozen_paper_matrix",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git("rev-parse", "HEAD"),
        "working_tree_dirty": False,
        "command": [sys.executable, "benchmarks/run_paper_matrix.py", "--execute"],
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
            "matrix_sha256": sha256_file(matrix_path),
            "matrix": matrix,
        },
        "dataset": "multiple frozen datasets and structural workloads",
        "metrics": {"command_count": len(results), "failed_command_count": sum(item["return_code"] != 0 for item in results)},
        "evidence": {
            "script_path": "benchmarks/run_paper_matrix.py",
            "raw_output_path": raw_path.relative_to(REPO_ROOT).as_posix(),
            "raw_output_sha256": sha256_file(raw_path),
            "timing_unit": "varies by experiment; each output labels units",
            "notes": "Candidate matrix run; promotion requires independent reproduction.",
        },
        "missing_evidence": [
            "Independent reproduction and reviewer attestation are required.",
            "An immutable artifact archive is required.",
        ],
    }
    errors = validate_manifest(manifest, repo_root=REPO_ROOT)
    if errors:
        raise RuntimeError(f"Invalid matrix manifest: {errors}")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
    args = parser.parse_args()
    matrix_path = args.matrix.resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    commands = build_commands(matrix, output_dir, set(args.families))
    if not args.execute:
        print(json.dumps({"command_count": len(commands), "commands": commands}, indent=2))
        return 0
    return execute(commands, matrix, matrix_path, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
