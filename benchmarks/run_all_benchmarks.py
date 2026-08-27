import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import benchmarks.manifest_schema as manifest_schema
except ModuleNotFoundError:
    import manifest_schema


BENCHMARKS = {
    "local_smoke": ["benchmarks/bench_local_index.py"],
    "local_pilot": ["benchmarks/bench_local_pilot.py"],
    "durability_smoke": ["benchmarks/bench_durability.py", "--mutations", "100"],
    "crash_recovery_smoke": [
        "benchmarks/bench_crash_recovery.py",
        "--trials",
        "3",
        "--delay-ms",
        "100",
    ],
    "dynamic_workload_smoke": [
        "benchmarks/bench_dynamic_workload.py",
        "--duration",
        "2",
        "--routes",
        "25",
        "--query-workers",
        "2",
        "--mutation-rate",
        "10",
        "--engine",
        "numpy",
    ],
    "backpressure_smoke": [
        "benchmarks/bench_async_backpressure.py",
        "--concurrency",
        "1",
        "8",
        "32",
        "--queue-size",
        "8",
        "--max-in-flight-batches",
        "1",
        "--batch-size",
        "4",
        "--encoder-delay-ms",
        "20",
    ],
    "banking77_pilot": [
        "benchmarks/run_intent_experiment.py",
        "--dataset",
        "banking77",
        "--evaluation-limit",
        "500",
    ],
    "clinc150_pilot": [
        "benchmarks/run_intent_experiment.py",
        "--dataset",
        "clinc150",
        "--evaluation-limit",
        "500",
    ],
    "banking77_multiseed": [
        "benchmarks/run_multiseed_intent.py",
        "--dataset",
        "banking77",
    ],
    "clinc150_multiseed": [
        "benchmarks/run_multiseed_intent.py",
        "--dataset",
        "clinc150",
    ],
    "accuracy": ["benchmarks/eval_accuracy.py"],
    "latency": ["benchmarks/eval_latency.py"],
    "realworld": ["benchmarks/bench_realworld.py"],
}
MODEL_AWARE_BENCHMARKS = {
    "banking77_pilot",
    "clinc150_pilot",
    "banking77_multiseed",
    "clinc150_multiseed",
}


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def git_worktree_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def machine_id() -> str:
    value = f"{platform.node()}|{platform.platform()}|{platform.processor()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _command_for(command: list[str]) -> list[str]:
    return [sys.executable, *command]


def benchmark_command(name: str, model: str) -> list[str]:
    command = [*BENCHMARKS[name]]
    if name in MODEL_AWARE_BENCHMARKS:
        command.extend(["--model", model])
    return command


def write_manifest(
    output_dir: Path,
    selected: list[str],
    model: str,
    results: dict | None = None,
    exit_status: int | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "benchmark_manifest.json"
    run_id = str(uuid.uuid4())
    if manifest_path.exists():
        try:
            run_id = json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"]
        except (KeyError, json.JSONDecodeError):
            pass
    raw_outputs = {
        name: str((output_dir / f"{name}.log").as_posix())
        for name in selected
    }
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "benchmark": "synaptoroute_benchmark_run",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "git_commit": git_revision(),
        "working_tree_dirty": git_worktree_dirty(),
        "timestamp_utc": _utc_now(),
        "command": [sys.executable, "benchmarks/run_all_benchmarks.py", "--benchmarks", *selected, "--model", model, "--output-dir", str(output_dir)],
        "exit_status": exit_status,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor() or "unknown",
            "gpu": "unknown",
            "machine_id": machine_id(),
        },
        "dependency_lock": {
            "path": "paper/requirements-linux-py311.lock",
            "sha256": manifest_schema.sha256_file(
                "paper/requirements-linux-py311.lock"
            ),
        },
        "configuration": {
            "benchmarks": selected,
            "encoder": model,
            "adaptive_memory": False,
            "redis": False,
        },
        "dataset": "varies by selected benchmark",
        "encoder": model,
        "benchmarks": selected,
        "metrics": {},
        "evidence": {
            "script_path": "benchmarks/run_all_benchmarks.py",
            "raw_output_path": None,
            "timing_unit": "varies; benchmark scripts must label units explicitly",
            "notes": "Run-level manifest. Per-benchmark raw stdout/stderr logs are recorded under raw_outputs.",
        },
        "raw_outputs": raw_outputs,
        "results": results or {},
        "missing_evidence": [
            "Metrics are not verified until raw logs are reviewed and promoted into a claim manifest.",
        ],
        "note": "Results are raw command output. Do not publish metrics unless this run completed successfully.",
    }
    errors = manifest_schema.validate_manifest(manifest, repo_root=Path.cwd())
    if errors:
        raise RuntimeError(f"Invalid benchmark manifest: {errors}")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def run_command(command: list[str], output_path: Path) -> int:
    environment = os.environ.copy()
    repo_root = str(Path.cwd().resolve())
    source_root = str((Path(repo_root) / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, repo_root, environment.get("PYTHONPATH", "")) if part
    )
    environment["SYNAPTOROUTE_RUN_DIR"] = str(output_path.parent.resolve())
    with output_path.open("w", encoding="utf-8") as output:
        process = subprocess.run(
            [sys.executable, *command],
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=environment,
        )
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible SynaptoRoute benchmarks.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=sorted(BENCHMARKS),
        default=["accuracy", "latency"],
        help="Benchmark scripts to run.",
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--dry-run", action="store_true", help="Write and validate the run manifest without executing benchmarks.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_path = write_manifest(output_dir, args.benchmarks, args.model)
    print(f"Wrote manifest: {manifest_path}")
    if args.dry_run:
        print("Dry run complete; no benchmarks executed.")
        return 0

    failures = []
    results = {}
    for name in args.benchmarks:
        output_path = output_dir / f"{name}.log"
        print(f"Running {name}; output -> {output_path}")
        command = benchmark_command(name, args.model)
        return_code = run_command(command, output_path)
        results[name] = {
            "command": _command_for(command),
            "return_code": return_code,
            "raw_output_path": str(output_path.as_posix()),
            "completed_at_utc": _utc_now(),
            "raw_output_sha256": manifest_schema.sha256_file(output_path),
        }
        if return_code != 0:
            failures.append((name, return_code))

    write_manifest(
        output_dir,
        args.benchmarks,
        args.model,
        results=results,
        exit_status=1 if failures else 0,
    )

    if failures:
        for name, return_code in failures:
            print(f"FAILED: {name} exited with {return_code}")
        return 1

    print("All selected benchmarks completed. Inspect logs before publishing any metric.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
