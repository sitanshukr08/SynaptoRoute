"""Fault-inject abrupt exits after memory and durable mutation acknowledgements."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from synaptoroute import SQLiteStorage


def _run_trial(
    *,
    output_dir: Path,
    mode: str,
    trial: int,
    delay_ms: float,
    timeout_seconds: float,
    mutation: str,
    synchronous: str,
) -> dict[str, Any]:
    prefix = f"{mutation}-{synchronous.lower()}-{mode}-{trial}"
    database_path = output_dir / f"{prefix}.sqlite3"
    marker_path = output_dir / f"{prefix}.ack"
    command = [
        sys.executable,
        "-m",
        "benchmarks.durability_crash_worker",
        "--database",
        str(database_path),
        "--marker",
        str(marker_path),
        "--mode",
        mode,
        "--delay-ms",
        str(delay_ms),
        "--mutation",
        mutation,
        "--synchronous",
        synchronous,
    ]
    environment = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(source_root), str(repo_root), environment.get("PYTHONPATH", ""))
        if part
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_seconds,
        check=False,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    marker = marker_path.read_text(encoding="utf-8").strip() if marker_path.exists() else None
    storage = SQLiteStorage(str(database_path), synchronous=synchronous)
    routes, _ = storage.load_all_routes()
    routes_by_name = {route.name: route for route in routes}
    if mutation == "add_route":
        survived = "crash_route" in routes_by_name
    elif mutation == "add_utterance":
        survived = "target utterance" in routes_by_name["base_route"].utterances
    elif mutation == "update_threshold":
        survived = routes_by_name["base_route"].threshold == 0.9
    else:
        survived = "base_route" not in routes_by_name
    storage.close()
    return {
        "mode": mode,
        "mutation": mutation,
        "sqlite_synchronous": synchronous,
        "trial": trial,
        "return_code": completed.returncode,
        "marker": marker,
        "acknowledged": marker is not None,
        "survived_restart": survived,
        "wall_ms": wall_ms,
        "database_path": database_path.as_posix(),
    }


def run_benchmark(
    *,
    output_dir: Path,
    trials: int,
    delay_ms: float,
    timeout_seconds: float = 10.0,
    mutation: str = "add_route",
    synchronous: str = "FULL",
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be positive")
    if delay_ms <= 0:
        raise ValueError("delay_ms must be positive")
    if mutation not in {"add_route", "add_utterance", "update_threshold", "delete_route"}:
        raise ValueError("unsupported mutation")
    synchronous = synchronous.upper()
    if synchronous not in {"FULL", "NORMAL"}:
        raise ValueError("synchronous must be FULL or NORMAL")
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_results = [
        _run_trial(
            output_dir=output_dir,
            mode=mode,
            trial=trial,
            delay_ms=delay_ms,
            timeout_seconds=timeout_seconds,
            mutation=mutation,
            synchronous=synchronous,
        )
        for mode in ("memory", "durable")
        for trial in range(trials)
    ]
    by_mode = {
        mode: [result for result in trial_results if result["mode"] == mode]
        for mode in ("memory", "durable")
    }
    metrics = {}
    for mode, results in by_mode.items():
        metrics[mode] = {
            "trial_count": len(results),
            "acknowledged_count": sum(result["acknowledged"] for result in results),
            "restart_survival_rate": (
                sum(result["survived_restart"] for result in results) / len(results)
            ),
            "all_children_exited_cleanly": all(result["return_code"] == 0 for result in results),
        }

    return {
        "benchmark": "abrupt_process_restart_durability",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "workload": {
            "trials_per_mode": trials,
            "injected_storage_delay_ms": delay_ms,
            "modes": ["memory", "durable"],
            "mutation": mutation,
            "sqlite_synchronous": synchronous,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "metrics": metrics,
        "trials": trial_results,
        "notes": [
            "Each mutation occurs in a child process terminated with os._exit immediately after acknowledgement.",
            "The injected pre-commit delay makes the memory/durable boundary deterministic.",
            "This benchmark does not simulate kernel panic, filesystem corruption, or power loss.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--delay-ms", type=float, default=250.0)
    parser.add_argument(
        "--mutation",
        choices=("add_route", "add_utterance", "update_threshold", "delete_route"),
        default="add_route",
    )
    parser.add_argument("--synchronous", choices=("FULL", "NORMAL"), default="FULL")
    default_dir = Path(os.environ.get("SYNAPTOROUTE_RUN_DIR", "benchmark_results/crash-recovery"))
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    args = parser.parse_args()

    result = run_benchmark(
        output_dir=args.output_dir,
        trials=args.trials,
        delay_ms=args.delay_ms,
        mutation=args.mutation,
        synchronous=args.synchronous,
    )
    output_path = args.output_dir / "crash_recovery_summary.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
