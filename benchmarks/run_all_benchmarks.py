import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BENCHMARKS = {
    "accuracy": ["benchmarks/eval_accuracy.py"],
    "latency": ["benchmarks/eval_latency.py"],
    "realworld": ["benchmarks/bench_realworld.py"],
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


def write_manifest(output_dir: Path, selected: list[str], model: str) -> Path:
    manifest = {
        "git_commit": git_revision(),
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "model": model,
        "benchmarks": selected,
        "note": "Results are raw command output. Do not publish metrics unless this run completed successfully.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def run_command(command: list[str], output_path: Path) -> int:
    with output_path.open("w", encoding="utf-8") as output:
        process = subprocess.run(
            [sys.executable, *command],
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_path = write_manifest(output_dir, args.benchmarks, args.model)
    print(f"Wrote manifest: {manifest_path}")

    failures = []
    for name in args.benchmarks:
        output_path = output_dir / f"{name}.log"
        print(f"Running {name}; output -> {output_path}")
        return_code = run_command(BENCHMARKS[name], output_path)
        if return_code != 0:
            failures.append((name, return_code))

    if failures:
        for name, return_code in failures:
            print(f"FAILED: {name} exited with {return_code}")
        return 1

    print("All selected benchmarks completed. Inspect logs before publishing any metric.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
