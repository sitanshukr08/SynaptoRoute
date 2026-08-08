"""Run the deterministic structural CI smoke and emit unverified evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.deterministic_encoder import DeterministicHashEncoder  # noqa: E402
from benchmarks.manifest_schema import sha256_file, validate_manifest  # noqa: E402
from synaptoroute import AdaptiveRouter, Route  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _machine_id() -> str:
    value = f"{platform.node()}|{platform.platform()}|{platform.processor()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def run_ci_smoke(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_dir / "ci_smoke.json"
    manifest_path = output_dir / "manifest.json"
    lock_path = REPO_ROOT / "paper" / "requirements-linux-py311.lock"

    route_count = 20
    query_count = 100
    router = AdaptiveRouter(DeterministicHashEncoder(dim=64))
    for route_index in range(route_count):
        router.add_route(
            Route(
                name=f"ci_intent_{route_index}",
                utterances=[f"deterministic intent {route_index} example {item}" for item in range(5)],
                threshold=0.7,
            )
        )
    router.durable_barrier(timeout=10.0)

    latencies_ms: list[float] = []
    matched = 0
    started = time.perf_counter()
    for query_index in range(query_count):
        target = query_index % route_count
        query = f"deterministic intent {target} example 0"
        query_started = time.perf_counter_ns()
        result = router.match(query)
        latencies_ms.append((time.perf_counter_ns() - query_started) / 1_000_000.0)
        matched += int(result.route_name == f"ci_intent_{target}")
    duration_seconds = time.perf_counter() - started
    router.close()

    metrics = {
        "top1_identity_accuracy": matched / query_count,
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "duration_seconds": duration_seconds,
    }
    raw_payload = {
        "benchmark": "ci_structural_smoke",
        "semantic_quality_eligible": False,
        "metrics": metrics,
    }
    raw_output_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    try:
        raw_output_reference = raw_output_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        raw_output_reference = raw_output_path.as_posix()

    command = [sys.executable, "benchmarks/run_ci_smoke_benchmark.py", "--output-dir", str(output_dir)]
    manifest = {
        "schema_version": 2,
        "run_id": str(uuid.uuid4()),
        "benchmark": "ci_structural_smoke",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "command": command,
        "exit_status": 0,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor() or "unknown",
            "gpu": "none",
            "machine_id": _machine_id(),
        },
        "dependency_lock": {
            "path": lock_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(lock_path),
        },
        "configuration": {
            "encoder": "deterministic-hash-64",
            "adaptive_memory": False,
            "redis": False,
        },
        "dataset": {
            "name": "deterministic_synthetic_ci",
            "version": "2.0",
            "revision": "repository",
            "split": "structural_smoke",
            "seed": 42,
            "route_count": route_count,
            "query_count": query_count,
            "license": "MIT",
        },
        "metrics": metrics,
        "evidence": {
            "script_path": "benchmarks/run_ci_smoke_benchmark.py",
            "raw_output_path": raw_output_reference,
            "raw_output_sha256": sha256_file(raw_output_path),
            "timing_unit": "milliseconds",
            "notes": "Offline structural smoke; timings and identity accuracy are not paper evidence.",
        },
        "missing_evidence": [
            "Independent reproduction and reviewer attestation are required.",
            "An immutable artifact archive is required.",
            "The deterministic encoder does not measure semantic quality.",
        ],
    }
    errors = validate_manifest(manifest, repo_root=REPO_ROOT)
    if errors:
        raise RuntimeError(f"Invalid CI smoke manifest: {errors}")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmark_results" / "ci_smoke",
    )
    args = parser.parse_args()
    manifest = run_ci_smoke(args.output_dir)
    print(json.dumps({"run_id": manifest["run_id"], "status": manifest["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
