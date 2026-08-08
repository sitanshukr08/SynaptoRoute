"""Run one bounded smoke cell per paper experiment family."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for source_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from benchmarks.bench_crash_recovery import run_benchmark as run_crash  # noqa: E402
from benchmarks.bench_dynamic_workload import run_benchmark as run_dynamic  # noqa: E402
from benchmarks.bench_scale_matrix import run_benchmark as run_scale  # noqa: E402
from benchmarks.bench_sustained_backpressure import (  # noqa: E402
    run_benchmark as run_backpressure,
)
from benchmarks.deterministic_encoder import DeterministicHashEncoder  # noqa: E402
from benchmarks.hf_intent_datasets import ExternalDatasetBundle, HFDatasetSpec  # noqa: E402
from benchmarks.manifest_schema import sha256_file, validate_manifest  # noqa: E402
from benchmarks.research_datasets import IntentExample, PreparedRoutingDataset  # noqa: E402
from benchmarks.run_intent_experiment import run_bundle_experiment  # noqa: E402


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


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _path_reference(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _quality_fixture() -> ExternalDatasetBundle:
    revision = "f" * 40
    labels = ("alpha", "beta", "gamma")
    prepared = PreparedRoutingDataset(
        name="protocol_smoke_quality",
        version=f"1@{revision}",
        license="MIT",
        seed=42,
        training_split="train",
        evaluation_split="test",
        route_examples={
            label: tuple(f"{label} route example {index}" for index in range(4))
            for label in labels
        },
        evaluation_examples=tuple(
            IntentExample(
                f"test-{label}",
                f"{label} evaluation request",
                label,
                "test",
            )
            for label in labels
        )
        + (
            IntentExample("test-ood-0", "unrelated evaluation zero", None, "test"),
            IntentExample("test-ood-1", "unrelated evaluation one", None, "test"),
        ),
        exact_text_overlap_count=0,
    )
    calibration_examples = tuple(
        IntentExample(
            f"validation-{label}-{index}",
            f"{label} validation request {index}",
            label,
            "validation",
        )
        for label in labels
        for index in range(4)
    ) + (
        IntentExample("validation-ood-0", "unrelated validation zero", None, "validation"),
        IntentExample("validation-ood-1", "unrelated validation one", None, "validation"),
    )
    spec = HFDatasetSpec(
        name="protocol_smoke_quality",
        dataset_id="synthetic/protocol-smoke",
        revision=revision,
        version="1",
        license="MIT",
        train_split="train",
        validation_split="validation",
        test_split="test",
        text_field="text",
        label_field="label",
    )
    return ExternalDatasetBundle(
        spec=spec,
        prepared=prepared,
        calibration_examples=calibration_examples,
        label_mapping={label: label for label in labels},
        split_fingerprints={
            "train": "synthetic-train-v1",
            "validation": "synthetic-validation-v1",
            "test": "synthetic-test-v1",
        },
        evaluation_limit=None,
    )


def evaluate_invariants(
    *,
    quality: dict[str, Any],
    dynamic: dict[str, Any],
    scale: dict[str, Any],
    crash: dict[str, Any],
    backpressure: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    required_quality_metrics = {
        "expected_calibration_error",
        "max_calibration_error",
        "brier_score",
    }
    for system_name, system in quality["systems"].items():
        if not required_quality_metrics.issubset(system["test"]):
            failures.append(f"quality:{system_name}:missing calibration metrics")
        for field in ("artifact_path", "source_predictions_path"):
            path = Path(system["probability_calibration"][field])
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"quality:{system_name}:missing {field}")
        for field in ("data_path", "diagram_path"):
            path = Path(system["reliability"][field])
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"quality:{system_name}:missing reliability {field}")
    if dynamic["metrics"]["correctness_violations"] != 0:
        failures.append("dynamic:correctness violations")
    if dynamic["metrics"]["restart_state_equal"] is not True:
        failures.append("dynamic:restart state diverged")
    if scale["metrics"]["top1_identity_accuracy"] != 1.0:
        failures.append("scale:identity retrieval failed")
    if crash["metrics"]["durable"]["restart_survival_rate"] != 1.0:
        failures.append("crash:durable acknowledgement did not survive")
    if crash["metrics"]["memory"]["restart_survival_rate"] != 0.0:
        failures.append("crash:memory acknowledgement unexpectedly survived delay injection")
    for scenario in backpressure["scenarios"]:
        if scenario["error_count"] != 0:
            failures.append(f"backpressure:{scenario['load_fraction']}:request errors")
        if scenario["successful_count"] and scenario["successful_accuracy"] != 1.0:
            failures.append(f"backpressure:{scenario['load_fraction']}:incorrect success")
    return failures


def run_protocol_smoke(output_root: Path) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    run_dir = output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    quality = run_bundle_experiment(
        bundle=_quality_fixture(),
        encoder=DeterministicHashEncoder(dim=32),
        output_dir=run_dir / "quality",
        min_known_coverage=0.5,
        max_ood_false_acceptance_rate=None,
    )
    dynamic = run_dynamic(
        database_path=run_dir / "dynamic" / "state.sqlite3",
        duration_seconds=0.5,
        route_count=10,
        query_workers=2,
        mutation_rate=5.0,
        dim=32,
        warmup_seconds=0.05,
    )
    scale = run_scale(engine="numpy", route_count=100, query_count=200, dim=32, seed=42)
    crash = run_crash(
        output_dir=run_dir / "crash_recovery",
        trials=1,
        delay_ms=10.0,
        mutation="add_route",
        synchronous="FULL",
    )
    backpressure = asyncio.run(
        run_backpressure(
            load_fractions=[0.5, 1.5],
            duration_seconds=0.25,
            queue_size=8,
            batch_size=2,
            max_in_flight_batches=2,
            encoder_delay_ms=10.0,
            calibration_seconds=0.25,
        )
    )

    family_paths = {
        "quality": run_dir / "quality" / "experiment_summary.json",
        "dynamic": _write_json(run_dir / "dynamic" / "summary.json", dynamic),
        "scale": _write_json(run_dir / "scale" / "summary.json", scale),
        "crash_recovery": _write_json(
            run_dir / "crash_recovery" / "summary.json",
            crash,
        ),
        "backpressure": _write_json(
            run_dir / "backpressure" / "summary.json",
            backpressure,
        ),
    }
    failures = evaluate_invariants(
        quality=quality,
        dynamic=dynamic,
        scale=scale,
        crash=crash,
        backpressure=backpressure,
    )
    raw_payload = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "unverified",
        "paper_evidence_eligible": False,
        "family_artifacts": {
            name: {
                "path": _path_reference(path),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(family_paths.items())
        },
        "invariants": {
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
        },
    }
    raw_path = _write_json(run_dir / "protocol_smoke.json", raw_payload)
    lock_path = REPO_ROOT / "paper" / "requirements-linux-py311.lock"
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "benchmark": "five_family_protocol_smoke",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git("status", "--porcelain")),
        "command": [
            sys.executable,
            "benchmarks/run_protocol_smoke.py",
            "--output-dir",
            str(output_root),
        ],
        "exit_status": 0 if not failures else 1,
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
            "families": sorted(family_paths),
            "bounded_smoke": True,
            "encoder": "deterministic-hash-32",
            "adaptive_memory": False,
            "redis": False,
        },
        "dataset": {
            "name": "deterministic_synthetic_protocol_smoke",
            "version": "1",
            "revision": "repository",
            "split": "synthetic_disjoint_train_validation_test",
            "seed": 42,
            "route_count": 10,
            "query_count": 200,
            "license": "MIT",
        },
        "metrics": {
            "family_count": len(family_paths),
            "invariant_failure_count": len(failures),
            "all_invariants_passed": not failures,
        },
        "evidence": {
            "script_path": "benchmarks/run_protocol_smoke.py",
            "raw_output_path": _path_reference(raw_path),
            "raw_output_sha256": sha256_file(raw_path),
            "timing_unit": "mixed; each family artifact declares its units",
            "notes": "Bounded structural smoke only; no metric is publication evidence.",
        },
        "missing_evidence": [
            "External quality datasets and the frozen model are not used by this smoke.",
            "The full frozen matrix has not been executed.",
            "Independent reproduction and immutable archival are required.",
        ],
    }
    errors = validate_manifest(manifest, repo_root=REPO_ROOT)
    if errors:
        raise RuntimeError(f"invalid protocol-smoke manifest: {errors}")
    _write_json(run_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmark_results" / "protocol-smoke",
    )
    args = parser.parse_args()
    manifest = run_protocol_smoke(args.output_dir)
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "status": manifest["status"],
                "exit_status": manifest["exit_status"],
            }
        )
    )
    return int(manifest["exit_status"])


if __name__ == "__main__":
    raise SystemExit(main())
