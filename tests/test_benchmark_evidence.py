import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.manifest_schema import sha256_file, validate_manifest, validate_manifest_file
from benchmarks.promote_evidence import promote
from benchmarks.run_all_benchmarks import BENCHMARKS, benchmark_command
from paper.generate_tables import generate


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_RUNTIME = REPO_ROOT / ".test-runtime"


def make_runtime_dir(name: str) -> Path:
    path = TEST_RUNTIME / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_historical_benchmark_manifests_are_schema_valid():
    manifest_dir = REPO_ROOT / "benchmarks" / "manifests"
    for manifest_path in manifest_dir.glob("*.json"):
        errors = validate_manifest_file(manifest_path, repo_root=REPO_ROOT)
        assert errors == [], f"{manifest_path.name}: {errors}"


def test_verified_manifest_requires_runnable_script_and_raw_output():
    temp_path = make_runtime_dir("manifest-missing-evidence")
    raw_output_path = temp_path / "missing.log"

    manifest = {
        "schema_version": 1,
        "benchmark": "example",
        "status": "verified",
        "timestamp_utc": "2026-06-04T00:00:00Z",
        "git_commit": "a" * 40,
        "command": ["python", "missing.py"],
        "environment": {
            "python_version": "3.12",
            "platform": "test",
            "cpu": "test",
            "gpu": "none",
        },
        "dataset": "test",
        "metrics": {"accuracy": 1.0},
        "evidence": {
            "script_path": "missing.py",
            "raw_output_path": str(raw_output_path),
            "timing_unit": "not applicable",
            "notes": "test",
        },
    }

    errors = validate_manifest(manifest, repo_root=temp_path)

    assert any("script_path" in error for error in errors)
    assert any("raw_output_path" in error for error in errors)


def test_verified_manifest_requires_clean_source_dataset_metadata_and_log_hash():
    temp_path = make_runtime_dir("manifest-verification-gates")
    script_path = temp_path / "benchmark.py"
    raw_output_path = temp_path / "benchmark.log"
    script_path.write_text("print('benchmark')\n", encoding="utf-8")
    raw_output_path.write_text("result=1\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "benchmark": "example",
        "status": "verified",
        "timestamp_utc": "2026-07-13T00:00:00Z",
        "git_commit": "a" * 40,
        "working_tree_dirty": False,
        "command": ["python", str(script_path)],
        "environment": {
            "python_version": "3.12",
            "platform": "test",
            "cpu": "test",
            "gpu": "none",
        },
        "dataset": {
            "name": "fixture",
            "version": "1",
            "split": "test",
            "seed": 42,
            "route_count": 2,
            "query_count": 2,
            "license": "test-only",
        },
        "metrics": {"accuracy": 1.0},
        "evidence": {
            "script_path": str(script_path),
            "raw_output_path": str(raw_output_path),
            "raw_output_sha256": sha256_file(raw_output_path),
            "timing_unit": "not applicable",
            "notes": "test",
        },
    }

    assert validate_manifest(manifest, repo_root=temp_path) == []

    manifest["working_tree_dirty"] = True
    manifest["evidence"]["raw_output_sha256"] = "invalid"
    errors = validate_manifest(manifest, repo_root=temp_path)

    assert any("working_tree_dirty=false" in error for error in errors)
    assert any("raw_output_sha256" in error for error in errors)


def test_verified_manifest_rejects_placeholder_commit():
    manifest = {
        "schema_version": 1,
        "benchmark": "placeholder",
        "status": "verified",
        "timestamp_utc": "2026-08-04T00:00:00Z",
        "git_commit": "ci_commit_build",
        "working_tree_dirty": False,
        "command": ["python", "benchmark.py"],
        "environment": {
            "python_version": "3.11",
            "platform": "test",
            "cpu": "test",
            "gpu": "none",
        },
        "dataset": {},
        "metrics": {},
        "evidence": {
            "script_path": "benchmark.py",
            "raw_output_path": "benchmark.log",
            "timing_unit": "milliseconds",
            "notes": "test",
        },
    }

    errors = validate_manifest(manifest)

    assert any("40-character" in error for error in errors)


def test_promotion_requires_independent_machine_and_review(tmp_path):
    script = tmp_path / "benchmark.py"
    raw = tmp_path / "raw.json"
    lock = tmp_path / "constraints.txt"
    script.write_text("print('benchmark')\n", encoding="utf-8")
    raw.write_text('{"value": 1}\n', encoding="utf-8")
    lock.write_text("example==1.0\n", encoding="utf-8")

    def candidate(run_id, machine_id):
        return {
            "schema_version": 2,
            "run_id": run_id,
            "benchmark": "example",
            "status": "unverified",
            "paper_evidence_eligible": False,
            "timestamp_utc": "2026-08-04T00:00:00Z",
            "git_commit": "b" * 40,
            "working_tree_dirty": False,
            "command": ["python", str(script)],
            "exit_status": 0,
            "environment": {
                "python_version": "3.11",
                "platform": "test",
                "cpu": "test",
                "gpu": "none",
                "machine_id": machine_id,
            },
            "dependency_lock": {"path": str(lock), "sha256": sha256_file(lock)},
            "configuration": {"seed": 42},
            "dataset": {
                "name": "fixture",
                "version": "1",
                "split": "test",
                "seed": 42,
                "route_count": 2,
                "query_count": 2,
                "license": "test-only",
            },
            "metrics": {"accuracy": 1.0},
            "evidence": {
                "script_path": str(script),
                "raw_output_path": str(raw),
                "raw_output_sha256": sha256_file(raw),
                "timing_unit": "milliseconds",
                "notes": "test",
            },
            "missing_evidence": ["independent review"],
        }

    original = candidate("original", "machine-a")
    reproduction = candidate("reproduction", "machine-b")
    promoted = promote(
        original,
        reproduction,
        reviewer="reviewer@example.com",
        claim="The invariant reproduced.",
        archive_uri="https://doi.org/10.0000/example",
        archive_sha256="c" * 64,
    )

    assert promoted["status"] == "verified"
    assert promoted["review"]["reproduction_run_id"] == "reproduction"
    assert validate_manifest(promoted, repo_root=tmp_path) == []

    reproduction["environment"]["machine_id"] = "machine-a"
    with pytest.raises(ValueError, match="different machine_id"):
        promote(
            original,
            reproduction,
            reviewer="reviewer@example.com",
            claim="claim",
            archive_uri="https://doi.org/10.0000/example",
            archive_sha256="c" * 64,
            repo_root=tmp_path,
        )

    reproduction = candidate("reproduction-2", "machine-b")
    reproduction["evidence"]["raw_output_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw output hash"):
        promote(
            original,
            reproduction,
            reviewer="reviewer@example.com",
            claim="claim",
            archive_uri="https://doi.org/10.0000/example",
            archive_sha256="c" * 64,
            repo_root=tmp_path,
        )


def test_verified_schema_v2_rechecks_lock_archive_and_reproduction(tmp_path):
    script = tmp_path / "benchmark.py"
    raw = tmp_path / "raw.json"
    lock = tmp_path / "constraints.txt"
    script.write_text("print('benchmark')\n", encoding="utf-8")
    raw.write_text('{"value": 1}\n', encoding="utf-8")
    lock.write_text("example==1.0\n", encoding="utf-8")
    base = {
        "schema_version": 2,
        "run_id": "original",
        "benchmark": "example",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "timestamp_utc": "2026-08-04T00:00:00Z",
        "git_commit": "b" * 40,
        "working_tree_dirty": False,
        "command": ["python", str(script)],
        "exit_status": 0,
        "environment": {"python_version": "3.11", "platform": "test", "cpu": "test", "gpu": "none", "machine_id": "machine-a"},
        "dependency_lock": {"path": str(lock), "sha256": sha256_file(lock)},
        "configuration": {"seed": 42},
        "dataset": {"name": "fixture", "version": "1", "split": "test", "seed": 42, "route_count": 2, "query_count": 2, "license": "test-only"},
        "metrics": {"accuracy": 1.0},
        "evidence": {"script_path": str(script), "raw_output_path": str(raw), "raw_output_sha256": sha256_file(raw), "timing_unit": "milliseconds", "notes": "test"},
        "missing_evidence": ["independent review"],
    }
    reproduction = json.loads(json.dumps(base))
    reproduction["run_id"] = "reproduction"
    reproduction["environment"]["machine_id"] = "machine-b"
    promoted = promote(base, reproduction, reviewer="reviewer", claim="claim", archive_uri="https://doi.org/example", archive_sha256="c" * 64)

    assert validate_manifest(promoted, repo_root=tmp_path) == []
    promoted["dependency_lock"]["sha256"] = "0" * 64
    promoted["archive"]["sha256"] = "invalid"
    errors = validate_manifest(promoted, repo_root=tmp_path)
    assert any("dependency_lock.sha256" in error for error in errors)
    assert any("archive.sha256" in error for error in errors)


def test_paper_table_generator_refuses_unverified_manifest(tmp_path):
    manifest_path = tmp_path / "candidate.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "candidate",
                "benchmark": "candidate",
                "status": "unverified",
                "paper_evidence_eligible": False,
                "timestamp_utc": "2026-08-04T00:00:00Z",
                "git_commit": "unknown",
                "working_tree_dirty": True,
                "command": ["python", "candidate.py"],
                "exit_status": 0,
                "environment": {"python_version": "3.11", "platform": "test", "cpu": "test", "gpu": "none", "machine_id": "machine-a"},
                "dependency_lock": {"path": "paper/constraints.txt", "sha256": "unknown"},
                "configuration": {},
                "dataset": "candidate",
                "metrics": {},
                "evidence": {"script_path": "benchmarks/run_all_benchmarks.py", "raw_output_path": None, "timing_unit": "none", "notes": "candidate"},
                "missing_evidence": ["independent review"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="verified evidence"):
        generate([manifest_path])


def test_benchmark_registry_scripts_exist_and_are_non_empty():
    for commands in BENCHMARKS.values():
        assert commands
        script_path = REPO_ROOT / commands[0]
        assert script_path.exists(), commands[0]
        assert script_path.stat().st_size > 0, commands[0]


def test_paper_dependency_lock_is_complete_and_exactly_pinned():
    lock_path = REPO_ROOT / "paper" / "requirements-linux-py311.lock"
    requirements = [
        line.strip()
        for line in lock_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert requirements
    assert all(line.count("==") == 1 for line in requirements)
    names = {line.split("==", 1)[0].lower() for line in requirements}
    assert {
        "datasets",
        "faiss-cpu",
        "fastembed",
        "mypy",
        "numpy",
        "pytest",
        "ruff",
        "scikit-learn",
        "semantic-router",
    } <= names


def test_external_benchmark_command_forwards_model():
    command = benchmark_command("banking77_pilot", "owner/model-revision")

    assert command[-2:] == ["--model", "owner/model-revision"]


def test_benchmark_runner_dry_run_writes_schema_valid_manifest():
    output_dir = make_runtime_dir("benchmark-dry-run") / "benchmark-output"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/run_all_benchmarks.py",
            "--benchmarks",
            "latency",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    manifest_path = output_dir / "benchmark_manifest.json"
    errors = validate_manifest_file(manifest_path, repo_root=REPO_ROOT)
    assert errors == []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "unverified"
    assert manifest["schema_version"] == 2
    assert manifest["paper_evidence_eligible"] is False
    assert isinstance(manifest["working_tree_dirty"], bool)
    assert manifest["raw_outputs"]["latency"].endswith("latency.log")


def test_benchmark_runner_executes_local_smoke_and_records_raw_output():
    output_dir = make_runtime_dir("benchmark-local-smoke") / "benchmark-output"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/run_all_benchmarks.py",
            "--benchmarks",
            "local_smoke",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    raw_output = output_dir / "local_smoke.log"
    payload = json.loads(raw_output.read_text(encoding="utf-8"))
    assert payload["status"] == "structural_only"
    assert payload["dataset"]["semantic_quality_eligible"] is False
    assert payload["metrics"]["top1_identity_accuracy"] == 1.0

    manifest = json.loads((output_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["results"]["local_smoke"]["return_code"] == 0


def test_large_scale_latency_claim_is_retracted():
    manifest_path = REPO_ROOT / "benchmarks" / "manifests" / "large_scale_retrieval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "retracted"
    assert manifest["metrics"]["corrected_interpretation"]["1000000_p95_ms"] == pytest.approx(3.1)
