import json
import sys
from pathlib import Path

import pytest

from benchmarks.manifest_schema import sha256_file
from paper.verify_matrix_run import (
    MatrixRunVerificationError,
    _verify_family_invariants,
    verify_matrix_run,
)


COMMIT = "a" * 40


def _json_sha256(value):
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _make_crash_run(tmp_path: Path, *, durable_survived: bool = True) -> tuple[Path, Path]:
    source = tmp_path / "source"
    run_dir = tmp_path / "artifact" / "matrix"
    (source / "paper").mkdir(parents=True)
    (source / "benchmarks").mkdir()
    (source / "benchmarks" / "run_paper_matrix.py").write_text("# fixture\n", encoding="utf-8")
    lock_path = source / "paper" / "requirements-linux-py311.lock"
    lock_path.write_text("fixture==1.0\n", encoding="utf-8")
    matrix = {
        "crash_recovery": {
            "mutations": ["add_route"],
            "acknowledgement_modes": ["memory", "durable"],
            "sqlite_synchronous": ["FULL"],
            "injected_commit_delay_ms": [10],
            "trials_per_cell": 1,
        }
    }
    matrix_path = source / "paper" / "experiment_matrix.json"
    _write_json(matrix_path, matrix)

    name = "add_route-full-10ms"
    cell_dir = run_dir / "crash_recovery" / name
    trials = []
    for mode, survived, state in (
        ("memory", False, "queued"),
        ("durable", durable_survived, "durable"),
    ):
        prefix = f"add_route-full-{mode}-0"
        marker = f"{mode}:add_route:{state}:1"
        (cell_dir / f"{prefix}.sqlite3").parent.mkdir(parents=True, exist_ok=True)
        (cell_dir / f"{prefix}.sqlite3").write_bytes(b"sqlite")
        (cell_dir / f"{prefix}.ack").write_text(marker + "\n", encoding="utf-8")
        trials.append(
            {
                "mode": mode,
                "mutation": "add_route",
                "sqlite_synchronous": "FULL",
                "trial": 0,
                "return_code": 0,
                "marker": marker,
                "acknowledged": True,
                "survived_restart": survived,
                "wall_ms": 1.0,
                "database_path": f"/original/matrix/crash_recovery/{name}/{prefix}.sqlite3",
            }
        )
    summary = {
        "benchmark": "abrupt_process_restart_durability",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "workload": {
            "trials_per_mode": 1,
            "injected_storage_delay_ms": 10.0,
            "modes": ["memory", "durable"],
            "mutation": "add_route",
            "sqlite_synchronous": "FULL",
        },
        "metrics": {
            "memory": {
                "trial_count": 1,
                "acknowledged_count": 1,
                "restart_survival_rate": 0.0,
                "all_children_exited_cleanly": True,
            },
            "durable": {
                "trial_count": 1,
                "acknowledged_count": 1,
                "restart_survival_rate": 1.0 if durable_survived else 0.0,
                "all_children_exited_cleanly": True,
            },
        },
        "trials": trials,
    }
    summary_path = cell_dir / "crash_recovery_summary.json"
    _write_json(summary_path, summary)
    log_path = run_dir / "logs" / f"0000-crash_recovery-{name}.log"
    _write_json(log_path, summary)
    command = [
        sys.executable,
        "benchmarks/bench_crash_recovery.py",
        "--mutation",
        "add_route",
        "--synchronous",
        "FULL",
        "--delay-ms",
        "10",
        "--trials",
        "1",
        "--output-dir",
        f"/original/matrix/crash_recovery/{name}",
    ]
    result = {
        "index": 0,
        "family": "crash_recovery",
        "name": name,
        "command": command,
        "started_at_utc": "2026-08-08T00:00:00Z",
        "finished_at_utc": "2026-08-08T00:00:01Z",
        "duration_seconds": 1.0,
        "return_code": 0,
        "timed_out": False,
        "log_path": f"benchmark_results/pilot/matrix/logs/{log_path.name}",
        "log_sha256": sha256_file(log_path),
    }
    plan_hash = _json_sha256([{key: result[key] for key in ("family", "name", "command")}])
    invocation = ["python", "benchmarks/run_paper_matrix.py", "--execute"]
    state = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "status": "completed",
        "started_at_utc": "2026-08-08T00:00:00Z",
        "updated_at_utc": "2026-08-08T00:00:01Z",
        "git_commit": COMMIT,
        "matrix_sha256": sha256_file(matrix_path),
        "command_plan_sha256": plan_hash,
        "command_count": 1,
        "command_timeout_seconds": 1800.0,
        "resume_count": 0,
        "invocations": [
            {
                "timestamp_utc": "2026-08-08T00:00:00Z",
                "command": invocation,
                "resume": False,
                "stop_on_failure": False,
                "command_timeout_seconds": 1800.0,
            }
        ],
        "results": [result],
    }
    state_path = run_dir / "run_state.json"
    raw_path = run_dir / "matrix_results.json"
    _write_json(state_path, state)
    _write_json(raw_path, [result])
    manifest = {
        "schema_version": 2,
        "run_id": "fixture-run",
        "benchmark": "synaptoroute_frozen_paper_matrix",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "timestamp_utc": "2026-08-08T00:00:01Z",
        "git_commit": COMMIT,
        "working_tree_dirty": False,
        "command": invocation,
        "exit_status": 0,
        "environment": {
            "python_version": "3.11.9",
            "platform": "fixture-linux",
            "cpu": "fixture-cpu",
            "gpu": "none",
            "machine_id": "fixture-machine",
        },
        "dependency_lock": {
            "path": "paper/requirements-linux-py311.lock",
            "sha256": sha256_file(lock_path),
        },
        "configuration": {
            "matrix": matrix,
            "matrix_sha256": sha256_file(matrix_path),
            "command_plan_sha256": plan_hash,
            "resume_count": 0,
            "command_timeout_seconds": 1800.0,
            "runner_invocations": state["invocations"],
        },
        "dataset": "structural fixture",
        "metrics": {
            "command_count": 1,
            "completed_command_count": 1,
            "successful_command_count": 1,
            "failed_command_count": 0,
            "skipped_successful_command_count": 0,
        },
        "evidence": {
            "script_path": "benchmarks/run_paper_matrix.py",
            "raw_output_path": "benchmark_results/pilot/matrix/matrix_results.json",
            "raw_output_sha256": sha256_file(raw_path),
            "timing_unit": "varies by experiment",
            "notes": "fixture",
            "run_state_path": "benchmark_results/pilot/matrix/run_state.json",
            "run_state_sha256": sha256_file(state_path),
        },
        "missing_evidence": ["independent reproduction"],
    }
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir, source


def test_verifier_accepts_complete_unverified_crash_run(tmp_path):
    run_dir, source = _make_crash_run(tmp_path)

    report = verify_matrix_run(
        run_dir,
        source_root=source,
        expected_commit=COMMIT,
        expected_family="crash_recovery",
    )

    assert report["verification_status"] == "valid_unverified_matrix_run"
    assert report["paper_evidence_eligible"] is False
    assert report["command_count"] == 1
    assert report["invariants"]["crash_trial_record_count"] == 2


def test_verifier_rejects_a_changed_hashed_log(tmp_path):
    run_dir, source = _make_crash_run(tmp_path)
    log = next((run_dir / "logs").glob("*.log"))
    log.write_text(log.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(MatrixRunVerificationError, match="command log SHA-256 mismatch"):
        verify_matrix_run(run_dir, source_root=source)


def test_verifier_rejects_an_internally_consistent_durability_violation(tmp_path):
    run_dir, source = _make_crash_run(tmp_path, durable_survived=False)

    with pytest.raises(MatrixRunVerificationError, match="restart survival violated"):
        verify_matrix_run(run_dir, source_root=source)


def test_verifier_requires_requested_environment_evidence(tmp_path):
    run_dir, source = _make_crash_run(tmp_path)

    with pytest.raises(MatrixRunVerificationError, match="environment evidence directory is required"):
        verify_matrix_run(run_dir, source_root=source, require_environment=True)


@pytest.mark.parametrize(
    ("family", "name", "matrix", "summary", "expected_error"),
    [
        (
            "dynamic",
            "r100-w1-m0-rep0",
            {"dynamic": {}},
            {
                "status": "unverified",
                "paper_evidence_eligible": False,
                "metrics": {
                    "correctness_violations": 1,
                    "pre_restart_state_equal": True,
                    "restart_state_equal": True,
                },
            },
            "dynamic:r100-w1-m0-rep0: correctness violations",
        ),
        (
            "scale",
            "numpy-r1000-rep0",
            {"scale": {}},
            {
                "status": "unverified",
                "paper_evidence_eligible": False,
                "metrics": {"top1_identity_accuracy": 0.99},
            },
            "scale:numpy-r1000-rep0: identity retrieval failed",
        ),
        (
            "backpressure",
            "balanced-rep0",
            {"backpressure": {"offered_load_fraction": [1.0]}},
            {
                "status": "unverified",
                "paper_evidence_eligible": False,
                "scenarios": [
                    {
                        "load_fraction": 1.0,
                        "offered_count": 3,
                        "successful_count": 1,
                        "overloaded_count": 1,
                        "error_count": 0,
                        "successful_accuracy": 1.0,
                    }
                ],
            },
            "backpressure:balanced-rep0: offered-load denominator mismatch",
        ),
    ],
)
def test_family_handlers_reject_structural_invariant_failures(
    tmp_path,
    family,
    name,
    matrix,
    summary,
    expected_error,
):
    run_dir = tmp_path / "matrix"
    result = {"index": 0, "family": family, "name": name}
    if family == "dynamic":
        summary_path = run_dir / family / name / "dynamic_workload_summary.json"
    else:
        summary_path = run_dir / family / f"{name}.json"
    log_path = run_dir / "logs" / f"0000-{family}-{name}.log"
    _write_json(summary_path, summary)
    _write_json(log_path, summary)
    errors = []

    _verify_family_invariants(run_dir, [result], matrix, {0: log_path}, errors)

    assert expected_error in errors


def test_quality_handler_checks_seed_artifact_hashes(tmp_path):
    run_dir = tmp_path / "matrix"
    name = "banking77"
    seed_summary = run_dir / "quality" / name / "seed-13" / "experiment_summary.json"
    _write_json(seed_summary, {"systems": {"fixture": {}}})
    study = {
        "status": "unverified",
        "paper_evidence_eligible": False,
        "per_seed": [
            {
                "seed": 13,
                "summary_path": "/original/matrix/quality/banking77/seed-13/experiment_summary.json",
                "summary_sha256": "0" * 64,
            }
        ],
    }
    study_path = run_dir / "quality" / name / "multiseed_summary.json"
    log_path = run_dir / "logs" / "0000-quality-banking77.log"
    _write_json(study_path, study)
    _write_json(log_path, {"study": study, "analysis": {}})
    errors = []

    _verify_family_invariants(
        run_dir,
        [{"index": 0, "family": "quality", "name": name}],
        {"quality": {"seeds": [13]}},
        {0: log_path},
        errors,
    )

    assert "quality:banking77: per-seed summary hash mismatch" in errors


def test_quality_handler_runs_deep_seed_verification(tmp_path, monkeypatch):
    run_dir = tmp_path / "matrix"
    name = "banking77"
    seed_summary = run_dir / "quality" / name / "seed-13" / "experiment_summary.json"
    _write_json(seed_summary, {"systems": {"fixture": {}}})
    study = {
        "status": "unverified",
        "paper_evidence_eligible": False,
        "per_seed": [
            {
                "seed": 13,
                "summary_path": "/original/matrix/quality/banking77/seed-13/experiment_summary.json",
                "summary_sha256": sha256_file(seed_summary),
            }
        ],
    }
    log_path = run_dir / "logs" / "0000-quality-banking77.log"
    _write_json(run_dir / "quality" / name / "multiseed_summary.json", study)
    _write_json(log_path, {"study": study, "analysis": {}})

    def reject_seed(_path):
        from paper.verify_quality_artifacts import QualityArtifactVerificationError

        raise QualityArtifactVerificationError(["fixture deep-verification failure"])

    monkeypatch.setattr("paper.verify_matrix_run.verify_quality_artifacts", reject_seed)
    errors = []

    _verify_family_invariants(
        run_dir,
        [{"index": 0, "family": "quality", "name": name}],
        {"quality": {"seeds": [13]}},
        {0: log_path},
        errors,
    )

    assert "quality:banking77:seed-13: fixture deep-verification failure" in errors
