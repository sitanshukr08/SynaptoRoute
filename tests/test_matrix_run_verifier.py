import json
import sys
from pathlib import Path

import pytest

from benchmarks.manifest_schema import sha256_file
from paper.verify_matrix_run import (
    MatrixRunVerificationError,
    _verify_family_invariants,
    _verify_scale_summary,
    verify_matrix_run,
)


COMMIT = "a" * 40


def test_scale_handler_validates_recorded_hnsw_configuration():
    summary = {
        "configuration": {
            "engine": "faiss",
            "route_count": 100,
            "query_count": 10,
            "index_parameters": {
                "construction_add_calls": 100,
                "vectors_per_add_call": 1,
                "metric": "normalized_inner_product",
                "implementation": "faiss_hnsw",
                "faiss_version": "1.14.3",
                "omp_threads": 4,
                "hnsw_m": 32,
                "hnsw_ef_construction": 40,
                "hnsw_ef_search": 0,
                "search_candidate_floor": 2048,
            },
        },
        "metrics": {
            "query_count": 10,
            "correct_count": 10,
            "incorrect_count": 0,
            "top1_identity_accuracy": 1.0,
            "build_seconds": 1.0,
            "query_seconds": 1.0,
            "throughput_qps": 10.0,
            "latency": {"p50_ms": 1.0, "p95_ms": 1.0, "p99_ms": 1.0, "max_ms": 1.0},
            "rss_before_mb": None,
            "rss_after_mb": None,
            "rss_delta_mb": None,
        },
    }
    errors = []

    _verify_scale_summary("faiss-r100-rep0", summary, errors, [])

    assert errors == ["scale:faiss-r100-rep0: hnsw_ef_search must be a positive integer"]


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
                "database_sha256": sha256_file(cell_dir / f"{prefix}.sqlite3"),
                "database_bytes": (cell_dir / f"{prefix}.sqlite3").stat().st_size,
                "marker_path": f"/original/matrix/crash_recovery/{name}/{prefix}.ack",
                "marker_sha256": sha256_file(cell_dir / f"{prefix}.ack"),
            }
        )
    summary = {
        "schema_version": 2,
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
                "acknowledgement_rate": 1.0,
                "survived_count": 0,
                "restart_survival_rate": 0.0,
                "clean_exit_count": 1,
                "clean_exit_rate": 1.0,
                "all_children_exited_cleanly": True,
                "expected_restart_survival": False,
                "contract_violation_count": 0,
                "contract_success_rate": 1.0,
            },
            "durable": {
                "trial_count": 1,
                "acknowledged_count": 1,
                "acknowledgement_rate": 1.0,
                "survived_count": 1 if durable_survived else 0,
                "restart_survival_rate": 1.0 if durable_survived else 0.0,
                "clean_exit_count": 1,
                "clean_exit_rate": 1.0,
                "all_children_exited_cleanly": True,
                "expected_restart_survival": True,
                "contract_violation_count": 0 if durable_survived else 1,
                "contract_success_rate": 1.0 if durable_survived else 0.0,
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
    assert report["outcome_observation_count"] == 0


def test_verifier_rejects_a_changed_hashed_log(tmp_path):
    run_dir, source = _make_crash_run(tmp_path)
    log = next((run_dir / "logs").glob("*.log"))
    log.write_text(log.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(MatrixRunVerificationError, match="command log SHA-256 mismatch"):
        verify_matrix_run(run_dir, source_root=source)


def test_verifier_rejects_changed_crash_database_evidence(tmp_path):
    run_dir, source = _make_crash_run(tmp_path)
    database = next((run_dir / "crash_recovery").rglob("*.sqlite3"))
    database.write_bytes(b"tamper")

    with pytest.raises(MatrixRunVerificationError, match="database hash differs"):
        verify_matrix_run(run_dir, source_root=source)


def test_verifier_retains_an_internally_consistent_durability_violation(tmp_path):
    run_dir, source = _make_crash_run(tmp_path, durable_survived=False)

    report = verify_matrix_run(run_dir, source_root=source)

    assert report["outcome_observation_count"] == 1
    assert report["outcome_observations"][0]["code"] == "durable_restart_contract_violations"


def test_verifier_requires_requested_environment_evidence(tmp_path):
    run_dir, source = _make_crash_run(tmp_path)

    with pytest.raises(MatrixRunVerificationError, match="environment evidence directory is required"):
        verify_matrix_run(run_dir, source_root=source, require_environment=True)


@pytest.mark.parametrize(
    ("family", "name", "matrix", "summary", "expected_code"),
    [
        (
            "dynamic",
            "r100-w1-m0-rep0",
            {"dynamic": {}},
            {
                "schema_version": 2,
                "benchmark": "concurrent_dynamic_routing_workload",
                "status": "unverified",
                "paper_evidence_eligible": False,
                "workload": {
                    "route_count": 100,
                    "query_workers": 1,
                    "target_mutations_per_second": 0,
                    "warmup_seconds": 10,
                    "duration_seconds": 60,
                },
                "metrics": {
                    "measurement_wall_seconds": 1.0,
                    "query_attempts": 2,
                    "completed_queries": 2,
                    "query_correct": 1,
                    "query_incorrect": 1,
                    "query_errors": 0,
                    "query_accuracy": 0.5,
                    "query_success_rate": 1.0,
                    "query_attempt_throughput_per_second": 2.0,
                    "query_success_throughput_per_second": 2.0,
                    "query_throughput_per_second": 2.0,
                    "query_latency": {
                        "p50_ms": 1.0,
                        "p95_ms": 1.0,
                        "p99_ms": 1.0,
                        "max_ms": 1.0,
                    },
                    "mutation_attempts": 0,
                    "mutation_successes": 0,
                    "mutation_errors": 0,
                    "mutation_success_rate": None,
                    "mutation_error_rate": None,
                    "mutation_shedding_count": 0,
                    "mutation_attempt_throughput_per_second": 0.0,
                    "mutation_success_throughput_per_second": 0.0,
                    "mutation_throughput_per_second": 0.0,
                    "mutation_memory_ack": None,
                    "mutation_durable_commit": None,
                    "mutation_receipt_count": 0,
                    "durable_latency_count": 0,
                    "durable_receipt_count": 0,
                    "storage_queue_depth": {"max": 0, "samples": 0},
                    "visibility_failures": 0,
                    "deletion_visibility_failures": 0,
                    "correctness_violations": 1,
                    "operation_failures": 0,
                    "total_adverse_outcomes": 1,
                    "pre_restart_state_equal": True,
                    "restart_state_equal": True,
                },
                "errors": {"query": [], "mutation": []},
            },
            "incorrect_query_results",
        ),
        (
            "scale",
            "numpy-r1000-rep0",
            {"scale": {}},
            {
                "schema_version": 2,
                "benchmark": "precomputed_vector_scale",
                "status": "unverified",
                "paper_evidence_eligible": False,
                "configuration": {
                    "engine": "numpy",
                    "route_count": 1000,
                    "query_count": 100,
                    "seed": 42,
                },
                "metrics": {
                    "query_count": 100,
                    "correct_count": 99,
                    "incorrect_count": 1,
                    "top1_identity_accuracy": 0.99,
                    "build_seconds": 1.0,
                    "query_seconds": 1.0,
                    "throughput_qps": 100.0,
                    "latency": {
                        "p50_ms": 1.0,
                        "p95_ms": 1.0,
                        "p99_ms": 1.0,
                        "max_ms": 1.0,
                    },
                },
            },
            "identity_retrieval_misses",
        ),
        (
            "backpressure",
            "balanced-rep0",
            {"backpressure": {"offered_load_fraction": [1.0]}},
            {
                "schema_version": 2,
                "benchmark": "sustained_async_backpressure",
                "status": "unverified",
                "paper_evidence_eligible": False,
                "configuration": {
                    "duration_seconds": 60,
                    "saturation_calibration_target_seconds": 10,
                    "queue_size": 32,
                    "batch_size": 8,
                    "saturation_calibration_attempts": 10,
                    "saturation_calibration_successes": 10,
                    "saturation_calibration_overloaded": 0,
                    "saturation_calibration_error_count": 0,
                    "saturation_calibration_error_types": [],
                    "saturation_calibration_seconds": 0.1,
                    "measured_saturation_qps": 100.0,
                },
                "scenarios": [
                    {
                        "load_fraction": 1.0,
                        "offered_count": 3,
                        "successful_count": 1,
                        "successful_correct_count": 1,
                        "successful_incorrect_count": 0,
                        "overloaded_count": 1,
                        "error_count": 1,
                        "success_rate": 1 / 3,
                        "shedding_rate": 1 / 3,
                        "error_rate": 1 / 3,
                        "successful_accuracy": 1.0,
                        "target_qps": 100.0,
                        "offering_wall_seconds": 1.0,
                        "drain_seconds": 0.0,
                        "scenario_wall_seconds": 1.0,
                        "offered_qps": 3.0,
                        "successful_qps": 1.0,
                        "completed_qps": 1.0,
                        "resolved_qps": 3.0,
                        "successful_latency": {
                            "p50_ms": 1.0,
                            "p95_ms": 1.0,
                            "p99_ms": 1.0,
                            "max_ms": 1.0,
                        },
                        "overload_latency": {
                            "p50_ms": 1.0,
                            "p95_ms": 1.0,
                            "p99_ms": 1.0,
                            "max_ms": 1.0,
                        },
                        "error_types": ["error:RuntimeError"],
                    }
                ],
            },
            "request_errors",
        ),
    ],
)
def test_family_handlers_retain_unfavorable_outcomes(
    tmp_path,
    family,
    name,
    matrix,
    summary,
    expected_code,
):
    run_dir = tmp_path / "matrix"
    commands = {
        "dynamic": [
            "python",
            "benchmark.py",
            "--routes",
            "100",
            "--query-workers",
            "1",
            "--mutation-rate",
            "0",
            "--warmup",
            "10",
            "--duration",
            "60",
        ],
        "scale": [
            "python",
            "benchmark.py",
            "--engine",
            "numpy",
            "--routes",
            "1000",
            "--queries",
            "100",
            "--seed",
            "42",
        ],
        "backpressure": [
            "python",
            "benchmark.py",
            "--duration",
            "60",
            "--calibration-duration",
            "10",
            "--queue-size",
            "32",
            "--batch-size",
            "8",
        ],
    }
    result = {"index": 0, "family": family, "name": name, "command": commands[family]}
    if family == "dynamic":
        summary_path = run_dir / family / name / "dynamic_workload_summary.json"
    else:
        summary_path = run_dir / family / f"{name}.json"
    log_path = run_dir / "logs" / f"0000-{family}-{name}.log"
    if family == "dynamic":
        database = run_dir / family / name / "state.sqlite3"
        database.parent.mkdir(parents=True, exist_ok=True)
        database.write_bytes(b"sqlite")
        summary["evidence"] = {
            "database_path": f"/original/matrix/{family}/{name}/{database.name}",
            "database_sha256": sha256_file(database),
            "database_bytes": database.stat().st_size,
        }
    _write_json(summary_path, summary)
    _write_json(log_path, summary)
    errors = []

    report = _verify_family_invariants(run_dir, [result], matrix, {0: log_path}, errors)

    assert errors == []
    assert expected_code in {
        observation["code"] for observation in report["outcome_observations"]
    }


def test_backpressure_handler_rejects_a_falsified_denominator(tmp_path):
    run_dir = tmp_path / "matrix"
    name = "balanced-rep0"
    summary = {
        "schema_version": 2,
        "benchmark": "sustained_async_backpressure",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "configuration": {
            "duration_seconds": 60,
            "saturation_calibration_target_seconds": 10,
            "queue_size": 32,
            "batch_size": 8,
            "saturation_calibration_attempts": 1,
            "saturation_calibration_successes": 1,
            "saturation_calibration_overloaded": 0,
            "saturation_calibration_error_count": 0,
            "saturation_calibration_error_types": [],
            "saturation_calibration_seconds": 1.0,
            "measured_saturation_qps": 1.0,
        },
        "scenarios": [
            {
                "load_fraction": 1.0,
                "target_qps": 1.0,
                "offered_count": 4,
                "successful_count": 1,
                "successful_correct_count": 1,
                "successful_incorrect_count": 0,
                "overloaded_count": 1,
                "error_count": 1,
                "success_rate": 0.25,
                "shedding_rate": 0.25,
                "error_rate": 0.25,
                "offering_wall_seconds": 1.0,
                "drain_seconds": 0.0,
                "scenario_wall_seconds": 1.0,
                "offered_qps": 4.0,
                "successful_qps": 1.0,
                "completed_qps": 1.0,
                "resolved_qps": 4.0,
                "successful_accuracy": 1.0,
                "successful_latency": {
                    "p50_ms": 1.0,
                    "p95_ms": 1.0,
                    "p99_ms": 1.0,
                    "max_ms": 1.0,
                },
                "overload_latency": {
                    "p50_ms": 1.0,
                    "p95_ms": 1.0,
                    "p99_ms": 1.0,
                    "max_ms": 1.0,
                },
                "error_types": ["error:RuntimeError"],
            }
        ],
    }
    summary_path = run_dir / "backpressure" / f"{name}.json"
    log_path = run_dir / "logs" / f"0000-backpressure-{name}.log"
    _write_json(summary_path, summary)
    _write_json(log_path, summary)
    errors = []

    _verify_family_invariants(
        run_dir,
        [
            {
                "index": 0,
                "family": "backpressure",
                "name": name,
                "command": [
                    "python",
                    "benchmark.py",
                    "--duration",
                    "60",
                    "--calibration-duration",
                    "10",
                    "--queue-size",
                    "32",
                    "--batch-size",
                    "8",
                ],
            }
        ],
        {"backpressure": {"offered_load_fraction": [1.0]}},
        {0: log_path},
        errors,
    )

    assert "backpressure:balanced-rep0:load=1.0: offered-load denominator mismatch" in errors


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
