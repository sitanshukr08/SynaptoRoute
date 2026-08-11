from benchmarks.bench_dynamic_workload import run_benchmark


def test_dynamic_workload_preserves_visibility_and_restart_state(tmp_path):
    result = run_benchmark(
        database_path=tmp_path / "dynamic.sqlite3",
        duration_seconds=0.25,
        route_count=8,
        query_workers=2,
        mutation_rate=12.0,
        dim=8,
    )

    metrics = result["metrics"]
    assert metrics["completed_queries"] > 0
    assert metrics["query_attempts"] == metrics["completed_queries"] + metrics["query_errors"]
    assert metrics["completed_queries"] == metrics["query_correct"] + metrics["query_incorrect"]
    assert metrics["query_errors"] == 0
    assert metrics["query_incorrect"] == 0
    assert metrics["query_accuracy"] == 1.0
    assert metrics["query_success_rate"] == 1.0
    assert metrics["mutation_attempts"] == (
        metrics["mutation_successes"] + metrics["mutation_errors"]
    )
    assert metrics["mutation_errors"] == 0
    assert metrics["visibility_failures"] == 0
    assert metrics["deletion_visibility_failures"] == 0
    assert metrics["correctness_violations"] == 0
    assert metrics["operation_failures"] == 0
    assert metrics["total_adverse_outcomes"] == 0
    assert metrics["pre_restart_state_equal"] is True
    assert metrics["restart_state_equal"] is True
    assert metrics["restart_recovery_ms"] >= 0.0
    assert metrics["storage_queue_depth"]["samples"] > 0
    parameters = result["workload"]["index_parameters"]
    assert result["workload"]["index_engine_requested"] == "auto"
    assert parameters["resolved_engine"] in {"numpy", "faiss"}
    assert parameters["metric"] == "normalized_inner_product"
    assert result["environment"]["faiss"] == parameters.get("faiss_version")


def test_dynamic_workload_supports_read_only_baseline(tmp_path):
    result = run_benchmark(
        database_path=tmp_path / "read-only.sqlite3",
        duration_seconds=0.1,
        route_count=4,
        query_workers=1,
        mutation_rate=0.0,
        dim=8,
        warmup_seconds=0.01,
    )

    assert result["metrics"]["completed_queries"] > 0
    assert result["metrics"]["mutation_attempts"] == 0
    assert result["metrics"]["mutation_success_rate"] is None
    assert result["metrics"]["mutation_error_rate"] is None
