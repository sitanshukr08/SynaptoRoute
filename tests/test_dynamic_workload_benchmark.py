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
    assert metrics["query_errors"] == 0
    assert metrics["query_accuracy"] == 1.0
    assert metrics["mutation_errors"] == 0
    assert metrics["visibility_failures"] == 0
    assert metrics["deletion_visibility_failures"] == 0
    assert metrics["restart_state_equal"] is True
