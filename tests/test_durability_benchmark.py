from benchmarks.bench_durability import run_benchmark


def test_durability_benchmark_checks_restart_and_failure_visibility(tmp_path):
    result = run_benchmark(
        mutation_count=12,
        database_path=tmp_path / "durability.sqlite3",
        dim=8,
    )

    assert result["paper_evidence_eligible"] is False
    assert result["metrics"]["restart_state_equal"] is True
    assert result["metrics"]["receipt_failure_observed"] is True
    assert result["metrics"]["barrier_failure_observed"] is True
    assert result["metrics"]["failure_resynced"] is True
    assert result["metrics"]["burst_throughput_mutations_per_second"] > 0
