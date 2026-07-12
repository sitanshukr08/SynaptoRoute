from benchmarks.bench_crash_recovery import run_benchmark


def test_crash_recovery_distinguishes_memory_and_durable_acknowledgement(tmp_path):
    result = run_benchmark(output_dir=tmp_path, trials=1, delay_ms=100.0)

    assert result["metrics"]["memory"]["acknowledged_count"] == 1
    assert result["metrics"]["memory"]["restart_survival_rate"] == 0.0
    assert result["metrics"]["durable"]["acknowledged_count"] == 1
    assert result["metrics"]["durable"]["restart_survival_rate"] == 1.0
    assert result["metrics"]["durable"]["all_children_exited_cleanly"] is True
