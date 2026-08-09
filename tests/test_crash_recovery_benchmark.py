from benchmarks.bench_crash_recovery import run_benchmark


def test_crash_recovery_distinguishes_memory_and_durable_acknowledgement(tmp_path):
    result = run_benchmark(output_dir=tmp_path, trials=1, delay_ms=100.0)

    assert result["metrics"]["memory"]["acknowledged_count"] == 1
    assert result["metrics"]["memory"]["acknowledgement_rate"] == 1.0
    assert result["metrics"]["memory"]["survived_count"] == 0
    assert result["metrics"]["memory"]["restart_survival_rate"] == 0.0
    assert result["metrics"]["memory"]["contract_violation_count"] == 0
    assert result["metrics"]["durable"]["acknowledged_count"] == 1
    assert result["metrics"]["durable"]["survived_count"] == 1
    assert result["metrics"]["durable"]["restart_survival_rate"] == 1.0
    assert result["metrics"]["durable"]["contract_success_rate"] == 1.0
    assert result["metrics"]["durable"]["all_children_exited_cleanly"] is True


def test_crash_recovery_covers_each_mutation_type(tmp_path):
    for mutation in ("add_route", "add_utterance", "update_threshold", "delete_route"):
        result = run_benchmark(
            output_dir=tmp_path / mutation,
            trials=1,
            delay_ms=50.0,
            mutation=mutation,
            synchronous="NORMAL",
        )
        assert result["metrics"]["memory"]["restart_survival_rate"] == 0.0
        assert result["metrics"]["durable"]["restart_survival_rate"] == 1.0
        durable_markers = [
            trial["marker"]
            for trial in result["trials"]
            if trial["mode"] == "durable"
        ]
        assert all(marker.startswith("durable:") for marker in durable_markers)
