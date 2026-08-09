from benchmarks.bench_scale_matrix import run_benchmark


def test_numpy_scale_benchmark_is_structural_and_complete():
    result = run_benchmark(
        engine="numpy",
        route_count=20,
        query_count=25,
        dim=8,
        seed=42,
    )

    assert result["status"] == "unverified"
    assert result["paper_evidence_eligible"] is False
    assert result["metrics"]["query_count"] == 25
    assert result["metrics"]["correct_count"] == 25
    assert result["metrics"]["incorrect_count"] == 0
    assert result["metrics"]["top1_identity_accuracy"] == 1.0
    assert result["metrics"]["latency"]["p99_ms"] >= 0.0
