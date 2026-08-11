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
    assert result["configuration"]["index_parameters"] == {
        "construction_add_calls": 20,
        "vectors_per_add_call": 1,
        "metric": "normalized_inner_product",
        "resolved_engine": "numpy",
        "implementation": "numpy_exact",
        "max_capacity": 20,
    }
    assert result["environment"]["faiss"] is None


def test_faiss_scale_benchmark_records_hnsw_configuration():
    import pytest

    pytest.importorskip("faiss")
    result = run_benchmark(
        engine="faiss",
        route_count=20,
        query_count=25,
        dim=8,
        seed=42,
    )

    parameters = result["configuration"]["index_parameters"]
    assert parameters["implementation"] == "faiss_hnsw"
    assert parameters["faiss_version"]
    assert parameters["hnsw_m"] == 32
    assert parameters["hnsw_ef_construction"] > 0
    assert parameters["hnsw_ef_search"] > 0
    assert parameters["omp_threads"] > 0
    assert parameters["search_candidate_floor"] == 2048
    assert parameters["construction_add_calls"] == 20
    assert result["environment"]["faiss"] == parameters["faiss_version"]
