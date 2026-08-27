import copy

import pytest

from paper.analyze_scale import analyze_records, render_csv, render_markdown


def summary(
    engine,
    *,
    route_count=1000,
    seed=42,
    correct=100,
    build_seconds=1.0,
    throughput_qps=1000.0,
    p95_ms=1.0,
    rss_delta_mb=10.0,
):
    query_count = 100
    return {
        "schema_version": 2,
        "benchmark": "precomputed_vector_scale",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "configuration": {
            "engine": engine,
            "route_count": route_count,
            "query_count": query_count,
            "dimension": 64,
            "seed": seed,
        },
        "metrics": {
            "query_count": query_count,
            "correct_count": correct,
            "incorrect_count": query_count - correct,
            "top1_identity_accuracy": correct / query_count,
            "build_seconds": build_seconds,
            "query_seconds": query_count / throughput_qps,
            "throughput_qps": throughput_qps,
            "latency": {
                "p50_ms": p95_ms / 2,
                "p95_ms": p95_ms,
                "p99_ms": p95_ms * 1.2,
                "max_ms": p95_ms * 1.5,
            },
            "rss_before_mb": 100.0,
            "rss_after_mb": None if rss_delta_mb is None else 100.0 + rss_delta_mb,
            "rss_delta_mb": rss_delta_mb,
        },
    }


def paired_records():
    return {
        "numpy-r1000-rep0": summary("numpy", seed=42, build_seconds=1.0, throughput_qps=1000, p95_ms=1.0),
        "numpy-r1000-rep1": summary("numpy", seed=43, build_seconds=2.0, throughput_qps=800, p95_ms=1.2),
        "faiss-r1000-rep0": summary(
            "faiss",
            seed=42,
            correct=99,
            build_seconds=4.0,
            throughput_qps=2000,
            p95_ms=0.5,
            rss_delta_mb=15.0,
        ),
        "faiss-r1000-rep1": summary(
            "faiss",
            seed=43,
            correct=98,
            build_seconds=8.0,
            throughput_qps=1600,
            p95_ms=0.6,
            rss_delta_mb=17.0,
        ),
    }


def test_scale_analysis_retains_misses_and_pairs_matching_seeds():
    result = analyze_records(paired_records(), bootstrap_repetitions=1000, random_seed=7)
    faiss = result["engines"]["faiss"]["route_counts"]["1000"]
    comparison = result["paired_comparisons"]["1000"]

    assert faiss["pooled_counts"]["query_count"] == 200
    assert faiss["pooled_counts"]["incorrect_count"] == 3
    assert faiss["pooled_top1_identity_accuracy"] == pytest.approx(0.985)
    assert comparison["seeds"] == [42, 43]
    assert comparison["faiss_minus_numpy_accuracy"]["mean"] == pytest.approx(-0.015)
    assert comparison["faiss_to_numpy_build_time_ratio"]["mean"] == 4.0
    assert comparison["faiss_to_numpy_throughput_ratio"]["mean"] == 2.0
    assert comparison["faiss_to_numpy_p95_latency_ratio"]["mean"] == 0.5


def test_scale_analysis_rejects_missing_engine_and_repetition_gap():
    with pytest.raises(ValueError, match="requires both engines"):
        analyze_records({"numpy-r1000-rep0": summary("numpy")}, bootstrap_repetitions=100)

    records = paired_records()
    records["numpy-r1000-rep2"] = records.pop("numpy-r1000-rep1")
    with pytest.raises(ValueError, match="contiguous"):
        analyze_records(records, bootstrap_repetitions=100)


def test_scale_analysis_rejects_configuration_and_seed_mismatch():
    records = paired_records()
    changed = copy.deepcopy(records["faiss-r1000-rep1"])
    changed["configuration"]["route_count"] = 2000
    records["faiss-r1000-rep1"] = changed
    with pytest.raises(ValueError, match="configuration differs"):
        analyze_records(records, bootstrap_repetitions=100)

    records = paired_records()
    records["faiss-r1000-rep1"]["configuration"]["seed"] = 44
    with pytest.raises(ValueError, match="same seeds"):
        analyze_records(records, bootstrap_repetitions=100)


def test_scale_analysis_allows_unavailable_rss_evidence():
    records = paired_records()
    for value in records.values():
        value["metrics"]["rss_delta_mb"] = None

    result = analyze_records(records, bootstrap_repetitions=100)

    assert result["engines"]["numpy"]["route_counts"]["1000"]["repetition_statistics"]["rss_delta_mb"] is None
    assert result["paired_comparisons"]["1000"]["faiss_minus_numpy_rss_delta_mb"] is None


def test_scale_renderers_expose_status_counts_and_effect_direction():
    aggregate = analyze_records(paired_records(), bootstrap_repetitions=100)
    analysis = {"summary": aggregate}

    csv_text = render_csv(analysis)
    markdown = render_markdown(analysis)

    assert "query_count,correct_count,incorrect_count,pooled_accuracy" in csv_text
    assert "faiss,1000,2,200,197,3,0.985" in csv_text
    assert "Status: unverified development evidence" in markdown
    assert "Effects are FAISS minus or divided by NumPy" in markdown
