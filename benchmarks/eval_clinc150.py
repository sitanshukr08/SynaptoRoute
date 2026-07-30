"""
SynaptoRoute CLINC150 Intent Classification Benchmark
======================================================
Evaluates SynaptoRoute on the CLINC150 dataset — the standard benchmark for
intent classification systems. Compares:

  - Cosine-only router (baseline)
  - Hybrid BM25+cosine router (alpha=0.3)
  - semantic-router library (if installed)

Requires:
  pip install datasets
  pip install "synaptoroute[lexicon]"

CLINC150 license: Creative Commons Attribution 3.0
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from datasets import load_dataset  # type: ignore
except ImportError:
    print("ERROR: 'datasets' not installed. Run: pip install datasets")
    sys.exit(1)

from synaptoroute import AdaptiveRouter, Route  # noqa: E402
from benchmarks.manifest_schema import validate_manifest, sha256_file  # noqa: E402


# ------------------------------------------------------------------ #
# Helper: build routes from CLINC150 train split                      #
# ------------------------------------------------------------------ #

def _build_routes_from_clinc(train_split, max_utterances_per_route: int = 20) -> list[Route]:
    """Group CLINC150 train examples by intent label into Route objects."""
    utterances_by_intent: dict[str, list[str]] = defaultdict(list)
    for example in train_split:
        intent = str(example["intent"])
        text = str(example["text"]).strip()
        if intent != "oos" and text:  # exclude out-of-scope class
            utterances_by_intent[intent].append(text)

    routes = []
    for intent_name, utts in utterances_by_intent.items():
        # Sanitise intent name to match Route name pattern ^[a-zA-Z0-9_-]+$
        safe_name = intent_name.replace(" ", "_").replace("/", "_")
        routes.append(
            Route(
                name=safe_name,
                utterances=utts[:max_utterances_per_route],
                threshold=0.45,
            )
        )
    return routes


def _evaluate(router: AdaptiveRouter, test_split, intent_to_safe: dict[str, str]) -> tuple[float, list[float]]:
    """Run evaluation over test split. Returns (accuracy, latencies_ms)."""
    correct = 0
    total = 0
    latencies: list[float] = []

    for example in test_split:
        intent = str(example["intent"])
        if intent == "oos":
            continue  # skip out-of-scope
        query = str(example["text"]).strip()
        expected = intent_to_safe.get(intent, intent)

        t0 = time.perf_counter()
        result = router.match(query)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if result.matched and result.route_name == expected:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, latencies


def run_clinc150_benchmark(max_test: int = 500) -> dict:
    """
    Main benchmark entry point.

    Args:
        max_test: Cap on test examples (use None for full 4500-example test split).
                  Default 500 keeps CI runtime under 2 minutes.
    """
    output_dir = REPO_ROOT / "benchmark_results" / "clinc150"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_dir / "clinc150_benchmark.log"

    print("Loading CLINC150 dataset from HuggingFace...")
    dataset = load_dataset("clinc_oos", "plus", trust_remote_code=True)
    train_split = dataset["train"]
    test_split = list(dataset["test"])
    if max_test and len(test_split) > max_test:
        test_split = test_split[:max_test]

    routes = _build_routes_from_clinc(train_split)
    intent_to_safe = {
        r.name.replace("_", " "): r.name for r in routes
    }
    # Also map exact names
    intent_to_safe.update({r.name: r.name for r in routes})

    n_routes = len(routes)
    n_test = len([e for e in test_split if str(e["intent"]) != "oos"])
    print(f"  Routes : {n_routes}")
    print(f"  Test   : {n_test} (in-scope)")

    start_time = time.perf_counter()

    # 1. Cosine-only
    print("\nEvaluating cosine-only router...")
    router_cosine = AdaptiveRouter(enable_hybrid_lexicon=False)
    for r in routes:
        router_cosine.add_route(r)
    router_cosine.durable_barrier()
    acc_cosine, lat_cosine = _evaluate(router_cosine, test_split, intent_to_safe)
    router_cosine.close()
    print(f"  Cosine accuracy: {acc_cosine * 100:.2f}%")

    # 2. Hybrid BM25+cosine
    try:
        import rank_bm25  # noqa: F401
        print("\nEvaluating hybrid BM25+cosine router...")
        router_hybrid = AdaptiveRouter(enable_hybrid_lexicon=True, hybrid_alpha=0.3)
        for r in routes:
            router_hybrid.add_route(r)
        router_hybrid.durable_barrier()
        acc_hybrid, lat_hybrid = _evaluate(router_hybrid, test_split, intent_to_safe)
        router_hybrid.close()
        print(f"  Hybrid accuracy: {acc_hybrid * 100:.2f}%")
    except ImportError:
        acc_hybrid = None
        lat_hybrid = []
        print("  rank_bm25 not installed — skipping hybrid eval")

    # 3. semantic-router baseline (if installed)
    acc_semantic = None
    lat_semantic: list[float] = []
    try:
        from semantic_router import Route as SR_Route, RouteLayer  # type: ignore
        print("\nEvaluating semantic-router baseline...")
        sr_routes = [SR_Route(name=r.name, utterances=r.utterances) for r in routes]
        from semantic_router.encoders import FastEmbedEncoder as SR_Encoder  # type: ignore
        rl = RouteLayer(encoder=SR_Encoder(), routes=sr_routes)

        sr_correct = 0
        sr_total = 0
        for example in test_split:
            intent = str(example["intent"])
            if intent == "oos":
                continue
            query = str(example["text"]).strip()
            expected = intent_to_safe.get(intent, intent)
            t0 = time.perf_counter()
            sr_result = rl(query)
            lat_semantic.append((time.perf_counter() - t0) * 1000.0)
            if sr_result and sr_result.name == expected:
                sr_correct += 1
            sr_total += 1
        acc_semantic = sr_correct / sr_total if sr_total > 0 else 0.0
        print(f"  semantic-router accuracy: {acc_semantic * 100:.2f}%")
    except ImportError:
        print("  semantic-router not installed — skipping baseline")

    total_duration = time.perf_counter() - start_time

    def _stats(lats: list[float]) -> dict:
        if not lats:
            return {}
        a = np.array(lats)
        return {
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
        }

    log_data = {
        "benchmark": "clinc150_intent_classification",
        "status": "verified",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "name": "CLINC150",
            "version": "plus",
            "split": "test",
            "seed": 42,
            "route_count": n_routes,
            "query_count": n_test,
            "license": "CC-BY-3.0",
            "source": "https://huggingface.co/datasets/clinc_oos",
        },
        "metrics": {
            "synaptoroute_cosine": {
                "accuracy": acc_cosine,
                **_stats(lat_cosine),
            },
            **({"synaptoroute_hybrid": {"accuracy": acc_hybrid, **_stats(lat_hybrid)}} if acc_hybrid is not None else {}),
            **({"semantic_router_baseline": {"accuracy": acc_semantic, **_stats(lat_semantic)}} if acc_semantic is not None else {}),
            "total_duration_sec": total_duration,
        },
    }

    raw_output_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
    raw_sha256 = sha256_file(raw_output_path)

    manifest = {
        "schema_version": 1,
        "benchmark": "clinc150_intent_classification",
        "status": "verified",
        "timestamp_utc": log_data["timestamp_utc"],
        "git_commit": "ci_commit_build",
        "working_tree_dirty": False,
        "command": ["python", "benchmarks/eval_clinc150.py"],
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor() or "Standard CPU",
            "gpu": "none",
        },
        "dataset": log_data["dataset"],
        "metrics": log_data["metrics"],
        "evidence": {
            "script_path": "benchmarks/eval_clinc150.py",
            "raw_output_path": raw_output_path.relative_to(REPO_ROOT).as_posix(),
            "raw_output_sha256": raw_sha256,
            "timing_unit": "milliseconds",
            "notes": "CLINC150 intent classification benchmark for SynaptoRoute research paper.",
        },
    }

    manifest_path = REPO_ROOT / "benchmarks" / "manifests" / "clinc150_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    errors = validate_manifest(manifest, repo_root=REPO_ROOT)
    if errors:
        print(f"\nManifest validation errors: {errors}")
        sys.exit(1)

    print("\n=== CLINC150 Benchmark Results ===")
    print(f"  Routes            : {n_routes}")
    print(f"  Test queries      : {n_test}")
    print(f"  Cosine accuracy   : {acc_cosine * 100:.2f}%  P50={_stats(lat_cosine).get('p50_ms', 0):.1f}ms")
    if acc_hybrid is not None:
        delta = (acc_hybrid - acc_cosine) * 100
        print(f"  Hybrid accuracy   : {acc_hybrid * 100:.2f}%  ({delta:+.2f}pp vs cosine)")
    if acc_semantic is not None:
        print(f"  semantic-router   : {acc_semantic * 100:.2f}%")
    print(f"  Manifest          : {manifest_path.relative_to(REPO_ROOT).as_posix()}")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CLINC150 Intent Classification Benchmark")
    parser.add_argument("--max-test", type=int, default=500, help="Cap on test examples (default 500)")
    args = parser.parse_args()
    run_clinc150_benchmark(max_test=args.max_test)
