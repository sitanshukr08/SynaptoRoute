import time
import pytest
import numpy as np
from synaptoroute.adaptive_weights import (
    BoundedBayesianWeigher,
    LockFreeStatsCollector,
    VectorARCCache,
    ContextMetadata,
)

def test_bounded_bayesian_weigher_preserves_metric_space():
    weigher = BoundedBayesianWeigher(frequency_boost_cap=0.08, saturation_constant=50.0)
    meta = ContextMetadata(
        key="test_intent",
        route_name="billing",
        embedding=np.array([1.0, 0.0]),
        frequency_count=10000,  # Extremely high frequency
        last_accessed=time.time(),
    )

    raw_cosine = 0.70
    adjusted_score = weigher.evaluate_score(raw_cosine, meta)

    # Assert prior boost is capped at +0.08, preventing metric space explosion
    assert adjusted_score == pytest.approx(0.78, abs=0.005)

def test_negative_feedback_penalty():
    weigher = BoundedBayesianWeigher(penalty_weight=0.05)
    meta = ContextMetadata(
        key="flaky_intent",
        route_name="support",
        embedding=np.array([0.0, 1.0]),
        frequency_count=10,
        negative_feedback_count=3,  # 3 negative feedback hits
        last_accessed=time.time(),
    )

    prior = weigher.compute_prior_adjustment(meta)
    assert prior < 0.0  # Penalty outweighs modest frequency boost

def test_lock_free_stats_collector():
    collector = LockFreeStatsCollector()
    collector.record_hit("intent_a")
    collector.record_hit("intent_b", is_negative=True)

    drained = collector.flush()
    assert len(drained) == 2
    assert drained[0][0] == "intent_a"
    assert drained[1][2] is True  # is_negative

    # Assert queue is flushed empty
    assert len(collector.flush()) == 0

def test_vector_arc_cache_tuning():
    cache = VectorARCCache(capacity=2)
    meta1 = ContextMetadata("k1", "r1", np.array([1.0, 0.0]))
    meta2 = ContextMetadata("k2", "r2", np.array([0.0, 1.0]))
    meta3 = ContextMetadata("k3", "r3", np.array([0.5, 0.5]))

    cache.put("k1", meta1)
    cache.put("k2", meta2)

    # Access k1 so it moves to T2 (frequent)
    assert cache.get("k1") is not None

    # Insert k3, evicting k2 (from T1)
    cache.put("k3", meta3)

    assert cache.get("k1") is not None
    assert cache.get("k3") is not None
    assert cache.get("k2") is None


def test_adaptive_router_integration(fake_encoder):
    from synaptoroute import AdaptiveRouter, Route, SQLiteStorage

    storage = SQLiteStorage(":memory:")
    router = AdaptiveRouter(fake_encoder, storage, enable_adaptive_memory=True)

    r1 = Route(name="route1", utterances=["help"], threshold=0.5)
    router.add_route(r1)

    # Initial match
    res1 = router.match("help")
    assert res1.matched
    assert res1.route_name == "route1"

    # Verify hit was recorded in stats collector and route context
    assert router.stats_collector is not None
    drained = router.stats_collector.flush()
    assert len(drained) == 1
    assert drained[0][0] == "route1"

    meta = router._route_metadata_context["route1"]
    assert meta.frequency_count == 1
    router.close()

