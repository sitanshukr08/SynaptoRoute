from synaptoroute.exceptions import RouteNotFoundError, RouterCapacityError
import pytest
from synaptoroute.router import AdaptiveRouter
from synaptoroute.models import Route
from synaptoroute.storage import SQLiteStorage

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    return str(db_path)

@pytest.fixture
def storage(temp_db):
    return SQLiteStorage(db_path=temp_db)


def test_static_routing(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    
    route1 = Route(name="greeting", utterances=["hello", "hi there"], threshold=0.5)
    route2 = Route(name="farewell", utterances=["bye", "see you"], threshold=0.5)
    
    router.add_route(route1)
    router.add_route(route2)
    
    match = router("hello my friend")
    assert match is not None
    assert match.name == "greeting"
    
    match = router("goodbye")
    assert match is not None
    assert match.name == "farewell"

def test_hot_reload_utterance(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    
    route1 = Route(name="support", utterances=["help me"], threshold=0.8)
    router.add_route(route1)
    
    # query that should fail threshold
    match = router("what is the weather today?")
    assert match is None
    
    # hot reload new utterance
    router.add_utterance("support", "I need assistance")
    
    # query should now match exactly
    match = router("I need assistance")
    assert match is not None
    assert match.name == "support"
    
    # Verify the route object has the new utterance
    assert "I need assistance" in match.utterances


def test_add_utterance_unknown_route(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    with pytest.raises(RouteNotFoundError):
        router.add_utterance("unknown_route", "test utterance")

def test_top_1_masking_fallback(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    
    route1 = Route(name="strict_route", utterances=["exactly this"], threshold=0.99)
    route2 = Route(name="fallback_route", utterances=["exactly this"], threshold=0.1)
    
    router.add_route(route1)
    router.add_route(route2)
    
    # "almost exactly this" will have high similarity but not 0.99
    match = router("almost exactly this")
    assert match is not None
    assert match.name == "fallback_route"

def test_max_capacity_add_route(storage, encoder):
    router = AdaptiveRouter(encoder, storage, max_capacity=2)
    route = Route(name="r1", utterances=["u1", "u2", "u3"], threshold=0.5)
    with pytest.raises(RouterCapacityError):
        router.add_route(route)

def test_max_capacity_add_utterance(storage, encoder):
    router = AdaptiveRouter(encoder, storage, max_capacity=2)
    route = Route(name="r1", utterances=["u1", "u2"], threshold=0.5)
    router.add_route(route)
    with pytest.raises(RouterCapacityError):
        router.add_utterance("r1", "u3")

def test_delete_route_memory(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    route1 = Route(name="r1", utterances=["u1", "u2"], threshold=0.5)
    route2 = Route(name="r2", utterances=["u3"], threshold=0.5)
    router.add_route(route1)
    router.add_route(route2)
    router._flush_storage_batch()
    assert router.index.total_vectors == 3
    
    router.delete_route("r1")
    router._flush_storage_batch()
    # FAISS remove_ids does shrink total_vectors in IndexFlatIP
    assert router.index.total_vectors == 1
    
    # Inference should ignore deleted route
    match = router("u1")
    if match is not None:
        assert match.name != "r1"

def test_duplicate_utterance_ignored(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    route = Route(name="r1", utterances=["u1"], threshold=0.5)
    router.add_route(route)
    router._flush_storage_batch()
    assert router.index.total_vectors == 1
    
    # add same utterance again
    router.add_utterance("r1", "u1")
    router._flush_storage_batch()
    assert router.index.total_vectors == 1
    assert len(router._route_map["r1"].utterances) == 1

def test_max_capacity_load_routes(temp_db, encoder):
    storage1 = SQLiteStorage(temp_db)
    router1 = AdaptiveRouter(encoder, storage1, max_capacity=10)
    route = Route(name="r1", utterances=["u1", "u2", "u3"], threshold=0.5)
    receipt = router1.add_route(route)
    receipt.wait_durable(timeout=5.0)
    router1.close()
    storage1.close()

    storage2 = SQLiteStorage(temp_db)
    try:
        with pytest.raises(RouterCapacityError):
            AdaptiveRouter(encoder, storage2, max_capacity=2)
    finally:
        storage2.close()

def test_overwrite_route_capacity(storage, encoder):
    router = AdaptiveRouter(encoder, storage, max_capacity=2)
    route1 = Route(name="r1", utterances=["u1"], threshold=0.5)
    router.add_route(route1)
    
    # overwrite with 3 utterances
    route1_new = Route(name="r1", utterances=["u1", "u2", "u3"], threshold=0.5)
    with pytest.raises(RouterCapacityError):
        router.add_route(route1_new)

def test_delete_nonexistent_route(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    # Should not raise
    router.delete_route("nonexistent")

def test_fit_thresholds_mismatched_lengths(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    with pytest.raises(ValueError):
        router.fit_thresholds(["q1"], ["l1", "l2"])

def test_zero_state_inference(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    assert router.index.total_vectors == 0
    assert router("hello") is None
    
    # Should return early
    router.fit_thresholds(["q1"], ["l1"])

    # Removed test_add_route_overwrite_rollback_on_index_failure because
    # FAISS insertion is now asynchronous via a background batch queue,
    # making synchronous rollback impossible and obsolete.

def test_load_routes_discards_wrong_dimension_blob(temp_db, encoder):
    import numpy as np
    storage = SQLiteStorage(temp_db)
    route = Route(name="bad_blob", utterances=["test 1"], threshold=0.5)
    np.zeros(128, dtype=np.float32).tobytes()
    
    # Need to manually insert bad blob since save_route might not allow mismatch if it checks
    # But storage.save_route doesn't check dimensions, it just blindly saves.
    # Actually wait, save_route takes a list of embeddings.
    # The current save_route takes np arrays or anything with tobytes(), but wait, save_route expects e.tobytes().
    # So we pass an array of length 128
    storage.save_route(route, [np.zeros(128, dtype=np.float32)])
    
    # Boot new router
    router = AdaptiveRouter(encoder, storage)
    assert "bad_blob" in router._route_map
    # Query should work, meaning it was re-encoded
    assert router("test 1") is not None
    assert router("test 1").name == "bad_blob"
