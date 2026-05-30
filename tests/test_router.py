import pytest
import os
import sqlite3
from synaptoroute.router import AdaptiveRouter
from synaptoroute.models import Route
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    return str(db_path)

@pytest.fixture
def storage(temp_db):
    return SQLiteStorage(db_path=temp_db)

@pytest.fixture
def encoder():
    return Encoder(model_name="BAAI/bge-small-en-v1.5")

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

from synaptoroute.exceptions import RouteNotFoundError, RouterCapacityError

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
    assert router._cursor == 3
    
    router.delete_route("r1")
    assert router._cursor == 1
    # u3 should now be at index 0
    assert router._meta[0].name == "r2"
    
    # Inference should ignore deleted route
    match = router("u1")
    if match is not None:
        assert match.name != "r1"

def test_duplicate_utterance_ignored(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    route = Route(name="r1", utterances=["u1"], threshold=0.5)
    router.add_route(route)
    assert router._cursor == 1
    
    # add same utterance again
    router.add_utterance("r1", "u1")
    assert router._cursor == 1
    assert len(router._route_map["r1"].utterances) == 1

def test_max_capacity_load_routes(temp_db, encoder):
    storage1 = SQLiteStorage(temp_db)
    router1 = AdaptiveRouter(encoder, storage1, max_capacity=10)
    route = Route(name="r1", utterances=["u1", "u2", "u3"], threshold=0.5)
    router1.add_route(route)
    
    storage2 = SQLiteStorage(temp_db)
    with pytest.raises(RouterCapacityError):
        AdaptiveRouter(encoder, storage2, max_capacity=2)

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
    assert router._cursor == 0
    assert router("hello") is None
    
    # Should return early
    router.fit_thresholds(["q1"], ["l1"])
