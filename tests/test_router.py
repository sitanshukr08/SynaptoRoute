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

from synaptoroute.exceptions import RouteNotFoundError

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
