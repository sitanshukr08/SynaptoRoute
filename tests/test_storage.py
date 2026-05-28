import pytest
from synaptoroute.models import Route
from synaptoroute.storage import SQLiteStorage

@pytest.fixture
def memory_db():
    storage = SQLiteStorage(":memory:")
    return storage

def test_save_and_load_route(memory_db):
    route1 = Route(
        name="test_route_1",
        utterances=["hello", "hi there"],
        threshold=0.8,
        metadata={"category": "greeting"}
    )
    
    route2 = Route(
        name="test_route_2",
        utterances=["bye", "see you"],
        threshold=0.7,
        metadata=None
    )

    memory_db.save_route(route1)
    memory_db.save_route(route2)

    routes, _ = memory_db.load_all_routes()
    assert len(routes) == 2

    # Map by name for easy checking
    routes_by_name = {r.name: r for r in routes}

    assert "test_route_1" in routes_by_name
    r1 = routes_by_name["test_route_1"]
    assert r1.threshold == 0.8
    assert r1.metadata == {"category": "greeting"}
    assert set(r1.utterances) == {"hello", "hi there"}

    assert "test_route_2" in routes_by_name
    r2 = routes_by_name["test_route_2"]
    assert r2.threshold == 0.7
    assert r2.metadata is None
    assert set(r2.utterances) == {"bye", "see you"}

def test_add_utterance(memory_db):
    route = Route(
        name="dynamic_route",
        utterances=["start"],
        threshold=0.5
    )
    
    memory_db.save_route(route)
    
    # Add new utterances dynamically
    memory_db.add_utterance("dynamic_route", "new utterance 1")
    memory_db.add_utterance("dynamic_route", "new utterance 2")

    routes, _ = memory_db.load_all_routes()
    assert len(routes) == 1
    loaded_route = routes[0]

    assert loaded_route.name == "dynamic_route"
    assert set(loaded_route.utterances) == {"start", "new utterance 1", "new utterance 2"}

def test_save_route_replace(memory_db):
    route = Route(
        name="replaceable",
        utterances=["v1"],
        threshold=0.5
    )
    memory_db.save_route(route)

    # Save again with same name but new data
    route_v2 = Route(
        name="replaceable",
        utterances=["v2", "v3"],
        threshold=0.9,
        metadata={"version": 2}
    )
    memory_db.save_route(route_v2)

    routes, _ = memory_db.load_all_routes()
    assert len(routes) == 1
    loaded_route = routes[0]

    assert loaded_route.threshold == 0.9
    assert loaded_route.metadata == {"version": 2}
    assert set(loaded_route.utterances) == {"v2", "v3"}
