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

def test_delete_route_storage(memory_db):
    route = Route(name="r1", utterances=["u1", "u2"], threshold=0.5)
    memory_db.save_route(route)
    memory_db.delete_route("r1")
    
    routes, _ = memory_db.load_all_routes()
    assert len(routes) == 0
    
    # Verify utterances were cascade deleted
    with memory_db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM utterances WHERE route_name = 'r1'")
        assert cursor.fetchone()[0] == 0

def test_corrupt_json_metadata(memory_db):
    route = Route(name="r1", utterances=["u1"], threshold=0.5)
    memory_db.save_route(route)
    
    with memory_db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE routes SET metadata = '{corrupt_json}' WHERE name = 'r1'")
        conn.commit()
        
    routes, _ = memory_db.load_all_routes()
    assert len(routes) == 1
    assert routes[0].metadata is None

def test_sqlite_storage_creates_directory(tmp_path):
    nested_path = tmp_path / "nested" / "dir" / "routes.db"
    assert not nested_path.parent.exists()
    
    storage = SQLiteStorage(str(nested_path))
    assert nested_path.parent.exists()
    
    # Cleanup connection to avoid file locking on Windows
    del storage

    # Removed test_fit_thresholds_db_error because update_threshold is now
    # asynchronous and processed by the batch worker, so it cannot raise
    # synchronous exceptions to the caller.
