import pytest
import sqlite3

from synaptoroute.exceptions import StorageVersionConflictError
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

def test_sqlite_connections_use_explicit_transactions(memory_db):
    with memory_db._get_connection() as conn:
        assert conn.isolation_level is None
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_schema_migrations_are_ordered_and_complete(memory_db):
    with memory_db._get_connection() as conn:
        versions = [
            row[0]
            for row in conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]
    assert versions == [1, 2, 3]


def test_legacy_schema_migrates_without_losing_routes(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE routes (
            name TEXT PRIMARY KEY,
            threshold REAL NOT NULL,
            metadata TEXT
        );
        CREATE TABLE utterances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            route_name TEXT NOT NULL,
            utterance TEXT NOT NULL,
            FOREIGN KEY(route_name) REFERENCES routes(name) ON DELETE CASCADE,
            UNIQUE(route_name, utterance)
        );
        INSERT INTO routes(name, threshold, metadata)
        VALUES ('legacy', 0.7, '{"source": "legacy"}');
        INSERT INTO utterances(route_name, utterance) VALUES ('legacy', 'first');
        INSERT INTO utterances(route_name, utterance) VALUES ('legacy', 'second');
        """
    )
    connection.commit()
    connection.close()

    storage = SQLiteStorage(str(database))
    routes, _ = storage.load_all_routes()

    assert routes == [
        Route(
            name="legacy",
            utterances=["first", "second"],
            threshold=0.7,
            version=1,
            metadata={"source": "legacy"},
        )
    ]
    storage.close()


def test_stale_route_replace_cannot_overwrite_newer_version(memory_db):
    memory_db.save_route(Route(name="versioned", utterances=["v1"], version=1))
    memory_db.save_route(
        Route(name="versioned", utterances=["v2"], version=2),
        expected_version=1,
    )

    with pytest.raises(StorageVersionConflictError):
        memory_db.save_route(
            Route(name="versioned", utterances=["stale"], version=2),
            expected_version=1,
        )

    route, _ = memory_db.load_route("versioned")
    assert route is not None
    assert route.version == 2
    assert route.utterances == ["v2"]


def test_utterance_order_is_deterministic(memory_db):
    route = Route(name="ordered", utterances=["third", "first", "second"])
    memory_db.save_route(route)

    loaded, _ = memory_db.load_all_routes()

    assert loaded[0].utterances == ["third", "first", "second"]

def test_delete_utterance_storage(memory_db):
    route = Route(name="dynamic_route", utterances=["start"], threshold=0.5)
    memory_db.save_route(route)
    memory_db.add_utterance("dynamic_route", "temporary")

    memory_db.delete_utterance("dynamic_route", "temporary")

    routes, _ = memory_db.load_all_routes()
    assert routes[0].utterances == ["start"]

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
