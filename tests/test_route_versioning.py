import pytest
from synaptoroute import Route, SQLiteStorage

def test_route_version_default_and_validation():
    r1 = Route(name="support", utterances=["help me"])
    assert r1.version == 1

    r2 = Route(name="billing", utterances=["pay bill"], version=3)
    assert r2.version == 3

    with pytest.raises(ValueError):
        Route(name="invalid", utterances=["test"], version=0)

def test_sqlite_storage_preserves_route_version(tmp_path):
    db_file = str(tmp_path / "test_version.sqlite3")
    storage = SQLiteStorage(db_path=db_file)

    route = Route(name="versioned_intent", utterances=["query v1"], version=5)
    storage.save_route(route)

    loaded_routes, _ = storage.load_all_routes()
    assert len(loaded_routes) == 1
    assert loaded_routes[0].name == "versioned_intent"
    assert loaded_routes[0].version == 5
