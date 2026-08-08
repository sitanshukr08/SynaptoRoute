import pytest
from synaptoroute import AdaptiveRouter, Route, SQLiteStorage

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


def test_router_mutations_advance_and_persist_route_versions(tmp_path, fake_encoder):
    database = tmp_path / "route_versions.sqlite3"
    router = AdaptiveRouter(fake_encoder, SQLiteStorage(str(database)))

    added = router.add_route(Route(name="support", utterances=["help"]))
    utterance = router.add_utterance("support", "assist")
    threshold = router.update_threshold("support", 0.8)
    threshold.wait_durable(timeout=2.0)

    assert [added.route_version, utterance.route_version, threshold.route_version] == [1, 2, 3]
    assert router._route_map["support"].version == 3
    router.close()

    restarted = AdaptiveRouter(fake_encoder, SQLiteStorage(str(database)))
    assert restarted._route_map["support"].version == 3
    assert restarted._route_map["support"].utterances == ["help", "assist"]
    restarted.close()
