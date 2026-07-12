import pytest

from synaptoroute.models import Route
from synaptoroute.router import AdaptiveRouter
from synaptoroute.storage import SQLiteStorage


class FailableSQLiteStorage(SQLiteStorage):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.fail_save_route = False
        self.fail_add_utterance = False
        self.fail_update_threshold = False
        self.fail_delete_route = False

    def save_route(self, route, embeddings=None):
        if self.fail_save_route:
            raise RuntimeError("forced save_route failure")
        return super().save_route(route, embeddings)

    def add_utterance(self, route_name: str, utterance: str, embedding=None):
        if self.fail_add_utterance:
            raise RuntimeError("forced add_utterance failure")
        return super().add_utterance(route_name, utterance, embedding)

    def update_threshold(self, route_name: str, threshold: float):
        if self.fail_update_threshold:
            raise RuntimeError("forced update_threshold failure")
        return super().update_threshold(route_name, threshold)

    def delete_route(self, route_name: str):
        if self.fail_delete_route:
            raise RuntimeError("forced delete_route failure")
        return super().delete_route(route_name)


@pytest.fixture
def failable_storage():
    return FailableSQLiteStorage(":memory:")


def test_failed_add_route_resyncs_memory_and_index(failable_storage, fake_encoder):
    router = AdaptiveRouter(fake_encoder, failable_storage)
    failable_storage.fail_save_route = True

    router.add_route(Route(name="bad", utterances=["hello"], threshold=0.5))
    router._flush_storage_batch()

    assert "bad" not in router._route_map
    assert router.index.total_vectors == 0
    assert failable_storage.load_all_routes()[0] == []
    router.close()


def test_failed_add_utterance_resyncs_to_persisted_route(failable_storage, fake_encoder):
    router = AdaptiveRouter(fake_encoder, failable_storage)
    router.add_route(Route(name="support", utterances=["help"], threshold=0.5))
    router._flush_storage_batch()

    failable_storage.fail_add_utterance = True
    router.add_utterance("support", "assistance")
    router._flush_storage_batch()

    assert router._route_map["support"].utterances == ["help"]
    assert router.index.total_vectors == 1
    routes, _ = failable_storage.load_all_routes()
    assert routes[0].utterances == ["help"]
    router.close()


def test_failed_threshold_update_resyncs_old_threshold(failable_storage, fake_encoder):
    router = AdaptiveRouter(fake_encoder, failable_storage)
    router.add_route(Route(name="support", utterances=["help"], threshold=0.5))
    router._flush_storage_batch()

    failable_storage.fail_update_threshold = True
    router.update_threshold("support", 0.9)
    router._flush_storage_batch()

    assert router._route_map["support"].threshold == 0.5
    routes, _ = failable_storage.load_all_routes()
    assert routes[0].threshold == 0.5
    router.close()


def test_failed_delete_route_resyncs_persisted_route(failable_storage, fake_encoder):
    router = AdaptiveRouter(fake_encoder, failable_storage)
    router.add_route(Route(name="support", utterances=["help"], threshold=0.5))
    router._flush_storage_batch()

    failable_storage.fail_delete_route = True
    router.delete_route("support")
    router._flush_storage_batch()

    assert "support" in router._route_map
    assert router.index.total_vectors == 1
    routes, _ = failable_storage.load_all_routes()
    assert [route.name for route in routes] == ["support"]
    router.close()


@pytest.mark.asyncio
async def test_stop_flushes_queued_storage_writes(fake_encoder):
    storage = SQLiteStorage(":memory:")
    router = AdaptiveRouter(fake_encoder, storage)

    router.add_route(Route(name="support", utterances=["help"], threshold=0.5))
    await router.stop()

    routes, _ = storage.load_all_routes()
    assert [route.name for route in routes] == ["support"]


@pytest.mark.asyncio
async def test_sync_and_async_paths_apply_the_same_margin_policy(fake_encoder):
    storage = SQLiteStorage(":memory:")
    router = AdaptiveRouter(fake_encoder, storage, margin=0.1)
    router.add_route(Route(name="first", utterances=["hello"], threshold=0.5))
    router.add_route(Route(name="second", utterances=["hi"], threshold=0.5))

    assert router("hello") is None

    await router.start()
    assert await router.aquery("hello") is None
    await router.stop()
