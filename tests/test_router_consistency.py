import threading

import pytest

from synaptoroute.durability import MutationReceipt, QueuedStorageMutation
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

    def save_route(self, route, embeddings=None, expected_version=None):
        if self.fail_save_route:
            raise RuntimeError("forced save_route failure")
        return super().save_route(route, embeddings, expected_version)

    def add_utterance(self, route_name: str, utterance: str, embedding=None, version=None):
        if self.fail_add_utterance:
            raise RuntimeError("forced add_utterance failure")
        return super().add_utterance(route_name, utterance, embedding, version)

    def update_threshold(self, route_name: str, threshold: float, version=None):
        if self.fail_update_threshold:
            raise RuntimeError("forced update_threshold failure")
        return super().update_threshold(route_name, threshold, version)

    def delete_route(self, route_name: str, expected_version=None):
        if self.fail_delete_route:
            raise RuntimeError("forced delete_route failure")
        return super().delete_route(route_name, expected_version)


class BlockingSaveStorage(SQLiteStorage):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.started = threading.Event()
        self.release = threading.Event()

    def save_route(self, route, embeddings=None, expected_version=None):
        self.started.set()
        self.release.wait(timeout=5.0)
        return super().save_route(route, embeddings, expected_version)


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


@pytest.mark.asyncio
async def test_rebuild_failure_retries_then_clears_pending_state(fake_encoder):
    from unittest.mock import patch
    storage = SQLiteStorage(":memory:")
    router = AdaptiveRouter(
        fake_encoder,
        storage,
        max_rebuild_retries=2,
        index_engine="numpy",
    )
    router.add_route(Route(name="r1", utterances=["utterance1"], threshold=0.5))
    router.durable_barrier()

    router._rebuild_pending = True
    with patch(
        "synaptoroute.index.NumpyIndex.rebuild",
        side_effect=RuntimeError("rebuild failed"),
    ) as rebuild:
        await router._rebuild_index()

    assert router._rebuild_pending is False
    assert rebuild.call_count == 2
    router.close()


def test_route_resync_does_not_replace_unrelated_memory_state(fake_encoder):
    storage = SQLiteStorage(":memory:")
    router = AdaptiveRouter(fake_encoder, storage)
    router.add_route(Route(name="r1", utterances=["utterance1"], threshold=0.5))
    router.add_route(Route(name="r2", utterances=["utterance2"], threshold=0.5))
    router.durable_barrier()

    router._route_map["r2"].threshold = 0.9
    router._resync_route_from_storage("r1")

    assert router._route_map["r2"].threshold == 0.9
    router.close()


def test_failed_older_mutation_cannot_resync_over_newer_memory_version(fake_encoder):
    storage = SQLiteStorage(":memory:")
    router = AdaptiveRouter(fake_encoder, storage)
    router.add_route(Route(name="r1", utterances=["hello"], threshold=0.5))
    router.durable_barrier(timeout=2.0)
    with router._route_map_lock:
        router._route_map["r1"] = router._route_map["r1"].model_copy(
            update={"threshold": 0.9, "version": 3}
        )
        router._route_versions["r1"] = 3

    stale = QueuedStorageMutation(
        action="update_threshold",
        args=("r1", 0.7, 2),
        receipt=MutationReceipt(
            sequence=2,
            action="update_threshold",
            route_name="r1",
            route_version=2,
        ),
    )
    router._reconcile_failed_mutation(stale)

    assert router._route_map["r1"].version == 3
    assert router._route_map["r1"].threshold == 0.9
    router.close()
    storage.close()


def test_durable_barrier_accumulates_multiple_storage_failures(failable_storage, fake_encoder):
    from synaptoroute.exceptions import StorageFlushError
    router = AdaptiveRouter(fake_encoder, failable_storage)
    router.add_route(Route(name="r1", utterances=["hello"], threshold=0.5))
    router._flush_storage_batch()

    failable_storage.fail_add_utterance = True
    router.add_utterance("r1", "u1")
    router.add_utterance("r1", "u2")

    router._flush_storage_batch()

    with pytest.raises(StorageFlushError) as exc_info:
        router.durable_barrier()

    assert len(exc_info.value.failures) == 2
    assert "2 storage mutation(s) failed" in str(exc_info.value)
    router.close()


def test_full_storage_queue_rejects_without_partial_memory_application(fake_encoder):
    from synaptoroute.exceptions import RouterOverloadedError

    storage = BlockingSaveStorage(":memory:")
    router = AdaptiveRouter(
        fake_encoder,
        storage,
        max_storage_queue_size=1,
    )
    first = router.add_route(Route(name="first", utterances=["hello"]))
    assert storage.started.wait(timeout=2.0)
    second = router.add_route(Route(name="second", utterances=["bye"]))

    with pytest.raises(RouterOverloadedError, match="Storage mutation queue is full"):
        router.add_route(Route(name="rejected", utterances=["support"]))

    assert "rejected" not in router._route_map
    storage.release.set()
    first.wait_durable(timeout=2.0)
    second.wait_durable(timeout=2.0)
    router.close()


def test_numpy_index_tombstone_compaction_reclaims_capacity():
    import numpy as np
    from synaptoroute.index import NumpyIndex

    # Capacity 5, add 5 vectors
    idx = NumpyIndex(dim=2, max_capacity=5)
    embs = np.ones((5, 2), dtype=np.float32)
    idx.add(embs[:3], "route1")
    idx.add(embs[3:], "route2")

    assert idx._next_id == 5
    assert idx.total_vectors == 5

    # Delete route1 (tombstoning 3 vectors)
    idx.delete("route1")
    assert idx.total_vectors == 2
    assert len(idx.tombstones) == 3

    # Add 2 new vectors for route3: should compact tombstones instead of throwing ID_OVERFLOW
    new_embs = np.ones((2, 2), dtype=np.float32)
    idx.add(new_embs, "route3")

    assert idx.total_vectors == 4
    assert len(idx.tombstones) == 0
    assert idx._next_id == 4


def test_delete_route_index_failure_rolls_back_in_memory_state(fake_encoder):
    from unittest.mock import patch
    storage = SQLiteStorage(":memory:")
    router = AdaptiveRouter(fake_encoder, storage)
    router.add_route(Route(name="support", utterances=["help"], threshold=0.5))
    router._flush_storage_batch()

    with patch.object(router.index, "delete", side_effect=RuntimeError("forced index delete error")):
        with pytest.raises(RuntimeError, match="forced index delete error"):
            router.delete_route("support")

    # Assert route was restored in memory map after index deletion failure
    assert "support" in router._route_map
    assert router._route_map["support"].utterances == ["help"]
    router.close()
