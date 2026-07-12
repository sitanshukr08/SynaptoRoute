import pytest

from synaptoroute import (
    AdaptiveRouter,
    Route,
    SQLiteStorage,
    StorageFlushError,
    StorageMutationError,
)


class FailingSaveStorage(SQLiteStorage):
    def save_route(self, route, embeddings=None):
        raise RuntimeError("forced durable write failure")


def test_mutation_receipt_reaches_durable_state_and_survives_restart(tmp_path, fake_encoder):
    database = tmp_path / "routes.db"
    router = AdaptiveRouter(fake_encoder, SQLiteStorage(str(database)))

    receipt = router.add_route(Route(name="support", utterances=["help"]))
    latency_ms = receipt.wait_durable(timeout=2.0)

    assert receipt.state == "durable"
    assert latency_ms >= 0.0
    router.close()

    restarted = AdaptiveRouter(fake_encoder, SQLiteStorage(str(database)))
    assert "support" in restarted._route_map
    restarted.close()


def test_failed_receipt_is_observable_and_barrier_reports_failure(fake_encoder):
    router = AdaptiveRouter(fake_encoder, FailingSaveStorage(":memory:"))
    receipt = router.add_route(Route(name="bad", utterances=["broken"]))

    with pytest.raises(StorageMutationError, match="forced durable write failure"):
        receipt.wait_durable(timeout=2.0)

    assert receipt.state == "failed"
    assert "bad" not in router._route_map
    with pytest.raises(StorageFlushError) as error:
        router.durable_barrier(timeout=2.0)
    assert error.value.failures[0][0] == receipt.sequence

    router.durable_barrier(timeout=2.0)
    router.close()


def test_receipt_sequences_preserve_enqueue_order(tmp_path, fake_encoder):
    router = AdaptiveRouter(fake_encoder, SQLiteStorage(str(tmp_path / "ordered.db")))

    added = router.add_route(Route(name="support", utterances=["help"], threshold=0.5))
    threshold = router.update_threshold("support", 0.8)
    utterance = router.add_utterance("support", "assistance")
    utterance.wait_durable(timeout=2.0)

    assert [added.sequence, threshold.sequence, utterance.sequence] == [1, 2, 3]
    assert added.state == threshold.state == utterance.state == "durable"
    router.durable_barrier(timeout=2.0)
    router.close()
