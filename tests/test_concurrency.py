import pytest
import asyncio
import concurrent.futures
import threading
import time
from synaptoroute.router import AdaptiveRouter
from synaptoroute.models import Route
from synaptoroute.storage import SQLiteStorage
from synaptoroute.exceptions import RouteNotFoundError

@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "test_concurrency.db")

@pytest.fixture
def storage(temp_db):
    return SQLiteStorage(db_path=temp_db)


@pytest.mark.asyncio
async def test_encoder_lock_concurrency(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    await router.start()
    
    async def add_routes():
        def sync_add_routes():
            for i in range(50):
                route = Route(name=f"route_{i}", utterances=[f"hello {i}", f"test {i}"], threshold=0.5)
                router.add_route(route)
        await asyncio.to_thread(sync_add_routes)
        
    async def run_queries():
        tasks = []
        for i in range(500):
            tasks.append(router.aquery(f"test query {i}"))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                raise res

    try:
        await asyncio.gather(add_routes(), run_queries())
    finally:
        await router.stop()

def test_add_utterance_route_deleted_during_encoding(storage, encoder, monkeypatch):
    router = AdaptiveRouter(encoder, storage)
    route = Route(name="r1", utterances=["start"], threshold=0.5)
    router.add_route(route)
    
    original_encode = encoder.encode
    encode_started = threading.Event()
    
    def slow_encode(text):
        encode_started.set()
        time.sleep(0.1) # Simulate slow encoding
        return original_encode(text)
        
    monkeypatch.setattr(encoder, "encode", slow_encode)
    
    def delete_while_encoding():
        encode_started.wait()
        router.delete_route("r1")
        
    t = threading.Thread(target=delete_while_encoding)
    t.start()
    
    with pytest.raises(RouteNotFoundError, match="not found"):
        router.add_utterance("r1", "new utterance")
        
    t.join()

@pytest.mark.asyncio
async def test_rebuild_replays_pending_mutations(storage, fake_encoder):
    from unittest.mock import patch

    router = AdaptiveRouter(fake_encoder, storage)
    await router.start()
    router.add_route(Route(name="r1", utterances=["hello"], threshold=0.5))
    router.durable_barrier()
    original_rebuild = type(router.index).rebuild
    rebuild_calls = 0

    def mutate_during_first_rebuild(index, route_map, embeddings_map):
        nonlocal rebuild_calls
        rebuild_calls += 1
        original_rebuild(index, route_map, embeddings_map)
        if rebuild_calls == 1:
            receipt = router.add_route(
                Route(name="r2", utterances=["bye"], threshold=0.5)
            )
            receipt.wait_durable(timeout=2.0)

    router._rebuild_pending = True

    try:
        with patch.object(type(router.index), "rebuild", mutate_during_first_rebuild):
            await router._rebuild_index()
        pending_embeddings = fake_encoder.encode_batch(["bye"])
        results = router.index.search(pending_embeddings, top_k=1)
        assert results[0][0][1] == "r2"
        assert rebuild_calls >= 2
        assert router._rebuild_pending is False
    finally:
        await router.stop()


def test_query_cannot_observe_route_replacement_mid_index_update(
    fake_encoder,
    monkeypatch,
):
    storage = SQLiteStorage(":memory:")
    router = AdaptiveRouter(fake_encoder, storage)
    router.add_route(Route(name="support", utterances=["hello"], threshold=0.5))
    router.durable_barrier(timeout=2.0)

    entered = threading.Event()
    release = threading.Event()
    original_add = router.index.add

    def blocking_add(embeddings, route_name):
        if route_name == "support":
            entered.set()
            release.wait(timeout=2.0)
        return original_add(embeddings, route_name)

    monkeypatch.setattr(router.index, "add", blocking_add)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        mutation = executor.submit(
            router.add_route,
            Route(name="support", utterances=["bye"], threshold=0.5),
        )
        assert entered.wait(timeout=2.0)
        query = executor.submit(router.match, "hello")
        with pytest.raises(concurrent.futures.TimeoutError):
            query.result(timeout=0.05)
        release.set()
        mutation.result(timeout=2.0).wait_durable(timeout=2.0)
        result = query.result(timeout=2.0)

    assert result.route_name != "support"
    router.close()
    storage.close()
