import pytest
import asyncio
import threading
import time
from synaptoroute.router import AdaptiveRouter
from synaptoroute.models import Route
from synaptoroute.storage import SQLiteStorage
from synaptoroute.exceptions import RouterOverloadedError

@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "test_async.db")

@pytest.fixture
def storage(temp_db):
    return SQLiteStorage(db_path=temp_db)


@pytest.mark.asyncio
async def test_aquery(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    route1 = Route(name="greeting", utterances=["hello", "hi there"], threshold=0.99)
    router.add_route(route1)
    
    await router.start()
    try:
        match = await router.aquery("hello")
        assert match is not None
        assert match.name == "greeting"
        
        match = await router.aquery("completely unrelated random text about space and planets")
        assert match is None
    finally:
        await router.stop()

@pytest.mark.asyncio
async def test_aquery_overload(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    await router.start()
    
    # Mock the queue to pretend it's full
    def mock_put(*args, **kwargs):
        raise asyncio.QueueFull()
        
    router._batch_queue.put_nowait = mock_put
    
    with pytest.raises(RouterOverloadedError, match="Router queue is full"):
        await router.aquery("test query")
        
    await router.stop()

@pytest.mark.asyncio
async def test_batch_worker_shutdown(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    await router.start()

    # Send a query but immediately stop the router.
    task = asyncio.create_task(router.aquery("hello"))
    
    # Yield control to let aquery put the future in the queue
    await asyncio.sleep(0.01)
    
    await router.stop()

    await task


@pytest.mark.asyncio
async def test_repeated_start_and_stop_are_idempotent(storage, fake_encoder):
    router = AdaptiveRouter(fake_encoder, storage)

    await router.start()
    first_worker = router._worker_task
    await router.start()
    assert router._worker_task is first_worker

    await router.stop()
    await router.stop()
    with pytest.raises(RuntimeError, match="closed"):
        await router.start()


@pytest.mark.asyncio
async def test_real_backpressure_bounds_inflight_batches(storage, fake_encoder, monkeypatch):
    router = AdaptiveRouter(
        fake_encoder,
        storage,
        max_queue_size=2,
        max_in_flight_batches=1,
    )
    router.batch_size = 1
    router.add_route(Route(name="greeting", utterances=["hello"], threshold=0.5))

    original_encode_batch = fake_encoder.encode_batch
    lock = threading.Lock()
    active = 0
    max_active = 0

    def slow_encode_batch(texts):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        try:
            return original_encode_batch(texts)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(fake_encoder, "encode_batch", slow_encode_batch)
    await router.start()
    try:
        results = await asyncio.gather(
            *(router.amatch("hello") for _ in range(30)),
            return_exceptions=True,
        )
    finally:
        await router.stop()

    overloaded = [result for result in results if isinstance(result, RouterOverloadedError)]
    successful = [result for result in results if not isinstance(result, Exception)]
    unexpected = [
        result
        for result in results
        if isinstance(result, Exception) and not isinstance(result, RouterOverloadedError)
    ]
    assert overloaded
    assert successful
    assert unexpected == []
    assert max_active == 1

@pytest.mark.asyncio
async def test_aquery_without_start(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    with pytest.raises(RuntimeError, match="Router must be started"):
        await router.aquery("hello")

@pytest.mark.asyncio
async def test_async_mutation_wrappers(storage, fake_encoder):
    router = AdaptiveRouter(fake_encoder, storage)

    await router.aadd_route(Route(name="async_route", utterances=["hello"], threshold=0.5))
    await router.aadd_utterance("async_route", "hi")
    await router.aupdate_threshold("async_route", 0.4)
    await router.adelete_route("async_route")

    assert router("hello") is None

@pytest.mark.asyncio
async def test_aquery_worker_crashed(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    await router.start()
    
    # Manually cancel the worker
    router._worker_task.cancel()
    try:
        await router._worker_task
    except asyncio.CancelledError:
        pass
        
    with pytest.raises(RuntimeError, match="Router worker has crashed or stopped"):
        await router.aquery("hello")

@pytest.mark.asyncio
async def test_aquery_raises_if_worker_crashes_while_pending(storage, encoder, monkeypatch):
    router = AdaptiveRouter(encoder, storage)
    await router.start()

    # Force the worker to crash when it tries to encode
    def crash_encode(*args, **kwargs):
        raise ValueError("ONNX Engine Exploded!")
        
    monkeypatch.setattr(encoder, "encode_batch", crash_encode)

    # When it crashes, the worker's except block sets the exception on the future
    with pytest.raises(ValueError, match="ONNX Engine Exploded!"):
        await router.aquery("test query")
