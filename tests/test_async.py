import pytest
import asyncio
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
    
    # Overload the queue. The queue size is 10000.
    # To flood it, we might need to block the worker temporarily or just blast it.
    # Blasting 15000 requests without awaiting them.
    # The queue will fill up and then aquery should raise RouterOverloadedError.
    
    overloaded = False
    # We need to use asyncio.gather or create_tasks to execute them concurrently
    # Because router.aquery is an async function, just calling it returns a coroutine.
    tasks = [asyncio.create_task(router.aquery("test query")) for _ in range(15000)]
    try:
        await asyncio.gather(*tasks)
    except RouterOverloadedError:
        overloaded = True
    
    # Let the worker process some or cancel
    await router.stop()
    # Wait for remaining futures to cancel
    for t in tasks:
        if not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        else:
            try:
                t.exception()
            except asyncio.CancelledError:
                pass
                
    assert overloaded, "RouterOverloadedError was not raised"

@pytest.mark.asyncio
async def test_batch_worker_shutdown(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    await router.start()
    
    # Send a query but immediately stop the router.
    # The future should be cancelled.
    task = asyncio.create_task(router.aquery("hello"))
    
    # Yield control to let aquery put the future in the queue
    await asyncio.sleep(0.01)
    
    await router.stop()
    
    with pytest.raises(RuntimeError, match="Router worker stopped before completing this query"):
        await task

@pytest.mark.asyncio
async def test_aquery_without_start(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    with pytest.raises(RuntimeError, match="Router must be started"):
        await router.aquery("hello")

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
async def test_aquery_raises_if_worker_crashes_while_pending(storage, encoder):
    router = AdaptiveRouter(encoder, storage)
    await router.start()

    async def mock_worker():
        # Keep worker alive for a moment, then crash it
        await asyncio.sleep(0.05)
        raise ValueError("Worker crashed!")

    router._worker_task.cancel()
    try:
        await router._worker_task
    except asyncio.CancelledError:
        pass
        
    router._worker_task = asyncio.create_task(mock_worker())

    with pytest.raises(RuntimeError, match="Router worker stopped before completing this query"):
        await asyncio.wait_for(router.aquery("test query"), timeout=2.0)
