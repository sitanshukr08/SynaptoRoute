import pytest
import asyncio
import threading
import time
from synaptoroute.router import AdaptiveRouter
from synaptoroute.models import Route
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.exceptions import RouteNotFoundError

@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "test_concurrency.db")

@pytest.fixture
def storage(temp_db):
    return SQLiteStorage(db_path=temp_db)

@pytest.fixture
def encoder():
    return Encoder(model_name="BAAI/bge-small-en-v1.5")

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
    
    with pytest.raises(RouteNotFoundError, match="deleted during encoding"):
        router.add_utterance("r1", "new utterance")
        
    t.join()
