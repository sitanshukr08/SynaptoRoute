import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

# Attempt to import RedisSyncManager. We assume the Editor subagent will implement this.
try:
    from synaptoroute.sync import RedisSyncManager
except ImportError:
    RedisSyncManager = None

@pytest.fixture
def mock_router():
    """Mock router that will record any method calls made to it by the sync manager."""
    return MagicMock()

@pytest.mark.asyncio
async def test_broadcast_sends_message(mock_router):
    if RedisSyncManager is None:
        pytest.skip("RedisSyncManager not yet implemented")

    with patch('synaptoroute.sync.redis.from_url') as mock_redis_from_url:
        mock_redis = MagicMock()
        mock_redis.aclose = AsyncMock()
        mock_pubsub = MagicMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_redis.publish = AsyncMock()
        mock_pubsub.close = AsyncMock()
        
        async def mock_listen():
            await asyncio.Event().wait()
            yield {}
        mock_pubsub.listen = mock_listen
        
        mock_redis.pubsub.return_value = mock_pubsub
        mock_redis_from_url.return_value = mock_redis
        manager = RedisSyncManager(redis_url="redis://localhost")
        mock_router.encoder = MagicMock()
        mock_router.encoder.model_name = "mock-model"
        manager.register(mock_router)
        await manager.start()
        
        payload = {"data": "test_data"}
        manager.broadcast("add_route", payload)
        
        # Yield to allow the background publisher loop to process the queue
        await asyncio.sleep(0.01)
        
        mock_redis.publish.assert_called()
        args, kwargs = mock_redis.publish.call_args
        
        # The first argument is typically the channel name
        channel = args[0]
        assert channel == getattr(manager, '_channel', 'synaptoroute:sync')
        
        # The second argument should contain our payload
        message_data = args[1]
        assert "test_data" in str(message_data)
        
        await manager.stop()

@pytest.mark.asyncio
async def test_listener_ignores_own_sender_id(mock_router):
    if RedisSyncManager is None:
        pytest.skip("RedisSyncManager not yet implemented")

    with patch('synaptoroute.sync.redis.from_url') as mock_redis_from_url:
        mock_redis = MagicMock()
        mock_redis.aclose = AsyncMock()
        mock_pubsub = MagicMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.get_message = AsyncMock()
        mock_pubsub.close = AsyncMock()
        mock_redis.pubsub.return_value = mock_pubsub
        mock_redis_from_url.return_value = mock_redis
        
        manager = RedisSyncManager(redis_url="redis://localhost")
        manager.register(mock_router)
        
        # Prepare a message from self, and a message from a foreign sender
        own_payload = {"sender_id": manager.sender_id, "action": "add_utterance", "payload": {"route_name": "r1", "utterance": "u1"}}
        foreign_payload = {"sender_id": "other-id", "action": "add_utterance", "payload": {"route_name": "r2", "utterance": "u2"}}
        
        own_message = {
            "type": "message",
            "data": json.dumps(own_payload).encode()
        }
        foreign_message = {
            "type": "message",
            "data": json.dumps(foreign_payload).encode()
        }
        
        async def mock_listen():
            yield own_message
            yield foreign_message
            await asyncio.Event().wait()
        
        mock_pubsub.listen = mock_listen
        
        await manager.start()
        
        # Yield to let the background listener loop process the pubsub messages
        await asyncio.sleep(0.05)
        
        # We don't know the exact method name the Editor will choose (e.g. apply_sync, handle_sync_message).
        # But we can assert that "sync_from_foreign" was passed to SOME method on the router,
        # and "sync_from_self" was completely ignored.
        calls = str(mock_router.mock_calls)
        
        assert "u1" not in calls, "Listener should have ignored its own sender_id"
        assert "u2" in calls, "Listener should have routed the foreign message to the router"
        
        await manager.stop()

def test_dispatch_rejects_mismatched_encoder_model():
    if RedisSyncManager is None:
        pytest.skip("RedisSyncManager not yet implemented")

    manager = RedisSyncManager(redis_url="redis://localhost")
    
    mock_router = MagicMock()
    mock_router.encoder = MagicMock()
    mock_router.encoder.model_name = "model-A"
    manager.register(mock_router)
    
    # Message with mismatched encoder
    data = {
        "action": "add_route",
        "encoder_model": "model-B",
        "payload": {"name": "test"}
    }
    
    manager._dispatch(data)
    
    # Should be rejected, router methods should not be called
    mock_router.add_route.assert_not_called()
    mock_router.add_utterance.assert_not_called()
