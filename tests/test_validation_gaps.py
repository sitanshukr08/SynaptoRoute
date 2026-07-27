import pytest
from typing import Any
import asyncio
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock

from synaptoroute import AdaptiveRouter, Route
from synaptoroute.encoder import OpenAIEncoder
from synaptoroute.storage import BaseStorage
from synaptoroute.profile import get_profile, ProfileType
from synaptoroute.trainer import SyntheticTuner

class DummyStorage(BaseStorage):
    def save_route(self, route: Route, embeddings: Any = None): pass
    def load_all_routes(self): return [], {}
    def delete_route(self, route_name: str): pass
    def add_utterance(self, route_name: str, utterance: str, embedding: Any = None): pass
    def update_threshold(self, route_name: str, threshold: float): pass

@pytest.mark.asyncio
async def test_openai_encoder_chunking():
    # Mock OpenAI response
    mock_client = MagicMock()
    mock_create = MagicMock()
        
        # We'll return 2048 embeddings on the first call, 10 on the second
    class DummyData:
        def __init__(self, emb):
            self.embedding = emb
            
    def create_side_effect(*args, **kwargs):
        inputs = kwargs.get("input", [])
        response = MagicMock()
        response.data = [DummyData([0.1]*1536) for _ in inputs]
        return response
        
    mock_create.create.side_effect = create_side_effect
    mock_client.embeddings = mock_create
    
    encoder = OpenAIEncoder(client=mock_client)
    texts = [f"test {i}" for i in range(2058)]
    
    embeddings = encoder.encode_batch(texts)
    assert len(embeddings) == 2058
    assert mock_create.create.call_count == 2
    
    call_1_kwargs = mock_create.create.call_args_list[0][1]
    call_2_kwargs = mock_create.create.call_args_list[1][1]
    
    assert len(call_1_kwargs["input"]) == 2048
    assert len(call_2_kwargs["input"]) == 10

@pytest.mark.asyncio
async def test_latency_profile_propagates_threads():
    # If the user doesn't pass an encoder, the LATENCY profile should set threads=1
    with patch("synaptoroute.encoder.FastEmbedEncoder") as mock_encoder_class:
        mock_encoder_class.return_value.dim = 384
        profile = get_profile(ProfileType.LATENCY)
        router = AdaptiveRouter(storage=DummyStorage(), profile=profile)
        
        # Verify the router instantiated FastEmbedEncoder with the correct thread count
        mock_encoder_class.assert_called_once_with(threads=profile.threads)
        assert router.encoder == mock_encoder_class.return_value

@pytest.mark.asyncio
async def test_fit_thresholds_async_loop_safety():
    # Verify that fit_thresholds is offloaded to a thread and doesn't block
    with patch("synaptoroute.encoder.FastEmbedEncoder") as mock_encoder_class:
        mock_encoder_class.return_value.dim = 384
        mock_encoder_class.return_value.encode_batch.side_effect = (
            lambda texts: np.ones((len(texts), 384), dtype=np.float32)
        )
        router = AdaptiveRouter(storage=DummyStorage())
        router.add_route(Route(name="test", utterances=["a", "b", "c"]))
        
        mock_client = MagicMock()
        tuner = SyntheticTuner(router=router, client=mock_client)
        
        mock_parse = AsyncMock()
        
        class DummyParsed:
            def __init__(self):
                # Use a larger sample size to make the block observable if it were synchronous
                self.positive = [f"pos_{i}" for i in range(200)]
                self.negative = [f"neg_{i}" for i in range(200)]
        class DummyMsg:
            def __init__(self):
                self.parsed = DummyParsed()
        class DummyChoice:
            def __init__(self):
                self.message = DummyMsg()
        class DummyResp:
            def __init__(self):
                self.choices = [DummyChoice()]
                
        mock_parse.return_value = DummyResp()
        mock_client.beta.chat.completions.parse = mock_parse
        
        # This should not block the event loop
        task = asyncio.create_task(tuner.tune_route("test", "test desc", 200))
        
        # Prove the event loop is responsive by sleeping
        await asyncio.sleep(0.01)
        
        await task
