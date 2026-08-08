from typing import Any
import numpy as np
from unittest.mock import MagicMock

from synaptoroute.encoder import OpenAIEncoder
from synaptoroute.router import AdaptiveRouter
from synaptoroute.storage import BaseStorage
from synaptoroute.models import Route

def test_encoder_contract_with_network_free_fixture(encoder):
    embedding = encoder.encode("Hello world")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == encoder.dim
    assert encoder.dim > 0

    embeddings = encoder.encode_batch(["Hello world", "Another sentence"])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, encoder.dim)

class DummyStorage(BaseStorage):
    def save_route(self, route: Route, embeddings: Any = None, expected_version=None): pass
    def load_all_routes(self): return [], {}
    def delete_route(self, name: str): pass
    def add_utterance(self, route_name: str, utterance: str, embedding: Any = None): pass
    def update_threshold(self, name: str, threshold: float): pass
    def clear(self): pass

def test_openai_encoder_mocked():
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_response_single = MagicMock()
    mock_response_single.data = [MagicMock(embedding=[0.1] * 1536)]
    
    mock_response_batch = MagicMock()
    mock_response_batch.data = [
        MagicMock(embedding=[0.1] * 1536),
        MagicMock(embedding=[0.2] * 1536)
    ]
    
    def side_effect(input, model):
        if isinstance(input, str) or (isinstance(input, list) and len(input) == 1):
            return mock_response_single
        return mock_response_batch

    mock_client.embeddings.create.side_effect = side_effect

    encoder = OpenAIEncoder(client=mock_client)
    assert encoder.dim == 1536
    
    # Test router with OpenAIEncoder
    storage = DummyStorage()
    router = AdaptiveRouter(encoder=encoder, storage=storage, max_capacity=100)
    
    route = Route(name="support", utterances=["help me", "I need help"], threshold=0.8)
    router.add_route(route)
    
    match = router("assistance please")
    # Due to dummy embeddings, it will match
    assert match is not None
    assert match.name == "support"
