import pytest
from synaptoroute.reranker import CrossEncoderReranker, CrossEncoder
from synaptoroute.models import Route

class MockModel:
    def __init__(self):
        class Config:
            label2id = {"entailment": 1, "contradiction": 0}
        self.config = Config()

class MockCrossEncoder:
    def __init__(self, model_name):
        self.model = MockModel()

    def predict(self, pairs):
        # Return mock scores based on simple heuristic
        results = []
        for q, u in pairs:
            if q == u:
                results.append([0.1, 0.9]) # entailment is high
            else:
                results.append([0.9, 0.1])
        return results

@pytest.fixture
def mock_cross_encoder(monkeypatch):
    if CrossEncoder is not None:
        monkeypatch.setattr("synaptoroute.reranker.CrossEncoder", MockCrossEncoder)
    return MockCrossEncoder

@pytest.mark.skipif(CrossEncoder is None, reason="sentence-transformers not installed")
def test_reranker_selects_best_route(mock_cross_encoder):
    reranker = CrossEncoderReranker(threshold=0.5)
    
    route_a = Route(name="route_a", utterances=["hello world"])
    route_b = Route(name="route_b", utterances=["goodbye world"])
    
    candidates = [(0.8, route_a), (0.9, route_b)]
    
    best = reranker.rerank("hello world", candidates)
    assert best is not None
    assert best.name == "route_a"

@pytest.mark.skipif(CrossEncoder is None, reason="sentence-transformers not installed")
def test_reranker_returns_none_if_below_threshold(mock_cross_encoder):
    reranker = CrossEncoderReranker(threshold=0.99)
    
    route_a = Route(name="route_a", utterances=["completely unrelated"])
    candidates = [(0.8, route_a)]
    
    best = reranker.rerank("hello world", candidates)
    assert best is None

@pytest.mark.skipif(CrossEncoder is None, reason="sentence-transformers not installed")
def test_reranker_empty_candidates(mock_cross_encoder):
    reranker = CrossEncoderReranker(threshold=0.5)
    assert reranker.rerank("hello world", []) is None
