from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.models import Route
import numpy as np

class MockEncoder(Encoder):
    def __init__(self):
        self._dim = 2

    @property
    def dim(self) -> int:
        return self._dim


    def encode(self, text: str) -> np.ndarray:
        if "finance" in text:
            return np.array([1.0, 0.0])
        elif "support" in text:
            return np.array([0.0, 1.0])
        else:
            return np.array([0.5, 0.5])

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.encode(t) for t in texts])

def test_fit_thresholds():
    encoder = MockEncoder()
    storage = SQLiteStorage(':memory:')
    router = AdaptiveRouter(encoder, storage)
    
    router.add_route(Route(name="finance", utterances=["finance is great", "finance topic"]))
    router.add_route(Route(name="support", utterances=["support ticket", "customer support"]))
    
    samples = [
        "finance question",
        "finance related",
        "support help",
        "support needed",
        "random stuff"
    ]
    labels = ["finance", "finance", "support", "support", "none"]
    
    router.fit_thresholds(samples, labels)
    
    r_finance = router._route_map["finance"]
    r_support = router._route_map["support"]
    
    assert r_finance.threshold > 0.0
    assert r_support.threshold > 0.0
