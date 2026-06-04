import pytest
import numpy as np


class FakeEncoder:
    model_name = "fake-test-encoder"
    requires_lock = False
    dim = 8

    def encode(self, text: str):
        return self.encode_batch([text])[0]

    def encode_batch(self, texts):
        return np.array([self._vector(text) for text in texts], dtype=np.float32)

    def _vector(self, text: str):
        lowered = text.lower()
        vector = np.zeros(self.dim, dtype=np.float32)
        keyword_groups = [
            ("greeting", ["hello", "hi"]),
            ("farewell", ["bye", "goodbye", "see you"]),
            ("support", ["support", "help", "assistance"]),
            ("finance", ["finance"]),
            ("billing", ["bill", "billing"]),
            ("exact", ["exactly this"]),
            ("test", ["test"]),
            ("async", ["async"]),
        ]
        for index, (_, keywords) in enumerate(keyword_groups):
            if any(keyword in lowered for keyword in keywords):
                vector[index] = 1.0
        if np.any(vector):
            return vector
        return np.full(self.dim, 0.01, dtype=np.float32)


@pytest.fixture(scope="session")
def encoder():
    return FakeEncoder()


@pytest.fixture
def fake_encoder():
    return FakeEncoder()
