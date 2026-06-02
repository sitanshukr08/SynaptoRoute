import pytest
import numpy as np
from synaptoroute.encoder import Encoder

@pytest.fixture(scope="session")
def encoder():
    """
    Session-scoped encoder fixture to prevent reloading the heavy ONNX weights
    for every single test. This stops GitHub Actions from running out of memory
    or hanging for 300+ minutes due to CPU thrashing.
    """
    return Encoder(model_name="BAAI/bge-small-en-v1.5")

@pytest.fixture
def fake_encoder():
    class FakeEncoder:
        model_name = "fake-test-encoder"
        requires_lock = False
        dim = 3

        def encode(self, text: str):
            return self.encode_batch([text])[0]

        def encode_batch(self, texts):
            vectors = []
            for text in texts:
                total = sum(ord(char) for char in text)
                vectors.append([
                    float(total % 7 + 1),
                    float(total % 11 + 1),
                    float(total % 13 + 1),
                ])
            return np.array(vectors, dtype=np.float32)

    return FakeEncoder()
