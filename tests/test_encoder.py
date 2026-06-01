import numpy as np
from synaptoroute.encoder import Encoder

def test_encoder_initialization(encoder):
    assert encoder is not None

def test_encode(encoder):
    embedding = encoder.encode("Hello world")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] > 0

def test_encode_batch(encoder):
    texts = ["Hello world", "Another sentence"]
    embeddings = encoder.encode_batch(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0
