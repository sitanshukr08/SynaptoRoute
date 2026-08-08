import os

import numpy as np
import pytest

from synaptoroute.encoder import FastEmbedEncoder


pytestmark = pytest.mark.model


def test_fastembed_model_loads_and_returns_finite_embeddings():
    if os.environ.get("SYNAPTOROUTE_RUN_MODEL_TESTS") != "1":
        pytest.skip("set SYNAPTOROUTE_RUN_MODEL_TESTS=1 to run model integration tests")

    encoder = FastEmbedEncoder(model_name="BAAI/bge-small-en-v1.5", threads=1)
    embeddings = encoder.encode_batch(["billing issue", "technical support"])

    assert embeddings.shape == (2, encoder.dim)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert np.linalg.norm(embeddings, axis=1).min() > 0.0
