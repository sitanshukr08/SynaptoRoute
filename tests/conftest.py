import pytest
from synaptoroute.encoder import Encoder

@pytest.fixture(scope="session")
def encoder():
    """
    Session-scoped encoder fixture to prevent reloading the heavy ONNX weights
    for every single test. This stops GitHub Actions from running out of memory
    or hanging for 300+ minutes due to CPU thrashing.
    """
    return Encoder(model_name="BAAI/bge-small-en-v1.5")
