import numpy as np
import numpy.typing as npt
from fastembed import TextEmbedding
from typing import List, Optional

class Encoder:
    """
    Handles local intent embeddings using fastembed ONNX models.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", providers: List[str] = None, threads: Optional[int] = None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.model = TextEmbedding(model_name=model_name, providers=providers, threads=threads)
        # Probe dimensionality directly from the model using a dummy token
        self.dim = len(list(self.model.embed(["test"]))[0])
    
    def encode(self, text: str) -> npt.NDArray[np.float32]:
        embeddings = list(self.model.embed([text]))
        return embeddings[0]
        
    def encode_batch(self, texts: List[str]) -> npt.NDArray[np.float32]:
        if not texts:
            # Dynamically use the model's true dimensionality
            return np.empty((0, self.dim), dtype=np.float32)
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings)
