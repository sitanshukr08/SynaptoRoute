import numpy as np
import numpy.typing as npt
from fastembed import TextEmbedding
from typing import List

class Encoder:
    """
    Handles local intent embeddings using fastembed ONNX models.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", providers: List[str] = None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.model = TextEmbedding(model_name=model_name, providers=providers)
    
    def encode(self, text: str) -> npt.NDArray[np.float32]:
        embeddings = list(self.model.embed([text]))
        return embeddings[0]
        
    def encode_batch(self, texts: List[str]) -> npt.NDArray[np.float32]:
        if not texts:
            # Return empty array with correct shape for BGE models (384)
            # If using another model, shape mismatches will be caught downstream
            return np.empty((0, 384), dtype=np.float32)
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings)
