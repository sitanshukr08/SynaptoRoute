import numpy as np
from fastembed import TextEmbedding
from typing import List

class Encoder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", providers: List[str] = None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.model = TextEmbedding(model_name=model_name, providers=providers)
    
    def encode(self, text: str) -> np.ndarray:
        embeddings = list(self.model.embed([text]))
        return embeddings[0]
        
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings)
