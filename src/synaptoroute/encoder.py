import abc
import numpy as np
import numpy.typing as npt
from typing import List, Optional

class BaseEncoder(abc.ABC):
    """
    Abstract base class for all SynaptoRoute encoders.
    """
    @property
    @abc.abstractmethod
    def dim(self) -> int:
        pass

    @abc.abstractmethod
    def encode(self, text: str) -> npt.NDArray[np.float32]:
        pass

    @abc.abstractmethod
    def encode_batch(self, texts: List[str]) -> npt.NDArray[np.float32]:
        pass

class FastEmbedEncoder(BaseEncoder):
    """
    Handles local intent embeddings using fastembed ONNX models.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", providers: List[str] = None, threads: Optional[int] = None):
        from fastembed import TextEmbedding
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.model = TextEmbedding(model_name=model_name, providers=providers, threads=threads)
        # Probe dimensionality directly from the model using a dummy token
        self._dim = len(list(self.model.embed(["test"]))[0])
    
    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> npt.NDArray[np.float32]:
        embeddings = list(self.model.embed([text]))
        return embeddings[0]
        
    def encode_batch(self, texts: List[str]) -> npt.NDArray[np.float32]:
        if not texts:
            # Dynamically use the model's true dimensionality
            return np.empty((0, self.dim), dtype=np.float32)
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings)

class OpenAIEncoder(BaseEncoder):
    """
    Handles remote intent embeddings using OpenAI models.
    """
    def __init__(self, model_name: str = "text-embedding-3-small", dim: Optional[int] = None, client=None):
        import openai
        self.model_name = model_name
        self.client = client or openai.OpenAI()
        
        if dim is not None:
            self._dim = dim
        else:
            # Hardcode based on known models to save an API call
            if model_name == "text-embedding-3-small":
                self._dim = 1536
            elif model_name == "text-embedding-3-large":
                self._dim = 3072
            elif model_name == "text-embedding-ada-002":
                self._dim = 1536
            else:
                raise ValueError("dim must be provided explicitly for unknown models.")
                
    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> npt.NDArray[np.float32]:
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        return np.array(response.data[0].embedding, dtype=np.float32)
        
    def encode_batch(self, texts: List[str]) -> npt.NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        embeddings = [data.embedding for data in response.data]
        return np.array(embeddings, dtype=np.float32)

# Preserve backwards compatibility
Encoder = FastEmbedEncoder
