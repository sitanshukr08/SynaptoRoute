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
    def requires_lock(self) -> bool:
        pass

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
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", providers: Optional[List[str]] = None, threads: Optional[int] = None):
        import threading
        self._lock = threading.Lock()
        from fastembed import TextEmbedding
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.model = TextEmbedding(model_name=model_name, providers=providers, threads=threads)
        # Probe dimensionality directly from the model using a dummy token
        self._dim = len(list(self.model.embed(["test"]))[0])
    
    @property
    def requires_lock(self) -> bool:
        return False

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> npt.NDArray[np.float32]:
        with self._lock:
            embeddings = list(self.model.embed([text]))
            return embeddings[0]  # type: ignore
        
    def encode_batch(self, texts: List[str]) -> npt.NDArray[np.float32]:
        if not texts:
            # Dynamically use the model's true dimensionality
            return np.empty((0, self.dim), dtype=np.float32)
            
        chunk_size = 500
        all_embeddings = []
        with self._lock:
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                embeddings = list(self.model.embed(chunk))
                all_embeddings.extend(embeddings)
            
        return np.array(all_embeddings, dtype=np.float32)

class OpenAIEncoder(BaseEncoder):
    """
    Handles remote intent embeddings using OpenAI models.
    """
    def __init__(self, model_name: str = "text-embedding-3-small", dim: Optional[int] = None, dimensions: Optional[int] = None, client=None):
        try:
            import openai  # type: ignore
        except ImportError as e:
            if client is None:
                raise RuntimeError("Please install synaptoroute[openai] to use the OpenAIEncoder. Run `pip install synaptoroute[openai]`.") from e
            openai = None
        self.model_name = model_name
        self._openai_error = openai.OpenAIError if openai is not None else Exception
        self.client = client or openai.OpenAI()
        
        self.dimensions = dimensions
        if dimensions is not None:
            self._dim = dimensions
        elif dim is not None:
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
    def requires_lock(self) -> bool:
        return False

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> npt.NDArray[np.float32]:

        from synaptoroute.exceptions import SynaptoRouteError
        try:
            kwargs: dict = {"input": [text], "model": self.model_name}
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions  # type: ignore
                # type: ignore
            response = self.client.embeddings.create(**kwargs)  # type: ignore  # type: ignore
            return np.array(response.data[0].embedding, dtype=np.float32)
        except self._openai_error as e:
            raise SynaptoRouteError(f"OpenAI API Error: {e}") from e
        
    def encode_batch(self, texts: List[str]) -> npt.NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        

        from synaptoroute.exceptions import SynaptoRouteError
        try:
            chunk_size = 2048
            all_embeddings = []
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                kwargs: dict = {"input": chunk, "model": self.model_name}
                if self.dimensions is not None:
                    kwargs["dimensions"] = self.dimensions  # type: ignore
                # type: ignore
                response = self.client.embeddings.create(**kwargs)  # type: ignore  # type: ignore
                embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(embeddings)
            return np.array(all_embeddings, dtype=np.float32)
        except self._openai_error as e:
            raise SynaptoRouteError(f"OpenAI API Error: {e}") from e

# Preserve backwards compatibility
Encoder = FastEmbedEncoder
