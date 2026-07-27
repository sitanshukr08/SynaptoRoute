"""Exact embedding memoization for quality-only benchmark runs."""

from __future__ import annotations

import numpy as np

from synaptoroute.encoder import BaseEncoder


class MemoizingEncoder(BaseEncoder):
    """Reuse deterministic embeddings without changing encoder outputs."""

    def __init__(self, backend: BaseEncoder):
        self.backend = backend
        self.model_name = getattr(backend, "model_name", type(backend).__name__)
        self._cache: dict[str, np.ndarray] = {}
        self._hits = 0
        self._misses = 0

    @property
    def requires_lock(self) -> bool:
        return self.backend.requires_lock

    @property
    def dim(self) -> int:
        return self.backend.dim

    def encode(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            self._hits += 1
            return cached.copy()
        embedding = np.asarray(self.backend.encode(text), dtype=np.float32)
        self._cache[text] = embedding.copy()
        self._misses += 1
        return embedding.copy()

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            embeddings = np.asarray(self.backend.encode_batch(missing), dtype=np.float32)
            for text, embedding in zip(missing, embeddings):
                self._cache[text] = embedding.copy()
            self._misses += len(missing)
        self._hits += len(texts) - len(missing)
        return np.asarray([self._cache[text].copy() for text in texts], dtype=np.float32)

    def cache_stats(self) -> dict[str, int]:
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
        }
