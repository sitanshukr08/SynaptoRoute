"""Deterministic non-semantic encoder for structural systems benchmarks."""

from __future__ import annotations

import hashlib

import numpy as np

from synaptoroute.encoder import BaseEncoder


class DeterministicHashEncoder(BaseEncoder):
    model_name = "sha256-expand-structural-only"

    def __init__(self, dim: int = 64):
        if dim < 2:
            raise ValueError("dimension must be at least 2")
        self._dim = dim

    @property
    def requires_lock(self) -> bool:
        return False

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        digest = hashlib.shake_256(text.encode("utf-8")).digest(self._dim * 4)
        integers = np.frombuffer(digest, dtype=np.uint32).astype(np.float32)
        vector = (integers / np.float32(2**31)) - 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.asarray([self.encode(text) for text in texts], dtype=np.float32)
