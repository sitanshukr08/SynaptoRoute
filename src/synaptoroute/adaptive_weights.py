"""
SynaptoRoute Refined Adaptive Weighted Embedding & MFU/LRU Context Engine
===========================================================================
Experimental architecture providing:
1. Bounded additive prior scoring after cosine retrieval.
2. Saturation Dampening & Negative Feedback Penalties (prevents popularity entrenchment).
3. Thread-safe buffered access statistics.
4. Adaptive Replacement Cache (ARC) for vector embedding memory.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class ContextMetadata:
    """Tracking metadata for a route or utterance context."""
    key: str
    route_name: str
    embedding: np.ndarray
    frequency_count: int = 0
    negative_feedback_count: int = 0
    last_accessed: float = 0.0
    base_priority: float = 0.0


class BoundedBayesianWeigher:
    """
    Computes bounded score adjustments from MFU and LRU signals.

    The adjustment can change candidate ordering and is not a metric-space
    preservation guarantee.
    """

    def __init__(
        self,
        frequency_boost_cap: float = 0.08,     # Max +0.08 added to cosine score
        saturation_constant: float = 50.0,       # Saturation threshold K_s
        recency_decay_lambda: float = 1e-4,     # Exponential decay per second
        penalty_weight: float = 0.05,            # Penalty per negative feedback hit
    ):
        self.frequency_boost_cap = frequency_boost_cap
        self.saturation_constant = saturation_constant
        self.recency_decay_lambda = recency_decay_lambda
        self.penalty_weight = penalty_weight

    def compute_prior_adjustment(self, meta: ContextMetadata, now: Optional[float] = None) -> float:
        """
        Compute bounded additive prior adjustment:
        Prior = Boost_cap * (freq / (freq + K_s)) - lambda * delta_t - penalty * neg_hits
        """
        if now is None:
            now = time.time()

        # 1. Saturation-dampened frequency boost (MFU)
        freq = max(0, meta.frequency_count)
        freq_boost = self.frequency_boost_cap * (freq / (freq + self.saturation_constant))

        # 2. Recency decay (LRU)
        delta_t = max(0.0, now - meta.last_accessed) if meta.last_accessed > 0 else 0.0
        recency_penalty = self.recency_decay_lambda * delta_t

        # 3. Negative feedback penalty
        neg_penalty = self.penalty_weight * meta.negative_feedback_count

        adjustment = meta.base_priority + freq_boost - recency_penalty - neg_penalty
        # Bound the amount by which contextual signals can change a score.
        return max(-0.15, min(self.frequency_boost_cap, adjustment))

    def evaluate_score(
        self,
        raw_cosine_score: float,
        meta: ContextMetadata,
        now: Optional[float] = None,
    ) -> float:
        """
        Combine raw cosine similarity with bounded prior adjustment.
        Raw cosine score governs metric search; prior adjustment acts as a tie-breaker.
        """
        prior = self.compute_prior_adjustment(meta, now=now)
        return float(np.clip(raw_cosine_score + prior, -1.0, 1.0))


class BufferedStatsCollector:
    """
    Thread-safe in-memory statistics buffer.

    This implementation uses a lock and makes no lock-free throughput claim.
    """

    def __init__(self, batch_flush_size: int = 100):
        self.batch_flush_size = batch_flush_size
        self._buffer: deque[Tuple[str, float, bool]] = deque()
        self._lock = threading.Lock()

    def record_hit(self, key: str, is_negative: bool = False):
        """Non-blocking append to ring buffer."""
        now = time.time()
        with self._lock:
            self._buffer.append((key, now, is_negative))

    def flush(self) -> List[Tuple[str, float, bool]]:
        """Drain buffered statistics for background processing."""
        with self._lock:
            drained = list(self._buffer)
            self._buffer.clear()
        return drained


# Backward-compatible alias. The old name is retained for imports only; the
# implementation has always used a lock.
LockFreeStatsCollector = BufferedStatsCollector


class VectorARCCache:
    """
    Adaptive Replacement Cache (ARC) for vector embeddings.
    Dynamically balances between Recent (T1) and Frequent (T2) vector caches.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.p = 0  # Target size for T1

        self.t1: Dict[str, ContextMetadata] = {}  # Recent items
        self.t2: Dict[str, ContextMetadata] = {}  # Frequent items
        self.b1: Dict[str, bool] = {}             # Ghost entries for T1 evictions
        self.b2: Dict[str, bool] = {}             # Ghost entries for T2 evictions

    def get(self, key: str, now: Optional[float] = None) -> Optional[ContextMetadata]:
        now_time = now if now is not None else time.time()

        if key in self.t1:
            meta = self.t1.pop(key)
            meta.frequency_count += 1
            meta.last_accessed = now_time
            self.t2[key] = meta
            return meta
        elif key in self.t2:
            meta = self.t2[key]
            meta.frequency_count += 1
            meta.last_accessed = now_time
            return meta
        return None

    def put(self, key: str, meta: ContextMetadata):
        now_time = time.time()
        meta.last_accessed = now_time

        if key in self.t1 or key in self.t2:
            self.get(key, now=now_time)
            return

        # Handle ghost cache hits for dynamic tuning of target size p
        if key in self.b1:
            delta = 1 if len(self.b1) >= len(self.b2) else len(self.b2) / max(1, len(self.b1))
            self.p = min(self.capacity, self.p + int(delta))
            self._replace(key)
            del self.b1[key]
            self.t2[key] = meta
            return
        elif key in self.b2:
            delta = 1 if len(self.b2) >= len(self.b1) else len(self.b1) / max(1, len(self.b2))
            self.p = max(0, self.p - int(delta))
            self._replace(key)
            del self.b2[key]
            self.t2[key] = meta
            return

        # Cache Miss logic
        total = len(self.t1) + len(self.b1)
        if total == self.capacity:
            if len(self.t1) < self.capacity:
                del self.b1[next(iter(self.b1))]
                self._replace(key)
            else:
                del self.t1[next(iter(self.t1))]
        elif total < self.capacity:
            total_all = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
            if total_all >= self.capacity:
                if total_all == 2 * self.capacity and self.b2:
                    del self.b2[next(iter(self.b2))]
                self._replace(key)

        self.t1[key] = meta

    def _replace(self, key: str):
        if self.t1 and (len(self.t1) > self.p or (key in self.b2 and len(self.t1) == self.p)):
            old_key = next(iter(self.t1))
            self.b1[old_key] = True
            del self.t1[old_key]
        elif self.t2:
            old_key = next(iter(self.t2))
            self.b2[old_key] = True
            del self.t2[old_key]
