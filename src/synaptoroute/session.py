"""
SynaptoRoute Session-Aware Routing
====================================
Provides lightweight multi-turn conversation context to the AdaptiveRouter.
Routes consistent with recent session history receive a small bounded recency boost.

Design constraints (backward-compatible):
- Off by default (enable_session_routing=False in AdaptiveRouter)
- Zero external dependencies
- Session store is in-process; no DB persistence (ephemeral by design)
- Score adjustments are additive and bounded to ±session_alpha to prevent
  session history from overriding strong semantic mismatches
"""

from __future__ import annotations

import time
import threading
from collections import deque
from typing import Dict


class SessionContext:
    """Tracks the recent route history for one session."""

    def __init__(self, session_id: str, window: int = 5, ttl_seconds: float = 1800.0) -> None:
        self.session_id = session_id
        self.window = window
        self.ttl_seconds = ttl_seconds
        # Each entry: (route_name, normalized_score, timestamp)
        self._history: deque[tuple[str, float, float]] = deque(maxlen=window)
        self._lock = threading.Lock()
        self.created_at = time.monotonic()
        self.last_active = time.monotonic()

    def record(self, route_name: str, score: float) -> None:
        with self._lock:
            self._history.append((route_name, score, time.monotonic()))
            self.last_active = time.monotonic()

    def is_expired(self) -> bool:
        return (time.monotonic() - self.last_active) > self.ttl_seconds

    def recency_weights(self) -> Dict[str, float]:
        """
        Return a dict of route_name -> recency weight using exponential decay.
        Most recent entry has weight 1.0, each prior entry decays by 0.5.
        """
        weights: Dict[str, float] = {}
        with self._lock:
            history = list(self._history)
        for i, (route_name, _score, _ts) in enumerate(reversed(history)):
            # position 0 = most recent, weight 1.0; position 1 = 0.5; etc.
            w = 0.5 ** i
            if route_name not in weights or weights[route_name] < w:
                weights[route_name] = w
        return weights


class SessionStore:
    """
    Thread-safe in-process store of SessionContext objects.
    Sessions expire after ttl_seconds of inactivity.
    """

    def __init__(self, default_window: int = 5, default_ttl: float = 1800.0) -> None:
        self.default_window = default_window
        self.default_ttl = default_ttl
        self._sessions: Dict[str, SessionContext] = {}
        self._lock = threading.Lock()
        self._last_gc = time.monotonic()
        self._gc_interval = 300.0  # run GC every 5 minutes

    def get_or_create(self, session_id: str) -> SessionContext:
        with self._lock:
            self._maybe_gc()
            if session_id not in self._sessions or self._sessions[session_id].is_expired():
                self._sessions[session_id] = SessionContext(
                    session_id=session_id,
                    window=self.default_window,
                    ttl_seconds=self.default_ttl,
                )
            return self._sessions[session_id]

    def record(self, session_id: str, route_name: str, score: float) -> None:
        ctx = self.get_or_create(session_id)
        ctx.record(route_name, score)

    def recency_weights(self, session_id: str) -> Dict[str, float]:
        with self._lock:
            ctx = self._sessions.get(session_id)
        if ctx is None or ctx.is_expired():
            return {}
        return ctx.recency_weights()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for ctx in self._sessions.values() if not ctx.is_expired())

    def _maybe_gc(self) -> None:
        """Purge expired sessions. Called under _lock."""
        now = time.monotonic()
        if now - self._last_gc < self._gc_interval:
            return
        expired = [sid for sid, ctx in self._sessions.items() if ctx.is_expired()]
        for sid in expired:
            del self._sessions[sid]
        self._last_gc = now


def apply_session_boost(
    best_by_route: dict[str, tuple[float, object]],
    session_weights: Dict[str, float],
    session_alpha: float,
) -> dict[str, tuple[float, object]]:
    """
    Apply additive session recency boost to candidate scores.

    session_alpha is clamped to [0.0, 0.15] to prevent session history
    from overriding a strong semantic mismatch (analogous to AMSR beta bound).

    Args:
        best_by_route: dict of route_name -> (score, Route)
        session_weights: dict of route_name -> recency weight in [0.0, 1.0]
        session_alpha: max possible score boost per candidate

    Returns:
        Updated best_by_route with session boosts applied.
    """
    alpha = max(0.0, min(0.15, session_alpha))
    if not session_weights or alpha == 0.0:
        return best_by_route
    result = {}
    for route_name, (score, route) in best_by_route.items():
        w = session_weights.get(route_name, 0.0)
        boosted = min(1.0, score + alpha * w)
        result[route_name] = (boosted, route)
    return result
