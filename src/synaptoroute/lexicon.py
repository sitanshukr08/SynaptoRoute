"""
BM25 lexicographic index for SynaptoRoute hybrid routing.

This module provides a sparse, token-frequency-based lexicographic index that
runs in parallel with the dense vector (NumpyIndex / FAISS) index. Scores from
both signals are fused via a weighted alpha parameter in the router:

    hybrid_score = alpha * cosine + (1 - alpha) * bm25_normalized

BM25 (Best Match 25) captures exact-token and rare-term relevance that dense
embeddings deliberately smooth over. Combining both signals yields higher
routing accuracy on queries containing proper nouns, order IDs, model names,
and other exact lexical tokens.

No GPU or heavy ML dependencies required: uses `rank_bm25` (pure Python, MIT).
"""

from __future__ import annotations

import math
import re
import threading
from typing import Dict, List, Optional, Tuple

try:
    from rank_bm25 import BM25Okapi

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace + punctuation tokenizer compatible with BM25Okapi."""
    return _TOKEN_RE.findall(text.lower())


class BM25LexiconIndex:
    """
    Sparse BM25 lexicographic index over route utterances.

    Each route is represented by the concatenation of its utterances. A query
    is scored against each route document and the raw BM25 score is normalized
    into [-1.0, 1.0] for fusion with cosine similarity.

    Thread safety: All public methods acquire ``_lock`` to protect the shared
    BM25 model and corpus state. ``build()`` is called automatically by
    ``add_route`` and ``remove_route`` and is safe to call from multiple threads.

    Example::

        index = BM25LexiconIndex()
        index.add_route("billing", ["my invoice", "payment failed", "refund"])
        index.add_route("support", ["app crash", "api timeout", "error 500"])
        scores = index.search("I need a refund for my order", top_k=3)
        # [("billing", 0.72), ("support", 0.08)]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """
        Args:
            k1: BM25 term saturation parameter (default 1.5, typical range 1.2-2.0).
            b:  BM25 length normalization parameter (default 0.75, 0=off, 1=full).
        """
        if not HAS_BM25:
            raise ImportError(
                "rank_bm25 is required for hybrid lexicographic routing. "
                "Install it with: pip install 'synaptoroute[lexicon]'"
            )
        self.k1 = k1
        self.b = b
        self._lock = threading.Lock()

        # corpus[i] = tokenized document for route _route_order[i]
        self._route_utterances: Dict[str, List[str]] = {}  # route_name -> utterances
        self._route_order: List[str] = []  # ordered list of route names
        self._corpus: List[List[str]] = []  # tokenized documents per route
        self._bm25: Optional["BM25Okapi"] = None  # type: ignore[type-arg]
        self._max_raw_score: float = 1.0  # used for score normalization

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_route(self, route_name: str, utterances: List[str]) -> None:
        """Add or replace a route's utterances and rebuild the BM25 index."""
        with self._lock:
            self._route_utterances[route_name] = list(utterances)
            self._rebuild_unlocked()

    def remove_route(self, route_name: str) -> None:
        """Remove a route and rebuild the BM25 index."""
        with self._lock:
            if route_name in self._route_utterances:
                del self._route_utterances[route_name]
                self._rebuild_unlocked()

    def clear(self) -> None:
        """Remove all routes from the index."""
        with self._lock:
            self._route_utterances.clear()
            self._rebuild_unlocked()

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Tuple[float, str]]:
        """
        Return ``top_k`` (normalized_score, route_name) pairs in descending order.

        Normalized scores are in the range [0.0, 1.0]. Routes with a raw BM25
        score of 0 are excluded from results.

        Args:
            query:  The natural language query string.
            top_k:  Maximum number of results to return.

        Returns:
            List of (normalized_score, route_name) tuples, sorted descending.
        """
        with self._lock:
            if self._bm25 is None or not self._route_order:
                return []
            tokens = _tokenize(query)
            if not tokens:
                return []
            raw_scores = self._bm25.get_scores(tokens)
            results: List[Tuple[float, str]] = []
            for i, raw in enumerate(raw_scores):
                if raw <= 0.0:
                    continue
                norm = self._normalize_unlocked(float(raw))
                results.append((norm, self._route_order[i]))
            results.sort(key=lambda x: x[0], reverse=True)
            return results[:top_k]

    def score_route(self, query: str, route_name: str) -> float:
        """Return the normalized BM25 score for a single route (0.0 if absent)."""
        with self._lock:
            if self._bm25 is None or route_name not in self._route_utterances:
                return 0.0
            if route_name not in self._route_order:
                return 0.0
            idx = self._route_order.index(route_name)
            tokens = _tokenize(query)
            if not tokens:
                return 0.0
            raw_scores = self._bm25.get_scores(tokens)
            return self._normalize_unlocked(float(raw_scores[idx]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_unlocked(self) -> None:
        """Rebuild the BM25 model from the current route utterance corpus."""
        self._route_order = sorted(self._route_utterances.keys())
        self._corpus = []
        for route_name in self._route_order:
            utterances = self._route_utterances[route_name]
            # Concatenate all utterances into one document per route.
            combined_tokens = []
            for utt in utterances:
                combined_tokens.extend(_tokenize(utt))
            self._corpus.append(combined_tokens)

        if not self._corpus:
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(self._corpus, k1=self.k1, b=self.b)

        # Pre-compute max raw score across all documents for normalization.
        # Use each document as its own query to estimate upper bound.
        max_score = 1e-9
        for doc_tokens in self._corpus:
            if doc_tokens:
                scores = self._bm25.get_scores(doc_tokens)
                max_score = max(max_score, float(max(scores)))
        self._max_raw_score = max_score

    def _normalize_unlocked(self, raw: float) -> float:
        """Normalize a raw BM25 score to [0.0, 1.0] using log dampening."""
        if raw <= 0.0:
            return 0.0
        # Log dampening prevents one extremely high-scoring document from
        # compressing all other scores near zero.
        log_raw = math.log1p(raw)
        log_max = math.log1p(self._max_raw_score)
        return min(log_raw / log_max, 1.0)

    @property
    def route_count(self) -> int:
        """Number of routes currently indexed."""
        with self._lock:
            return len(self._route_utterances)

    @property
    def is_ready(self) -> bool:
        """True if the index has been built and is ready to serve queries."""
        with self._lock:
            return self._bm25 is not None
