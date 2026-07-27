"""Reproducible routing baselines that emit the same observable result schema."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from synaptoroute.encoder import BaseEncoder
from synaptoroute.models import DecisionReason, Route, RouteCandidate, RouterResult

from benchmarks.research_datasets import normalized_text


def _decision_from_scores(
    routes: dict[str, Route],
    scored_routes: Sequence[tuple[float, str]],
    *,
    threshold_for: Callable[[Route], float],
    margin: float,
    candidate_limit: int,
) -> RouterResult:
    ranked = sorted(scored_routes, key=lambda item: (-item[0], item[1]))
    if not ranked:
        return RouterResult(decision_reason=DecisionReason.NO_CANDIDATES)

    result_candidates = [
        RouteCandidate(
            route_name=route_name,
            score=score,
            threshold=threshold_for(routes[route_name]),
            passed_threshold=score >= threshold_for(routes[route_name]),
        )
        for score, route_name in ranked[:candidate_limit]
    ]
    raw_margin = ranked[0][0] - ranked[1][0] if len(ranked) > 1 else None
    eligible = [
        (score, route_name)
        for score, route_name in ranked
        if score >= threshold_for(routes[route_name])
    ]
    if not eligible:
        return RouterResult(
            score=ranked[0][0],
            margin=raw_margin,
            candidates=result_candidates,
            decision_reason=DecisionReason.BELOW_THRESHOLD,
        )

    decision_margin = eligible[0][0] - eligible[1][0] if len(eligible) > 1 else None
    if decision_margin is not None and decision_margin < margin:
        return RouterResult(
            score=eligible[0][0],
            margin=decision_margin,
            candidates=result_candidates,
            decision_reason=DecisionReason.AMBIGUOUS_MARGIN,
        )

    return RouterResult(
        route=routes[eligible[0][1]],
        score=eligible[0][0],
        margin=decision_margin,
        candidates=result_candidates,
        decision_reason=DecisionReason.MATCHED,
    )


class ExactStringBaseline:
    """Normalized exact matching with abstention on misses and collisions."""

    def __init__(self, routes: Sequence[Route]):
        self.routes = {route.name: route for route in routes}
        if len(self.routes) != len(routes):
            raise ValueError("route names must be unique")
        self._utterance_routes: dict[str, set[str]] = defaultdict(set)
        for route in routes:
            for utterance in route.utterances:
                self._utterance_routes[normalized_text(utterance)].add(route.name)

    def match(self, query: str) -> RouterResult:
        route_names = sorted(self._utterance_routes.get(normalized_text(query), set()))
        if not route_names:
            return RouterResult(decision_reason=DecisionReason.NO_CANDIDATES)

        candidates = [
            RouteCandidate(
                route_name=route_name,
                score=1.0,
                threshold=1.0,
                passed_threshold=True,
            )
            for route_name in route_names
        ]
        if len(route_names) > 1:
            return RouterResult(
                score=1.0,
                margin=0.0,
                candidates=candidates,
                decision_reason=DecisionReason.AMBIGUOUS_MARGIN,
            )
        return RouterResult(
            route=self.routes[route_names[0]],
            score=1.0,
            candidates=candidates,
            decision_reason=DecisionReason.MATCHED,
        )

    def __call__(self, query: str) -> Route | None:
        return self.match(query).route


class ExactCosineBaseline:
    """Exact utterance retrieval over the same fixed encoder embeddings."""

    def __init__(
        self,
        encoder: BaseEncoder,
        routes: Sequence[Route],
        *,
        margin: float = 0.0,
        candidate_limit: int = 5,
    ):
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self.encoder = encoder
        self.routes = {route.name: route for route in routes}
        if len(self.routes) != len(routes):
            raise ValueError("route names must be unique")
        self.margin = margin
        self.candidate_limit = candidate_limit

        utterances: list[str] = []
        labels: list[str] = []
        for route in routes:
            utterances.extend(route.utterances)
            labels.extend([route.name] * len(route.utterances))
        if not utterances:
            raise ValueError("at least one route utterance is required")

        embeddings = np.asarray(self.encoder.encode_batch(utterances), dtype=np.float32)
        self._embeddings = self._normalize_rows(embeddings)
        self._labels = labels

    @staticmethod
    def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def match(self, query: str) -> RouterResult:
        query_embedding = np.asarray(self.encoder.encode(query), dtype=np.float32).reshape(1, -1)
        query_embedding = self._normalize_rows(query_embedding)[0]
        utterance_scores = self._embeddings @ query_embedding

        route_scores: dict[str, float] = {}
        for score, route_name in zip(utterance_scores, self._labels):
            route_scores[route_name] = max(route_scores.get(route_name, -1.0), float(score))

        return _decision_from_scores(
            self.routes,
            [(score, route_name) for route_name, score in route_scores.items()],
            threshold_for=lambda route: route.threshold,
            margin=self.margin,
            candidate_limit=self.candidate_limit,
        )

    def __call__(self, query: str) -> Route | None:
        return self.match(query).route


class LogisticRegressionBaseline:
    """Multiclass logistic regression over fixed route-utterance embeddings."""

    def __init__(
        self,
        encoder: BaseEncoder,
        routes: Sequence[Route],
        *,
        threshold: float = 0.0,
        margin: float = 0.0,
        random_state: int = 42,
        candidate_limit: int = 5,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self.encoder = encoder
        self.routes = {route.name: route for route in routes}
        if len(self.routes) != len(routes):
            raise ValueError("route names must be unique")
        if len(self.routes) < 2:
            raise ValueError("logistic regression requires at least two routes")
        self.threshold = threshold
        self.margin = margin
        self.candidate_limit = candidate_limit

        utterances: list[str] = []
        labels: list[str] = []
        for route in routes:
            utterances.extend(route.utterances)
            labels.extend([route.name] * len(route.utterances))
        embeddings = np.asarray(self.encoder.encode_batch(utterances), dtype=np.float32)

        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )
        self.classifier.fit(embeddings, labels)

    def match(self, query: str) -> RouterResult:
        embedding = np.asarray(self.encoder.encode(query), dtype=np.float32).reshape(1, -1)
        probabilities = self.classifier.predict_proba(embedding)[0]
        scored_routes = [
            (float(probability), str(route_name))
            for probability, route_name in zip(probabilities, self.classifier.classes_)
        ]
        return _decision_from_scores(
            self.routes,
            scored_routes,
            threshold_for=lambda route: self.threshold,
            margin=self.margin,
            candidate_limit=self.candidate_limit,
        )

    def __call__(self, query: str) -> Route | None:
        return self.match(query).route


class SemanticRouterBaseline:
    """Aurelio Semantic Router adapter using the experiment's shared encoder."""

    def __init__(
        self,
        encoder: BaseEncoder,
        routes: Sequence[Route],
        *,
        margin: float = 0.0,
        candidate_limit: int = 5,
    ):
        try:
            from semantic_router import Route as SemanticRoute
            from semantic_router import SemanticRouter
            from semantic_router.encoders.base import DenseEncoder
        except ImportError as error:
            raise RuntimeError("Install synaptoroute[benchmark] for the Semantic Router baseline") from error

        class SharedDenseEncoder(DenseEncoder):
            backend: Any

            def __call__(self, docs: list[Any]) -> list[list[float]]:
                embeddings = self.backend.encode_batch([str(doc) for doc in docs])
                return np.asarray(embeddings, dtype=np.float32).tolist()

            async def acall(self, docs: list[Any]) -> list[list[float]]:
                embeddings = await self.backend.aencode_batch([str(doc) for doc in docs])
                return np.asarray(embeddings, dtype=np.float32).tolist()

        self.routes = {route.name: route for route in routes}
        if len(self.routes) != len(routes):
            raise ValueError("route names must be unique")
        self.margin = margin
        self.candidate_limit = candidate_limit
        shared_encoder = SharedDenseEncoder(name="shared-experiment-encoder", backend=encoder)
        semantic_routes = [
            SemanticRoute(
                name=route.name,
                utterances=list(route.utterances),
                score_threshold=-1.0,
            )
            for route in routes
        ]
        self.router = SemanticRouter(
            encoder=shared_encoder,
            routes=[],
            top_k=candidate_limit,
            aggregation="max",
        )
        self.router.add(semantic_routes)

    def match(self, query: str) -> RouterResult:
        choices = self.router(text=query, limit=self.candidate_limit)
        if not isinstance(choices, list):
            choices = [choices]
        scored_routes = [
            (float(choice.similarity_score), str(choice.name))
            for choice in choices
            if choice.name is not None and choice.similarity_score is not None
        ]
        return _decision_from_scores(
            self.routes,
            scored_routes,
            threshold_for=lambda route: route.threshold,
            margin=self.margin,
            candidate_limit=self.candidate_limit,
        )

    def __call__(self, query: str) -> Route | None:
        return self.match(query).route
