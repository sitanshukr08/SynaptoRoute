"""Validation-only threshold and margin calibration for selective routing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from synaptoroute.models import DecisionReason, Route, RouteCandidate, RouterResult


CALIBRATION_SCHEMA_VERSION = 1
DEFAULT_CALIBRATION_SPLITS = frozenset({"calibration", "dev", "validation", "val"})


@dataclass(frozen=True)
class ScoredExample:
    example_id: str
    split: str
    expected_route: str | None
    candidates: tuple[tuple[str, float], ...]

    def __post_init__(self):
        if not self.example_id:
            raise ValueError("example_id must not be empty")
        if not self.split:
            raise ValueError("split must not be empty")
        candidate_names = [name for name, _ in self.candidates]
        if len(candidate_names) != len(set(candidate_names)):
            raise ValueError("candidate route names must be unique")

    @classmethod
    def from_result(
        cls,
        *,
        example_id: str,
        split: str,
        expected_route: str | None,
        result: RouterResult,
    ) -> "ScoredExample":
        candidates = tuple(
            sorted(
                ((candidate.route_name, candidate.score) for candidate in result.candidates),
                key=lambda item: (-item[1], item[0]),
            )
        )
        return cls(
            example_id=example_id,
            split=split,
            expected_route=expected_route,
            candidates=candidates,
        )


@dataclass(frozen=True)
class GlobalPolicy:
    threshold: float
    margin: float

    def __post_init__(self):
        if self.margin < 0:
            raise ValueError("margin must be non-negative")


@dataclass(frozen=True)
class PolicyMetrics:
    example_count: int
    known_count: int
    ood_count: int
    accepted_count: int
    correct_count: int
    known_coverage: float
    ood_false_acceptance_rate: float
    overall_accuracy: float
    selective_accuracy: float


@dataclass(frozen=True)
class CalibrationResult:
    policy: GlobalPolicy
    metrics: PolicyMetrics


@dataclass(frozen=True)
class PerRoutePolicy:
    default_threshold: float
    thresholds: dict[str, float]
    margin: float

    def __post_init__(self):
        if not np.isfinite(self.default_threshold):
            raise ValueError("default threshold must be finite")
        if not np.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and non-negative")
        for route_name, threshold in self.thresholds.items():
            if not route_name:
                raise ValueError("route threshold names must not be empty")
            if not np.isfinite(threshold):
                raise ValueError("route thresholds must be finite")

    def threshold_for(self, route_name: str) -> float:
        return self.thresholds.get(route_name, self.default_threshold)


@dataclass(frozen=True)
class PerRouteCalibrationResult:
    policy: PerRoutePolicy
    metrics: PolicyMetrics


@dataclass(frozen=True)
class _ScoredArrays:
    top_scores: np.ndarray
    margins: np.ndarray
    known: np.ndarray
    top_correct: np.ndarray
    top_routes: np.ndarray


def _scored_arrays(examples: Sequence[ScoredExample]) -> _ScoredArrays:
    top_scores = np.full(len(examples), float("-inf"), dtype=np.float64)
    margins = np.full(len(examples), float("inf"), dtype=np.float64)
    known = np.zeros(len(examples), dtype=bool)
    top_correct = np.zeros(len(examples), dtype=bool)
    top_routes = np.full(len(examples), None, dtype=object)
    for index, example in enumerate(examples):
        known[index] = example.expected_route is not None
        if not example.candidates:
            continue
        top_name, top_score = example.candidates[0]
        top_routes[index] = top_name
        top_scores[index] = top_score
        top_correct[index] = example.expected_route is not None and top_name == example.expected_route
        if len(example.candidates) > 1:
            margins[index] = top_score - example.candidates[1][1]
    return _ScoredArrays(top_scores, margins, known, top_correct, top_routes)


def _metrics_from_acceptance(arrays: _ScoredArrays, accepted: np.ndarray) -> PolicyMetrics:
    known_count = int(np.count_nonzero(arrays.known))
    ood_count = len(accepted) - known_count
    accepted_known = int(np.count_nonzero(accepted & arrays.known))
    accepted_ood = int(np.count_nonzero(accepted & ~arrays.known))
    correct_accepted = int(np.count_nonzero(accepted & arrays.top_correct))
    correct_rejected_ood = int(np.count_nonzero(~accepted & ~arrays.known))
    accepted_count = int(np.count_nonzero(accepted))
    correct_count = correct_accepted + correct_rejected_ood
    return PolicyMetrics(
        example_count=len(accepted),
        known_count=known_count,
        ood_count=ood_count,
        accepted_count=accepted_count,
        correct_count=correct_count,
        known_coverage=accepted_known / known_count if known_count else 0.0,
        ood_false_acceptance_rate=accepted_ood / ood_count if ood_count else 0.0,
        overall_accuracy=correct_count / len(accepted),
        selective_accuracy=correct_accepted / accepted_count if accepted_count else 1.0,
    )


def _metrics_for_global_policy(arrays: _ScoredArrays, policy: GlobalPolicy) -> PolicyMetrics:
    accepted = (arrays.top_scores >= policy.threshold) & (arrays.margins >= policy.margin)
    return _metrics_from_acceptance(arrays, accepted)


def _prediction(example: ScoredExample, policy: GlobalPolicy) -> str | None:
    if not example.candidates:
        return None
    top_name, top_score = example.candidates[0]
    if top_score < policy.threshold:
        return None
    if len(example.candidates) > 1:
        score_margin = top_score - example.candidates[1][1]
        if score_margin < policy.margin:
            return None
    return top_name


def evaluate_global_policy(examples: Sequence[ScoredExample], policy: GlobalPolicy) -> PolicyMetrics:
    if not examples:
        raise ValueError("at least one scored example is required")
    return _metrics_for_global_policy(_scored_arrays(examples), policy)


def _per_route_prediction(example: ScoredExample, policy: PerRoutePolicy) -> str | None:
    if not example.candidates:
        return None
    top_name, top_score = example.candidates[0]
    if top_score < policy.threshold_for(top_name):
        return None
    if len(example.candidates) > 1:
        score_margin = top_score - example.candidates[1][1]
        if score_margin < policy.margin:
            return None
    return top_name


def evaluate_per_route_policy(
    examples: Sequence[ScoredExample],
    policy: PerRoutePolicy,
) -> PolicyMetrics:
    if not examples:
        raise ValueError("at least one scored example is required")

    arrays = _scored_arrays(examples)
    thresholds = np.asarray(
        [
            policy.threshold_for(str(route_name)) if route_name is not None else float("inf")
            for route_name in arrays.top_routes
        ],
        dtype=np.float64,
    )
    accepted = (arrays.top_scores >= thresholds) & (arrays.margins >= policy.margin)
    return _metrics_from_acceptance(arrays, accepted)


def _candidate_grid(values: Sequence[float], max_points: int) -> list[float]:
    if max_points < 3:
        raise ValueError("max_points must be at least 3")
    if not values:
        return [0.0]
    unique = np.array(sorted(set(float(value) for value in values)), dtype=float)
    if len(unique) > max_points - 2:
        quantiles = np.linspace(0.0, 1.0, max_points - 2)
        unique = np.unique(np.quantile(unique, quantiles))
    scale = max(1.0, abs(float(unique[0])), abs(float(unique[-1])))
    epsilon = np.finfo(float).eps * scale * 8.0
    return [float(unique[0] - epsilon), *[float(value) for value in unique], float(unique[-1] + epsilon)]


def fit_global_policy(
    examples: Sequence[ScoredExample],
    *,
    min_known_coverage: float = 0.8,
    max_ood_false_acceptance_rate: float | None = None,
    allowed_splits: frozenset[str] = DEFAULT_CALIBRATION_SPLITS,
    max_grid_points: int = 101,
) -> CalibrationResult:
    """Fit a policy without accepting test-split examples."""
    if not examples:
        raise ValueError("at least one calibration example is required")
    if not 0.0 <= min_known_coverage <= 1.0:
        raise ValueError("min_known_coverage must be between 0 and 1")
    if max_ood_false_acceptance_rate is not None and not 0.0 <= max_ood_false_acceptance_rate <= 1.0:
        raise ValueError("max_ood_false_acceptance_rate must be between 0 and 1")

    invalid_splits = sorted(
        {
            example.split
            for example in examples
            if example.split.casefold() not in allowed_splits
        }
    )
    if invalid_splits:
        raise ValueError(f"non-calibration splits are not allowed for fitting: {invalid_splits}")

    top_scores = [example.candidates[0][1] for example in examples if example.candidates]
    score_margins = [
        example.candidates[0][1] - example.candidates[1][1]
        for example in examples
        if len(example.candidates) > 1
    ]
    thresholds = _candidate_grid(top_scores, max_grid_points)
    margins = sorted(set([0.0, *_candidate_grid(score_margins, max_grid_points)]))
    arrays = _scored_arrays(examples)

    feasible: list[CalibrationResult] = []
    for threshold in thresholds:
        for margin in margins:
            if margin < 0:
                continue
            policy = GlobalPolicy(threshold=threshold, margin=margin)
            metrics = _metrics_for_global_policy(arrays, policy)
            if metrics.known_coverage < min_known_coverage:
                continue
            if (
                max_ood_false_acceptance_rate is not None
                and metrics.ood_false_acceptance_rate > max_ood_false_acceptance_rate
            ):
                continue
            feasible.append(CalibrationResult(policy=policy, metrics=metrics))

    if not feasible:
        raise ValueError("no policy satisfies the calibration constraints")

    return max(
        feasible,
        key=lambda result: (
            result.metrics.overall_accuracy,
            -result.metrics.ood_false_acceptance_rate,
            result.metrics.selective_accuracy,
            result.metrics.known_coverage,
            result.policy.threshold,
            result.policy.margin,
        ),
    )


def fit_per_route_policy(
    examples: Sequence[ScoredExample],
    *,
    min_known_coverage: float = 0.8,
    max_ood_false_acceptance_rate: float | None = None,
    allowed_splits: frozenset[str] = DEFAULT_CALIBRATION_SPLITS,
    max_grid_points: int = 101,
    min_top_examples_per_route: int = 5,
    max_rounds: int = 3,
) -> PerRouteCalibrationResult:
    """Fit route-specific thresholds with a shared validation-fitted margin."""
    if min_top_examples_per_route < 1:
        raise ValueError("min_top_examples_per_route must be positive")
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    global_result = fit_global_policy(
        examples,
        min_known_coverage=min_known_coverage,
        max_ood_false_acceptance_rate=max_ood_false_acceptance_rate,
        allowed_splits=allowed_splits,
        max_grid_points=max_grid_points,
    )
    top_scores: dict[str, list[float]] = {}
    for example in examples:
        if example.candidates:
            route_name, score = example.candidates[0]
            top_scores.setdefault(route_name, []).append(score)

    thresholds = {route_name: global_result.policy.threshold for route_name in sorted(top_scores)}
    current_policy = PerRoutePolicy(
        default_threshold=global_result.policy.threshold,
        thresholds=thresholds,
        margin=global_result.policy.margin,
    )
    arrays = _scored_arrays(examples)
    current_thresholds = np.asarray(
        [
            current_policy.threshold_for(str(route_name))
            if route_name is not None
            else float("inf")
            for route_name in arrays.top_routes
        ],
        dtype=np.float64,
    )
    current_metrics = _metrics_from_acceptance(
        arrays,
        (arrays.top_scores >= current_thresholds) & (arrays.margins >= current_policy.margin),
    )

    def result_key(result: PerRouteCalibrationResult, route_name: str) -> tuple[float, ...]:
        return (
            result.metrics.overall_accuracy,
            -result.metrics.ood_false_acceptance_rate,
            result.metrics.selective_accuracy,
            result.metrics.known_coverage,
            result.policy.threshold_for(route_name),
        )

    for _ in range(max_rounds):
        changed = False
        for route_name in sorted(top_scores):
            if len(top_scores[route_name]) < min_top_examples_per_route:
                continue
            feasible: list[PerRouteCalibrationResult] = []
            route_mask = arrays.top_routes == route_name
            for threshold in _candidate_grid(top_scores[route_name], max_grid_points):
                candidate_thresholds = dict(current_policy.thresholds)
                candidate_thresholds[route_name] = threshold
                candidate_policy = PerRoutePolicy(
                    default_threshold=current_policy.default_threshold,
                    thresholds=candidate_thresholds,
                    margin=current_policy.margin,
                )
                candidate_thresholds_for_examples = current_thresholds.copy()
                candidate_thresholds_for_examples[route_mask] = threshold
                accepted = (
                    (arrays.top_scores >= candidate_thresholds_for_examples)
                    & (arrays.margins >= candidate_policy.margin)
                )
                metrics = _metrics_from_acceptance(arrays, accepted)
                if metrics.known_coverage < min_known_coverage:
                    continue
                if (
                    max_ood_false_acceptance_rate is not None
                    and metrics.ood_false_acceptance_rate > max_ood_false_acceptance_rate
                ):
                    continue
                feasible.append(PerRouteCalibrationResult(candidate_policy, metrics))
            if not feasible:
                continue
            selected = max(feasible, key=lambda result: result_key(result, route_name))
            if selected.policy.threshold_for(route_name) != current_policy.threshold_for(route_name):
                changed = True
            current_policy = selected.policy
            current_metrics = selected.metrics
            current_thresholds[route_mask] = current_policy.threshold_for(route_name)
        if not changed:
            break
    return PerRouteCalibrationResult(policy=current_policy, metrics=current_metrics)


def apply_global_policy(
    result: RouterResult,
    *,
    routes: Mapping[str, Route],
    policy: GlobalPolicy,
) -> RouterResult:
    """Re-evaluate raw candidates under a frozen calibrated policy."""
    if not result.candidates:
        reason = (
            DecisionReason.EMPTY_INDEX
            if result.decision_reason is DecisionReason.EMPTY_INDEX
            else DecisionReason.NO_CANDIDATES
        )
        return RouterResult(decision_reason=reason)

    ranked = sorted(result.candidates, key=lambda candidate: (-candidate.score, candidate.route_name))
    top = ranked[0]
    score_margin = top.score - ranked[1].score if len(ranked) > 1 else None
    calibrated_candidates = [
        RouteCandidate(
            route_name=candidate.route_name,
            score=candidate.score,
            threshold=policy.threshold,
            passed_threshold=candidate.score >= policy.threshold,
        )
        for candidate in ranked
    ]

    if top.score < policy.threshold:
        return RouterResult(
            score=top.score,
            margin=score_margin,
            candidates=calibrated_candidates,
            decision_reason=DecisionReason.BELOW_THRESHOLD,
        )
    if score_margin is not None and score_margin < policy.margin:
        return RouterResult(
            score=top.score,
            margin=score_margin,
            candidates=calibrated_candidates,
            decision_reason=DecisionReason.AMBIGUOUS_MARGIN,
        )
    if top.route_name not in routes:
        raise KeyError(f"candidate route is missing from route lookup: {top.route_name}")
    return RouterResult(
        route=routes[top.route_name],
        score=top.score,
        margin=score_margin,
        candidates=calibrated_candidates,
        decision_reason=DecisionReason.MATCHED,
    )


def apply_per_route_policy(
    result: RouterResult,
    *,
    routes: Mapping[str, Route],
    policy: PerRoutePolicy,
) -> RouterResult:
    if not result.candidates:
        reason = (
            DecisionReason.EMPTY_INDEX
            if result.decision_reason is DecisionReason.EMPTY_INDEX
            else DecisionReason.NO_CANDIDATES
        )
        return RouterResult(decision_reason=reason)

    ranked = sorted(result.candidates, key=lambda candidate: (-candidate.score, candidate.route_name))
    top = ranked[0]
    threshold = policy.threshold_for(top.route_name)
    score_margin = top.score - ranked[1].score if len(ranked) > 1 else None
    calibrated_candidates = [
        RouteCandidate(
            route_name=candidate.route_name,
            score=candidate.score,
            threshold=policy.threshold_for(candidate.route_name),
            passed_threshold=candidate.score >= policy.threshold_for(candidate.route_name),
        )
        for candidate in ranked
    ]
    if top.score < threshold:
        return RouterResult(
            score=top.score,
            margin=score_margin,
            candidates=calibrated_candidates,
            decision_reason=DecisionReason.BELOW_THRESHOLD,
        )
    if score_margin is not None and score_margin < policy.margin:
        return RouterResult(
            score=top.score,
            margin=score_margin,
            candidates=calibrated_candidates,
            decision_reason=DecisionReason.AMBIGUOUS_MARGIN,
        )
    if top.route_name not in routes:
        raise KeyError(f"candidate route is missing from route lookup: {top.route_name}")
    return RouterResult(
        route=routes[top.route_name],
        score=top.score,
        margin=score_margin,
        candidates=calibrated_candidates,
        decision_reason=DecisionReason.MATCHED,
    )


def calibration_artifact(
    result: CalibrationResult,
    *,
    dataset: Mapping[str, Any],
    source_predictions_sha256: str,
) -> dict[str, Any]:
    if len(source_predictions_sha256) != 64:
        raise ValueError("source_predictions_sha256 must be a SHA-256 hex digest")
    try:
        int(source_predictions_sha256, 16)
    except ValueError as error:
        raise ValueError("source_predictions_sha256 must be a SHA-256 hex digest") from error
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "method": "global_threshold_margin_grid_search",
        "policy": asdict(result.policy),
        "calibration_metrics": asdict(result.metrics),
        "dataset": dict(dataset),
        "source_predictions_sha256": source_predictions_sha256,
    }


def per_route_calibration_artifact(
    result: PerRouteCalibrationResult,
    *,
    dataset: Mapping[str, Any],
    source_predictions_sha256: str,
) -> dict[str, Any]:
    artifact = calibration_artifact(
        CalibrationResult(
            policy=GlobalPolicy(
                threshold=result.policy.default_threshold,
                margin=result.policy.margin,
            ),
            metrics=result.metrics,
        ),
        dataset=dataset,
        source_predictions_sha256=source_predictions_sha256,
    )
    artifact["method"] = "per_route_threshold_shared_margin_coordinate_search"
    artifact["policy"] = {
        "default_threshold": result.policy.default_threshold,
        "thresholds": dict(sorted(result.policy.thresholds.items())),
        "margin": result.policy.margin,
    }
    return artifact


def write_calibration_artifact(path: Path | str, artifact: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True), encoding="utf-8")
    return output_path
