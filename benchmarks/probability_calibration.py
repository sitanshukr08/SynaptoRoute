"""Held-out correctness-probability calibration for quality experiments."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from synaptoroute.calibration import CalibrationMetrics, evaluate_calibration
from synaptoroute.models import RouterResult


FEATURE_NAMES = (
    "top_score",
    "margin",
    "acceptance_confidence",
    "score_present",
    "margin_present",
    "decision_matched",
)


def split_calibration_examples(
    examples: Sequence[Any],
    *,
    seed: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Create deterministic, label-stratified policy and probability splits."""
    if len(examples) < 2:
        raise ValueError("at least two calibration examples are required for a held-out split")

    grouped: dict[str, list[Any]] = defaultdict(list)
    for example in examples:
        label_key = "__OOD__" if example.label is None else str(example.label)
        grouped[label_key].append(example)

    policy_examples: list[Any] = []
    probability_examples: list[Any] = []
    for label_key, values in sorted(grouped.items()):
        ranked = sorted(
            values,
            key=lambda example: hashlib.sha256(
                f"{seed}\0{label_key}\0{example.example_id}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) == 1:
            policy_examples.extend(ranked)
            continue
        probability_count = len(ranked) // 2
        probability_examples.extend(ranked[:probability_count])
        policy_examples.extend(ranked[probability_count:])

    if not probability_examples:
        ranked_policy = sorted(
            policy_examples,
            key=lambda example: hashlib.sha256(
                f"{seed}\0fallback\0{example.example_id}".encode("utf-8")
            ).hexdigest(),
        )
        probability_examples.append(ranked_policy[0])
        policy_examples.remove(ranked_policy[0])
    if not policy_examples:
        raise ValueError("held-out split left no examples for routing-policy calibration")

    return (
        tuple(sorted(policy_examples, key=lambda example: example.example_id)),
        tuple(sorted(probability_examples, key=lambda example: example.example_id)),
    )


def correctness_features(
    raw_result: RouterResult,
    final_result: RouterResult,
    *,
    acceptance_confidence: float,
) -> tuple[float, ...]:
    """Extract finite features without pretending raw scores are probabilities."""
    score_present = raw_result.score is not None and math.isfinite(raw_result.score)
    margin_present = raw_result.margin is not None and math.isfinite(raw_result.margin)
    confidence_present = math.isfinite(acceptance_confidence)
    return (
        float(raw_result.score) if score_present else -1.0,
        float(raw_result.margin) if margin_present else 0.0,
        float(acceptance_confidence) if confidence_present else -2.0,
        float(score_present),
        float(margin_present),
        float(final_result.matched),
    )


@dataclass(frozen=True)
class CorrectnessProbabilityCalibrator:
    """Serializable logistic calibrator for final-decision correctness."""

    method: str
    coefficients: tuple[float, ...]
    intercept: float
    constant_probability: float | None = None

    def predict(self, features: Sequence[float]) -> float:
        if len(features) != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} confidence features")
        if self.constant_probability is not None:
            return self.constant_probability
        linear = self.intercept + sum(
            coefficient * float(value)
            for coefficient, value in zip(self.coefficients, features)
        )
        if linear >= 0:
            probability = 1.0 / (1.0 + math.exp(-linear))
        else:
            exponential = math.exp(linear)
            probability = exponential / (1.0 + exponential)
        return min(1.0, max(0.0, probability))

    def artifact(
        self,
        *,
        fit_count: int,
        positive_count: int,
        fit_metrics: CalibrationMetrics,
        source_predictions_path: str,
        source_predictions_sha256: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "unverified",
            "paper_evidence_eligible": False,
            "target": "final routing decision is correct",
            "method": self.method,
            "feature_names": list(FEATURE_NAMES),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "constant_probability": self.constant_probability,
            "fit_count": fit_count,
            "positive_count": positive_count,
            "fit_metrics": asdict(fit_metrics),
            "source_predictions": {
                "path": source_predictions_path,
                "sha256": source_predictions_sha256,
            },
            "notes": [
                "The probability model is fitted on examples disjoint from policy fitting and test evaluation.",
                "Fit metrics are descriptive in-sample diagnostics, not reported test performance.",
                "Raw similarity scores and margins are input features, not probabilities.",
            ],
        }


def fit_correctness_probability(
    features: Sequence[Sequence[float]],
    labels: Sequence[bool | int],
    *,
    random_state: int,
) -> tuple[CorrectnessProbabilityCalibrator, CalibrationMetrics]:
    """Fit logistic calibration, with a declared fallback for one-class data."""
    if not features or len(features) != len(labels):
        raise ValueError("features and labels must have the same non-zero length")
    matrix = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"features must have shape (n, {len(FEATURE_NAMES)})")
    if not np.isfinite(matrix).all():
        raise ValueError("confidence features must be finite")
    if not set(targets.tolist()).issubset({0, 1}):
        raise ValueError("correctness labels must be binary")

    if len(set(targets.tolist())) == 1:
        probability = (float(targets.sum()) + 1.0) / (len(targets) + 2.0)
        calibrator = CorrectnessProbabilityCalibrator(
            method="laplace_constant_one_class",
            coefficients=tuple(0.0 for _ in FEATURE_NAMES),
            intercept=0.0,
            constant_probability=probability,
        )
    else:
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )
        model.fit(matrix, targets)
        calibrator = CorrectnessProbabilityCalibrator(
            method="logistic_correctness_calibration",
            coefficients=tuple(float(value) for value in model.coef_[0]),
            intercept=float(model.intercept_[0]),
        )

    probabilities = [calibrator.predict(row) for row in matrix]
    metrics = evaluate_calibration(targets.tolist(), probabilities, num_bins=10)
    return calibrator, metrics
