"""Shared, deterministic statistics for unverified matrix-run analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def summarize_values(
    values: Sequence[float],
    *,
    bootstrap_repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Summarize repetition-level values with a percentile bootstrap interval."""

    if not values:
        raise ValueError("cannot summarize an empty sequence")
    if bootstrap_repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("summary values must be finite")

    mean = float(np.mean(array))
    if len(array) == 1 or np.all(array == array[0]):
        lower = upper = mean
    else:
        sampled_indexes = rng.integers(
            0,
            len(array),
            size=(bootstrap_repetitions, len(array)),
        )
        bootstrap_means = np.mean(array[sampled_indexes], axis=1)
        lower, upper = (float(value) for value in np.percentile(bootstrap_means, [2.5, 97.5]))
    return {
        "n": len(array),
        "mean": mean,
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else None,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "confidence_interval_95": [lower, upper],
    }


def concise_matrix_verification(report: Mapping[str, Any]) -> dict[str, Any]:
    """Retain integrity outcomes without copying a potentially large observation list."""

    return {
        "verification_status": report["verification_status"],
        "command_count": report["command_count"],
        "log_hashes_verified": report["log_hashes_verified"],
        "raw_and_state_hashes_verified": report["raw_and_state_hashes_verified"],
        "invariants_passed": report["invariants"]["passed"],
        "outcome_observation_count": report["outcome_observation_count"],
        "environment_evidence_verified": report["environment_evidence_verified"],
    }
