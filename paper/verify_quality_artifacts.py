"""Independently verify one intent-quality experiment artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


SHA256 = re.compile(r"^[0-9a-f]{64}$")
FEATURE_NAMES = (
    "top_score",
    "margin",
    "acceptance_confidence",
    "score_present",
    "margin_present",
    "decision_matched",
)
MAX_REPORTED_ERRORS = 200


class QualityArtifactVerificationError(RuntimeError):
    """Raised when an experiment's quality evidence is incomplete or inconsistent."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("quality artifact verification failed:\n- " + "\n- ".join(errors))


def _add_error(errors: list[str], message: str) -> None:
    if len(errors) < MAX_REPORTED_ERRORS:
        errors.append(message)
    elif len(errors) == MAX_REPORTED_ERRORS:
        errors.append("additional verification errors omitted")


def _sha256_file(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in {".log", ".json", ".txt", ".csv", ".py", ".md"}:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def _load_json(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        _add_error(errors, f"{label} is missing or empty: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _add_error(errors, f"{label} is not readable JSON: {error}")
        return None
    if not isinstance(value, dict):
        _add_error(errors, f"{label} must be a JSON object")
        return None
    return value


def _artifact_path(
    run_dir: Path,
    reference: Any,
    expected_name: str,
    label: str,
    errors: list[str],
) -> Path:
    """Resolve a relocatable artifact while requiring its frozen filename."""
    path = run_dir / expected_name
    if not isinstance(reference, str) or not reference.strip():
        _add_error(errors, f"{label} path is missing")
    else:
        normalized = reference.replace("\\", "/")
        reference_path = PurePosixPath(normalized)
        if ".." in reference_path.parts or reference_path.name != expected_name:
            _add_error(errors, f"{label} path does not name {expected_name}")
    if not path.is_file() or path.stat().st_size == 0:
        _add_error(errors, f"{label} is missing or empty: {path}")
    return path


def _verify_sha(path: Path, expected: Any, label: str, errors: list[str]) -> None:
    if not isinstance(expected, str) or not SHA256.fullmatch(expected):
        _add_error(errors, f"{label} does not declare a lowercase SHA-256 digest")
    elif path.is_file() and _sha256_file(path) != expected:
        _add_error(errors, f"{label} SHA-256 mismatch")


def _verify_unverified_status(value: Mapping[str, Any], label: str, errors: list[str]) -> None:
    if value.get("status") != "unverified" or value.get("paper_evidence_eligible") is not False:
        _add_error(errors, f"{label} is not explicitly unverified and paper-ineligible")


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _read_predictions(
    path: Path,
    *,
    system_name: str,
    phase: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file() or path.stat().st_size == 0:
        return records
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        _add_error(errors, f"{system_name} {phase} predictions are unreadable: {error}")
        return records

    required = {
        "schema_version",
        "example_id",
        "query_sha256",
        "expected_route",
        "predicted_route",
        "matched",
        "correct",
        "score",
        "margin",
        "decision_reason",
        "candidates",
        "latency_ms",
        "metadata",
    }
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        prefix = f"{system_name} {phase} line {line_number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            _add_error(errors, f"{prefix} is invalid JSON: {error}")
            continue
        if not isinstance(record, dict):
            _add_error(errors, f"{prefix} must be an object")
            continue
        missing = sorted(required - record.keys())
        if missing:
            _add_error(errors, f"{prefix} is missing fields: {', '.join(missing)}")
            continue
        if "query" in record:
            _add_error(errors, f"{prefix} contains raw query text")
        if record.get("schema_version") != 1:
            _add_error(errors, f"{prefix} has an unsupported schema version")
        if not isinstance(record.get("example_id"), str) or not record["example_id"]:
            _add_error(errors, f"{prefix} has an invalid example_id")
        if not isinstance(record.get("query_sha256"), str) or not SHA256.fullmatch(
            record["query_sha256"]
        ):
            _add_error(errors, f"{prefix} has an invalid query_sha256")
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            _add_error(errors, f"{prefix} metadata must be an object")
            metadata = {}
        if metadata.get("system") != system_name:
            _add_error(errors, f"{prefix} names a different system")
        if metadata.get("phase") != phase:
            _add_error(errors, f"{prefix} names a different phase")
        if not isinstance(record.get("matched"), bool):
            _add_error(errors, f"{prefix} matched must be boolean")
        if not isinstance(record.get("correct"), bool):
            _add_error(errors, f"{prefix} correct must be boolean")
        elif record["correct"] != (record.get("predicted_route") == record.get("expected_route")):
            _add_error(errors, f"{prefix} correctness does not match the recorded routes")
        if isinstance(record.get("matched"), bool) and record["matched"] != (
            record.get("predicted_route") is not None
        ):
            _add_error(errors, f"{prefix} matched state does not match predicted_route")
        if not _is_finite_number(record.get("latency_ms")) or float(record["latency_ms"]) < 0:
            _add_error(errors, f"{prefix} latency_ms must be finite and non-negative")
        for field in ("score", "margin"):
            if record.get(field) is not None and not _is_finite_number(record[field]):
                _add_error(errors, f"{prefix} {field} must be finite or null")
        if not isinstance(record.get("decision_reason"), str) or not record["decision_reason"]:
            _add_error(errors, f"{prefix} decision_reason must be a non-empty string")
        if not isinstance(record.get("candidates"), list):
            _add_error(errors, f"{prefix} candidates must be an array")
        records.append(record)

    if not records:
        _add_error(errors, f"{system_name} {phase} predictions are empty")
        return records
    identifiers = [record.get("example_id") for record in records]
    if len(set(identifiers)) != len(identifiers):
        _add_error(errors, f"{system_name} {phase} contains duplicate example IDs")
    return records


def _compare_values(
    recorded: Any,
    recomputed: Any,
    label: str,
    errors: list[str],
    *,
    tolerance: float = 1e-7,
) -> None:
    if recomputed is None:
        if recorded is not None:
            _add_error(errors, f"{label} must be null")
        return
    if isinstance(recomputed, dict):
        if not isinstance(recorded, dict):
            _add_error(errors, f"{label} must be an object")
            return
        for key, value in recomputed.items():
            _compare_values(recorded.get(key), value, f"{label}.{key}", errors, tolerance=tolerance)
        return
    if isinstance(recomputed, (list, tuple)):
        if not isinstance(recorded, list) or len(recorded) != len(recomputed):
            _add_error(errors, f"{label} has a different length")
            return
        for index, value in enumerate(recomputed):
            _compare_values(
                recorded[index],
                value,
                f"{label}[{index}]",
                errors,
                tolerance=tolerance,
            )
        return
    if isinstance(recomputed, bool):
        if recorded is not recomputed:
            _add_error(errors, f"{label} differs: expected {recomputed!r}, found {recorded!r}")
        return
    if isinstance(recomputed, (int, float)) and not isinstance(recomputed, bool):
        if not _is_finite_number(recorded) or not math.isclose(
            float(recorded), float(recomputed), rel_tol=tolerance, abs_tol=tolerance
        ):
            _add_error(errors, f"{label} differs: expected {recomputed!r}, found {recorded!r}")
        return
    if recorded != recomputed:
        _add_error(errors, f"{label} differs: expected {recomputed!r}, found {recorded!r}")


def _calibration_metrics(labels: Sequence[bool], probabilities: Sequence[float]) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int32)
    y_prob = np.asarray(probabilities, dtype=np.float32)
    num_bins = 10
    boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    accuracies: list[float] = []
    confidences: list[float] = []
    counts: list[int] = []
    ece = 0.0
    mce = 0.0
    for index in range(num_bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == num_bins - 1:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        else:
            in_bin = (y_prob >= lower) & (y_prob < upper)
        count = int(np.sum(in_bin))
        counts.append(count)
        if count:
            accuracy = float(np.mean(y_true[in_bin]))
            confidence = float(np.mean(y_prob[in_bin]))
            gap = abs(accuracy - confidence)
            ece += (count / len(y_true)) * gap
            mce = max(mce, gap)
            accuracies.append(accuracy)
            confidences.append(confidence)
        else:
            accuracies.append(0.0)
            confidences.append(0.0)
    return {
        "expected_calibration_error": ece,
        "max_calibration_error": mce,
        "brier_score": float(np.mean((y_prob - y_true) ** 2)),
        "num_samples": len(y_true),
        "num_bins": num_bins,
        "bin_accuracies": accuracies,
        "bin_confidences": confidences,
        "bin_counts": counts,
    }


def _validate_probability_model(
    artifact: Mapping[str, Any],
    *,
    system_name: str,
    errors: list[str],
) -> tuple[tuple[float, ...], float, float | None] | None:
    if artifact.get("feature_names") != list(FEATURE_NAMES):
        _add_error(errors, f"{system_name} probability model has unexpected features")
    coefficients = artifact.get("coefficients")
    intercept = artifact.get("intercept")
    constant = artifact.get("constant_probability")
    if (
        not isinstance(coefficients, list)
        or len(coefficients) != len(FEATURE_NAMES)
        or any(not _is_finite_number(value) for value in coefficients)
        or not _is_finite_number(intercept)
    ):
        _add_error(errors, f"{system_name} probability model parameters are invalid")
        return None
    if constant is not None and (not _is_finite_number(constant) or not 0.0 <= constant <= 1.0):
        _add_error(errors, f"{system_name} constant probability is invalid")
        return None
    normalized_coefficients = tuple(float(cast(float, value)) for value in coefficients)
    normalized_intercept = float(cast(float, intercept))
    normalized_constant = float(cast(float, constant)) if constant is not None else None
    method = artifact.get("method")
    if method == "logistic_correctness_calibration" and constant is not None:
        _add_error(errors, f"{system_name} logistic model must not declare a constant probability")
    elif method == "laplace_constant_one_class":
        if constant is None:
            _add_error(errors, f"{system_name} one-class model lacks its constant probability")
        if any(value != 0.0 for value in normalized_coefficients) or normalized_intercept != 0.0:
            _add_error(errors, f"{system_name} one-class model has non-zero logistic parameters")
    elif method not in {"logistic_correctness_calibration", "laplace_constant_one_class"}:
        _add_error(errors, f"{system_name} probability calibration method is unsupported")
    return normalized_coefficients, normalized_intercept, normalized_constant


def _predict_probability(
    model: tuple[tuple[float, ...], float, float | None],
    features: Sequence[float],
) -> float:
    coefficients, intercept, constant = model
    if constant is not None:
        return constant
    linear = intercept + sum(coefficient * value for coefficient, value in zip(coefficients, features))
    if linear >= 0:
        return 1.0 / (1.0 + math.exp(-linear))
    exponential = math.exp(linear)
    return exponential / (1.0 + exponential)


def _features_from_test_record(record: Mapping[str, Any]) -> tuple[float, ...]:
    score = record.get("score")
    margin = record.get("margin")
    confidence = record.get("metadata", {}).get("acceptance_confidence")
    score_present = _is_finite_number(score)
    margin_present = _is_finite_number(margin)
    confidence_present = _is_finite_number(confidence)
    return (
        float(cast(float, score)) if score_present else -1.0,
        float(cast(float, margin)) if margin_present else 0.0,
        float(cast(float, confidence)) if confidence_present else -2.0,
        float(score_present),
        float(margin_present),
        float(bool(record.get("matched"))),
    )


def _summary_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = [record.get("expected_route") or "OOD" for record in records]
    predicted = [record.get("predicted_route") or "OOD" for record in records]
    known_indices = [index for index, label in enumerate(expected) if label != "OOD"]
    ood_indices = [index for index, label in enumerate(expected) if label == "OOD"]
    accepted_indices = [index for index, label in enumerate(predicted) if label != "OOD"]
    accepted_known = [index for index in known_indices if predicted[index] != "OOD"]
    confidences_raw = [
        record.get("metadata", {}).get("acceptance_confidence") for record in records
    ]
    finite = [float(value) for value in confidences_raw if _is_finite_number(value)]
    floor = min(finite, default=0.0) - 1.0
    confidences = np.asarray(
        [float(value) if _is_finite_number(value) else floor for value in confidences_raw],
        dtype=np.float64,
    )
    ranked = sorted(range(len(records)), key=lambda index: (-confidences[index], index))
    cumulative_errors = 0
    selective_risks: list[float] = []
    for rank, index in enumerate(ranked, start=1):
        cumulative_errors += not bool(records[index].get("metadata", {}).get("raw_top_correct"))
        selective_risks.append(cumulative_errors / rank)

    ood_targets = np.asarray([label == "OOD" for label in expected], dtype=np.int8)
    if len(set(ood_targets.tolist())) == 2:
        ood_scores = -confidences
        false_positive_rates, true_positive_rates, _ = roc_curve(ood_targets, ood_scores)
        candidates = false_positive_rates[true_positive_rates >= 0.95]
        ood_metrics = {
            "ood_auroc": float(roc_auc_score(ood_targets, ood_scores)),
            "ood_auprc": float(average_precision_score(ood_targets, ood_scores)),
            "ood_fpr_at_95_tpr": float(np.min(candidates)) if len(candidates) else None,
        }
    else:
        ood_metrics = {"ood_auroc": None, "ood_auprc": None, "ood_fpr_at_95_tpr": None}

    latencies = [float(record["latency_ms"]) for record in records]
    correct = [predicted[index] == expected[index] for index in range(len(records))]
    selective_accuracy = (
        sum(correct[index] for index in accepted_indices) / len(accepted_indices)
        if accepted_indices
        else None
    )
    return {
        "query_count": len(records),
        "overall_accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "known_accuracy": (
            float(
                accuracy_score(
                    [expected[index] for index in known_indices],
                    [predicted[index] for index in known_indices],
                )
            )
            if known_indices
            else None
        ),
        "ood_recall": (
            sum(predicted[index] == "OOD" for index in ood_indices) / len(ood_indices)
            if ood_indices
            else None
        ),
        "coverage": len(accepted_indices) / len(records),
        "known_coverage": len(accepted_known) / len(known_indices) if known_indices else None,
        "selective_accuracy": selective_accuracy,
        "selective_risk_coverage_auc": float(np.mean(selective_risks)) if finite else None,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "decision_reasons": dict(sorted(Counter(record["decision_reason"] for record in records).items())),
        **ood_metrics,
    }


def _cohort(records: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple((str(record.get("example_id")), str(record.get("query_sha256"))) for record in records)


def _verify_policy_artifacts(
    run_dir: Path,
    *,
    system_name: str,
    system_summary: Mapping[str, Any],
    expected_count: int,
    errors: list[str],
) -> list[dict[str, Any]]:
    calibration = system_summary.get("calibration")
    if calibration is None:
        return []
    if not isinstance(calibration, dict):
        _add_error(errors, f"{system_name} calibration summary must be an object or null")
        return []
    predictions_path = run_dir / f"calibration_predictions_{system_name}.jsonl"
    records = _read_predictions(
        predictions_path,
        system_name=system_name,
        phase="policy_calibration",
        errors=errors,
    )
    if len(records) != expected_count:
        _add_error(
            errors,
            f"{system_name} policy prediction count differs: expected {expected_count}, found {len(records)}",
        )
    artifact_path = run_dir / f"calibration_{system_name}.json"
    artifact = _load_json(artifact_path, f"{system_name} policy artifact", errors)
    if artifact is not None:
        _verify_unverified_status(artifact, f"{system_name} policy artifact", errors)
        _verify_sha(
            predictions_path,
            artifact.get("source_predictions_sha256"),
            f"{system_name} policy predictions",
            errors,
        )
        dataset = artifact.get("dataset")
        if not isinstance(dataset, dict):
            _add_error(errors, f"{system_name} policy artifact dataset must be an object")
        else:
            if dataset.get("fit_count") != expected_count:
                _add_error(errors, f"{system_name} policy artifact fit_count differs")
            if dataset.get("exact_text_overlap_count") != 0:
                _add_error(errors, f"{system_name} policy artifact reports text overlap")
        _compare_values(
            artifact.get("policy"), calibration.get("policy"), f"{system_name} policy", errors
        )
        _compare_values(
            artifact.get("calibration_metrics"),
            calibration.get("metrics"),
            f"{system_name} policy metrics",
            errors,
        )
    return records


def _verify_probability_artifacts(
    run_dir: Path,
    *,
    system_name: str,
    system_summary: Mapping[str, Any],
    expected_count: int,
    errors: list[str],
) -> tuple[list[dict[str, Any]], tuple[tuple[float, ...], float, float | None] | None, str | None]:
    summary = system_summary.get("probability_calibration")
    if not isinstance(summary, dict):
        _add_error(errors, f"{system_name} probability calibration summary is missing")
        return [], None, None
    predictions_name = f"probability_calibration_predictions_{system_name}.jsonl"
    artifact_name = f"probability_calibration_{system_name}.json"
    predictions_path = _artifact_path(
        run_dir,
        summary.get("source_predictions_path"),
        predictions_name,
        f"{system_name} probability predictions",
        errors,
    )
    artifact_path = _artifact_path(
        run_dir,
        summary.get("artifact_path"),
        artifact_name,
        f"{system_name} probability model",
        errors,
    )
    _verify_sha(
        predictions_path,
        summary.get("source_predictions_sha256"),
        f"{system_name} probability predictions summary binding",
        errors,
    )
    _verify_sha(
        artifact_path,
        summary.get("artifact_sha256"),
        f"{system_name} probability model summary binding",
        errors,
    )
    records = _read_predictions(
        predictions_path,
        system_name=system_name,
        phase="probability_calibration",
        errors=errors,
    )
    if len(records) != expected_count:
        _add_error(
            errors,
            f"{system_name} probability prediction count differs: expected {expected_count}, found {len(records)}",
        )
    artifact = _load_json(artifact_path, f"{system_name} probability model", errors)
    if artifact is None:
        return records, None, None
    _verify_unverified_status(artifact, f"{system_name} probability model", errors)
    _verify_sha(
        predictions_path,
        artifact.get("source_predictions", {}).get("sha256")
        if isinstance(artifact.get("source_predictions"), dict)
        else None,
        f"{system_name} probability model source",
        errors,
    )
    source = artifact.get("source_predictions")
    if isinstance(source, dict):
        _artifact_path(
            run_dir,
            source.get("path"),
            predictions_name,
            f"{system_name} probability model source",
            errors,
        )
    if artifact.get("fit_count") != len(records) or summary.get("fit_count") != len(records):
        _add_error(errors, f"{system_name} probability fit_count differs from predictions")
    positive_count = sum(record.get("correct") is True for record in records)
    if artifact.get("positive_count") != positive_count or summary.get("positive_count") != positive_count:
        _add_error(errors, f"{system_name} probability positive_count differs from predictions")
    if summary.get("method") != artifact.get("method"):
        _add_error(errors, f"{system_name} probability method differs between summary and model")

    model = _validate_probability_model(artifact, system_name=system_name, errors=errors)
    features: list[tuple[float, ...]] = []
    for index, record in enumerate(records, start=1):
        values = record.get("metadata", {}).get("confidence_features")
        if (
            not isinstance(values, list)
            or len(values) != len(FEATURE_NAMES)
            or any(not _is_finite_number(value) for value in values)
        ):
            _add_error(errors, f"{system_name} probability line {index} has invalid confidence features")
            continue
        features.append(tuple(float(value) for value in values))
    if model is not None and len(features) == len(records):
        probabilities = [_predict_probability(model, row) for row in features]
        recomputed = _calibration_metrics(
            [bool(record["correct"]) for record in records], probabilities
        )
        _compare_values(
            artifact.get("fit_metrics"),
            recomputed,
            f"{system_name} probability fit metrics",
            errors,
        )
        if artifact.get("method") == "laplace_constant_one_class":
            expected_constant = (positive_count + 1.0) / (len(records) + 2.0)
            _compare_values(
                artifact.get("constant_probability"),
                expected_constant,
                f"{system_name} one-class Laplace probability",
                errors,
            )
    return records, model, str(artifact.get("method"))


def _verify_test_artifacts(
    run_dir: Path,
    *,
    system_name: str,
    system_summary: Mapping[str, Any],
    expected_count: int,
    model: tuple[tuple[float, ...], float, float | None] | None,
    method: str | None,
    errors: list[str],
) -> list[dict[str, Any]]:
    reliability_summary = system_summary.get("reliability")
    test_summary = system_summary.get("test")
    if not isinstance(reliability_summary, dict) or not isinstance(test_summary, dict):
        _add_error(errors, f"{system_name} test or reliability summary is missing")
        return []
    predictions_name = f"test_predictions_{system_name}.jsonl"
    predictions_path = run_dir / predictions_name
    records = _read_predictions(
        predictions_path,
        system_name=system_name,
        phase="test",
        errors=errors,
    )
    if len(records) != expected_count:
        _add_error(
            errors,
            f"{system_name} test prediction count differs: expected {expected_count}, found {len(records)}",
        )
    data_name = f"reliability_{system_name}.json"
    diagram_name = f"reliability_{system_name}.svg"
    data_path = _artifact_path(
        run_dir,
        reliability_summary.get("data_path"),
        data_name,
        f"{system_name} reliability data",
        errors,
    )
    diagram_path = _artifact_path(
        run_dir,
        reliability_summary.get("diagram_path"),
        diagram_name,
        f"{system_name} reliability diagram",
        errors,
    )
    _verify_sha(
        data_path,
        reliability_summary.get("data_sha256"),
        f"{system_name} reliability data summary binding",
        errors,
    )
    _verify_sha(
        diagram_path,
        reliability_summary.get("diagram_sha256"),
        f"{system_name} reliability diagram summary binding",
        errors,
    )
    _verify_sha(
        predictions_path,
        reliability_summary.get("source_predictions_sha256"),
        f"{system_name} test predictions summary binding",
        errors,
    )
    reliability = _load_json(data_path, f"{system_name} reliability data", errors)
    if reliability is None:
        return records
    _verify_unverified_status(reliability, f"{system_name} reliability data", errors)
    if reliability.get("system") != system_name:
        _add_error(errors, f"{system_name} reliability data names a different system")
    if reliability.get("probability_calibration_method") != method:
        _add_error(errors, f"{system_name} reliability data names a different method")
    source = reliability.get("source_predictions")
    if not isinstance(source, dict):
        _add_error(errors, f"{system_name} reliability source is missing")
    else:
        _artifact_path(
            run_dir,
            source.get("path"),
            predictions_name,
            f"{system_name} reliability source",
            errors,
        )
        _verify_sha(
            predictions_path,
            source.get("sha256"),
            f"{system_name} reliability source",
            errors,
        )
    try:
        root = ET.parse(diagram_path).getroot()
        if not root.tag.endswith("svg"):
            _add_error(errors, f"{system_name} reliability diagram is not SVG")
    except (OSError, ET.ParseError) as error:
        _add_error(errors, f"{system_name} reliability diagram is invalid XML: {error}")

    probabilities: list[float] = []
    for index, record in enumerate(records, start=1):
        probability = record.get("metadata", {}).get("correctness_probability")
        if not _is_finite_number(probability) or not 0.0 <= float(probability) <= 1.0:
            _add_error(errors, f"{system_name} test line {index} has an invalid correctness probability")
            continue
        probability = float(probability)
        probabilities.append(probability)
        if model is not None:
            expected_probability = _predict_probability(model, _features_from_test_record(record))
            if not math.isclose(probability, expected_probability, rel_tol=1e-10, abs_tol=1e-10):
                _add_error(errors, f"{system_name} test line {index} probability differs from model")
    if len(probabilities) == len(records):
        calibration = _calibration_metrics(
            [bool(record["correct"]) for record in records], probabilities
        )
        _compare_values(
            reliability.get("metrics"),
            calibration,
            f"{system_name} reliability metrics",
            errors,
        )
        _compare_values(
            test_summary,
            {
                "expected_calibration_error": calibration["expected_calibration_error"],
                "max_calibration_error": calibration["max_calibration_error"],
                "brier_score": calibration["brier_score"],
            },
            f"{system_name} test calibration",
            errors,
        )
    if records:
        _compare_values(
            test_summary,
            _summary_metrics(records),
            f"{system_name} test metrics",
            errors,
        )
    return records


def verify_quality_artifacts(summary_path: Path) -> dict[str, Any]:
    """Verify artifact bindings and independently recompute one quality run."""
    summary_path = summary_path.resolve()
    if summary_path.is_dir():
        summary_path = summary_path / "experiment_summary.json"
    run_dir = summary_path.parent
    errors: list[str] = []
    summary = _load_json(summary_path, "experiment summary", errors)
    if summary is None:
        raise QualityArtifactVerificationError(errors)
    _verify_unverified_status(summary, "experiment summary", errors)
    if summary.get("benchmark") != "external_intent_routing_experiment":
        _add_error(errors, "experiment summary has an unexpected benchmark identifier")
    dataset = summary.get("dataset")
    configuration = summary.get("configuration")
    systems = summary.get("systems")
    if not isinstance(dataset, dict):
        _add_error(errors, "experiment dataset metadata is missing")
        dataset = {}
    if not isinstance(configuration, dict):
        _add_error(errors, "experiment configuration is missing")
        configuration = {}
    if not isinstance(systems, dict) or not systems:
        _add_error(errors, "experiment systems are missing")
        systems = {}
    if dataset.get("exact_text_overlap_count") != 0:
        _add_error(errors, "dataset metadata reports exact text overlap")
    query_count = dataset.get("query_count")
    policy_count = configuration.get("policy_calibration_count")
    probability_count = configuration.get("probability_calibration_count")
    for value, label in (
        (query_count, "dataset query_count"),
        (policy_count, "policy calibration count"),
        (probability_count, "probability calibration count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            _add_error(errors, f"{label} must be a positive integer")

    phase_cohorts: dict[str, tuple[tuple[str, str], ...]] = {}
    total_records = 0
    for system_name, system_summary in sorted(systems.items()):
        if not isinstance(system_name, str) or not isinstance(system_summary, dict):
            _add_error(errors, "system summaries must map names to objects")
            continue
        policy_records = _verify_policy_artifacts(
            run_dir,
            system_name=system_name,
            system_summary=system_summary,
            expected_count=policy_count if isinstance(policy_count, int) else 0,
            errors=errors,
        )
        probability_records, model, method = _verify_probability_artifacts(
            run_dir,
            system_name=system_name,
            system_summary=system_summary,
            expected_count=probability_count if isinstance(probability_count, int) else 0,
            errors=errors,
        )
        test_records = _verify_test_artifacts(
            run_dir,
            system_name=system_name,
            system_summary=system_summary,
            expected_count=query_count if isinstance(query_count, int) else 0,
            model=model,
            method=method,
            errors=errors,
        )
        total_records += len(policy_records) + len(probability_records) + len(test_records)

        cohorts = {
            "policy": _cohort(policy_records),
            "probability": _cohort(probability_records),
            "test": _cohort(test_records),
        }
        for first, second in (("policy", "probability"), ("policy", "test"), ("probability", "test")):
            first_ids = {item[0] for item in cohorts[first]}
            second_ids = {item[0] for item in cohorts[second]}
            if first_ids & second_ids:
                _add_error(errors, f"{system_name} {first}/{second} example IDs overlap")
            first_hashes = {item[1] for item in cohorts[first]}
            second_hashes = {item[1] for item in cohorts[second]}
            if first_hashes & second_hashes:
                _add_error(errors, f"{system_name} {first}/{second} query hashes overlap")

        for phase, cohort in cohorts.items():
            if not cohort:
                continue
            reference = phase_cohorts.setdefault(phase, cohort)
            if cohort != reference:
                _add_error(errors, f"{system_name} {phase} cohort differs from other systems")

    if errors:
        raise QualityArtifactVerificationError(errors)
    return {
        "schema_version": 1,
        "verification_status": "valid_unverified_quality_run",
        "paper_evidence_eligible": False,
        "summary_path": summary_path.as_posix(),
        "summary_sha256": _sha256_file(summary_path),
        "dataset": {
            "name": dataset.get("name"),
            "revision": dataset.get("revision"),
            "seed": dataset.get("seed"),
            "query_count": query_count,
        },
        "system_count": len(systems),
        "systems": sorted(systems),
        "record_counts": {
            "policy_per_calibrated_system": policy_count,
            "probability_per_system": probability_count,
            "test_per_system": query_count,
            "total_records_checked": total_records,
        },
        "verified_invariants": [
            "artifact filename and SHA-256 bindings",
            "privacy-preserving prediction schema",
            "policy/probability/test split disjointness",
            "cross-system cohort alignment",
            "probability model application",
            "classification and selective metrics",
            "ECE, MCE, Brier score, and reliability bins",
            "unverified evidence status preserved",
        ],
        "warning": (
            "This validates an unverified run. It does not replace independent reproduction, "
            "immutable archival, or reviewer attestation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="experiment_summary.json or its directory")
    parser.add_argument("--report", type=Path, help="optional path for the verification report")
    args = parser.parse_args()
    try:
        report = verify_quality_artifacts(args.summary)
    except QualityArtifactVerificationError as error:
        print(str(error), file=sys.stderr)
        return 1
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
