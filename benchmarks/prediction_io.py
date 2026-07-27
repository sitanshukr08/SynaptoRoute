"""Machine-readable prediction artifacts for research evaluations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from synaptoroute.models import RouterResult


PREDICTION_SCHEMA_VERSION = 1


def query_digest(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def prediction_record(
    *,
    example_id: str,
    query: str,
    expected_route: str | None,
    result: RouterResult,
    latency_seconds: float,
    include_query: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one privacy-conscious prediction record.

    The query text is omitted by default. Public-dataset experiments may opt in
    to storing it, while private deployments can retain only its digest.
    """
    if not example_id:
        raise ValueError("example_id must not be empty")
    if latency_seconds < 0:
        raise ValueError("latency_seconds must be non-negative")

    record: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "example_id": example_id,
        "query_sha256": query_digest(query),
        "expected_route": expected_route,
        "predicted_route": result.route_name,
        "matched": result.matched,
        "correct": result.route_name == expected_route,
        "score": result.score,
        "margin": result.margin,
        "decision_reason": result.decision_reason.value,
        "candidates": [candidate.model_dump(mode="json") for candidate in result.candidates],
        "latency_ms": latency_seconds * 1000.0,
        "metadata": dict(metadata or {}),
    }
    if include_query:
        record["query"] = query
    return record


def validate_prediction_record(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
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
    missing = sorted(required - record.keys())
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
        return errors
    if record.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PREDICTION_SCHEMA_VERSION}")
    digest = record.get("query_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append("query_sha256 must be a SHA-256 hex digest")
    else:
        try:
            int(digest, 16)
        except ValueError:
            errors.append("query_sha256 must be a SHA-256 hex digest")
    if not isinstance(record.get("example_id"), str) or not record["example_id"]:
        errors.append("example_id must be a non-empty string")
    if not isinstance(record.get("matched"), bool):
        errors.append("matched must be boolean")
    if not isinstance(record.get("correct"), bool):
        errors.append("correct must be boolean")
    latency_ms = record.get("latency_ms")
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)) or latency_ms < 0:
        errors.append("latency_ms must be a non-negative number")
    if not isinstance(record.get("candidates"), list):
        errors.append("candidates must be a list")
    if not isinstance(record.get("metadata"), dict):
        errors.append("metadata must be an object")
    return errors


def write_prediction_jsonl(path: Path | str, records: Iterable[Mapping[str, Any]]) -> Path:
    """Validate and atomically write prediction records as JSON Lines."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    count = 0
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            for count, record in enumerate(records, start=1):
                errors = validate_prediction_record(record)
                if errors:
                    raise ValueError(f"invalid prediction record {count}: {errors}")
                output.write(json.dumps(dict(record), sort_keys=True, separators=(",", ":")))
                output.write("\n")
        if count == 0:
            raise ValueError("at least one prediction record is required")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def read_prediction_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            errors = validate_prediction_record(record)
            if errors:
                raise ValueError(f"invalid prediction record at line {line_number}: {errors}")
            records.append(record)
    if not records:
        raise ValueError("prediction artifact is empty")
    return records
