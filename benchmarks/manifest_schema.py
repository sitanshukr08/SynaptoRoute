"""Benchmark manifest validation for truth-first release gates."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


VALID_STATUSES = {"verified", "unverified", "retracted"}
BASE_REQUIRED = {
    "schema_version",
    "benchmark",
    "status",
    "timestamp_utc",
    "git_commit",
    "command",
    "environment",
    "dataset",
    "metrics",
    "evidence",
}
ENVIRONMENT_REQUIRED = {"python_version", "platform", "cpu", "gpu"}
EVIDENCE_REQUIRED = {"script_path", "raw_output_path", "timing_unit", "notes"}
VERIFIED_DATASET_REQUIRED = {
    "name",
    "version",
    "split",
    "seed",
    "route_count",
    "query_count",
    "license",
}


def _as_path(repo_root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = repo_root / path
    return path


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest for an evidence file, normalized for cross-platform line endings."""
    p = Path(path)
    content = p.read_bytes()
    if p.suffix.lower() in (".log", ".json", ".txt", ".csv", ".py", ".md"):
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def validate_manifest(manifest: dict[str, Any], repo_root: Path | str = ".") -> list[str]:
    """Return validation errors for a benchmark manifest."""
    root = Path(repo_root)
    errors: list[str] = []

    missing = sorted(BASE_REQUIRED - manifest.keys())
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
        return errors

    status = manifest.get("status")
    if status not in VALID_STATUSES:
        errors.append(f"status must be one of {sorted(VALID_STATUSES)}")

    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        errors.append("environment must be an object")
    else:
        env_missing = sorted(ENVIRONMENT_REQUIRED - environment.keys())
        if env_missing:
            errors.append(f"environment missing: {', '.join(env_missing)}")

    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object")
        evidence = {}
    else:
        evidence_missing = sorted(EVIDENCE_REQUIRED - evidence.keys())
        if evidence_missing:
            errors.append(f"evidence missing: {', '.join(evidence_missing)}")

    command = manifest.get("command")
    if command is not None and (
        not isinstance(command, list) or not command or not all(isinstance(part, str) and part for part in command)
    ):
        errors.append("command must be null or a non-empty list of strings")

    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics must be an object")

    script_path = _as_path(root, evidence.get("script_path"))
    if script_path is not None and (not script_path.exists() or script_path.stat().st_size == 0):
        errors.append(f"script_path does not exist or is empty: {evidence.get('script_path')}")

    if status == "verified":
        if command is None:
            errors.append("verified manifests require command")
        if not manifest.get("git_commit") or manifest.get("git_commit") == "unknown":
            errors.append("verified manifests require a concrete git_commit")
        if manifest.get("working_tree_dirty") is not False:
            errors.append("verified manifests require working_tree_dirty=false")
        if not script_path:
            errors.append("verified manifests require evidence.script_path")
        raw_output_path = _as_path(root, evidence.get("raw_output_path"))
        if raw_output_path is None or not raw_output_path.exists() or raw_output_path.stat().st_size == 0:
            errors.append("verified manifests require a non-empty raw_output_path")
        elif evidence.get("raw_output_sha256") != sha256_file(raw_output_path):
            errors.append("verified manifests require a matching evidence.raw_output_sha256")
        if not metrics:
            errors.append("verified manifests require non-empty metrics")
        if isinstance(environment, dict):
            for field in sorted(ENVIRONMENT_REQUIRED):
                value = environment.get(field)
                if value in (None, "", "unknown"):
                    errors.append(f"verified environment.{field} must be concrete")
        dataset = manifest.get("dataset")
        if not isinstance(dataset, dict):
            errors.append("verified manifests require dataset metadata as an object")
        else:
            dataset_missing = sorted(VERIFIED_DATASET_REQUIRED - dataset.keys())
            if dataset_missing:
                errors.append(f"verified dataset metadata missing: {', '.join(dataset_missing)}")
            for field in ("name", "version", "split", "license"):
                value = dataset.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"verified dataset.{field} must be a non-empty string")
            seed = dataset.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, (int, str)) or seed == "":
                errors.append("verified dataset.seed must be an integer or non-empty string")
            for count_field in ("route_count", "query_count"):
                count = dataset.get(count_field)
                if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                    errors.append(f"verified dataset.{count_field} must be a positive integer")
        timing_unit = evidence.get("timing_unit")
        if timing_unit in (None, "", "varies", "unknown"):
            errors.append("verified manifests require an explicit evidence.timing_unit")
    elif status == "unverified":
        missing_evidence = manifest.get("missing_evidence")
        if not isinstance(missing_evidence, list) or not missing_evidence:
            errors.append("unverified manifests require non-empty missing_evidence")
    elif status == "retracted":
        if not manifest.get("retraction_reason"):
            errors.append("retracted manifests require retraction_reason")

    return errors


def validate_manifest_file(path: Path | str, repo_root: Path | str = ".") -> list[str]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return validate_manifest(manifest, repo_root=repo_root)
