"""Benchmark manifest validation for truth-first release gates."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any


VALID_STATUSES = {"verified", "unverified", "retracted"}
FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
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
        if not FULL_GIT_SHA.fullmatch(str(manifest.get("git_commit", ""))):
            errors.append("verified manifests require a full 40-character git_commit")
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
        if manifest.get("schema_version", 1) >= 2:
            if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"].strip():
                errors.append("verified schema v2 manifests require run_id")
            if manifest.get("paper_evidence_eligible") is not True:
                errors.append("verified schema v2 manifests require paper_evidence_eligible=true")
            if not isinstance(manifest.get("claim"), str) or not manifest["claim"].strip():
                errors.append("verified schema v2 manifests require a non-empty claim")
            if manifest.get("exit_status") != 0:
                errors.append("verified schema v2 manifests require exit_status=0")
            if not isinstance(manifest.get("configuration"), dict):
                errors.append("verified schema v2 manifests require configuration")
            lock = manifest.get("dependency_lock")
            if not isinstance(lock, dict) or not {
                "path",
                "sha256",
            }.issubset(lock):
                errors.append("verified schema v2 manifests require dependency_lock")
            else:
                lock_path = _as_path(root, lock.get("path"))
                if lock_path is None or not lock_path.exists() or lock_path.stat().st_size == 0:
                    errors.append("verified schema v2 dependency_lock.path must be non-empty")
                elif lock.get("sha256") != sha256_file(lock_path):
                    errors.append("verified schema v2 dependency_lock.sha256 must match")
            review = manifest.get("review")
            if not isinstance(review, dict) or not {
                "reviewer",
                "reviewed_at_utc",
                "original_run_id",
                "reproduction_run_id",
                "decision",
                "notes",
                "attestation_path",
                "attestation_sha256",
            }.issubset(review):
                errors.append("verified schema v2 manifests require independent review")
            else:
                for field in (
                    "reviewer",
                    "reviewed_at_utc",
                    "original_run_id",
                    "reproduction_run_id",
                    "notes",
                ):
                    if not isinstance(review.get(field), str) or not review[field].strip():
                        errors.append(f"verified review.{field} must be non-empty")
                if review.get("original_run_id") == review.get("reproduction_run_id"):
                    errors.append("verified review requires distinct original and reproduction runs")
                if review.get("decision") != "approve":
                    errors.append("verified review.decision must be approve")
                attestation_path = _as_path(root, review.get("attestation_path"))
                if (
                    attestation_path is None
                    or not attestation_path.exists()
                    or attestation_path.stat().st_size == 0
                ):
                    errors.append("verified review.attestation_path must be non-empty")
                elif review.get("attestation_sha256") != sha256_file(attestation_path):
                    errors.append("verified review.attestation_sha256 must match")
                else:
                    try:
                        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        errors.append("verified review attestation must be readable JSON")
                    else:
                        expected_attestation = {
                            "schema_version": 1,
                            "decision": "approve",
                            "reviewer": review.get("reviewer"),
                            "reviewed_at_utc": review.get("reviewed_at_utc"),
                            "original_run_id": review.get("original_run_id"),
                            "reproduction_run_id": review.get("reproduction_run_id"),
                            "claim": manifest.get("claim"),
                            "archive_uri": manifest.get("archive", {}).get("uri"),
                            "archive_sha256": manifest.get("archive", {}).get("sha256"),
                            "notes": review.get("notes"),
                        }
                        if not isinstance(attestation, dict) or any(
                            attestation.get(field) != value
                            for field, value in expected_attestation.items()
                        ):
                            errors.append("verified review attestation does not match manifest")
            archive = manifest.get("archive")
            if not isinstance(archive, dict) or not {"uri", "sha256"}.issubset(archive):
                errors.append("verified schema v2 manifests require immutable archive metadata")
            else:
                if not isinstance(archive.get("uri"), str) or not archive["uri"].strip():
                    errors.append("verified archive.uri must be non-empty")
                if not re.fullmatch(r"[0-9a-f]{64}", str(archive.get("sha256", ""))):
                    errors.append("verified archive.sha256 must be a lowercase SHA-256 digest")
            if isinstance(environment, dict):
                machine_id = environment.get("machine_id")
                if not isinstance(machine_id, str) or not machine_id.strip():
                    errors.append("verified schema v2 environment.machine_id must be concrete")
            reproduction = manifest.get("reproduction")
            if not isinstance(reproduction, dict) or not {"environment", "metrics", "evidence"}.issubset(reproduction):
                errors.append("verified schema v2 manifests require reproduction evidence")
            elif isinstance(environment, dict):
                reproduced_machine = reproduction.get("environment", {}).get("machine_id")
                if not reproduced_machine or reproduced_machine == environment.get("machine_id"):
                    errors.append("verified schema v2 reproduction must use a different machine")
    elif status == "unverified":
        missing_evidence = manifest.get("missing_evidence")
        if not isinstance(missing_evidence, list) or not missing_evidence:
            errors.append("unverified manifests require non-empty missing_evidence")
        if manifest.get("schema_version", 1) >= 2 and manifest.get("paper_evidence_eligible") is not False:
            errors.append("unverified schema v2 manifests require paper_evidence_eligible=false")
    elif status == "retracted":
        if not manifest.get("retraction_reason"):
            errors.append("retracted manifests require retraction_reason")

    return errors


def validate_manifest_file(path: Path | str, repo_root: Path | str = ".") -> list[str]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return validate_manifest(manifest, repo_root=repo_root)
