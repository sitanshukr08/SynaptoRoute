"""Promote two independently reproduced candidate runs into a verified claim."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from benchmarks.manifest_schema import FULL_GIT_SHA, sha256_file, validate_manifest
except ModuleNotFoundError:
    from manifest_schema import FULL_GIT_SHA, sha256_file, validate_manifest


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _validate_candidate(label: str, run: dict, repo_root: Path) -> None:
    if run.get("status") != "unverified" or run.get("schema_version") != 2:
        raise ValueError(f"{label} must be an unverified schema v2 run")
    if not isinstance(run.get("run_id"), str) or not run["run_id"].strip():
        raise ValueError(f"{label} must record run_id")
    if run.get("working_tree_dirty") is not False:
        raise ValueError(f"{label} must come from a clean working tree")
    if run.get("exit_status") != 0:
        raise ValueError(f"{label} did not complete successfully")
    if not FULL_GIT_SHA.fullmatch(str(run.get("git_commit", ""))):
        raise ValueError(f"{label} must record a full git commit")
    if not isinstance(run.get("configuration"), dict):
        raise ValueError(f"{label} must record configuration")
    machine_id = run.get("environment", {}).get("machine_id")
    if not isinstance(machine_id, str) or not machine_id.strip():
        raise ValueError(f"{label} must record environment.machine_id")

    lock = run.get("dependency_lock")
    if not isinstance(lock, dict) or not {"path", "sha256"}.issubset(lock):
        raise ValueError(f"{label} must record dependency_lock")
    lock_path = _evidence_path(str(lock["path"]), repo_root)
    if not lock_path.is_file() or lock_path.stat().st_size == 0:
        raise ValueError(f"{label} dependency lock is missing or empty")
    if lock["sha256"] != sha256_file(lock_path):
        raise ValueError(f"{label} dependency lock hash does not match")

    evidence = run.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"{label} must record evidence")
    for field in ("script_path", "raw_output_path", "raw_output_sha256"):
        if not isinstance(evidence.get(field), str) or not evidence[field].strip():
            raise ValueError(f"{label} evidence.{field} is required")
    script_path = _evidence_path(evidence["script_path"], repo_root)
    raw_path = _evidence_path(evidence["raw_output_path"], repo_root)
    if not script_path.is_file() or script_path.stat().st_size == 0:
        raise ValueError(f"{label} benchmark script is missing or empty")
    if not raw_path.is_file() or raw_path.stat().st_size == 0:
        raise ValueError(f"{label} raw output is missing or empty")
    if evidence["raw_output_sha256"] != sha256_file(raw_path):
        raise ValueError(f"{label} raw output hash does not match")


def _validate_attestation(
    attestation: dict,
    *,
    original_run_id: str,
    reproduction_run_id: str,
    claim: str,
    archive_uri: str,
    archive_sha256: str,
) -> None:
    required = {
        "schema_version",
        "decision",
        "reviewer",
        "reviewed_at_utc",
        "original_run_id",
        "reproduction_run_id",
        "claim",
        "archive_uri",
        "archive_sha256",
        "notes",
    }
    missing = sorted(required - attestation.keys())
    if missing:
        raise ValueError(f"reviewer attestation missing: {', '.join(missing)}")
    if attestation["schema_version"] != 1:
        raise ValueError("reviewer attestation schema_version must be 1")
    if attestation["decision"] != "approve":
        raise ValueError("reviewer attestation decision must be 'approve'")
    for field in ("reviewer", "reviewed_at_utc", "notes"):
        if not isinstance(attestation[field], str) or not attestation[field].strip():
            raise ValueError(f"reviewer attestation {field} must be non-empty")
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        attestation["reviewed_at_utc"],
    ):
        raise ValueError("reviewer attestation reviewed_at_utc must be UTC ISO-8601")
    expected = {
        "original_run_id": original_run_id,
        "reproduction_run_id": reproduction_run_id,
        "claim": claim,
        "archive_uri": archive_uri,
        "archive_sha256": archive_sha256,
    }
    for field, value in expected.items():
        if attestation[field] != value:
            raise ValueError(f"reviewer attestation {field} does not match promotion input")


def _manifest_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def promote(
    original: dict,
    reproduction: dict,
    *,
    reviewer_attestation: Path | str,
    claim: str,
    archive_uri: str,
    archive_sha256: str,
    repo_root: Path | str = ".",
) -> dict:
    root = Path(repo_root)
    for label, run in (("original", original), ("reproduction", reproduction)):
        _validate_candidate(label, run, root)

    comparable_fields = ("benchmark", "git_commit", "dataset", "configuration")
    for field in comparable_fields:
        if original.get(field) != reproduction.get(field):
            raise ValueError(f"independent runs differ in {field}")
    original_machine = original.get("environment", {}).get("machine_id")
    reproduction_machine = reproduction.get("environment", {}).get("machine_id")
    if not original_machine or original_machine == reproduction_machine:
        raise ValueError("independent runs must use different machine_id values")
    if original["run_id"] == reproduction["run_id"]:
        raise ValueError("independent runs must have different run_id values")
    if not claim.strip():
        raise ValueError("claim is required")
    if not archive_uri.strip():
        raise ValueError("archive_uri is required")
    if not re.fullmatch(r"[0-9a-f]{64}", archive_sha256):
        raise ValueError("archive_sha256 must be a lowercase SHA-256 digest")
    if original["dependency_lock"]["sha256"] != reproduction["dependency_lock"]["sha256"]:
        raise ValueError("independent runs differ in dependency lock")

    attestation_path = _evidence_path(str(reviewer_attestation), root)
    if not attestation_path.is_file() or attestation_path.stat().st_size == 0:
        raise ValueError("reviewer attestation is missing or empty")
    try:
        attestation = _load(attestation_path)
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError("reviewer attestation must be readable JSON") from error
    if not isinstance(attestation, dict):
        raise ValueError("reviewer attestation must be a JSON object")
    _validate_attestation(
        attestation,
        original_run_id=original["run_id"],
        reproduction_run_id=reproduction["run_id"],
        claim=claim,
        archive_uri=archive_uri,
        archive_sha256=archive_sha256,
    )

    promoted = dict(original)
    promoted.update(
        {
            "status": "verified",
            "paper_evidence_eligible": True,
            "claim": claim,
            "review": {
                "reviewer": attestation["reviewer"],
                "reviewed_at_utc": attestation["reviewed_at_utc"],
                "original_run_id": original["run_id"],
                "reproduction_run_id": reproduction["run_id"],
                "decision": attestation["decision"],
                "notes": attestation["notes"],
                "attestation_path": _manifest_path(attestation_path, root),
                "attestation_sha256": sha256_file(attestation_path),
            },
            "reproduction": {
                "environment": reproduction["environment"],
                "metrics": reproduction["metrics"],
                "evidence": reproduction["evidence"],
            },
            "archive": {"uri": archive_uri, "sha256": archive_sha256},
        }
    )
    promoted.pop("missing_evidence", None)
    return promoted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--archive-uri", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    promoted = promote(
        _load(args.original),
        _load(args.reproduction),
        reviewer_attestation=args.attestation,
        claim=args.claim,
        archive_uri=args.archive_uri,
        archive_sha256=args.archive_sha256,
        repo_root=repo_root,
    )
    errors = validate_manifest(promoted, repo_root=repo_root)
    if errors:
        print(json.dumps({"errors": errors}, indent=2), file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(promoted, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
