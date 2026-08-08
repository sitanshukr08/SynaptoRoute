import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from benchmarks.manifest_schema import sha256_file
from paper.build_archive import ArchiveInput, build_archive
from paper.verify_archive import verify_archive


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _archive_fixture(tmp_path):
    repo = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    repo.mkdir()
    evidence.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "artifact@example.invalid")
    _git(repo, "config", "user.name", "Artifact Test")
    (repo / "benchmarks").mkdir()
    (repo / "paper").mkdir()
    (repo / "benchmarks" / "fixture.py").write_text("print('fixture')\n", encoding="utf-8")
    (repo / "paper" / "requirements-linux-py311.lock").write_text(
        "fixture==1.0\n",
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    raw_path = evidence / "raw.json"
    raw_path.write_text('{"result": true}\n', encoding="utf-8")
    lock_path = repo / "paper" / "requirements-linux-py311.lock"
    manifest = {
        "schema_version": 2,
        "run_id": "fixture-run",
        "benchmark": "fixture",
        "status": "unverified",
        "paper_evidence_eligible": False,
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "git_commit": commit,
        "working_tree_dirty": False,
        "command": [sys.executable, "benchmarks/fixture.py"],
        "exit_status": 0,
        "environment": {
            "python_version": "3.11.0",
            "platform": "test",
            "cpu": "test",
            "gpu": "none",
            "machine_id": "fixture-machine",
        },
        "dependency_lock": {
            "path": "paper/requirements-linux-py311.lock",
            "sha256": sha256_file(lock_path),
        },
        "configuration": {},
        "dataset": "fixture",
        "metrics": {"passed": True},
        "evidence": {
            "script_path": "benchmarks/fixture.py",
            "raw_output_path": raw_path.as_posix(),
            "raw_output_sha256": sha256_file(raw_path),
            "timing_unit": "milliseconds",
            "notes": "fixture",
        },
        "missing_evidence": ["independent reproduction"],
    }
    manifest_path = evidence / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return repo, evidence, raw_path


def test_archive_builder_is_deterministic_and_inventoried(tmp_path):
    repo, evidence, _ = _archive_fixture(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_report = build_archive(
        repo_root=repo,
        inputs=[ArchiveInput("original", evidence)],
        output_path=first,
    )
    second_report = build_archive(
        repo_root=repo,
        inputs=[ArchiveInput("original", evidence)],
        output_path=second,
    )

    assert first_report["archive_sha256"] == second_report["archive_sha256"]
    assert first_report["manifest_count"] == 1
    assert first.with_suffix(".zip.sha256").is_file()
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        assert "ARCHIVE_METADATA.json" in names
        assert "ARCHIVE_INVENTORY.json" in names
        assert "source/pyproject.toml" in names
        assert "evidence/original/manifest.json" in names

    verification = verify_archive(first)
    assert verification["sidecar_verified"] is True
    assert verification["manifest_count"] == 1


def test_archive_builder_rejects_tampered_raw_output(tmp_path):
    repo, evidence, raw_path = _archive_fixture(tmp_path)
    raw_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="raw-output hash mismatch"):
        build_archive(
            repo_root=repo,
            inputs=[ArchiveInput("original", evidence)],
            output_path=tmp_path / "artifact.zip",
        )


def test_archive_builder_rejects_raw_output_outside_evidence_input(tmp_path):
    repo, evidence, raw_path = _archive_fixture(tmp_path)
    external_raw = tmp_path / "external-raw.json"
    external_raw.write_bytes(raw_path.read_bytes())
    manifest_path = evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence"]["raw_output_path"] = external_raw.as_posix()
    manifest["evidence"]["raw_output_sha256"] = sha256_file(external_raw)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="raw output is not included"):
        build_archive(
            repo_root=repo,
            inputs=[ArchiveInput("original", evidence)],
            output_path=tmp_path / "artifact.zip",
        )


def test_archive_verifier_rejects_duplicate_members(tmp_path):
    repo, evidence, _ = _archive_fixture(tmp_path)
    archive_path = tmp_path / "artifact.zip"
    build_archive(
        repo_root=repo,
        inputs=[ArchiveInput("original", evidence)],
        output_path=archive_path,
    )
    tampered = tmp_path / "tampered.zip"
    shutil.copyfile(archive_path, tampered)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(tampered, mode="a") as archive:
            archive.writestr("source/pyproject.toml", "tampered")

    with pytest.raises(RuntimeError, match="duplicate member names"):
        verify_archive(tampered)
