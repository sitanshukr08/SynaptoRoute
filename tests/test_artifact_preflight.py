import json
from pathlib import Path

from paper import preflight


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_development_preflight_passes_current_artifact_foundation():
    report = preflight.run_preflight(REPO_ROOT, strict=False)

    assert report.passed, report.to_dict()
    assert report.strict is False
    assert {check.name for check in report.checks} == {
        "source_commit",
        "working_tree",
        "package_version",
        "dependency_lock",
        "experiment_matrix",
        "historical_manifests",
        "paper_container",
        "ci_package_smoke",
            "paper_files",
            "matrix_resume",
            "matrix_verifier",
            "evidence_promotion",
            "archive_builder",
    }


def test_strict_preflight_rejects_dirty_checkout(monkeypatch):
    def fake_git(_repo_root, *args):
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args == ("status", "--porcelain"):
            return " M README.md"
        raise AssertionError(args)

    monkeypatch.setattr(preflight, "_git_output", fake_git)

    report = preflight.run_preflight(REPO_ROOT, strict=True)

    working_tree = next(check for check in report.checks if check.name == "working_tree")
    assert working_tree.passed is False
    assert report.passed is False


def test_lock_validation_rejects_ranges_and_missing_required_packages(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("numpy>=1.24\npytest==9.0.0\n", encoding="utf-8")

    passed, detail = preflight.validate_lock_file(lock)

    assert passed is False
    assert "non-exact" in detail


def test_matrix_preflight_matches_frozen_protocol():
    passed, detail = preflight.validate_matrix(REPO_ROOT)

    assert passed is True
    assert "208 commands" in detail


def test_preflight_json_is_machine_readable():
    report = preflight.run_preflight(REPO_ROOT, strict=False)

    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["passed"] is True
    assert payload["strict"] is False
