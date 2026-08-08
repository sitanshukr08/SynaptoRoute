"""Validate that a checkout is eligible to become a paper artifact candidate."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.manifest_schema import FULL_GIT_SHA, sha256_file, validate_manifest_file  # noqa: E402
from benchmarks.run_paper_matrix import build_commands  # noqa: E402


EXPECTED_VERSION = "0.5.0.dev0"
EXPECTED_COMMAND_COUNTS = {
    "quality": 2,
    "dynamic": 135,
    "scale": 40,
    "crash_recovery": 16,
    "backpressure": 15,
}
REQUIRED_LOCK_PACKAGES = {
    "datasets",
    "faiss-cpu",
    "fastembed",
    "mypy",
    "numpy",
    "pytest",
    "ruff",
    "scikit-learn",
    "semantic-router",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    strict: bool
    commit: str
    working_tree_dirty: bool
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "strict": self.strict,
            "git_commit": self.commit,
            "working_tree_dirty": self.working_tree_dirty,
            "checks": [asdict(check) for check in self.checks],
        }


def _git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _normalized_package_name(requirement: str) -> str:
    return re.sub(r"[-_.]+", "-", requirement.strip().lower())


def validate_lock_file(path: Path) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, f"missing or empty lock file: {path}"
    requirements = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    malformed = [line for line in requirements if line.count("==") != 1]
    if malformed:
        return False, f"non-exact requirements: {malformed[:3]}"
    names = [_normalized_package_name(line.split("==", 1)[0]) for line in requirements]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        return False, f"duplicate packages: {duplicates}"
    missing = sorted(REQUIRED_LOCK_PACKAGES - set(names))
    if missing:
        return False, f"required packages missing: {missing}"
    return True, f"{len(requirements)} exact pins; sha256={sha256_file(path)}"


def validate_matrix(repo_root: Path) -> tuple[bool, str]:
    matrix_path = repo_root / "paper" / "experiment_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    families = set(EXPECTED_COMMAND_COUNTS)
    commands = build_commands(
        matrix,
        repo_root / "benchmark_results" / "paper-matrix",
        families,
    )
    observed = {
        family: sum(item["family"] == family for item in commands)
        for family in sorted(families)
    }
    if observed != EXPECTED_COMMAND_COUNTS:
        return False, f"matrix command counts changed: {observed}"
    command_keys = [(item["family"], item["name"]) for item in commands]
    if len(command_keys) != len(set(command_keys)):
        return False, "matrix contains duplicate family/name commands"
    missing_scripts = []
    for item in commands:
        script_path = repo_root / item["command"][1]
        if not script_path.is_file() or script_path.stat().st_size == 0:
            missing_scripts.append(item["command"][1])
    if missing_scripts:
        return False, f"missing or empty entrypoints: {sorted(set(missing_scripts))}"
    return True, f"{len(commands)} commands across {len(families)} families"


def validate_historical_manifests(repo_root: Path) -> tuple[bool, str]:
    manifest_paths = sorted((repo_root / "benchmarks" / "manifests").glob("*.json"))
    errors = {
        path.name: validate_manifest_file(path, repo_root=repo_root)
        for path in manifest_paths
    }
    errors = {name: messages for name, messages in errors.items() if messages}
    if errors:
        return False, json.dumps(errors, sort_keys=True)
    statuses = {
        path.name: json.loads(path.read_text(encoding="utf-8"))["status"]
        for path in manifest_paths
    }
    return True, f"{len(statuses)} valid audit manifests: {statuses}"


def _result(name: str, operation) -> CheckResult:
    try:
        passed, detail = operation()
    except Exception as error:
        return CheckResult(name=name, passed=False, detail=f"{type(error).__name__}: {error}")
    return CheckResult(name=name, passed=bool(passed), detail=str(detail))


def run_preflight(
    repo_root: Path = REPO_ROOT,
    *,
    strict: bool = True,
    expected_version: str = EXPECTED_VERSION,
) -> PreflightReport:
    repo_root = repo_root.resolve()
    try:
        commit = _git_output(repo_root, "rev-parse", "HEAD")
        dirty = bool(_git_output(repo_root, "status", "--porcelain"))
    except Exception:
        commit = "unknown"
        dirty = True

    def check_source() -> tuple[bool, str]:
        return bool(FULL_GIT_SHA.fullmatch(commit)), f"commit={commit}"

    def check_tree() -> tuple[bool, str]:
        if strict and dirty:
            return False, "strict artifact candidates require a clean working tree"
        return True, "clean" if not dirty else "dirty tree allowed for development preflight"

    def check_version() -> tuple[bool, str]:
        with (repo_root / "pyproject.toml").open("rb") as handle:
            version = tomllib.load(handle)["project"]["version"]
        return version == expected_version, f"expected={expected_version}; actual={version}"

    def check_container() -> tuple[bool, str]:
        content = (repo_root / "Dockerfile.paper").read_text(encoding="utf-8")
        required = (
            "FROM python:3.11",
            "paper/requirements-linux-py311.lock",
            "python -m pip check",
        )
        missing = [fragment for fragment in required if fragment not in content]
        return not missing, "CPU Python 3.11 lock and pip check configured" if not missing else f"missing: {missing}"

    def check_paper_files() -> tuple[bool, str]:
        required = (
            "paper/PAPER.md",
            "paper/ARTIFACT_EVALUATION.md",
            "paper/QUALITY_PROTOCOL.md",
            "paper/experiment_matrix.json",
            "benchmarks/run_protocol_smoke.py",
            "paper/build_archive.py",
            "docs/RESEARCH_PROTOCOL.md",
            "docs/CURRENT_EVIDENCE_STATUS.md",
        )
        missing = [relative for relative in required if not (repo_root / relative).is_file()]
        return not missing, "required paper/protocol files present" if not missing else f"missing: {missing}"

    def check_matrix_runner() -> tuple[bool, str]:
        content = (repo_root / "benchmarks" / "run_paper_matrix.py").read_text(
            encoding="utf-8"
        )
        required = (
            "--resume",
            "run_state.json",
            "command_plan_sha256",
            "_atomic_write_json",
            "successful checkpoint log is missing or changed",
        )
        missing = [fragment for fragment in required if fragment not in content]
        detail = "atomic checkpoint and candidate-bound resume configured"
        return not missing, detail if not missing else f"missing: {missing}"

    def check_archive_builder() -> tuple[bool, str]:
        content = (repo_root / "paper" / "build_archive.py").read_text(encoding="utf-8")
        required = (
            "ARCHIVE_INVENTORY.json",
            "archive requires a clean working tree",
            "manifest raw-output hash mismatch",
            "evidence symlinks are not allowed",
            "force_zip64=True",
        )
        missing = [fragment for fragment in required if fragment not in content]
        detail = "streamed content inventory and evidence integrity gates configured"
        return not missing, detail if not missing else f"missing: {missing}"

    checks = (
        _result("source_commit", check_source),
        _result("working_tree", check_tree),
        _result("package_version", check_version),
        _result(
            "dependency_lock",
            lambda: validate_lock_file(repo_root / "paper" / "requirements-linux-py311.lock"),
        ),
        _result("experiment_matrix", lambda: validate_matrix(repo_root)),
        _result("historical_manifests", lambda: validate_historical_manifests(repo_root)),
        _result("paper_container", check_container),
        _result("paper_files", check_paper_files),
        _result("matrix_resume", check_matrix_runner),
        _result("archive_builder", check_archive_builder),
    )
    return PreflightReport(
        strict=strict,
        commit=commit,
        working_tree_dirty=dirty,
        checks=checks,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run development checks without treating a dirty tree as a failure.",
    )
    parser.add_argument("--expected-version", default=EXPECTED_VERSION)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    report = run_preflight(
        strict=not args.allow_dirty,
        expected_version=args.expected_version,
    )
    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        for check in report.checks:
            label = "PASS" if check.passed else "FAIL"
            print(f"[{label}] {check.name}: {check.detail}")
        print(f"Preflight {'passed' if report.passed else 'failed'}.")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
