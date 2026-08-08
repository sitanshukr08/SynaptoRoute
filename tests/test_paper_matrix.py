import json
import sys
from pathlib import Path

import pytest

import benchmarks.run_paper_matrix as matrix_runner
from benchmarks.run_paper_matrix import build_commands, execute


def test_frozen_matrix_expands_every_declared_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    matrix = json.loads((root / "paper" / "experiment_matrix.json").read_text(encoding="utf-8"))
    families = {"quality", "dynamic", "scale", "crash_recovery", "backpressure"}

    commands = build_commands(matrix, tmp_path, families)

    assert {item["family"] for item in commands} == families
    assert len([item for item in commands if item["family"] == "quality"]) == 2
    assert len([item for item in commands if item["family"] == "dynamic"]) == 135
    assert len([item for item in commands if item["family"] == "scale"]) == 40
    crash_commands = [item for item in commands if item["family"] == "crash_recovery"]
    assert len(crash_commands) == 16
    # Each crash command runs both memory-visible and durable acknowledgement modes.
    assert all("--trials" in item["command"] for item in crash_commands)
    assert len([item for item in commands if item["family"] == "backpressure"]) == 15


def _runner_workspace(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "benchmarks").mkdir(parents=True)
    (root / "paper").mkdir()
    (root / "benchmarks" / "run_paper_matrix.py").write_text("# fixture\n", encoding="utf-8")
    (root / "paper" / "requirements-linux-py311.lock").write_text(
        "fixture==1.0\n",
        encoding="utf-8",
    )
    matrix_path = root / "paper" / "matrix.json"
    matrix = {"fixture": True}
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    commit = {"value": "a" * 40}

    def fake_git(*args):
        if args == ("status", "--porcelain"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return commit["value"]
        return "unknown"

    monkeypatch.setattr(matrix_runner, "REPO_ROOT", root)
    monkeypatch.setattr(matrix_runner, "_git", fake_git)
    return root, matrix_path, matrix, commit


def test_matrix_resume_skips_successful_hashed_logs(tmp_path, monkeypatch):
    root, matrix_path, matrix, _ = _runner_workspace(tmp_path, monkeypatch)
    marker = root / "executions.txt"
    script = (
        "from pathlib import Path; "
        f"p=Path({str(marker)!r}); "
        "p.write_text((p.read_text() if p.exists() else '') + 'x')"
    )
    commands = [
        {
            "family": "fixture",
            "name": "success",
            "command": [sys.executable, "-c", script],
        }
    ]
    output_dir = root / "results"

    assert execute(commands, matrix, matrix_path, output_dir) == 0
    first_state = json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))
    assert marker.read_text(encoding="utf-8") == "x"

    assert execute(commands, matrix, matrix_path, output_dir, resume=True) == 0
    second_state = json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert marker.read_text(encoding="utf-8") == "x"
    assert second_state["run_id"] == first_state["run_id"]
    assert second_state["resume_count"] == 1
    assert len(second_state["invocations"]) == 2
    assert manifest["metrics"]["skipped_successful_command_count"] == 1


def test_matrix_resume_retries_failed_command(tmp_path, monkeypatch):
    root, matrix_path, matrix, _ = _runner_workspace(tmp_path, monkeypatch)
    marker = root / "first-attempt.marker"
    script = (
        "from pathlib import Path; import sys; "
        f"p=Path({str(marker)!r}); first=not p.exists(); "
        "p.write_text('attempted'); sys.exit(2 if first else 0)"
    )
    commands = [
        {
            "family": "fixture",
            "name": "retry",
            "command": [sys.executable, "-c", script],
        }
    ]
    output_dir = root / "results"

    assert execute(commands, matrix, matrix_path, output_dir) == 1
    failed_state = json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))
    assert failed_state["results"][0]["return_code"] == 2

    assert execute(commands, matrix, matrix_path, output_dir, resume=True) == 0
    recovered_state = json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))
    assert recovered_state["status"] == "completed"
    assert recovered_state["results"][0]["return_code"] == 0


def test_matrix_resume_rejects_a_different_commit(tmp_path, monkeypatch):
    root, matrix_path, matrix, commit = _runner_workspace(tmp_path, monkeypatch)
    commands = [
        {
            "family": "fixture",
            "name": "success",
            "command": [sys.executable, "-c", "print('ok')"],
        }
    ]
    output_dir = root / "results"
    assert execute(commands, matrix, matrix_path, output_dir) == 0
    commit["value"] = "b" * 40

    with pytest.raises(RuntimeError, match="different matrix candidate"):
        execute(commands, matrix, matrix_path, output_dir, resume=True)
