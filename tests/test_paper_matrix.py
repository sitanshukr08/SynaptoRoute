import json
from pathlib import Path

from benchmarks.run_paper_matrix import build_commands


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
