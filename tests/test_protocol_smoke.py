from pathlib import Path

from benchmarks.run_protocol_smoke import evaluate_invariants


def test_protocol_smoke_invariants_accept_complete_artifacts(tmp_path):
    probability = tmp_path / "probability.json"
    predictions = tmp_path / "predictions.jsonl"
    reliability = tmp_path / "reliability.json"
    diagram = tmp_path / "reliability.svg"
    for path in (probability, predictions, reliability, diagram):
        path.write_text("evidence", encoding="utf-8")
    quality = {
        "systems": {
            "system": {
                "test": {
                    "expected_calibration_error": 0.1,
                    "max_calibration_error": 0.2,
                    "brier_score": 0.1,
                },
                "probability_calibration": {
                    "artifact_path": str(probability),
                    "source_predictions_path": str(predictions),
                },
                "reliability": {
                    "data_path": str(reliability),
                    "diagram_path": str(diagram),
                },
            }
        }
    }

    failures = evaluate_invariants(
        quality=quality,
        dynamic={"metrics": {"correctness_violations": 0, "restart_state_equal": True}},
        scale={"metrics": {"top1_identity_accuracy": 1.0}},
        crash={
            "metrics": {
                "durable": {"restart_survival_rate": 1.0},
                "memory": {"restart_survival_rate": 0.0},
            }
        },
        backpressure={
            "scenarios": [
                {
                    "load_fraction": 1.5,
                    "error_count": 0,
                    "successful_count": 1,
                    "successful_accuracy": 1.0,
                }
            ]
        },
    )

    assert failures == []


def test_protocol_smoke_invariants_report_failures(tmp_path):
    missing = Path(tmp_path / "missing")
    quality = {
        "systems": {
            "system": {
                "test": {},
                "probability_calibration": {
                    "artifact_path": str(missing),
                    "source_predictions_path": str(missing),
                },
                "reliability": {
                    "data_path": str(missing),
                    "diagram_path": str(missing),
                },
            }
        }
    }

    failures = evaluate_invariants(
        quality=quality,
        dynamic={"metrics": {"correctness_violations": 1, "restart_state_equal": False}},
        scale={"metrics": {"top1_identity_accuracy": 0.0}},
        crash={
            "metrics": {
                "durable": {"restart_survival_rate": 0.0},
                "memory": {"restart_survival_rate": 1.0},
            }
        },
        backpressure={
            "scenarios": [
                {
                    "load_fraction": 1.5,
                    "error_count": 1,
                    "successful_count": 1,
                    "successful_accuracy": 0.0,
                }
            ]
        },
    )

    assert len(failures) == 12
