import pytest
from synaptoroute.calibration import evaluate_calibration, export_reliability_diagram

def test_evaluate_calibration_perfect_alignment():
    # Perfect alignment: accuracy matches confidence
    y_true = [1, 1, 0, 0]
    y_prob = [1.0, 1.0, 0.0, 0.0]

    metrics = evaluate_calibration(y_true, y_prob, num_bins=5)
    assert metrics.expected_calibration_error == pytest.approx(0.0)
    assert metrics.max_calibration_error == pytest.approx(0.0)
    assert metrics.brier_score == pytest.approx(0.0)
    assert metrics.num_samples == 4

def test_evaluate_calibration_miscalibrated():
    # Overconfident predictions
    y_true = [1, 0, 0, 0]
    y_prob = [0.9, 0.9, 0.9, 0.9]

    metrics = evaluate_calibration(y_true, y_prob, num_bins=10)
    # Accuracy is 0.25, confidence is 0.9 => gap is 0.65
    assert metrics.expected_calibration_error == pytest.approx(0.65)
    assert metrics.max_calibration_error == pytest.approx(0.65)
    assert metrics.num_samples == 4

def test_export_reliability_diagram_svg_output(tmp_path):
    y_true = [1, 1, 0, 1, 0, 0]
    y_prob = [0.8, 0.7, 0.2, 0.9, 0.1, 0.4]

    metrics = evaluate_calibration(y_true, y_prob, num_bins=5)
    svg_file = tmp_path / "diagram.svg"

    path = export_reliability_diagram(metrics, svg_file)
    assert path.exists()
    assert path.stat().st_size > 0

    content = path.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "ECE:" in content
