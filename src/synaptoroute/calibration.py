"""
SynaptoRoute Confidence Calibration & Reliability Metrics
==========================================================
Provides ECE (Expected Calibration Error), MCE (Maximum Calibration Error),
Brier score calculations, and SVG reliability diagram generation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, List, Union
import numpy as np

@dataclass
class CalibrationMetrics:
    """Confidence calibration metrics for model risk analysis."""
    expected_calibration_error: float
    max_calibration_error: float
    brier_score: float
    num_samples: int
    num_bins: int
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]

def evaluate_calibration(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    num_bins: int = 10,
) -> CalibrationMetrics:
    """Calculate Expected Calibration Error (ECE), MCE, Brier score, and bin stats."""
    labels = np.asarray(y_true, dtype=np.int32)
    probs = np.asarray(y_prob, dtype=np.float32)

    if len(labels) != len(probs):
        raise ValueError("y_true and y_prob must have the same length.")
    if len(labels) == 0:
        return CalibrationMetrics(
            expected_calibration_error=0.0,
            max_calibration_error=0.0,
            brier_score=0.0,
            num_samples=0,
            num_bins=num_bins,
            bin_accuracies=[],
            bin_confidences=[],
            bin_counts=[],
        )

    brier_score = float(np.mean((probs - labels) ** 2))

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_accuracies: List[float] = []
    bin_confidences: List[float] = []
    bin_counts: List[int] = []

    ece = 0.0
    mce = 0.0
    total_samples = len(labels)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        bin_count = int(np.sum(in_bin))
        bin_counts.append(bin_count)

        if bin_count > 0:
            bin_acc = float(np.mean(labels[in_bin]))
            bin_conf = float(np.mean(probs[in_bin]))
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)

            gap = abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * gap
            mce = max(mce, gap)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    return CalibrationMetrics(
        expected_calibration_error=ece,
        max_calibration_error=mce,
        brier_score=brier_score,
        num_samples=total_samples,
        num_bins=num_bins,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )

def export_reliability_diagram(
    metrics: CalibrationMetrics,
    output_path: Union[str, Path],
    title: str = "SynaptoRoute Confidence Reliability Diagram",
) -> Path:
    """Export a standalone SVG reliability diagram showing empirical accuracy vs confidence."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 600, 500
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    svg_lines = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        '  <style>',
        '    text { font-family: system-ui, -apple-system, sans-serif; font-size: 12px; fill: #1e293b; }',
        '    .title { font-size: 16px; font-weight: bold; fill: #0f172a; }',
        '    .grid { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4,4; }',
        '    .ideal { stroke: #94a3b8; stroke-width: 2; stroke-dasharray: 6,6; }',
        '    .bar { fill: #3b82f6; fill-opacity: 0.7; stroke: #2563eb; stroke-width: 1.5; }',
        '    .metrics { font-size: 13px; font-weight: 500; fill: #334155; }',
        '  </style>',
        '  <rect width="100%" height="100%" fill="#ffffff"/>',
        f'  <text x="{width/2}" y="30" text-anchor="middle" class="title">{title}</text>',
    ]

    # Grid and axis lines
    for i in range(5):
        val = i / 4.0
        x = margin + val * plot_w
        y = height - margin - val * plot_h
        svg_lines.append(f'  <line x1="{x}" y1="{height-margin}" x2="{x}" y2="{margin}" class="grid"/>')
        svg_lines.append(f'  <line x1="{margin}" y1="{y}" x2="{width-margin}" y2="{y}" class="grid"/>')
        svg_lines.append(f'  <text x="{x}" y="{height-margin+20}" text-anchor="middle">{val:.2f}</text>')
        svg_lines.append(f'  <text x="{margin-15}" y="{y+4}" text-anchor="end">{val:.2f}</text>')

    # Ideal diagonal line
    svg_lines.append(f'  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{margin}" class="ideal"/>')

    # Plot bars
    num_bins = metrics.num_bins
    bin_width_px = plot_w / num_bins

    for i in range(num_bins):
        bin_count = metrics.bin_counts[i]
        if bin_count == 0:
            continue
        acc = metrics.bin_accuracies[i]
        bar_h = acc * plot_h
        x = margin + i * bin_width_px
        y = height - margin - bar_h

        svg_lines.append(
            f'  <rect x="{x+2}" y="{y}" width="{bin_width_px-4}" height="{bar_h}" class="bar"/>'
        )

    # Annotations
    svg_lines.append(f'  <text x="{width/2}" y="{height-15}" text-anchor="middle" font-weight="bold">Confidence</text>')
    svg_lines.append(f'  <text x="20" y="{height/2}" text-anchor="middle" font-weight="bold" transform="rotate(-90 20 {height/2})">Accuracy</text>')

    stats_text = f"ECE: {metrics.expected_calibration_error:.4f} | MCE: {metrics.max_calibration_error:.4f} | Brier: {metrics.brier_score:.4f}"
    svg_lines.append(f'  <text x="{margin}" y="{margin - 10}" class="metrics">{stats_text}</text>')

    svg_lines.append('</svg>')

    path.write_text('\n'.join(svg_lines), encoding='utf-8')
    return path
