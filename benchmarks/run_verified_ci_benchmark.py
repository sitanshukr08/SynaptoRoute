"""Compatibility wrapper for the renamed, unverified CI structural smoke."""

from pathlib import Path

from benchmarks.run_ci_smoke_benchmark import REPO_ROOT, main, run_ci_smoke


def run_ci_benchmark(output_dir: Path | None = None) -> dict:
    """Preserve the old callable while never producing verified evidence."""
    return run_ci_smoke(output_dir or REPO_ROOT / "benchmark_results" / "ci_smoke")


if __name__ == "__main__":
    raise SystemExit(main())
