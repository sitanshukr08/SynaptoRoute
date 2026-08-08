# Artifact Evaluation Guide

## Install

Build `Dockerfile.paper` on a CPU-only Linux host. Record the image digest and
retain `paper/resolved-environment.txt` from the image used for final runs.
The image installs `paper/requirements-linux-py311.lock`, including CPU FAISS,
and fails the build if `pip check` detects an inconsistent dependency closure.

## Smoke

```bash
python paper/preflight.py
python -m pytest tests -q
python benchmarks/run_ci_smoke_benchmark.py --output-dir benchmark_results/artifact-smoke
```

During development, `python paper/preflight.py --allow-dirty` checks the
artifact structure without waiving the clean-tree requirement for candidate
runs. Strict preflight must pass from the committed candidate.

The smoke manifest must remain `unverified` and
`paper_evidence_eligible=false`.

## Candidate Runs

Execute the frozen commands generated from `paper/experiment_matrix.json`.
Archive every manifest, raw log, prediction file, calibration record, and
analysis output without manual edits.

## Independent Reproduction

The reproducer must use the same source commit and configuration on a different
machine. Promote a claim only after both runs are archived:

```bash
python benchmarks/promote_evidence.py \
  --original ORIGINAL_MANIFEST \
  --reproduction REPRODUCTION_MANIFEST \
  --reviewer REVIEWER_ID \
  --claim "CLAIM TEXT" \
  --archive-uri ARCHIVE_URI \
  --archive-sha256 ARCHIVE_SHA256 \
  --output VERIFIED_CLAIM.json
```

Performance values may differ across hardware. Correctness invariants must
agree, and quality results must be interpreted using the frozen paired
statistical analysis rather than an ad hoc equality tolerance.
