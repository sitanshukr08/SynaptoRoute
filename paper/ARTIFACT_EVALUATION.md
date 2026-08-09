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
python benchmarks/run_protocol_smoke.py --output-dir benchmark_results/protocol-smoke
```

During development, `python paper/preflight.py --allow-dirty` checks the
artifact structure without waiving the clean-tree requirement for candidate
runs. Strict preflight must pass from the committed candidate.

The smoke manifest must remain `unverified` and
`paper_evidence_eligible=false`.

`run_protocol_smoke.py` executes bounded synthetic quality, dynamic mutation,
scale, process-crash, and sustained-backpressure cells. It is a wiring and
correctness gate, not a substitute for the frozen confirmatory matrix.

## Candidate Runs

Execute the frozen commands generated from `paper/experiment_matrix.json`.
Archive every manifest, raw log, prediction file, calibration record, and
analysis output without manual edits.

After extracting a matrix artifact, independently verify its candidate
identity, ledger consistency, hashes, output/log binding, and family-specific
correctness invariants:

```bash
python paper/verify_matrix_run.py EXTRACTED_ROOT/matrix \
  --expected-commit 0de734be8427aa3786e29062339a83b2ffb79bdd \
  --family crash_recovery \
  --require-environment
```

For quality-family output, the matrix verifier also inspects every per-seed
prediction, policy, probability model, and reliability artifact. A single seed
can be audited directly with:

```bash
python paper/verify_quality_artifacts.py \
  EXTRACTED_ROOT/matrix/quality/DATASET/seed-SEED/DATASET/experiment_summary.json
```

The verifier only accepts an `unverified`, paper-ineligible run. A successful
report does not promote a claim; independent reproduction, immutable archival,
and reviewer attestation remain separate gates.

Long runs checkpoint `run_state.json` after every command. Resume an interrupted
run only from the same clean commit, matrix, command plan, output directory,
and timeout configuration:

```bash
python benchmarks/run_paper_matrix.py \
  --execute \
  --resume \
  --output-dir benchmark_results/paper-matrix
```

Successful commands are skipped only when their logs still match the recorded
SHA-256. Failed commands are retried. A changed commit, matrix, command plan,
timeout, command identity, or successful log aborts resume instead of mixing
evidence. `--stop-on-failure` is available for supervised pilot runs.

## Independent Reproduction

The reproducer must use the same source commit and configuration on a different
machine. An independent reviewer must create a JSON attestation tied to both
run IDs, the exact claim, and the uploaded archive:

```json
{
  "schema_version": 1,
  "decision": "approve",
  "reviewer": "REVIEWER_ID",
  "reviewed_at_utc": "2026-08-08T00:00:00Z",
  "original_run_id": "ORIGINAL_RUN_ID",
  "reproduction_run_id": "REPRODUCTION_RUN_ID",
  "claim": "CLAIM TEXT",
  "archive_uri": "ARCHIVE_URI",
  "archive_sha256": "ARCHIVE_SHA256",
  "notes": "Raw logs, invariants, configuration, and analysis reviewed."
}
```

Promote a claim only after the attestation and both runs are archived:

```bash
python benchmarks/promote_evidence.py \
  --original ORIGINAL_MANIFEST \
  --reproduction REPRODUCTION_MANIFEST \
  --attestation REVIEW_ATTESTATION.json \
  --claim "CLAIM TEXT" \
  --archive-uri ARCHIVE_URI \
  --archive-sha256 ARCHIVE_SHA256 \
  --output VERIFIED_CLAIM.json
```

Performance values may differ across hardware. Correctness invariants must
agree, and quality results must be interpreted using the frozen paired
statistical analysis rather than an ad hoc equality tolerance.

## Archive Construction

After original and reproduction runs pass review, build a content-inventoried
deposit bundle from the clean candidate:

```bash
python paper/build_archive.py \
  --input original=benchmark_results/original \
  --input reproduction=benchmark_results/reproduction \
  --output dist/synaptoroute-artifact.zip
```

The builder includes the tracked source tree, rejects dirty candidates, failed
runs, changed raw-output hashes, changed lock hashes, manifest/commit
mismatches, duplicate labels, and symlinks. `ARCHIVE_INVENTORY.json` hashes
every payload entry and a `.sha256` sidecar records the final ZIP digest. Upload
that immutable ZIP before using its URI and digest in evidence promotion.

Reviewers can independently stream-check the sidecar, inventory membership,
member sizes, hashes, metadata, duplicate names, and unsafe paths:

```bash
python paper/verify_archive.py dist/synaptoroute-artifact.zip
```
