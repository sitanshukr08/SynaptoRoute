# Unattended Artifact Runbook

This runbook advances an immutable candidate without promoting unsupported
claims. Run final experiments on a controlled Linux host, not on a development
workstation with background load.

## 1. Candidate Gate

```bash
git status --short
python paper/preflight.py
ruff check src tests benchmarks paper
mypy src/synaptoroute
python -m pytest tests -m "not model" -q
SYNAPTOROUTE_RUN_MODEL_TESTS=1 \
  python -m pytest tests/integration/test_fastembed_model.py -m model -q
python -m build
```

Stop if the tree is dirty or any gate fails. Record the full commit SHA. Do not
reuse an output directory from another commit.

## 2. Paper Container

```bash
docker build --pull --no-cache \
  --file Dockerfile.paper \
  --tag synaptoroute-paper:CANDIDATE_SHA .
docker image inspect synaptoroute-paper:CANDIDATE_SHA
docker run --rm synaptoroute-paper:CANDIDATE_SHA
docker run --rm \
  --volume "$PWD/benchmark_results:/artifact/benchmark_results" \
  synaptoroute-paper:CANDIDATE_SHA \
  python benchmarks/run_protocol_smoke.py \
    --output-dir benchmark_results/container-protocol-smoke
```

Retain the image ID/digest, resolved package inventory, test log, smoke
manifest, and raw smoke outputs. These remain unverified development evidence.

## 3. Confirmatory Matrix

Use a new output root named with the candidate SHA and machine identifier:

```bash
python benchmarks/run_paper_matrix.py \
  --execute \
  --output-dir benchmark_results/paper-matrix-CANDIDATE-MACHINE
```

`run_state.json` is written atomically after every cell. If the process or host
is interrupted, use the exact same commit, matrix, Python executable, output
directory, and timeout setting:

```bash
python benchmarks/run_paper_matrix.py \
  --execute \
  --resume \
  --output-dir benchmark_results/paper-matrix-CANDIDATE-MACHINE
```

Do not edit logs, state, predictions, summaries, or manifests. The runner skips
only successful cells with matching log hashes and retries failed cells. It
refuses to resume a different candidate or command plan.

## 4. Original-Run Review

Require all 208 commands to complete, no correctness-invariant failures, a
schema-valid unverified manifest, complete raw logs, and generated analysis
artifacts. Statistical and paper outputs must be regenerated from archived
machine-readable inputs. A failed or incomplete run is diagnostic only.

## 5. Independent Reproduction

Transfer the candidate commit and runbook to a second contributor using a
different Linux machine. Rebuild the image and repeat the smoke and matrix.
Record a distinct machine ID and retain the reproduction manifest and raw
outputs. Do not copy the original output directory into the reproduction root.

## 6. Archive And Verify

```bash
python paper/build_archive.py \
  --input original=benchmark_results/paper-matrix-ORIGINAL \
  --input reproduction=benchmark_results/paper-matrix-REPRODUCTION \
  --output dist/synaptoroute-artifact.zip
python paper/verify_archive.py dist/synaptoroute-artifact.zip
```

Upload the exact ZIP and its `.sha256` sidecar to an immutable versioned
archive. Record the DOI/URI and digest; do not rebuild the ZIP after upload.

## 7. Claim Promotion

Promote claims individually only after reviewer attestation and archive upload:

The attestation must be a non-empty JSON object using the schema documented in
`paper/ARTIFACT_EVALUATION.md`. It must identify both run IDs, the exact claim,
archive URI and SHA-256, reviewer, UTC review time, approval decision, and
review notes. Promotion records and revalidates the attestation hash.

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

Generate paper tables from verified claim manifests. If code changes after a
candidate run, create a new commit and rerun every affected experiment.

## 8. Release Gate

Release `v0.5.0` only when the source commit, package version, container,
manifests, archive, documentation, paper tables, and release tag agree. Redis,
adaptive memory, hybrid routing, sessions, slots, and permissions remain
outside the primary artifact until separately specified and tested.
