# Changelog

All notable changes to SynaptoRoute will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Confidence Calibration & Reliability Metrics:** Added `evaluate_calibration` in `synaptoroute.calibration` returning Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score. Added `export_reliability_diagram` for rendering SVG reliability curves.
- **Framework Integration Adapters:** Added `examples/langchain_router.py` (`SynaptoRouteLangChainAdapter`) and `examples/llamaindex_selector.py` (`SynaptoRouteLlamaIndexSelector`).
- **Route Versioning:** Added `version: int` attribute to `Route` model and `version INTEGER DEFAULT 1` SQLite schema column with auto-migration support.
- **Node Sequence Numbering:** Added `sequence_id` to `RedisSyncManager` payload for message ordering across cluster nodes.
- **CI Verified Benchmark:** Added `benchmarks/run_verified_ci_benchmark.py` generating schema-valid `status: verified` manifests.
- **Public API Examples:** Added `examples/quickstart.py`, `examples/async_router.py`, and `examples/sqlite_persistence.py`.
- **Open-Source Release Artifacts:** Added `SECURITY.md`, `CODE_OF_CONDUCT.md`, and `docs/API_REFERENCE.md`.

### Fixed
- **In-Memory Mutation Rollback:** Added `_rollback_mutation_in_memory` helper in `AdaptiveRouter` to strip unpersisted routes/utterances prior to storage resync.
- **Index Rebuild Error Safety:** Fixed `_rebuild_index` to acquire `rwlock.write_lock()` in its `finally:` block, clearing pending mutations and resetting rebuild flags cleanly on failure.
- **Tombstone Compaction:** Added `_compact_unlocked` in `NumpyIndex` to reclaim tombstoned vector slots when `max_capacity` is reached instead of raising `ID_OVERFLOW`.
