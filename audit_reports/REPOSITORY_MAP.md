# Repository Map

## Overview

This document maps the files, their purposes, dependencies, and exported components within the repository.

### Databases & Models
- **src/synaptoroute/models.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: dataclasses, typing, typing_extensions, pydantic, json, numpy
  - **Exports**: validate_metadata_serializable, Dict, deduplicate_utterances, field_validator, Route, Field, BaseModel, ConfigDict, Annotated, RollbackSnapshot, dataclass, Any, Optional, List, StringConstraints

### Agent Workflows
- **src/synaptoroute/storage.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: typing, queue, sqlite3, os, threading, json, contextlib, synaptoroute.models, abc
  - **Exports**: _init_db, add_utterance, ABC, __init__, Route, load_all_routes, delete_route, __del__, save_route, abstractmethod, SQLiteStorage, update_threshold, BaseStorage, _get_connection, List

### Routers
- **src/synaptoroute/encoder.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.exceptions, typing, numpy.typing, abc, numpy, fastembed, openai
  - **Exports**: __init__, SynaptoRouteError, dim, requires_lock, OpenAIEncoder, encode, encode_batch, BaseEncoder, FastEmbedEncoder, Optional, List, TextEmbedding

- **src/synaptoroute/exceptions.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: None
  - **Exports**: RouteNotFoundError, RouterOverloadedError, SynaptoRouteError, ModelLoadError, RouterCapacityError

- **src/synaptoroute/index.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: threading, numpy, typing, faiss
  - **Exports**: Tuple, __init__, FaissIndex, get_index, add, delete, rebuild, _add_unlocked, total_vectors, List, NumpyIndex, search

- **src/synaptoroute/locks.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: threading
  - **Exports**: release_read, __init__, read_lock, write_lock, acquire_write, __enter__, __exit__, WriteLockContext, RWLock, release_write, acquire_read, ReadLockContext

- **src/synaptoroute/metrics.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: collections, prometheus_client
  - **Exports**: Gauge, CollectorRegistry, __init__, Counter, _MockHistogram, inc, generate_latest, export_metrics, dec, Histogram, MetricsRegistry, _MockCounter, set, observe, _MockGauge

- **src/synaptoroute/profile.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: dataclasses, enum, os
  - **Exports**: ProfileType, Enum, OptimizationProfile, dataclass, get_profile

- **src/synaptoroute/reranker.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.models, typing, sentence_transformers
  - **Exports**: Tuple, CrossEncoder, Route, __init__, rerank, CrossEncoderReranker, Optional, List

- **src/synaptoroute/router.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: logging, time, synaptoroute.index, synaptoroute.exceptions, concurrent.futures, synaptoroute.encoder, synaptoroute.locks, synaptoroute.profile, numpy, threading, synaptoroute.models, synaptoroute.storage, sklearn.metrics, sys, asyncio, typing, synaptoroute.metrics, queue, synaptoroute.sync
  - **Exports**: Encoder, f1_score, update_threshold, _flush_storage_batch, _load_routes, _dispatch_and_set, delete_route, start, aquery, _encode, _batch_worker, BaseSyncManager, BaseStorage, _dispatch_batch, Optional, _rebuild_index, add_route, RouterCapacityError, get_profile, add_utterance, fit_thresholds, get_index, RollbackSnapshot, OptimizationProfile, MetricsRegistry, _resolve_task, __call__, ProfileType, __init__, Route, _storage_worker, RouteNotFoundError, RouterOverloadedError, stop, RWLock, AdaptiveRouter, FastEmbedEncoder, SynaptoRouteError

- **src/synaptoroute/sync.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: logging, asyncio, base64, typing, uuid, time, redis.asyncio, json, traceback, synaptoroute.models, numpy
  - **Exports**: __init__, register, _publisher_loop, _dispatch_worker_loop, start, Route, BaseSyncManager, RedisSyncManager, _listener_loop, stop, broadcast, _dispatch, Optional

- **src/synaptoroute/trainer.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: asyncio, typing, synaptoroute.router, pydantic, json, openai
  - **Exports**: __init__, tune_route, BaseModel, AsyncOpenAI, AdaptiveRouter, SyntheticResponse, Optional, List, SyntheticTuner

- **src/synaptoroute/__init__.py**
  - **Purpose**: SynaptoRoute A high-throughput, local semantic routing engine.
  - **Imports**: importlib.metadata, synaptoroute.models, synaptoroute.router, synaptoroute.encoder, synaptoroute.storage
  - **Exports**: Route, Encoder, SQLiteStorage, BaseStorage, AdaptiveRouter

- **src/synaptoroute/integrations/langchain.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.router, typing, langchain_core.runnables
  - **Exports**: ainvoke, __init__, Runnable, SynaptoRouteChain, Any, invoke, AdaptiveRouter, Optional, RunnableConfig

- **src/synaptoroute/integrations/llamaindex.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: llama_index.core.tools.types, typing, llama_index.core.schema, synaptoroute.router, llama_index.core.selectors
  - **Exports**: _get_prompts, Sequence, ToolMetadata, __init__, _update_prompts, SingleSelection, _aselect, SelectorResult, Any, QueryBundle, AdaptiveRouter, BaseSelector, SynaptoRouteSelector, _select

- **src/synaptoroute/integrations/__init__.py**
  - **Purpose**: SynaptoRoute integrations for external frameworks like LangChain and LlamaIndex.
  - **Imports**: None
  - **Exports**: None

### Documentation
- **docs/ARCHITECTURE.md**
  - **Purpose**: # SynaptoRoute System Architecture  This document maps the core subsystems of SynaptoRoute v0.4.0, defining strict ownership, dependencies, and known failure modes to facilitate safe architectural reasoning for contributors.  ## Subsystem 1: Router (`AdaptiveRouter`)

- **docs/BENCHMARK_EVIDENCE_INDEX.md**
  - **Purpose**: # Benchmark Evidence Traceability Index  This matrix maps every metric in the `BENCHMARK_REGISTRY.md` to its origin benchmark script and raw output artifact.  | Metric | Registry Entry | Benchmark Script | Raw JSON Manifest | Status |

- **docs/BENCHMARK_REGISTRY.md**
  - **Purpose**: # SynaptoRoute Benchmark Registry  This document serves as the authoritative ledger for all SynaptoRoute performance, accuracy, and scaling benchmarks.  ## Verification Tiers

- **docs/CONTRIBUTOR_QUICKSTART.md**
  - **Purpose**: # Contributor Quickstart  Welcome to the SynaptoRoute project! This guide will get you oriented with the v0.4.0 architecture so you can run benchmarks, reproduce results, and contribute effectively.  ## 1. Architecture Overview

- **docs/SYNAPTOROUTE_TECHNICAL_REFERENCE.md**
  - **Purpose**: # SynaptoRoute Technical Reference  > **Authoritative Documentation – SynaptoRoute v0.4.0**  ## 1. Executive Summary

- **ARCHITECTURE_REPORT.md**
  - **Purpose**: # Architecture Report  ## Dependency Graph ```mermaid graph TD

- **BENCHMARKS.md**
  - **Purpose**: # SynaptoRoute Benchmarks  This file summarizes the benchmark performance of SynaptoRoute. All claims are backed by empirical telemetry found in the [BENCHMARK_REGISTRY.md](docs/BENCHMARK_REGISTRY.md).  ## Metric Definitions

- **COMPARISON.md**
  - **Purpose**: # Competitor Comparison Matrix  This document provides a realistic, evidence-backed assessment of how SynaptoRoute compares to alternative routing frameworks in the industry.  To prevent accidental overclaiming, every comparison is explicitly tagged with the constraints under which it was evaluated, and classified as `[MEASURED]`, `[ESTIMATED]`, or `[UNKNOWN]`.

- **CONTRIBUTING.md**
  - **Purpose**: # Contributing to SynaptoRoute  SynaptoRoute operates under an extremely rigid set of engineering principles. To maintain the project's reliability and architectural honesty, all contributors must strictly adhere to the following workflows.  ## 1. Development Flow

- **LIMITATIONS.md**
  - **Purpose**: # Verified and Unknown Boundaries  To maintain engineering trust, SynaptoRoute explicitly separates verified system limits from theoretical or unknown boundaries. This document outlines the absolute edge of our confidence in the current v0.4.0 architecture.  ---

- **README.md**
  - **Purpose**: # SynaptoRoute  SynaptoRoute is an adaptive, high-throughput semantic router. It is **not** a large language model (LLM), an embedding model, or a conversational agent. It is a highly optimized control plane that ingests natural language queries and deterministically routes them to predefined system actions ("routes") based on semantic similarity.  It is designed to sit at the edge of your infrastructure, intercepting user intents in milliseconds to bypass heavy LLM generation where predefined workflows (e.g., billing, password resets, API lookups) exist.

- **REPOSITORY_MAP.md**
  - **Purpose**: # Repository Map  ## Overview  This document maps the files, their purposes, dependencies, and exported components within the repository.

- **ROADMAP.md**
  - **Purpose**: # SynaptoRoute Roadmap  This roadmap documents the architectural and research trajectory for SynaptoRoute.  ## Completed (Verified in v0.4.0)

### Tests
- **tests/conftest.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.encoder, pytest
  - **Exports**: encoder, Encoder

- **tests/test_async.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.exceptions, asyncio, pytest, synaptoroute.router, synaptoroute.encoder, synaptoroute.storage, synaptoroute.models
  - **Exports**: test_aquery_worker_crashed, Route, test_aquery_without_start, test_aquery, RouterOverloadedError, temp_db, Encoder, mock_worker, test_aquery_raises_if_worker_crashes_while_pending, SQLiteStorage, AdaptiveRouter, test_batch_worker_shutdown, test_aquery_overload, storage

- **tests/test_concurrency.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.exceptions, asyncio, pytest, synaptoroute.router, time, threading, synaptoroute.encoder, synaptoroute.storage, synaptoroute.models
  - **Exports**: sync_add_routes, Route, delete_while_encoding, RouteNotFoundError, slow_encode, temp_db, Encoder, test_encoder_lock_concurrency, add_routes, test_add_utterance_route_deleted_during_encoding, run_queries, SQLiteStorage, AdaptiveRouter, storage

- **tests/test_encoder.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.encoder, numpy
  - **Exports**: test_encode_batch, test_encode, test_encoder_initialization, Encoder

- **tests/test_encoders.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: pytest, synaptoroute.models, synaptoroute.router, synaptoroute.encoder, synaptoroute.storage, numpy, unittest.mock
  - **Exports**: test_fastembed_encoder, add_utterance, Route, clear, Mock, load_all_routes, delete_route, side_effect, OpenAIEncoder, save_route, MagicMock, update_threshold, BaseEncoder, BaseStorage, AdaptiveRouter, DummyStorage, FastEmbedEncoder, test_openai_encoder_mocked

- **tests/test_langchain.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.models, synaptoroute.integrations.langchain, pytest, unittest.mock
  - **Exports**: test_synaptoroute_chain_ainvoke_no_route, Route, test_synaptoroute_chain_ainvoke_with_route, SynaptoRouteChain, test_synaptoroute_chain_invoke_with_route, MagicMock, test_synaptoroute_chain_invoke_no_route, AsyncMock

- **tests/test_llamaindex.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: llama_index.core.tools.types, pytest, llama_index.core.schema, synaptoroute.router, synaptoroute.integrations.llamaindex, llama_index.core.selectors, synaptoroute.models, unittest.mock
  - **Exports**: ToolMetadata, Route, Mock, SingleSelection, test_synaptoroute_selector_no_match, SelectorResult, test_synaptoroute_selector_select, MagicMock, QueryBundle, AdaptiveRouter, test_synaptoroute_selector_aselect, SynaptoRouteSelector, AsyncMock

- **tests/test_metrics.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.metrics, pytest, unittest.mock
  - **Exports**: test_metrics_registry_mock_fallback, MetricsRegistry, patch

- **tests/test_models.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.exceptions, synaptoroute.models, pytest
  - **Exports**: test_route_creation, RouteNotFoundError, Route, test_route_with_metadata_and_threshold

- **tests/test_optimization.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: pytest, synaptoroute.router, synaptoroute.encoder, synaptoroute.storage, synaptoroute.models, numpy
  - **Exports**: __init__, Route, test_fit_thresholds, dim, MockEncoder, Encoder, encode, SQLiteStorage, encode_batch, AdaptiveRouter

- **tests/test_profile.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: pytest, synaptoroute.router, os, synaptoroute.encoder, synaptoroute.profile, synaptoroute.storage
  - **Exports**: test_throughput_profile_defaults, Encoder, test_latency_profile_defaults, SQLiteStorage, AdaptiveRouter, get_profile, test_router_inherits_profile, ProfileType

- **tests/test_router.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.exceptions, pytest, sqlite3, synaptoroute.router, os, synaptoroute.encoder, synaptoroute.storage, synaptoroute.models, numpy
  - **Exports**: test_static_routing, temp_db, Encoder, test_top_1_masking_fallback, test_delete_route_memory, test_max_capacity_load_routes, test_overwrite_route_capacity, test_fit_thresholds_mismatched_lengths, SQLiteStorage, test_add_utterance_unknown_route, test_duplicate_utterance_ignored, RouterCapacityError, test_zero_state_inference, test_delete_nonexistent_route, test_load_routes_discards_wrong_dimension_blob, test_max_capacity_add_route, test_hot_reload_utterance, Route, RouteNotFoundError, AdaptiveRouter, test_max_capacity_add_utterance, storage

- **tests/test_storage.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.exceptions, pytest, sqlite3, os, synaptoroute.router, synaptoroute.encoder, synaptoroute.storage, synaptoroute.models
  - **Exports**: test_delete_route_storage, test_add_utterance, Route, test_save_and_load_route, test_sqlite_storage_creates_directory, memory_db, test_save_route_replace, Encoder, SQLiteStorage, test_corrupt_json_metadata, AdaptiveRouter, SynaptoRouteError

- **tests/test_sync.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: asyncio, pytest, json, unittest.mock, synaptoroute.sync
  - **Exports**: test_dispatch_rejects_mismatched_encoder_model, test_broadcast_sends_message, patch, mock_listen, RedisSyncManager, MagicMock, test_listener_ignores_own_sender_id, AsyncMock, mock_router

- **tests/test_trainer.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: synaptoroute.router, pytest, unittest.mock, synaptoroute.trainer
  - **Exports**: mock_openai_client, test_synthetic_tuner_tune_route, MagicMock, AdaptiveRouter, SyntheticResponse, SyntheticTuner, AsyncMock, mock_router

- **tests/test_validation_gaps.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: asyncio, pytest, synaptoroute, synaptoroute.trainer, synaptoroute.encoder, synaptoroute.profile, synaptoroute.storage, unittest.mock
  - **Exports**: create_side_effect, update_threshold, DummyStorage, delete_route, BaseStorage, SyntheticTuner, get_profile, DummyParsed, add_utterance, test_openai_encoder_chunking, patch, load_all_routes, OpenAIEncoder, DummyChoice, AsyncMock, ProfileType, DummyData, __init__, Route, test_fit_thresholds_async_loop_safety, DummyMsg, save_route, MagicMock, AdaptiveRouter, test_latency_profile_propagates_threads, FastEmbedEncoder, DummyResp

- **tests/__init__.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: None
  - **Exports**: None

### Other Components
- **analyze_repo2.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: ast, os, json
  - **Exports**: analyze_file, main, extract_docstring

- **generate_reports.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: os, json
  - **Exports**: generate_repo_map, detect_risks, main, generate_arch_report

- **test_fastembed.py**
  - **Purpose**: No explicit purpose documented.
  - **Imports**: onnxruntime, fastembed
  - **Exports**: TextEmbedding
