# SynaptoRoute Improvement Roadmap

This document summarizes what similar projects are doing well and turns that into a practical roadmap for improving SynaptoRoute as both an open-source library and a paper-worthy systems project.

Execution status and milestone gates live in [`../ROADMAP.md`](../ROADMAP.md).
The fixed experimental definitions live in
[`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md). This document remains the
competitive analysis and product backlog; it is not the experiment protocol.

## Similar Work

### Semantic Router

Closest product analogue: [aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router).

What it does well:

* Clear mental model: routes are named decisions with example utterances.
* Easy quickstart and install path.
* Multiple encoder integrations, including local and hosted providers.
* Integrations with vector stores such as Pinecone and Qdrant.
* Docs around route optimization, dynamic routes, multimodal routes, and framework integrations.

What SynaptoRoute can differentiate on:

* SQLite-backed local persistence as a first-class feature.
* Truth-first benchmark manifests and raw-output evidence.
* Mutation/recovery behavior as a core systems contribution.
* A narrower, more auditable scope: local semantic routing for workflow/tool dispatch.

### RouteLLM And Router Research

Relevant research analogue: [RouteLLM](https://github.com/lm-sys/RouteLLM) and [RouteLLM paper](https://arxiv.org/abs/2406.18665).

What it does well:

* Treats routing as a cost-quality tradeoff.
* Provides trained routers and an evaluation framework.
* Supports multiple router strategies, including matrix factorization, BERT, LLM classifiers, and random baselines.
* Makes evaluation a central part of the project, not an afterthought.

SynaptoRoute should borrow the evaluation discipline, not the exact problem framing. RouteLLM routes between LLMs; SynaptoRoute should route between local workflow/tool intents.

### RouterEval

Relevant benchmark analogue: [RouterEval](https://arxiv.org/abs/2503.10657).

What it teaches:

* Router research needs open benchmark records, not just headline numbers.
* Baselines matter because routing quality is meaningless without a comparison set.
* Scaling analysis is a legitimate paper dimension when the benchmark data is reproducible.

SynaptoRoute does not need RouterEval-scale data, but it needs the same attitude: publish scripts, raw outputs, manifests, and baselines.

### RouteNLP

Relevant systems/research direction: [RouteNLP](https://arxiv.org/abs/2604.23577).

What it teaches:

* Routing is valuable when tied to operational constraints: cost, latency, acceptance rate, escalation rate.
* Calibration and cascading are paper-worthy.
* Closed-loop improvement from failures is stronger than static benchmark claims.

SynaptoRoute can adapt this idea with local routing: route locally first, escalate uncertain queries, collect failures, and retune thresholds.

### LlamaIndex, Haystack, Semantic Kernel, OpenAI Tools

Mature ecosystem references:

* [LlamaIndex RouterQueryEngine and RouterRetriever](https://developers.llamaindex.ai/python/framework/module_guides/querying/router/)
* [Haystack routers](https://docs.haystack.deepset.ai/docs/routers)
* [Semantic Kernel function calling](https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/chat-completion/function-calling/)
* [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)

What they show:

* Mature frameworks present routing as composition over tools/components.
* Tool schemas, descriptions, and selectors are part of the user-facing API.
* Function/tool calling is the dominant default in hosted LLM workflows.

SynaptoRoute should position itself as the local pre-routing layer before expensive LLM tool selection, not as a replacement for every tool-calling system.

## Best Project Positioning

The strongest honest claim is:

> SynaptoRoute is a local, persistent semantic routing layer for low-latency workflow, tool, and intent dispatch.

Avoid these claims until evidence exists:

* "production ready"
* "research validated"
* "guaranteed OOD detection"
* "verified 3ms latency"
* "distributed consistency"

The paper should be a systems/tooling paper, not a model paper.

## Highest-Impact Improvements

### 1. Make The Public API Boring And Stable

Target user flow:

```python
from synaptoroute import AdaptiveRouter, Route

router = AdaptiveRouter()
router.add_route(Route(name="billing", utterances=["payment failed", "refund status"]))

match = router("my card was charged twice")
if match:
    print(match.name)
```

Needed work:

* Add `examples/quickstart.py`.
* Add `examples/async_router.py`.
* Add `examples/sqlite_persistence.py`.
* Add `examples/langchain_router.py` and `examples/llamaindex_selector.py`.
* Document `router.close()` / `await router.stop()` clearly.
* Add a `RouterResult` type with `route`, `score`, `margin`, `candidates`, and `decision_reason`.

Why: current `router("query") -> Optional[Route]` is simple, but it hides confidence, score, and ambiguity. For papers and production, the result object matters.

### 2. Introduce A First-Class Evaluation Harness

Current benchmark policy is improving, but the benchmark scripts are still fragmented.

Needed work:

* Create `benchmarks/registry.py` with typed benchmark definitions.
* Make every benchmark emit one manifest plus one raw log.
* Add `benchmarks/run_benchmark.py --name latency_small --output-dir ...`.
* Add validators for dataset split, seed, route count, query count, and timing units.
* Add a docs page mapping each claim to exact command and raw output.

First verified benchmark should be a small local benchmark:

* no network;
* no external dataset download;
* deterministic synthetic route set;
* runs in CI;
* produces a manifest that can be marked `verified`.

Why: a small verified benchmark is more valuable than five impressive unverified ones.

### 3. Add Baselines Before Chasing Bigger Numbers

Minimum baseline set:

* exact string/rule matching;
* sklearn cosine nearest-neighbor over the same embeddings;
* SynaptoRoute NumpyIndex;
* SynaptoRoute FAISS/HNSW when installed;
* semantic-router, if feasible in an optional benchmark environment;
* LLM function calling/tool selection as an optional cost-latency comparison.

Metrics:

* top-1 accuracy;
* reject accuracy;
* AUROC/AUPRC for OOD;
* false-positive rate at chosen threshold;
* latency p50/p95/p99;
* memory usage;
* route mutation latency;
* cold boot time from SQLite.

Why: reviewers and users will ask "better than what?"

### 4. Make OOD And Abstention A Core Feature

Current thresholding is useful but too thin for a paper.

Needed work:

* Return top-k candidates and score margins.
* Add a calibrated abstention policy.
* Add threshold fitting per route with held-out validation data.
* Add reliability diagrams and expected calibration error.
* Add failure buckets: no-match, ambiguous-match, low-confidence-match, high-confidence-wrong.
* Add optional cross-encoder reranking as a second-stage verifier.

Paper angle:

* local semantic routers are useful only if they know when not to route.

### 5. Strengthen Persistence And Mutation Semantics

This is SynaptoRoute's likely differentiator.

Needed work:

* Define exact consistency contract:
  * read-your-writes for in-memory index;
  * eventual persistence to SQLite;
  * `flush_storage()` for durability barrier;
  * `close()` drains queued writes.
* Add tests for worker crash, queue overflow, failed SQLite writes, and restart recovery.
* Add WAL/backpressure metrics.
* Consider moving queued storage work to an explicit `StorageWriter` class.
* Add a route/version column to avoid stale update ordering.

Why: "mutable persistent semantic routing" is more interesting than "vector search with thresholds."

### 6. Treat Redis Sync As Experimental Until Rebuilt

Current Redis Pub/Sub can broadcast mutations, but it is not a durable sync protocol.

Needed work:

* Document it as experimental everywhere.
* Add node IDs, route versions, and idempotency keys.
* Replace full-state broadcast storms with snapshot + incremental log.
* Add missed-window tests.
* Add duplicate/out-of-order message tests.
* Add "single-node stable, multi-node experimental" release boundary.

Paper angle:

* distributed routing is future work unless the sync layer is proven.

### 7. Create A Real Open-Source Surface

Needed work:

* Add `CHANGELOG.md`.
* Add `SECURITY.md`.
* Add `CODE_OF_CONDUCT.md`.
* Add `examples/`.
* Add API reference docs.
* Keep README text ASCII-clean and avoid encoding-damaged symbols.
* Add badges only after CI is reliable.
* Add a "What this is not" section.
* Publish a small `v0.1.0` once API and tests are clean.

Why: the project needs to look dependable before anyone will trust the claims.

## Paper Plan

Working title:

> SynaptoRoute: A Local Persistent Semantic Router for Low-Latency Tool and Workflow Dispatch

Main claim:

* A local embedding router can provide fast, private, and durable routing for known workflow/tool intents, while exposing explicit abstention and escalation behavior.

Core experiments:

* Accuracy on intent datasets.
* OOD rejection and calibration.
* Latency and memory versus route count.
* Dynamic mutation and restart recovery.
* Baseline comparison.
* Cost/latency comparison versus LLM tool calling for fixed tool sets.

Required tables:

* Feature comparison against Semantic Router, LlamaIndex routers, Haystack routers, and LLM function calling.
* Accuracy/rejection metrics.
* Latency/memory scaling.
* Persistence/mutation behavior.
* Ablations: thresholding, margin, reranking, encoder choice.

Required honesty section:

* not a security boundary;
* OOD remains hard;
* Redis sync is experimental;
* encoder dominates latency;
* semantic routing quality depends on route examples and embedding model.

## Suggested PR Sequence

### PR 1: Stabilization And Evidence Baseline

Already mostly in progress:

* benchmark manifest schema;
* conservative docs;
* router consistency fixes;
* regression tests.

### PR 2: Public API And Examples

* Add `RouterResult`.
* Keep `router("query")` compatibility.
* Add `router.match("query") -> RouterResult`.
* Add examples and README quickstart.

### PR 3: First Verified Local Benchmark

* Deterministic synthetic dataset.
* CI-safe run.
* Manifest marked `verified`.
* Raw output checked in or archived under a documented evidence path.

### PR 4: Baselines

* Exact match baseline.
* sklearn cosine baseline.
* semantic-router optional baseline.
* LLM tool-calling optional baseline.

### PR 5: OOD And Calibration

* Top-k result object.
* Margin/threshold calibration.
* OOD benchmark.
* Reliability diagram data.

### PR 6: Persistence And Mutation Paper Section

* Storage writer abstraction.
* Durability barrier tests.
* restart recovery tests.
* mutation latency benchmark.

### PR 7: Open-Source Release Polish

* changelog/security/code-of-conduct;
* docs cleanup;
* API reference;
* package build/release workflow;
* v0.1.0 tag.

### PR 8: Paper Draft

* Write paper skeleton.
* Fill only verified results.
* Keep unverified claims out of tables.

## Immediate Next Actions

1. Finish and merge the current stabilization branch.
2. Add `RouterResult` without breaking existing `Optional[Route]` behavior.
3. Add quickstart and persistence examples.
4. Build the first CI-safe verified benchmark.
5. Only then start drafting the results section of the paper.
