# SynaptoRoute Research Protocol

Version: 0.3

Status: confirmatory protocol draft; exploratory pilots already observed

Scope: single-node semantic routing; Redis synchronization is excluded from
the primary paper claim.

Earlier Banking77, CLINC150/OOS, and systems pilots informed this protocol and
are therefore exploratory rather than preregistered evidence. The protocol is
frozen before the final artifact runs. Any later deviation must be recorded in
the relevant manifest and in the paper's threats to validity.

## Research Questions

**RQ1 - Selective quality:** At matched coverage, how accurately does a local
embedding router classify known intents and reject unsupported requests?

**RQ2 - Dynamic behavior:** How do concurrent route mutations affect routing
correctness, mutation visibility, and p95/p99 query latency?

**RQ3 - Durability:** What latency and recovery cost is required to provide
explicit in-memory, flushed, and restart-durable mutation guarantees?

**RQ4 - Scale:** How do route count, utterances per route, index choice, and
batching affect quality, latency, throughput, and memory?

## Hypotheses

* **H1:** Validation-calibrated per-route thresholds and score margins reduce
  selective risk and OOD false acceptance relative to one global threshold at
  matched coverage.
* **H2:** Bounded microbatching improves throughput under burst load without
  increasing p99 latency beyond the declared service objective.
* **H3:** Explicit storage barriers recover every mutation acknowledged as
  durable after process restart, with measurable but bounded overhead.
* **H4:** HNSW lowers retrieval cost at large route counts, but exact retrieval
  remains a necessary quality baseline because approximation can change route
  rankings.

Hypotheses are not success criteria. Unsupported hypotheses will be reported
as negative results rather than retuned after test-set inspection.

## System Boundary

The primary artifact includes:

* local text encoding;
* NumPy exact and FAISS HNSW retrieval;
* route-level threshold and margin decisions;
* optional second-stage reranking;
* asynchronous query batching;
* SQLite persistence and recovery.

Mutation experiments distinguish in-memory acknowledgement from receipt-level
durable acknowledgement as defined in
[`DURABILITY_CONTRACT.md`](DURABILITY_CONTRACT.md). Queue drain alone is not
treated as proof of a successful commit.

The primary artifact excludes:

* Redis consistency claims;
* authorization or security enforcement;
* adaptive-memory, hybrid lexical, slot, session, and permission-filter claims;
* end-to-end LLM answer quality;
* claims that bundled generated fixtures represent real user traffic.

## Datasets And Splits

| Dataset | Purpose | Split policy |
|---|---|---|
| Banking77 | Known-intent classification | Official train/test split; route examples sampled only from train. |
| CLINC150/OOS | Intent classification and OOS rejection | Official train/validation/test splits; validation fits thresholds, test is evaluated once per frozen configuration. |
| BOLT tasks | Optional external-validity follow-up | Not required for the primary systems paper. |
| MASSIVE subset | Optional multilingual follow-up | Not required for the primary systems paper. |
| Synthetic vectors | Structural latency and scale only | Never used to support semantic accuracy claims. |

For every run, record dataset source, version or revision, license, split,
known-class selection, route-example count, query count, and random seed.

Exact normalized text is decontaminated before sampling. The official test
split has priority: validation examples duplicated in test are excluded, then
training examples duplicated in either held-out split are excluded. Duplicate
texts within a class are collapsed before route/calibration sampling, and text
mapped to multiple known labels is rejected. Every exclusion count is written
to dataset metadata. This rule is fixed before final experiments.

## Baselines

All feasible baselines must receive the same route examples, test queries,
encoder, warmup, and concurrency limits.

1. Exact normalized string matching.
2. Logistic regression over fixed embeddings.
3. Exact cosine nearest-route retrieval over the same embeddings.
4. SynaptoRoute with NumPy exact search.
5. SynaptoRoute with FAISS HNSW search.
6. Aurelio Semantic Router `>=0.1.15` using the same encoder.
7. Published open-intent methods when their implementations and licenses permit
   faithful reproduction.

Top-K metrics must be omitted for a baseline when its supported public API does
not expose ranked candidates. A Top-1 decision must never be copied into
Top-3 or Top-5 columns.

## Metrics

Quality:

* top-1 accuracy and macro-F1 on known intents;
* AUROC, AUPRC, and FPR@95 for OOD detection;
* coverage, selective risk, and risk-coverage AUC;
* expected calibration error and Brier score where probabilities are produced
  by a held-out calibration method fixed before the confirmatory run; raw
  cosine scores and margins are not treated as probabilities;
* explicit abstention, ambiguity, and high-confidence-error counts.

The frozen probability-calibration split, target, features, fallback, and
artifact contract are specified in `paper/QUALITY_PROTOCOL.md`.

Systems:

* p50, p95, p99, and maximum end-to-end latency;
* wall time and completed requests per second;
* process RSS and index memory;
* mutation visibility and durable-commit latency;
* recovery time and lost, duplicated, or stale mutations;
* cold-start and model/index build time.

Latency is reported in milliseconds. Raw measurements are retained in seconds
or nanoseconds with the conversion performed exactly once in analysis code.

## Experimental Design

* Use the predetermined seeds `13`, `29`, `42`, `71`, and `101` for sampled
  route examples or class selections.
* Run a documented warmup before timed measurements.
* Run compared systems sequentially on the same otherwise-idle host.
* Alternate system order between repetitions to reduce order effects.
* Apply the same offered concurrency and harness queueing policy to every
  compared system.
* Separate encoder, retrieval, decision, persistence, and end-to-end timings.
* Record failures and timeouts in the denominator; do not silently retry them.
* Freeze thresholds and hyperparameters using training/validation data before
  evaluating the final test split.

## Statistical Analysis

* Report bootstrap 95% confidence intervals for accuracy, F1, OOD, and
  percentile-latency summaries.
* Use paired bootstrap or McNemar tests for predictions on the same examples.
* Use paired per-repetition comparisons for systems measurements; report an
  effect size in addition to p-values.
* Correct for multiple comparisons within each experiment family.
* Publish per-query predictions and per-repetition measurements so alternative
  analyses remain possible.

## Evidence Lifecycle

Every run starts as `unverified`.

A run can be promoted to `verified` only when:

* it was executed from a concrete commit with a clean working tree;
* the command, environment, hardware, dependencies, and seed are recorded;
* dataset metadata and counts are complete;
* the benchmark script and raw log are non-empty;
* the raw log hash is recorded;
* metrics are machine-readable and use explicit units;
* all subprocesses completed successfully;
* another contributor reviewed the manifest against the raw output.
* an independent run from a different machine reproduced the same frozen
  configuration;
* the evidence bundle was archived immutably with a recorded digest.

Paper tables must be generated from verified machine-readable result files.
Values must not be transcribed manually from terminal output.

The exact final systems matrix is stored in
`paper/experiment_matrix.json`. Development smokes may use smaller parameters
but must never be promoted as substitutes for that matrix.

## Failure And Exclusion Policy

Runs may be excluded only for a reason fixed before analysis, such as host
suspension, unrelated system load, corrupted dataset download, or benchmark
process failure. The failed run and exclusion reason remain archived. Poor
results are not an exclusion reason.

## Ethics And Limitations

* A semantic route is not authorization to execute a tool.
* False positive routes may trigger unsafe workflows; abstention and downstream
  authorization are mandatory in deployments.
* User queries and embeddings may contain sensitive information and must not be
  included in public artifacts without consent and de-identification.
* Performance can vary across language, dialect, domain, hardware, and route
  wording.
* Generated or templated data must be labeled as synthetic and cannot replace
  evaluation on independently collected public data.

## Reproduction Checklist

Before a paper submission, a new machine must be able to:

1. install the declared environment;
2. download or validate licensed datasets;
3. run the test suite;
4. execute one smoke benchmark;
5. execute one complete quality experiment;
6. validate manifests and raw-log hashes;
7. regenerate every paper table and figure.
