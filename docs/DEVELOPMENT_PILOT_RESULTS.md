# Development Pilot Results

Date: 2026-07-13

Status: diagnostic only; not paper evidence

These pilots validate the external-data experiment path and identify the next
research question. They were run from a dirty working tree with one seed and a
500-example stratified test subset. The generated manifests therefore report
`status=unverified` and `paper_evidence_eligible=false`.

## Fixed Configuration

* encoder: `BAAI/bge-small-en-v1.5` via FastEmbed;
* seed: 42;
* route examples: 20 unique training utterances per intent;
* calibration: full held-out calibration data;
* test: 500 deterministic stratified examples;
* global policy constraint: at least 0.80 known-intent coverage on calibration;
* external comparator: `semantic-router==0.1.15` with the shared encoder and
  route examples.

Banking77 excluded seven training rows that duplicated official test text.
CLINC150 excluded four training rows that duplicated held-out text. The exact
counts are retained in each experiment summary.

## Banking77

| System | Accuracy | Macro-F1 | Known coverage | Selective accuracy | P95 latency |
|---|---:|---:|---:|---:|---:|
| Exact cosine | 0.892 | 0.882 | 0.994 | 0.897 | 12.22 ms |
| Logistic regression | 0.862 | 0.854 | 1.000 | 0.862 | 11.72 ms |
| Semantic Router | 0.884 | 0.878 | 0.984 | 0.898 | 12.96 ms |
| SynaptoRoute, global | 0.884 | 0.878 | 0.984 | 0.898 | 11.46 ms |
| SynaptoRoute, per-route | 0.804 | 0.837 | 0.872 | 0.922 | 11.94 ms |

Per-route thresholds traded coverage for higher selective accuracy but did not
improve overall known-intent classification. Exact cosine was the strongest
accuracy baseline in this single pilot.

## CLINC150/OOS

| System | Accuracy | Macro-F1 | Known coverage | OOD recall | OOD AUROC | FPR@95 | P95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact cosine | 0.796 | 0.827 | 0.993 | 0.418 | 0.923 | 0.345 | 8.88 ms |
| Logistic regression | 0.854 | 0.880 | 0.983 | 0.582 | 0.961 | 0.198 | 9.04 ms |
| Semantic Router | 0.796 | 0.827 | 0.993 | 0.418 | 0.933 | 0.267 | 13.11 ms |
| SynaptoRoute, global | 0.796 | 0.827 | 0.993 | 0.418 | 0.933 | 0.267 | 8.75 ms |
| SynaptoRoute, per-route | 0.836 | 0.845 | 0.905 | 0.868 | 0.957 | 0.139 | 7.79 ms |

Per-route thresholds substantially improved OOD rejection in this pilot, at
the cost of lower known-intent coverage. Logistic regression remained stronger
on overall accuracy and OOD AUROC.

## Decision

The current implementation does not support a claim of superior static intent
classification. SynaptoRoute and Semantic Router produced the same quality
results, while a simple supervised baseline was stronger on CLINC150.

The defensible next directions are:

1. evaluate global and per-route policies at matched known coverage across five
   seeds with paired confidence intervals;
2. retain the dynamic mutation, bounded concurrency, latency, and durability
   systems study as the primary paper direction;
3. treat any quality improvement as a calibration result, not a novel
   embedding or retrieval algorithm.

Latency values above come from one sequential development run. They are useful
for detecting regressions only and must not be presented as comparative
performance claims.
