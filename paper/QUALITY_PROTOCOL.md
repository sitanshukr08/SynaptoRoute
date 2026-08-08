# Quality And Probability Calibration Protocol

Status: frozen implementation protocol; all produced results remain unverified
until independent promotion.

## Data Boundaries

Route examples, calibration examples, and test examples must be text-disjoint.
The dataset loader enforces those boundaries before an experiment begins.

Calibration examples are partitioned deterministically within each intent. The
ordering key is SHA-256 over the experiment seed, label, and example ID. Half
of each label group is reserved for correctness-probability calibration; the
remainder is used for routing-policy fitting. A singleton label group remains
in the policy split. The two splits and test data must be disjoint.

## Routing Policy

The policy split selects global or per-route score thresholds and a decision
margin subject to the frozen coverage and OOD constraints. Exact-string
routing has no fitted threshold policy, but uses the same probability split as
the other systems.

## Correctness Probability

After the routing policy is fixed, a logistic model is fitted on the separate
probability split. Its binary target is whether the final route or abstention
decision is correct. Its finite input features are:

1. top retrieval score;
2. top-two margin;
3. policy-relative acceptance confidence;
4. score-present indicator;
5. margin-present indicator;
6. final-decision matched indicator.

Missing scores and acceptance confidence use declared finite sentinels. Raw
similarity scores and margins are features, not probabilities. If the held-out
target has only one class, the run records a Laplace-smoothed constant
probability instead of pretending a logistic model was identifiable.

## Test Evaluation

The fitted policy and probability model are applied to the untouched test set.
The runner emits test accuracy, selective/OOD metrics, ECE, maximum calibration
error, Brier score, reliability-bin JSON, and an SVG generated from that JSON's
metrics. Every probability artifact references its source prediction file by
SHA-256.

Selective accuracy is undefined and emitted as `null` when a system accepts no
test queries. Selective risk-coverage AUC is likewise `null` when the system
provides no finite confidence signal from which to rank queries. Reports must
not replace either value with a perfect score.

Per-system files are:

* `calibration_<system>.json` for the routing policy, when applicable;
* `probability_calibration_predictions_<system>.jsonl` for probability-fit data;
* `probability_calibration_<system>.json` for the fitted model;
* `test_predictions_<system>.jsonl` for untouched test decisions;
* `reliability_<system>.json` and `reliability_<system>.svg` for test reliability.

## Interpretation

Probability-fit metrics are in-sample diagnostics and are never publication
results. Test calibration metrics are descriptive unless the complete
multi-seed run, paired analysis, archive, and independent reproduction satisfy
the evidence-promotion gate. The method estimates final-decision correctness,
not semantic similarity or an authorization guarantee.
