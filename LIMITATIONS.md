# Known And Unknown Boundaries

SynaptoRoute separates implemented behavior from verified evidence. A working implementation is not the same thing as a release-grade performance or research claim.

## Known Engineering Boundaries

* **Encoder bottleneck:** local embedding inference is expected to dominate end-to-end latency, but exact throughput claims must be rerun.
* **Dynamic mutation at scale:** tombstones and index rebuilds can become expensive for very large route sets.
* **Redis sync:** Redis Pub/Sub sync is experimental. Bootstrap, replay, and missed-window behavior are not validated well enough for distributed consistency claims.
* **Retracted latency metric:** the old `0.003ms` claim was invalid because seconds were labeled as milliseconds. Treat the corrected `~3ms` interpretation as unverified until rerun.

## Unknown Boundaries

* Multi-million route deployments.
* Multi-region Redis sync behavior.
* Sustained GPU encoder throughput in production.
* Non-English semantic matching.
* OOD rejection reliability under adversarial or noisy text.

## Research Gaps

SynaptoRoute is a software system under active development, not a research-validated method yet.

Missing validation includes ablations, statistical significance, cross-encoder baselines, calibration analysis, multilingual evaluation, and reproducible OOD benchmarks.
