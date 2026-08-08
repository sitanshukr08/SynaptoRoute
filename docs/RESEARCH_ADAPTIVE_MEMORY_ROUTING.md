# Experimental Design Note: Adaptive Memory Routing

Status: experimental, disabled by default, and excluded from the primary paper artifact

## Implemented Mechanism

SynaptoRoute can adjust a retrieved route score using bounded frequency,
recency, negative-feedback, and priority signals. The current implementation
also contains an in-process buffered statistics collector and an adaptive
replacement cache prototype.

The feature changes decision scores after cosine retrieval. It therefore does
not preserve the original ranking in every case, and the current asymmetric
adjustment range can reverse candidates whose raw score gap is smaller than
the maximum difference between their adjustments.

## Claims Not Established

The implementation does not currently establish:

* metric-topology preservation;
* lock-free or zero-contention query execution;
* anti-starvation or safety guarantees;
* improved semantic-routing accuracy;
* reduced LLM token use, output entropy, cost, or behavioral variance;
* persistence of adaptive counters across restart.

Conditioning a probabilistic model on additional information does not by
itself prove that a particular deployed model will emit fewer tokens. Any
token or cost claim requires an end-to-end experiment with a fixed model,
prompt policy, tokenizer, workload, and billing definition.

## Required Validation

Before this feature can support a research claim, a separate protocol must:

1. define stationary and drifting workloads with ground-truth routes;
2. compare raw cosine, static priors, bounded adaptive priors, and a supervised
   online baseline;
3. measure accuracy, OOD false acceptance, calibration, route starvation, and
   p95/p99 latency;
4. report sensitivity to every adjustment bound and decay parameter;
5. test adversarial popularity feedback and delayed negative feedback;
6. use multi-seed runs and independently reproduced artifacts.

Until those requirements are met, adaptive memory is a product experiment and
must remain disabled in paper benchmarks.
