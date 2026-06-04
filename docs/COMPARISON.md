# Comparison Policy

This document no longer publishes static performance comparisons. Previous
comparison numbers were not reliably tied to reproducible benchmark logs and
should be treated as retracted.

## How To Compare SynaptoRoute

Use the benchmark harness and save raw outputs before making claims:

```bash
python benchmarks/run_all_benchmarks.py --benchmarks accuracy latency
```

For any comparison against another router, record:

- exact package versions
- exact embedding model
- dataset and split
- hardware and provider configuration
- command used
- raw output log
- failure and timeout behavior

## Documentation Rule

Do not add tables with accuracy, latency, throughput, or scale claims unless the
corresponding raw benchmark output is committed or linked from the release notes.
