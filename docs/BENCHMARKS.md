# Benchmarks

This project previously published benchmark numbers that were not reliably tied
to reproducible scripts and captured logs. Treat those historical numbers as
retracted until they are regenerated from the current repository state.

## Current Policy

- Do not publish benchmark metrics unless the exact script, commit, environment,
  dataset, and raw output log are available.
- Do not copy numbers from generated prose into the README.
- Prefer small, repeatable benchmark profiles before publishing large-scale
  claims.
- Keep benchmark output in `benchmark_results/` or another ignored directory.

## Reproducible Entry Points

Run the benchmark wrapper:

```bash
python benchmarks/run_all_benchmarks.py --benchmarks accuracy latency
```

Run external dataset evaluation directly:

```bash
python benchmarks/bench_realworld.py
```

Run synthetic scale evaluation directly:

```bash
python benchmarks/bench_extreme_scale_v2.py --dataset benchmarks/datasets/synthetic/routes_50k.json
```

## Publishing Requirements

Before adding a number to docs or release notes, include:

- git commit hash
- Python version and operating system
- CPU/GPU/provider configuration
- embedding model
- dataset name and split
- command used
- raw output log
- whether the run completed successfully

If any of those are missing, the number should be described as anecdotal or
omitted.
