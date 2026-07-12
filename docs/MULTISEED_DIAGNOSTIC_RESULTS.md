# Multi-Seed Diagnostic Results

Date: 2026-07-13  
Status: unverified development evidence; not paper evidence

These studies execute the fixed seeds `13`, `29`, `42`, `71`, and `101` with
20 unique route examples per intent, full held-out calibration data, and the
full official test splits. Exact-text embedding memoization is used only to
avoid recomputing deterministic vectors; latency is excluded.

The original runs occurred from a dirty working tree while implementation
continued. The complete study was subsequently repeated from clean commit
`df94df3`, producing exactly equal aggregate and paired statistical sections.
That provenance is recorded in `CLEAN_REPLICATION_RESULTS.md`. The artifacts
remain unverified and must not be used externally until they are archived and
independently reproduced.

## Banking77

Values are mean plus or minus sample standard deviation over five seeds.

| System | Accuracy | Macro-F1 | Known coverage |
|---|---:|---:|---:|
| Exact cosine | 0.8876 +/- 0.0051 | 0.8795 +/- 0.0087 | 0.9977 +/- 0.0026 |
| Logistic regression | 0.8659 +/- 0.0029 | 0.8518 +/- 0.0042 | 0.9968 +/- 0.0042 |
| Semantic Router | 0.8871 +/- 0.0045 | 0.8796 +/- 0.0085 | 0.9963 +/- 0.0046 |
| SynaptoRoute, global | 0.8871 +/- 0.0045 | 0.8796 +/- 0.0085 | 0.9963 +/- 0.0046 |
| SynaptoRoute, per-route | 0.7966 +/- 0.0081 | 0.8394 +/- 0.0032 | 0.8623 +/- 0.0131 |

Per-route minus global overall accuracy was `-0.0906`, with a hierarchical
bootstrap 95% interval of `[-0.0981, -0.0829]`. At matched 90% known coverage,
per-route selective accuracy was `-0.0170` lower, with a seed-bootstrap 95%
interval of `[-0.0223, -0.0116]`.

Conclusion: route-specific thresholds harm Banking77 performance under this
protocol. This negative result must be retained in subsequent reporting.

## CLINC150/OOS

| System | Accuracy | Macro-F1 | Known coverage | OOD recall | OOD AUROC | FPR@95 |
|---|---:|---:|---:|---:|---:|---:|
| Exact cosine | 0.8128 +/- 0.0075 | 0.8395 +/- 0.0032 | 0.9868 +/- 0.0045 | 0.4884 +/- 0.0739 | 0.9132 +/- 0.0063 | 0.3509 +/- 0.0178 |
| Logistic regression | 0.8447 +/- 0.0171 | 0.8782 +/- 0.0088 | 0.9862 +/- 0.0063 | 0.4996 +/- 0.1126 | 0.9426 +/- 0.0046 | 0.2367 +/- 0.0111 |
| Semantic Router | 0.8129 +/- 0.0076 | 0.8396 +/- 0.0032 | 0.9868 +/- 0.0045 | 0.4888 +/- 0.0744 | 0.9202 +/- 0.0055 | 0.3160 +/- 0.0124 |
| SynaptoRoute, global | 0.8128 +/- 0.0075 | 0.8395 +/- 0.0032 | 0.9868 +/- 0.0045 | 0.4884 +/- 0.0739 | 0.9201 +/- 0.0055 | 0.3160 +/- 0.0125 |
| SynaptoRoute, per-route | 0.8305 +/- 0.0035 | 0.8490 +/- 0.0034 | 0.9032 +/- 0.0034 | 0.8392 +/- 0.0107 | 0.9421 +/- 0.0032 | 0.3059 +/- 0.0209 |

Per-route minus global overall accuracy was `+0.0177`, with a hierarchical
bootstrap 95% interval of `[+0.0083, +0.0267]`. Per-route still trailed logistic
regression by `-0.0142`, interval `[-0.0276, -0.0012]`.

At matched 90% known coverage, per-route versus global produced:

* selective accuracy effect `+0.0133`, 95% interval `[+0.0096, +0.0165]`;
* selective-risk effect `-0.0133`, 95% interval `[-0.0165, -0.0096]`;
* OOD false-acceptance effect `-0.1354`, 95% interval
  `[-0.1512, -0.1136]`.

Against logistic regression at the same known coverage, per-route thresholds
had lower OOD false acceptance by `0.0204` but lower selective accuracy by
`0.0430`.

## Research Decision

The static quality hypothesis is only partially supported:

* route-specific calibration improves open-set behavior on CLINC150/OOS;
* the same method materially harms Banking77;
* a simple supervised classifier remains stronger on overall CLINC quality;
* SynaptoRoute and Semantic Router are effectively equivalent under shared
  route examples and encoder vectors.

The project should not pursue a broad claim of superior intent-routing
accuracy. The strongest paper direction remains explicit durability,
backpressure, dynamic mutation behavior, and measured quality/latency tradeoffs
for local routing. Calibration is an ablation within that systems paper.
