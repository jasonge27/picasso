# Multinomial Exact-KKT Scan Benchmark

Phase A compares the legacy exact active-KKT scan on every coordinate sweep
(`interval=1`) with a periodic scan (`interval=4`). Each timing is the median of
three fresh processes after one warm-up process. The candidate still scans on
the first and final allowed sweep, scans early when the coordinate-change proxy
is small, and uses only exact scans for convergence and inactive-set repair.

The frozen baseline binary has SHA-256
`6eef34b44dadd78147feb51fe2cd499328474d49c1e90c0b1c468dc26453ee22` and was
built from the retained pre-change Phase A source checkpoint. These external
checkpoint artifacts are not required to build or test the repository.

## Dense Primary Gate

| Penalty | Baseline (s) | Interval 4 (s) | Speedup | Sweep change | Visit change | RSS change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MCP | 9.7756 | 6.8697 | 29.73% | +0.50% | +0.41% | +0.70% |
| SCAD | 8.8047 | 6.0139 | 31.70% | +0.61% | +0.51% | +0.65% |

Both penalties exceed the required 15% speedup; sweep and coordinate-visit
growth remain below 2%, and peak RSS growth remains below 5%. The largest
absolute target-objective digest difference is `2.30e-7`; the largest final
objective difference is `6.24e-10`; and the largest final exact-KKT difference
is `1.03e-12`. An interval-1 rebuild exactly reproduced the frozen sweep counts,
visit counts, and numerical digests.

## Secondary No-Regression Gate

Negative changes are faster.

| Case | MCP time change | SCAD time change |
| --- | ---: | ---: |
| `sparse-k3` | -7.62% | -5.60% |
| `wide-k4` | -3.75% | -3.68% |
| `high-k12` | -7.88% | -8.40% |

All benchmark processes converged and passed their objective, majorization,
descent, finiteness, and stationarity checks. The candidate therefore passes
Phase A, and `MultinomialActNewtonOptions` now defaults to interval 4. Use
`--scan-interval 1` in the benchmark to reproduce the legacy schedule.
