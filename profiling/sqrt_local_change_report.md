# Square-root Lasso Local-change Report

The inner ActNewton loop previously evaluated the square-root-loss local
change with an `O(n)` column reduction after every active coordinate update.
The retained implementation first evaluates a conservative `O(1)` upper bound
using the cached column norm, coefficient change, sample count, and current
loss.  It skips the exact reduction only when the inflated bound already lies
below the solver threshold.  Other objectives and the intercept retain their
existing checks, and no persistent storage is added.

Release libraries from the complete pre-change backup and candidate tree were
measured in an interleaved sequence.  Times below are medians of four fresh
processes for the same cap-three adaptive-LLA path.

| Workload | Penalty | Before | After | Speedup |
|---|---|---:|---:|---:|
| n=600, d=120, 24 lambda | MCP | 0.019290 s | 0.017349 s | 1.112x |
| n=600, d=120, 24 lambda | SCAD | 0.017877 s | 0.016097 s | 1.111x |
| n=3000, d=200, 20 lambda | MCP | 0.067727 s | 0.062291 s | 1.087x |
| n=3000, d=200, 20 lambda | SCAD | 0.060007 s | 0.058300 s | 1.029x |
| n=400, d=1000, 20 lambda | MCP | 0.462025 s | 0.403643 s | 1.145x |
| n=400, d=1000, 20 lambda | SCAD | 0.425172 s | 0.373069 s | 1.140x |

Every paired run has the same output SHA-256, status, fitted path length, stage
count, and iteration vector.  Peak RSS is unchanged within process noise.  A
native test also recomputes every certified change and verifies it is below the
threshold.  All 12 native tests, targeted ASan/UBSan, R tests, and Python
feature tests pass.

The complete baseline is
`/private/tmp/picasso-master-baseline-20260716-preincremental`; direct source
backups are `/private/tmp/picasso-step4-*.before`.
