# Multinomial Compact Active-Set Benchmark

## Change

The restricted multinomial Proximal-Newton/IRLS subproblem now uses the same
two-tier working-set pattern as glmnet: a full sweep over the strong/KKT
candidate set, repeated partial sweeps over feature blocks that actually move,
and a final full-candidate sweep plus exact KKT certificate. The authoritative
outer active mask is never shrunk. Dense subproblems fall back to full sweeps
when the compact list reaches 75% of the candidate set.

The production C API enables the optimization. Direct C++ callers retain the
old default, which provides an isolated A/B switch and preserves compatibility.
The complete pre-change tree is backed up at
`/private/tmp/picasso-pre-glmnet-compact-active-20260717/repository`.

## Isolated A/B Results

Apple Clang 21, arm64 macOS, `-O3 -DNDEBUG`, fixed seeds, 24 lambdas, 15
alternating fresh-process runs. “Off” and “on” use identical current sources
except for the compact-list flag.

| Case | Off median | On median | Speedup | Coordinate visits |
|---|---:|---:|---:|---:|
| sparse, K=3 | 0.18245 s | 0.18250 s | 1.000x | 12,750 → 12,630 |
| wide, K=4 | 0.11363 s | 0.11178 s | 1.017x | 17,884 → 16,292 |
| high-K, K=12 | 0.10766 s | 0.10717 s | 1.005x | 25,032 → 24,588 |
| dense control | 0.36164 s | 0.36015 s | 1.004x | 457,560 → 456,508 |

Against the frozen pre-change executable, the primary wide case improved
1.020x. Wide-case median peak RSS was unchanged; the dense control differed by
16 KiB, one allocator/page-size step and not a measurable memory regression.

All runs passed independent objective, probability, support, and full-model KKT
checks; maximum independent KKT residual was `6.273e-8`. Native ActNewton, LLA,
C API, ASan/UBSan, complete R testthat, and complete Python regressions passed.
In that recorded run, `R CMD check` completed with only the external
R-header/Apple Clang warning already present before this change.
