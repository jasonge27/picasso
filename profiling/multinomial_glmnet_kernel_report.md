# Multinomial glmnet-Inspired Kernel Validation

## Scope and Backup

The pre-change repository is preserved at
`/private/tmp/picasso-pre-glmnet-fast-kernel-20260717/repository`. The frozen
production library has SHA-256
`7aa8b54a8ba8ed20e3fad66bee04f62014c312c72148c0b99d4d4f0168a4a340`.
Each candidate was compiled and timed independently before retention.

## Rejected Candidates

Class-major traversal had a `0.99x` geometric speedup and slowed the sparse
case by about 5.5%. Fixed modified-Newton `1/4` curvature slowed all four
fixtures (`0.90x`, `0.93x`, `0.63x`, and `0.44x`) because conservative steps
increased coordinate visits. A blocked `X^2` GEMM added an `O(nK)` buffer
without reliable speedup. Exact-curvature Eigen vectorization averaged only
about `1.006x` and produced repeatable wide/dense regressions. A runtime hybrid
putting both compact schedules in one hot function slowed the large-`n` case
by about 5.2%. All five candidates were rolled back.

## Retained Change

The feature-level strong/KKT set remains authoritative. Inside a restricted
fixed-IRLS quadratic, partial sweeps now use a precision- and shape-adaptive
second tier:

- scalar L1 with tolerance at least `1e-4`, `d < 96`, and `K < 8` keeps the
  lower-overhead feature-resolution list;
- strict, wide, high-class, and coordinate-weighted LLA subproblems use a
  coefficient-resolution list containing only coordinates that moved.

The two hot loops are compile-time specializations with one dispatch outside
the sweep loop. Every path still performs a complete candidate-feature/class
sweep before exact KKT certification. Compact storage stops at 75% density,
is released immediately, and falls back to full sweeps. Checked index
arithmetic falls back safely if `candidate_count * K` cannot fit in `int`.

## Performance and Accuracy

Fresh-process ABBA tests used the V4 C ABI, 24 lambdas, four warmups, 15
measured repeats, and one thread.

| Case | Fast `1e-4` | Strict `1e-7` |
|---|---:|---:|
| Tiny (`60 x 8`, K=3) | 1.016x | 1.118x |
| `p > n` (`96 x 600`, K=4) | 1.166x | 1.268x |
| High K (`240 x 32`, K=12) | 1.070x | 1.114x |
| Large n (`4000 x 24`, K=4) | 1.000x | 1.011x |
| Geometric mean | **1.061x** | **1.124x** |

Fast-mode peak-RSS ratios were `0.997-1.002`; strict ratios were
`0.994-1.000`. All paths fit 24/24 points and produced valid probabilities.
Maximum independent KKT residuals were `9.87e-5` and `8.69e-8`; maximum
objective differences from the frozen baseline were `2.07e-6` and `1.76e-12`.

Across 12 train/validation/test experiments, both libraries selected the same
lambda and stable-support Jaccard was `1.0`. Worst candidate increases were
`1.02e-4` test log-loss, `0.0005` classification error, and `5.51e-5` Brier
score.

## Regression Checks

- Native objective, ActNewton, LLA, and C API tests: PASS.
- ActNewton and C API under ASan/UBSan: PASS.
- Python end-to-end interface script against the candidate: PASS.
- In that recorded run, `R CMD check --no-manual` functional checks passed
  with one external R-header/Apple-Clang warning
  (`-Wfixed-enum-extension`).
