# R Scalar CV Lambda-Blocking Benchmark

Scalar cross-validation formerly materialized an entire test-fold-by-lambda
predictor matrix. The retained implementation preserves the original single
GEMM when the coefficient slice plus predictor fits in 8 MiB; larger paths are
evaluated in lambda-column blocks. It does not split rows, so per-column
reduction order remains unchanged.

The fresh-process fixture in `r_cv_lambda_blocking_benchmark.R` uses a
`120000 x 20` Gaussian design, 100 explicit lambdas, two fixed folds, the
covariance backend, and one BLAS thread. Baseline and candidate package
installations were selected through `R_LIBS_USER` and measured with
`/usr/bin/time -l`.

| implementation | 3-run median (s) | max RSS (bytes) | peak footprint (bytes) |
| --- | ---: | ---: | ---: |
| full predictor | 0.457 | 606,568,448 | 365,511,880 |
| 8 MiB lambda blocks | 0.307 | 576,438,272 | 232,227,848 |

The first comparison improved median time by 32.8%, reduced max RSS by
28.7 MiB, and reduced macOS peak footprint by 127.1 MiB. A reverse-order,
seven-run repeat gave medians of 0.309 seconds (baseline) and 0.276 seconds
(candidate), a 10.7% improvement. Saved result objects were byte-identical.

Focused tests force one- and two-column blocks across Gaussian, binomial,
Poisson, and square-root loss, every supported CV measure, offsets, truncated
paths, tail blocks, and one-row/one-lambda shapes. The unblocked branch is
byte-identical to the old oracle; blocked continuous losses agree within
`1e-13`, and class loss is exact. The public `cv.picasso` integration and the
complete R test suite passed.

These results isolate a tall, long-path case where the old allocation is
material. Small and wide problems deliberately retain the prior code path.
