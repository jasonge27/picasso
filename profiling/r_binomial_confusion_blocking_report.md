# R Binomial Confusion Lambda-Blocking Benchmark

Binomial `confusion.picasso()` formerly materialized full `n x nlambda`
linear-predictor and integer-prediction matrices. The retained implementation
preserves that single-GEMM path when their avoidable `12*n*nlambda`-byte
workspace fits in 8 MiB. Larger paths are evaluated in lambda-column blocks;
their width budgets the predictor and coefficient slice. It does not split
observations, so the classifier remains exactly `eta > 0`.

The fresh-process fixture in
`r_binomial_confusion_blocking_benchmark.R` uses a `100000 x 20` design and
100 lambdas. The final implementation, including the subsequent exact integer
tabulation kernel, was compared with the original implementation five times
using one BLAS thread and `/usr/bin/time -l`.

| implementation | 5-run median (s) | max RSS (bytes) | peak footprint (bytes) |
| --- | ---: | ---: | ---: |
| original full matrices | 2.154 | 695,205,888 | 451,691,600 |
| conditional blocks and tabulation | 0.168 | 516,980,736 | 213,730,192 |

The final path was 92.2% faster, reduced max RSS by 25.6%, and reduced macOS
peak footprint by 52.7%. Saved results were both `identical()` and
serialization-identical. A `100 x 100000` wide control remained on the
single-GEMM path and improved from 0.054 to 0.050 seconds; RSS differed by
0.2%. This control prevents coefficient size from triggering harmful blocks.

Focused tests force the exact full-path boundary, one-byte-below boundary,
exact block widths, and one-column blocks. They also cover repeated and
reordered lambdas, offsets, sparse coefficients, absent classes, and zero or
subnormal logits. The small path retains the previous matrix multiplication
and output structure.
