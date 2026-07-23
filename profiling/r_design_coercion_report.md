# R Design-Coercion Report

## Decision

Retain type-checked R design coercion. Assigning
`storage.mode(X) <- "double"` duplicates a shared matrix even when `X` is
already double. PICASSO now performs that assignment only for integer input,
and the redundant multinomial pre-coercion is removed. No solver, tolerance,
or public argument changed.

## Evidence

The A/B baseline was the completed scalar borrowed-design build; the candidate
changed only three R preprocessing sites. Tests used an arm64 macOS 26.5.2
host, R 4.5.2, Accelerate BLAS, `n=100000`, `d=100`, explicit lambda paths,
`standardize=FALSE`, and `fast.mode=TRUE`.

`Rprofmem()` isolated the Gaussian call and found that the candidate removed
exactly one 80,000,048-byte allocation. Its measured R allocations fell from
165,695,168 to 85,695,920 bytes. A fresh-process multinomial run reduced peak
RSS from 535.8 MB to 460.4 MB.

Two opposite-order timing passes gave these averages of per-process medians:

| Family | Before | After | Change |
|---|---:|---:|---:|
| Gaussian | 162.5 ms | 152.5 ms | -6.2% |
| Binomial | 302.5 ms | 291.5 ms | -3.6% |
| Poisson | 466.0 ms | 457.0 ms | -1.9% |
| sqrt-lasso | 171.0 ms | 156.5 ms | -8.5% |
| Multinomial | 108.5 ms | 99.5 ms | -8.3% |

Negative change means faster. Scalar output summaries were serialization-
identical. Multinomial coefficients, intercepts, deviance, status, iteration
diagnostics, and KKT diagnostics were identical after excluding the expected
wall-clock runtime column. Integer and double designs now have explicit
full-path parity tests, while no-op double preparation has a `tracemem()` guard
against reintroducing the copy. The complete R testthat suite passed.

Raw measurements are in `r_design_coercion_results.csv`. The scalar path can
be reproduced with `r_scalar_borrowed_design_benchmark.R`; use
`/usr/bin/time -l` for process RSS and `Rprofmem()` around `picasso()` for
R-owned allocation attribution.
