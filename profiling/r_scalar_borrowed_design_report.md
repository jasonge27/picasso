# R Scalar Borrowed-Design Report

## Decision

Retain the R-only borrowed column-major design view for Gaussian, binomial,
Poisson, and square-root lasso. R owns a stable column-major `double` matrix
for the duration of the synchronous native call, so the previous C++ copy was
redundant. Python remains on the owning row-major conversion path.

## A/B protocol

The baseline was the exact repository snapshot at
`/private/tmp/picasso-pre-huge-adoption-20260720-014625/repository`; the
candidate changed only scalar design storage. Each RSS measurement used a
fresh R process and `/usr/bin/time -l`. The Tall fixture was
`n=100000, d=100` (an 80,000,000-byte design). The 30-lambda timing fixture
used the same matrix, two opposite-order passes, and 7 then 9 in-process
repetitions. Both packages used R 4.5.2, arm64 macOS 26.5.2, Accelerate BLAS,
`standardize=FALSE`, Gaussian naive mode, and `fast.mode=TRUE`.

| Family | Owned RSS | Borrowed RSS | RSS reduction | Mean median runtime change |
|---|---:|---:|---:|---:|
| Gaussian | 460.3 MB | 392.0 MB | 68.3 MB | -6.7% |
| Binomial | 471.4 MB | 397.5 MB | 73.9 MB | -0.5% |
| Poisson | 471.2 MB | 395.4 MB | 75.7 MB | -1.0% |
| sqrt-lasso | 463.7 MB | 381.2 MB | 82.4 MB | -2.5% |

Negative runtime change means faster. The RSS reduction is 85%–103% of the
theoretical eliminated design copy; values above 100% reflect allocator and
OS sampling noise. No family showed a repeatable runtime regression.

Short and 30-lambda path summaries were serialization-identical between the
two installed packages. The focused native test additionally compares full
owned and borrowed solver paths, offsets, both Gaussian backends, MCP LLA,
copy/rvalue behavior, and invalid layouts. ASan/UBSan, the native suite, the R
testthat suite, and the Python suite passed. Raw measurements are in
`r_scalar_borrowed_design_results.csv`; reproduce a worker with, for example:

```sh
/usr/bin/time -l env R_LIBS_USER=/path/to/lib \
  Rscript --vanilla profiling/r_scalar_borrowed_design_benchmark.R \
  gaussian tall short 1 /tmp/gaussian.rds
```
