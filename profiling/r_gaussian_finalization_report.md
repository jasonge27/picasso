# Gaussian R Finalization Refactor

## Scope

This benchmark isolates the R-only Gaussian path-finalization change. The old
and new wrapper sources are loaded into separate environments above the same
installed PICASSO namespace and native library. This prevents native solver
changes from contaminating the comparison.

The refactor removes unreachable zero-feature handling, reshapes the committed
native prefix directly, and replaces row-wise `sweep()` scaling with ordinary
matrix/vector multiplication. It retains exact-zero `df` counting.

## Reproducibility

The recorded run used nine alternating old/new repetitions. Before measurement,
the driver calibrates one shared inner count per case against both
implementations, targeting a 0.2-second batch. It rejects a run if any measured
batch is shorter than 0.1 seconds. The recorded batches ranged from 0.229 to
0.529 seconds, so the sub-millisecond per-invocation results below are not
single-timer-tick estimates. `Rprofmem()` cumulative allocation bytes and wall
time are reported per invocation; these bytes are not peak RSS.

```sh
Rscript --vanilla profiling/r_gaussian_finalization_benchmark.R \
  /private/tmp/picasso-r-refactor-step1-before-20260719/R-package/R/picasso.gaussian.R \
  R-package/R/picasso.gaussian.R \
  /private/tmp/picasso-refactor-p0-r-check/picasso.Rcheck \
  profiling/r_gaussian_finalization_results.csv 9
```

- Old source MD5: `2ad4e71da8132eb8065167c6a6b7ac57`
- New source MD5: `210b86a6ef54c501fbfb8d8c15104fa7`
- Native library MD5: `aca1284e9a328609975c738a5ab33535`
- Benchmark driver MD5: `1dae0f9401606e65652336db916eefd7`
- R/Matrix: R 4.5.2, Matrix 1.7.4
- CPU/platform: Apple M3 Pro; Darwin 25.5.0 arm64
- BLAS: Apple Accelerate `libBLAS.dylib`
- Thread variables: OMP, OpenBLAS, MKL, vecLib, BLIS, and Rcpp Parallel all
  unset

The raw CSV additionally records the complete R version and platform strings,
native-library path, BLAS path, per-case inner count, every batch duration, and
all thread-variable values on every row.

The script verifies byte-identical serialization after normalizing only the
wall-clock `runtime` field. Oracles cover naive/covariance backends,
standardize/intercept combinations, integer input, a one-lambda path, and a
`dfmax`-truncated prefix. All 11 source-wrapper oracles passed. Every old/new
pair within a benchmark case also has the same result checksum.

## Results

Median results from the raw CSV:

| Case | Old time | New time | Time change | Old bytes | New bytes | Byte change |
|---|---:|---:|---:|---:|---:|---:|
| `large_d_standardized` | 7.171 ms | 4.029 ms | -43.8% | 34,022,832 | 24,002,192 | -29.5% |
| `large_d_unstandardized` | 5.344 ms | 3.133 ms | -41.4% | 26,022,736 | 20,002,144 | -23.1% |
| `small_d_standardized` | 0.179 ms | 0.128 ms | -28.9% | 438,288 | 309,392 | -29.4% |
| `small_d_unstandardized` | 0.146 ms | 0.123 ms | -16.0% | 335,792 | 258,144 | -23.1% |
| Tall covariance wrapper | 0.486 ms | 0.466 ms | -4.3% | 972,576 | 947,616 | -2.6% |
| Tall naive wrapper | 1.180 ms | 1.160 ms | -1.7% | 2,045,680 | 2,028,960 | -0.8% |
| Wide naive wrapper | 0.503 ms | 0.481 ms | -4.4% | 943,744 | 846,000 | -10.4% |

No priority case regressed. The isolated finalization kernel is materially
faster and allocates 23–29% fewer cumulative bytes; complete wrapper fits retain
median runtime while reducing allocation.
