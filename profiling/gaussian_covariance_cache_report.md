# Gaussian Covariance Lazy-Cache Report

The benchmark compares the frozen full-Gram covariance objective with the
lazy-column implementation using the same dense input generator, lambda path,
compiler (`Apple clang`, `-O3 -DNDEBUG`), and `ActGDSolver`. Times and process
peak RSS are medians of five isolated runs. Peak RSS is read with `getrusage`.

| Dense scenario | Variant | Construct (ms) | Solve (ms) | Total (ms) | Peak RSS | Final nnz |
|---|---:|---:|---:|---:|---:|---:|
| `n=20000, p=120` | full Gram | 32.64 | 8.09 | 41.87 | 40.86 MB | 5 |
| | lazy columns | 5.60 | 12.19 | 17.70 | 41.27 MB | 5 |
| `n=128, p=3000` | full Gram | 113.74 | 1.97 | 115.83 | 80.04 MB | 5 |
| | lazy columns | 0.85 | 2.18 | 3.04 | 8.21 MB | 5 |
| `n=1000, p=600`, lambda down to zero | full Gram | 31.35 | 4.71 | 36.02 | 14.11 MB | 600 |
| | lazy columns | 1.18 | 39.28 | 40.52 | 14.39 MB | 600 |

For the large-`p` sparse path, lazy construction reduces peak RSS by about
89.7% and total time by about 97.4%. In the `n >> p` case the Gram matrix is a
small fraction of the dense design, so peak RSS is essentially unchanged, but
total time drops about 57.7%. The deliberately dense worst case eventually
caches every column: peak memory is comparable and total time is about 12.5%
slower because separate matrix-vector products do not match the efficiency of
one up-front Gram construction.

Every intercept and coefficient was dumped and compared for all six lambdas:
`6 x 25`, `6 x 121`, `6 x 3001`, and dense-worst `6 x 601` path values were
bit-for-bit identical to the pre-change binary. The dense-worst regression is
accepted without a fallback threshold: a late switch would repeat most Gram
work and temporarily duplicate memory, while the measured total-time penalty
is modest and sparse high-dimensional paths receive the intended gains.
