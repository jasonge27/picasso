# Python Scalar Path Evaluation Benchmark

## Change

Scalar-family deviance calculation previously issued one matrix-vector product
per lambda. `Solver.assess()` then repeated the same products for MSE, MAE, or
classification error. The candidate evaluates `beta_block @ X.T` once and
reuses each predictor row for all requested metrics. Predictor blocks are
limited to 8 MiB; a one-model block still uses the original matrix-vector path.

The pre-change Python package is backed up at
`/private/tmp/picasso-step6-python-baseline`. The benchmark driver is
`profiling/python_path_evaluation_benchmark.py`.

## Correctness and memory tests

`python-package/test_pycasso.py` compares all Gaussian, sqrt-lasso, binomial,
and Poisson path metrics with the original scalar formulas. It forces multiple
two-model blocks, covers offsets, a non-contiguous design, a one-model tail,
binomial clipping and the `eta > 0` class boundary, Poisson overflow, and input
immutability. The full Python feature suite passes. ABBA benchmark checksums
were identical or differed only by `2.84e-14` in their summed metrics.

## ABBA results

Fresh processes used one BLAS thread. Each median below combines four baseline
and four candidate processes; native model fitting is excluded.

| Case | Baseline | Candidate | Speedup | Peak RSS change |
|---|---:|---:|---:|---:|
| `n=200,d=50,L=100`, Gaussian assess | 1.172 ms | 0.519 ms | 2.26x | +0.03 MiB |
| `n=10000,d=500,L=100`, Gaussian assess | 100.885 ms | 6.583 ms | 15.33x | +7.86 MiB |
| `n=10000,d=500,L=100`, binomial assess | 130.116 ms | 22.573 ms | 5.76x | +7.74 MiB |
| `n=100000,d=100,L=100`, Gaussian assess | 386.938 ms | 47.432 ms | 8.16x | +6.57 MiB |
| same normal case, strided `X` | 1005.348 ms | 9.325 ms | 107.81x | +47.53 MiB |
| `n=10000,d=500,L=1`, Gaussian assess | 1.830 ms | 1.238 ms | 1.48x | -0.02 MiB |

The strided case permits BLAS to pack the 40 MiB design, explaining both its
exceptional speedup and extra RSS. Ordinary contiguous inputs stay within the
8 MiB predictor cap.

## Block-size decision

On the tall case, 1/2/4/8/16/32 MiB caps gave 2.04x, 2.44x, 5.42x, 7.95x,
11.21x, and 13.60x speedups respectively. Eight MiB was retained: it captures
most of the practical improvement while bounding ordinary-input working memory
well below the faster 16--32 MiB alternatives.
