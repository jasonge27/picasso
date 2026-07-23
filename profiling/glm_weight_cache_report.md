# Scalar GLM Weighted-Column Cache Benchmark

## Question and setup

This benchmark compares the current lazy `wXX` cache in
`src/objective/glm.cpp` with the backed-up eager implementation in
`/tmp/glm-eager.cpp`. The files differ only in cache refresh: eager recomputes
all weighted column norms after every IRLS auxiliary update; lazy invalidates
the cache and computes a norm only when the active-set solver requests it.

Both complete dylibs were built from the same current sources with Apple
Clang, C++11, `-O3 -DNDEBUG -funroll-loops`; only `glm.cpp` was swapped. The
benchmark calls `SolveLogisticRegressionV2` or `SolvePoissonRegressionV2`
directly with identical standardized data, nonzero offsets, MCP (`gamma=3`),
28 or 20 lambdas, tolerance `1e-6`, and at most three adaptive LLA stages.
Each timing is a fresh, single-threaded subprocess. Seven repetitions per
implementation are interleaved, and the median solver-only wall time is
reported. Peak RSS includes Python, the dense input, and the loaded dylib.

## Results

| Case (`n x d`) | Max active | Iterations | Eager median | Lazy median | Speedup | Eager/Lazy peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Binomial sparse (900 x 3,000) | 49 | 332 | 377.1 ms | 188.4 ms | 2.00x | 80.61 / 80.66 MiB |
| Binomial wide (320 x 12,000) | 23 | 200 | 291.4 ms | 161.9 ms | 1.80x | 101.02 / 101.09 MiB |
| Poisson sparse (900 x 3,000) | 30 | 336 | 381.6 ms | 186.1 ms | 2.05x | 80.66 / 80.59 MiB |
| Poisson wide (320 x 12,000) | 86 | 256 | 358.4 ms | 196.2 ms | 1.83x | 100.84 / 101.08 MiB |

All paths returned every requested lambda. Status `10`
(`lla_stationarity_limit`) was identical and denotes a usable model at the
configured three-stage cap. For every case, coefficients, intercepts,
objectives, KKT/stationarity diagnostics, lambdas, iteration counts, active
sizes, LLA-stage counts, fitted-path length, and failure metadata were
bit-for-bit identical; every maximum absolute floating-point difference was
zero.

## Decision

Keep the lazy cache; do not roll it back. It reduces wall time by 44–51% on
the tested sparse active paths without changing numerical or iteration
behavior. It does not materially reduce peak memory (differences are below
0.3%), because both implementations retain the same cache arrays.

Reproduce with `profiling/glm_weight_cache_benchmark.py`; raw measurements
from this run are in
`/private/tmp/picasso-glm-weight-cache-bench/results.json`.
