# Multinomial Performance and glmnet Comparison

## Scope

Measurements were run on arm64 macOS with R 4.5.2, glmnet 5.0, Accelerate
BLAS, and one compute thread. Dense, ungrouped multinomial L1 paths used the
same pre-standardized data, intercept, and lambda values. Objectives, KKT
residuals, probabilities, and held-out metrics were recomputed independently.
Timings are medians from isolated or alternating-order runs after warm-up.

## Retained PICASSO Improvements

Against the backed-up original multinomial implementation, the final core
uses adaptive inexact Proximal Newton subproblems, branch-free Eigen coordinate
kernels, and accepted-softmax reuse. On strict 24-lambda paths at
`prec=1e-7`, fresh-process ABBA measurements were:

| Workload | Speedup | Peak-RSS ratio | Max objective difference |
|---|---:|---:|---:|
| Tiny (`60 x 8`, K=3) | 2.918x | 1.000 | `1.62e-13` |
| Wide (`96 x 600`, K=4) | 5.000x | 1.003 | `1.38e-11` |
| High-K (`240 x 32`, K=12) | 5.659x | 1.004 | `8.70e-13` |
| Tall (`4000 x 24`, K=4) | 3.219x | 0.994 | `1.97e-13` |

The geometric-mean speedup is 4.04x; maximum external KKT is `9.36e-8`.

V4 additionally applies glmnet's generated-path tail rules after five points:
stop at explained deviance above `0.999` or adjacent gain below `1e-5`.
At `prec=1e-4`, 100 requested lambdas, and ratio `1e-4`, the retained prefix
was 58--60 points. Versus the same final library with stopping disabled,
speedups were 1.35x (wide), 12.47x (high-K), and 1.34x (tall), with a 2.82x
geometric mean. Common-prefix objectives were identical and peak RSS changed
by at most 0.3%. Eight held-out simulations retained the full-path validation
optimum every time; test log-loss agreed within `1.12e-16` and classification
error was identical. Explicit lambda paths intentionally remain complete.

R now streams post-fit deviance evaluation one lambda at a time. For
`n=15000`, `K=8`, and 80 lambdas, median peak RSS fell from 400.7 MB to
302.7 MB (24.5%) while the deviance vector was bit-identical. Median fit-plus-
postprocessing time changed from 2.378 s to 1.833 s, so the memory reduction
did not trade away speed.

## PICASSO Versus glmnet

Using the same numeric setting (`1e-7`) is not an equal-accuracy comparison:
glmnet was 3.61--10.60x faster, but its independently measured relative KKT
was roughly three orders looser. At matched usable external KKT below
`6.9e-4` (`PICASSO prec=3e-6`, `glmnet thresh=1e-11`):

| Shape | PICASSO | glmnet | Faster implementation |
|---|---:|---:|---:|
| Tall (`4000 x 120`) | 0.903 s | 0.595 s | glmnet, 1.52x |
| Balanced (`1200 x 500`) | 1.514 s | 3.214 s | PICASSO, 2.12x |
| Wide (`350 x 2200`) | 1.078 s | 1.415 s | PICASSO, 1.31x |

PICASSO is 1.22x faster by geometric mean at matched accuracy. Maximum
relative objective difference is `4.36e-8`. All three held-out experiments
selected the same lambda; maximum test log-loss difference is `1.47e-7`, and
classification error is identical. The historical 1.6--4.3x returned-object
size claim is withdrawn because the comparison script and raw object payloads
needed to rule out a captured glmnet training call are not preserved.
Historical peak-RSS percentages are also not used as solver-memory conclusions
because the package baselines differed and the durable repository does not
contain the raw process records needed for a fresh audit.

The glmnet comparison used an explicit 45-lambda path and therefore directly
represents final explicit-path behavior; V4 tail stopping is reported
separately above. The later streamed R postprocessing can only reduce wrapper
overhead relative to those measurements.

## Decisions and Reproduction

For exploratory model selection, `prec=1e-4` is the recommended fast setting;
use `1e-6` or `1e-7` for a tighter certificate. Curvature GEMM caching,
full-scan KKT GEMM, and in-place gradient restoration were benchmarked and
rolled back because their gains were negligible or memory/regression costs
outweighed them. Use `profiling/multinomial_benchmark.py` for isolated native
comparisons; pass `--path-early-stop` only when measuring generated-path V4
behavior.
