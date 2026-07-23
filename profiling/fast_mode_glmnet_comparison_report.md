# PICASSO Fast Mode vs glmnet

> Historical benchmark snapshot from July 2026. It predates the current
> `type.gaussian="auto"` default: rows labelled `public defaults at benchmark
> time` used PICASSO's former naive Gaussian default. All measurements are
> dense L1 results on the environment recorded below, not universal speed
> claims.

## Protocol

This benchmark compares PICASSO 1.6 with glmnet 5.0 on the shared L1
objectives. Tall uses `n=4000, p=120`; Wide uses `n=350, p=2200`; multinomial
has four classes. Each case uses three seeds, one numerical thread,
pre-standardized dense matrices, an intercept, and 45 explicit decreasing
lambda values. Timings pool 21 alternating-order blocks. All shared-family
fits completed 45/45 points and used the same lambda path.

PICASSO used `fast.mode=TRUE`: Gaussian retained `1e-7`, binomial,
multinomial, and square-root lasso used `1e-4`, and Poisson used `4e-4`,
calibrated to glmnet's observed KKT certificate. glmnet used `thresh=1e-7`.
Square-root lasso has no glmnet counterpart and is reported separately.

## Runtime

`PIC/glmnet > 1` means glmnet is faster.

| Family/configuration | Shape | PICASSO (s) | glmnet (s) | PIC/glmnet |
|---|---|---:|---:|---:|
| Gaussian, public defaults at benchmark time | Tall | 0.01467 | 0.00844 | 1.737 |
| Gaussian, matched naive | Tall | 0.01442 | 0.01725 | 0.836 |
| Gaussian, matched covariance | Tall | 0.01183 | 0.00837 | 1.414 |
| Gaussian, public defaults at benchmark time | Wide | 0.02550 | 0.02400 | 1.063 |
| Binomial | Tall | 0.03750 | 0.05367 | 0.699 |
| Binomial | Wide | 0.02750 | 0.03900 | 0.705 |
| Poisson | Tall | 0.02600 | 0.03233 | 0.804 |
| Poisson | Wide | 0.05933 | 0.05733 | 1.035 |
| Multinomial | Tall | 0.15500 | 0.14600 | 1.062 |
| Multinomial | Wide | 0.13250 | 0.15300 | 0.866 |

PICASSO is faster for binomial in both shapes, Poisson Tall, Gaussian Tall
when both packages use the naive algorithm, and multinomial Wide. Poisson
Wide, Gaussian Wide, and multinomial Tall are within 3.5%, 6.3%, and 6.2% of
glmnet respectively. At the time of this benchmark, the apparent Gaussian
Tall default gap was primarily a configuration difference: glmnet used
covariance updates while PICASSO defaulted to naive updates. PICASSO now
resolves a guarded automatic backend, so the historical default rows do not
measure the current policy.

The archived CSV `object_mb` column is not a valid cross-package memory
comparison. The historical glmnet call was constructed with `do.call` and
retained the dense training matrix inside `fit$call$x`, whereas the PICASSO
fit did not. Those values and the earlier 40--200x interpretation are
withdrawn. The benchmark driver now removes captured calls before measuring
returned-object size; peak RSS still requires an isolated-process benchmark.

## Optimization and prediction quality

External checks recomputed objectives and KKT residuals from returned
coefficients. Maximum absolute KKT residuals were:

| Family | Shape | PICASSO | glmnet |
|---|---|---:|---:|
| Gaussian | Tall | 2.20e-4 | 1.22e-4 |
| Gaussian | Wide | 7.02e-4 | 5.46e-4 |
| Binomial | Tall | 9.77e-5 | 1.21e-5 |
| Binomial | Wide | 9.99e-5 | 7.65e-5 |
| Poisson | Tall | 3.97e-4 | 3.92e-5 |
| Poisson | Wide | 3.99e-4 | 3.88e-4 |
| Multinomial | Tall | 9.89e-5 | 5.68e-5 |
| Multinomial | Wide | 9.98e-5 | 1.10e-4 |

All 24 shared-family validation experiments selected the same lambda.
Binomial and multinomial classification error was identical in every case.
Selected-model metric differences were small: multinomial log-loss differed
by at most `4.46e-6`, binomial log-loss by `3.68e-6`, and Poisson deviance by
`5.18e-5`. The largest selected-model difference was Gaussian Wide MSE
(`+0.00487`, with R-squared `-0.000894`).

## Retained performance changes

Profiling showed that the important gaps were in native kernels, not R
wrapping. The retained changes therefore target solver work directly:

- scalar GLMs cache fixed weighted column norms and fuse accepted coordinate
  state updates;
- fast Poisson uses adaptive deferred predictor rebuilding plus packetized
  residual and curvature reductions;
- multinomial uses sequential active-set screening, cached smooth state, and
  reuses the inner Proximal-Newton direction `X * delta_beta + delta_b` during
  line search instead of recomputing `X * beta_candidate`;
- LLA is adaptive for non-Gaussian MCP/SCAD families, with a default maximum
  of three stages. Gaussian MCP/SCAD uses direct coordinate updates.

The final multinomial line-search change alone reduced the formal Wide median
from `0.1580` to `0.1325` seconds (about 16%) and Tall from `0.1600` to `0.1550`
seconds (about 3%). Its independent objective difference was at most
`4.44e-16`; work counters and selected models were unchanged. At `1e-7`, old
and new native outputs were byte-identical. Peak RSS changed by less than 1.2%
in the isolated A/B benchmark.

A global Eigen-to-BLAS switch was tested but not retained: it helped selected
shapes, but introduced R-specific linkage, Fortran ABI, wheel-packaging, and
threading risks. A future portable BLAS dispatch should be opt-in and
shape-gated.

## Remaining work

1. Stabilize Wide square-root lasso: its three paths completed 44/45, 38/45,
   and 45/45 points, and runtime remains highly variable.
2. Re-run the full comparison with the current automatic Gaussian policy.
3. Close the remaining small Poisson-Wide and multinomial-Tall gaps with a
   portable, fast-mode-only GEMV/GEMM dispatch rather than a global BLAS macro.
4. Continue reducing full-gradient/KKT refreshes, while retaining exact final
   certification and strict-mode behavior.

## Reproduction and artifacts

Use `profiling/fast_mode_glmnet_benchmark.R` for a new fast-mode explicit-path
run with the current Gaussian automatic backend and corrected call-stripped
object measurement. The configuration label is
`fast_explicit_path_auto`; it does not mean that every public argument keeps
its default.
The repository aggregate CSV files in
`profiling/fast_mode_glmnet_results/` are the historical snapshot reported
above; they retain the invalid raw `object_mb` field for provenance and
should not be used for memory claims. The full historical RDS files and A/B
artifacts were stored under
`/private/tmp/picasso-wide-round3-20260717/` and are not part of the
repository, so the exact old source state is not durably reproducible.

Environment: R 4.5.2, arm64 macOS, Accelerate BLAS, Apple Clang 21, one
numerical thread.
