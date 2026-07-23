# Incremental Performance Round 2

> Historical July 2026 optimization snapshot. Its `public` Gaussian result
> used the former naive default, before `type.gaussian="auto"`. Later
> reports supersede its cross-package runtime conclusions.

## Scope and Safety

All changes were made as isolated increments against the same fast-mode,
single-thread, 45-lambda benchmark. The complete pre-change source archive is
`/private/tmp/picasso-performance-round2-backup-20260717/source-before.tar.gz`
(SHA-256 `53f1d19d...8869`); every source step also has a direct before-file in
that directory. Full raw results and samples are under
`/private/tmp/picasso-performance-round2/`.

## Retained Changes

- Square-root Lasso now measures path saturation with `0.5 * L^2`, matching
  the reported deviance. Two Wide interpolation-boundary failures now end as
  clean, certified early stops with exactly the same retained models.
- Its invalid `sum(x^2*r^2)` curvature was replaced by the global MM
  curvature `||x||^2/(nL)`, removing one O(n) reduction per coordinate while
  guaranteeing that a weighted-L1 coordinate step does not increase its
  objective.
- Scalar Newton paths reuse the last certified master state. Binomial,
  Poisson, and square-root fits first test exact KKT conditions on the active
  set; a full gradient is computed only when convergence is possible. The
  full KKT scan remains the only final certificate.
- GLM gradients reuse the maintained residual, and rejected sub-`1e-8`
  coordinate candidates no longer desynchronize `beta`, `Xb`, and residuals.
- Multinomial paths reuse the full gradient already computed by the converged
  proximal-Newton solve. No public struct, class layout, C API, or ABI changed.

## Runtime Results

`PICASSO / glmnet` below is smaller-is-better; values below one mean PICASSO
is faster. Times are medians over 21 measurements (three seeds, seven blocks).

| Family | Shape | Before | After | PICASSO speedup | Final PICASSO / glmnet |
|---|---|---:|---:|---:|---:|
| Binomial | Tall | 0.0420 | 0.0388 | 1.08x | 0.76 |
| Binomial | Wide | 0.0345 | 0.0270 | 1.28x | 0.73 |
| Poisson | Tall | 0.0425 | 0.0388 | 1.10x | 1.25 |
| Poisson | Wide | 0.1900 | 0.1400 | 1.36x | 2.69 |
| Multinomial | Tall | 0.2060 | 0.1900 | 1.08x | 1.25 |
| Multinomial | Wide | 0.2620 | 0.2220 | 1.18x | 1.37 |
| Sqrt-Lasso | Tall | 0.0268 | 0.0236 | 1.14x | n/a |
| Sqrt-Lasso | Wide | 0.2980 | 0.2580 | 1.16x | n/a |

Gaussian was intentionally unchanged in this round. Its then-public Tall
comparison was 1.82; with both packages forced to naive updates, PICASSO was
1.20x faster. Neither value describes the current automatic backend.

## Validation and Remaining Hotspots

All 12 native tests, the complete R testthat suite, and Python end-to-end tests
passed in that recorded run. `R CMD check --as-cran` then completed with
only the known external R-header/Apple-Clang warning. Every affected final
path had independently recomputed
absolute KKT at or below `1e-4`; selected test metrics are unchanged up to
solver tolerance.

Sampling confirms Wide Poisson full-gradient GEMV fell from 36.2% to 9.7% of
samples. The deleted multinomial post-solve gradient previously consumed 9.2%
(Tall) and 18.8% (Wide). The next useful targets are multinomial Tall scalar
coordinate reductions and Wide GEMM packing; these require a separate round
because they change lower-level kernels rather than remove redundant work.
