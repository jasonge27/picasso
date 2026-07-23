# picasso 1.6

## New models and solver controls

- Added multinomial logistic regression for L1, MCP, and SCAD penalties in the
  R and Python interfaces. The solver uses class-coupled active-set Proximal
  Newton/IRLS updates, pathwise warm starts, strong screening, full KKT checks,
  and deviance-tail early stopping for automatically generated paths.
- MCP/SCAD binomial, Poisson, square-root-lasso, and multinomial fits now use
  adaptive local linear approximation (LLA). `lla.max.stages = 3L` is the
  default maximum, including the initial L1 master; larger budgets are
  available when stricter stationarity is required.
- Added opt-in `fast.mode`. The default remains high precision. Fast mode uses
  benchmark-calibrated achieved-accuracy tolerances of `4e-4` for Poisson and
  `1e-4` for binomial, square-root-lasso, and multinomial; Gaussian retains
  `1e-7`.
- Gaussian fits now default to `type.gaussian = "auto"`. A calibrated,
  memory-bounded policy selects between residual-based naive updates and lazy
  covariance updates using the design shape and lambda-path reuse. Explicit
  `"naive"` and `"covariance"` requests retain their previous behavior.
- Added `dfmax` path-size stopping consistently across families.

## Interfaces and diagnostics

- Added binomial and Poisson offsets throughout fitting, prediction,
  assessment, and cross-validation. New-data offsets are applied on the link
  scale and are required whenever prediction or assessment evaluates the
  linear predictor; nonzero-support extraction does not need them.
- Added `cv.picasso()`, `assess.picasso()`, and `confusion.picasso()` support
  for all applicable families, including stratified categorical folds and
  stable class maps for factor and multinomial responses.
- Prediction methods now support response, link, class, and nonzero outputs as
  applicable, plus interpolation by lambda value through `s`.
- Versioned native solvers now report termination status and failure location.
  Gaussian iteration-limit failures retain only the previously converged path
  prefix. Non-Gaussian fits additionally report per-lambda runtime, objective,
  KKT residual, stationarity, and LLA stage counts; usable stage-budget
  termination remains distinct from a hard failure.
- Gaussian fit objects record both requested and resolved solver modes.
  Multinomial fits record the requested path length and whether a saturated
  generated-path tail was omitted.

## Correctness, performance, and portability

- Categorical class prediction now compares finite link-scale scores directly,
  matching glmnet while avoiding probability-rounding ties and unnecessary
  sigmoid/softmax temporaries. Confusion tables use predicted classes in rows,
  observed classes in columns, and retain every fitted class level.
- Fixed binomial response/class prediction for requests containing multiple
  lambda values.
- Scalar coefficient and prediction defaults now select at most the first
  three available path points or coefficients, so short paths and low-
  dimensional models no longer fail with out-of-range default indices.
- Scalar prediction row selectors now use `NULL` for all rows. Explicit
  `Y.pred.idx` or `p.pred.idx` values are always applied instead of treating
  `1:5` as a hidden default sentinel.
- Invalid MCP/SCAD concavity parameters now fail instead of being silently
  replaced by 3; every family validates the documented LLA stage budget.
  Square-root-lasso reports its algorithm as active-set quadratic MM.
- Moved scalar and multinomial path-loss evaluation into the native solvers,
  avoiding repeated R-side matrix products after fitting.
- Cached fixed weighted column norms inside scalar active-set subproblems,
  compacted active working sets, vectorized multinomial curvature/KKT kernels,
  and reduced temporary allocations in prediction and path rescaling.
- Scalar-family assessment now keeps small paths on one matrix multiplication
  and evaluates larger paths in approximately 8 MiB link-predictor blocks,
  reducing peak memory without changing metric values.
- Strengthened validation for numeric matrix storage, finite inputs, decreasing
  lambda paths, dimensions, class maps, offsets, and no-intercept
  standardization. Constant-response and zero-gradient paths now have explicit
  finite behavior.
- Kept R-package and standalone native source mirrors under build-time parity
  checks and added scalar, multinomial, interface, and numerical-stability
  regression coverage.
- Removed the unused `MASS` attachment from the R package dependency surface;
  only `Matrix` is attached at load time.
