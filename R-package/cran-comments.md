## Submission

This submission updates `picasso` from CRAN version 1.5, published on
2026-03-12, to version 2.0.0.

## Main changes since 1.5

- Added multinomial L1/MCP/SCAD paths with an active-set Proximal Newton/IRLS
  solver and R methods for printing, plotting, coefficient extraction,
  prediction, assessment, confusion matrices, and cross-validation.
- Added adaptive LLA stage budgets and stationarity diagnostics for nonconvex
  binomial, Poisson, square-root-lasso, and multinomial fits.
- Added opt-in, benchmark-calibrated `fast.mode`; the default remains the
  previous high-precision behavior.
- Added binomial and Poisson offsets consistently across fitting, prediction,
  assessment, and cross-validation.
- Added `dfmax`, stratified categorical cross-validation, lambda-value
  interpolation, native termination/failure diagnostics, and robust input and
  standardization checks.
- Added a calibrated automatic Gaussian naive/covariance backend with an
  8 MiB numeric-cache guard; explicit backend selection remains available.
- Reduced solver and wrapper overhead through cached curvature quantities,
  compact active sets, vectorized multinomial kernels, native path losses, and
  bounded prediction/rescaling temporaries.
- Updated all R help pages, examples, package metadata, release notes, and the
  installed vignette to describe the current interfaces and defaults.

The release also retains the earlier CRAN-oriented fixes: explicit values for
exported S3 methods, quiet non-verbose fitting, corrected Gaussian and GLM
intercepts, `.Call()` bridges, bundled Eigen copyright attribution, and removal
of source-level warning-suppression macros.

## Validation

The final isolated source tarball was installed and tested before this
submission. The complete `testthat` suite contains 97 `test_that()` cases and
1,481 expectations covering all families, interfaces, solver statuses,
offsets, adaptive LLA, Gaussian backend selection, path loss reuse, and
numerical stability.

Local `R CMD check --as-cran` with R 4.5.2 on macOS reports:

- 0 ERROR
- 1 WARNING
- 0 NOTEs

All functional checks, examples, tests, Rd checks, and vignette rebuilds pass.
The remaining warning comes from R 4.5.2's installed
`R_ext/Boolean.h`, whose `-Wfixed-enum-extension` diagnostic is unknown to
Apple Clang 21. It is emitted while compiling the R system header, not by
package source. CRAN incoming feasibility and URL checks completed normally.
