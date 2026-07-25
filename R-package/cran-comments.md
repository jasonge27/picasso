## Resubmission

This is a resubmission. The 2.0.0 incoming check reported install-time
WARNINGs on R-devel (C++20 default, GCC 14/16) originating entirely in the
bundled Eigen 3.3.5 headers
(`-Wdeprecated-enum-enum-conversion`), plus a spelling NOTE.

Changes in 2.0.1:

- Updated the bundled Eigen headers from 3.3.5 to 3.4.0. Eigen 3.4.0 adds the
  explicit `int()` casts that eliminate the deprecated enum-enum arithmetic
  under C++20, so the install warnings no longer occur. This is the same
  root-cause fix adopted by the `RcppEigen` package. No `picasso` source, API,
  or numerical behavior changed; the full test suite is unchanged and passes.

Regarding the spelling NOTE: "majorization" (DESCRIPTION Description field) is
a standard optimization term (as in majorization-minimization / quadratic
majorization) and is spelled correctly. It is not a typo.

## First 2.0.0 submission summary (unchanged features)

This release updates `picasso` from CRAN version 1.5 (published 2026-03-12)
with multinomial L1/MCP/SCAD paths, adaptive LLA for nonconvex fits, an
automatic Gaussian naive/covariance backend, link-scale offsets, and
cross-validation / assessment / confusion-matrix helpers, all documented in
the manuals, vignette, and release notes.

## Validation

The isolated source tarball was installed and tested before this submission.
The `testthat` suite contains 97 `test_that()` cases and 1,481 expectations
covering all families, interfaces, solver statuses, offsets, adaptive LLA,
Gaussian backend selection, path-loss reuse, and numerical stability.

Local `R CMD check --as-cran` (R 4.5.2, macOS, Apple Clang) reports 0 ERRORs
and 0 NOTEs. The only WARNING is emitted while compiling the installed R
system header `R_ext/Boolean.h`, whose `-Wfixed-enum-extension` pragma is
unknown to Apple Clang; it is not produced by package or Eigen sources and
does not occur on the CRAN GCC/MinGW machines. All examples, tests, Rd checks,
and vignette rebuilds pass.
