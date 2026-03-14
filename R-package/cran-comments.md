## Resubmission

This is a resubmission of `picasso` (version 1.5), which was archived on CRAN on 2025-02-09.

### Changes addressing previously reported CRAN issues

1. Added explicit `\value` sections for exported S3 methods and documented output structure/meaning, including:
   - `coef.*`, `plot.*`, `print.*` Rd files for gaussian/logit/poisson/sqrtlasso.
   - updated `predict.*` return-value descriptions.
2. Removed leftover Rd template/placeholder text and cleaned help-page wording.
3. Adjusted R-side output behavior:
   - core methods now return objects (e.g., coefficient/prediction matrices) for downstream use;
   - console writes in fitting code are now behind `verbose` or converted to `warning()/stop()` where appropriate;
   - `print.*` methods remain user-facing side-effect methods and return invisibly.
4. Updated authorship/copyright metadata:
   - expanded `Authors@R` with third-party contributor/copyright roles for bundled Eigen headers;
   - added `inst/COPYRIGHTS` and referenced it from `DESCRIPTION`.
5. Fixed invalid reference URL in package documentation and updated it to:
   `https://arxiv.org/abs/2006.15261`.
6. Removed source-level warning-suppression macro definitions from package headers while keeping build portability and clean checks.

### Bug fixes

7. Fixed Poisson regression with `intercept = TRUE` producing all-zero coefficients.
   The shared GLM base class constructor initialized the intercept using the logistic
   link function (`log(y/(1-y))`), which produces `NaN` for Poisson data where
   `mean(y) > 1`. Moved intercept initialization to each subclass with the correct
   link: logit for binomial, log for Poisson.
8. Fixed Gaussian regression returning incorrect intercept values when
   `standardize = TRUE`. The response mean was subtracted before fitting but
   not added back to the intercept during solution rescaling.
9. Fixed GLM solvers (logistic, Poisson) starting from `intercept = 0` instead of
   the analytic null-model intercept. The constructors computed the correct initial
   intercept for gradient estimation but then reset it to zero. Removing the reset
   improves convergence at the start of the regularization path.

### Performance improvements

10. Migrated R-to-C++ interface from `.C()` to `.Call()`, eliminating redundant
    data copies on each solver invocation.
11. Replaced row-scatter `eval()` in Gaussian naive-update objective with a single
    cache-friendly matrix–vector multiply (Eigen), yielding 1.7–3× speedup for
    the Gaussian family.
12. Used Eigen vectorized array assignment and `std::copy` instead of element-wise
    loops for warm-start copies in both solvers.
13. Added `vector::reserve()` for solution-path storage to avoid reallocation.
14. Replaced R-side `do.call(cbind, ...)` and `vapply` with direct `matrix()`
    construction and `colSums()` for building coefficient output.
15. Implemented incremental `sum_r` / `sum_r2` updates in the sqrt-lasso
    coordinate descent, replacing O(n) recomputation with O(1) algebra.
16. Removed redundant `X %*% beta` prediction matrix computation from logistic
    fitting (was unused by downstream code).

## R CMD check results

`R CMD check --as-cran` on the local submission tarball reports:

- 0 ERROR
- 0 WARNING
- 2 NOTEs

The remaining NOTEs are:

- CRAN incoming feasibility:
  - "Package was archived on CRAN" and the related CRAN repository db override.
- Installed package size (5.2 MB) due to bundled Eigen headers.

These NOTEs are expected for this resubmission.
