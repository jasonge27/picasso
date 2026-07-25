## Submission

This is a metadata-only update of `picasso`, from 2.0.1 (published
2026-07-25) to 2.0.2. No R code, C++ code, or documented behavior changed.

The bundled Eigen library's authors were previously credited with the `ctb`
role. In R's role vocabulary `ctb` denotes a contributor to *this* package,
which misstates the relationship: those authors wrote the independent Eigen
library that this package bundles, and have never contributed to `picasso`
itself. They are now credited as `cph` with comments recording that they
authored and hold copyright in the bundled Eigen library rather than in
`picasso`, and a third `cph` entry covers the remaining Eigen authors and
refers to `inst/COPYRIGHTS`. The `picasso` authors remain the package's `aut`
entries, and `inst/COPYRIGHTS` now states explicitly that the `picasso`
authors claim no copyright over the Eigen sources and that copyright in those
files remains with the Eigen authors. Licensing pointers are unchanged.

## Validation

Local `R CMD check --as-cran` (R 4.5.2, macOS, Apple Clang) reports 0 ERRORs
and 0 NOTEs. The only WARNING is emitted while compiling the installed R
system header `R_ext/Boolean.h`, whose `-Wfixed-enum-extension` pragma is
unknown to Apple Clang; it is not produced by package or Eigen sources and
does not occur on the CRAN GCC/MinGW machines, as the clean 2.0.1 results on
those flavors confirmed.

The `testthat` suite (97 `test_that()` cases, 1,481 expectations) passes, as do
all 15 native C++ tests and the C ABI export gate.
