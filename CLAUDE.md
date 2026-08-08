# CLAUDE.md

## Project overview

PICASSO is a C++11 sparse-learning library shared by an R package (development
version 2.0.0) and a Python package (`pycasso`, development version 2.0.0).
Public families are Gaussian, binomial, Poisson, square-root-lasso, and
multinomial; every family supports lasso, MCP, and SCAD. See the
[JMLR paper](https://www.jmlr.org/papers/v20/17-722.html) for the original
framework and [README.md](README.md) for current interfaces.

As of 2026-07-19, CRAN carries picasso 1.5, published 2026-03-12. Do not
describe it as archived. Published R/Python releases may lag this development
tree.

High precision (`prec = 1e-7`) is the default. `fast.mode` / `fast_mode`
selects calibrated achieved-accuracy presets (`4e-4` Poisson; `1e-4` binomial,
square-root-lasso, and multinomial; Gaussian stays `1e-7`) — never describe
these as glmnet's literal `thresh` value.

## Repository structure

```text
include/picasso/       public and shared C++ headers
src/objective/         Gaussian, GLM, square-root, and multinomial objectives
src/solver/            scalar and multinomial active-set solvers
src/c_api/             C ABI used by R and Python
src/internal/          private native headers (mirrored like the rest)
tests/                 native C++ tests plus the verify_c_api_exports.py ABI gate
R-package/             R API, Rd/vignette/testthat, and mirrored native sources
python-package/        Python ctypes API, Sphinx sources, and integration tests
profiling/             benchmark programs, raw aggregates, and reports
amalgamation/          standalone unity-build translation unit
cmake/                 source inventory, mirror checks, Eigen warning-pair check
```

Treat `R-package/src/include/eigen3/` as vendored code; it is the only
bundled Eigen tree, and both the CMake and Makefile builds resolve it when no
system Eigen is configured.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure        # native tests + mirror/Eigen gates
ctest --test-dir build -R multinomial_c_api_test  # single native test
cmake --build build --target check_mirrors
cmake --build build --target stage_picasso        # stages build/stage/libpicasso.*
python3 tests/verify_c_api_exports.py build/libpicasso.dylib
```

R (testthat requires the package installed first):

```bash
R CMD INSTALL --preclean R-package
Rscript -e 'library(picasso); testthat::test_dir("R-package/tests/testthat", reporter = "summary", stop_on_failure = TRUE)'
Rscript -e 'library(picasso); testthat::test_file("R-package/tests/testthat/test-multinomial.R")'
R CMD build R-package
R CMD check --as-cran picasso_2.0.0.tar.gz
```

Python (point the wrapper at a staged native build):

```bash
PICASSO_NATIVE_LIBRARY="$PWD/build/stage/libpicasso.dylib" \
  python -m pip install ./python-package
python python-package/test_pycasso.py             # monolithic; no per-test selector
python python-package/test_reproducibility.py     # library selection + benchmark isolation
```

Replace the staged suffix with `.so` on Linux or `picasso.dll` on Windows.
`PICASSO_NATIVE_LIBRARY` also overrides library discovery at import time;
without it exactly one bundled library may be present (ambiguity is refused).
Add `-DPICASSO_ENABLE_SANITIZERS=ON` for an ASan/UBSan CMake build. The root
Makefile is retained for compatibility and packaging (`make dylib`,
`make pippack`, `make Rpack`); use CMake for native development.
CI (`.github/workflows/cpp-tests.yml`) runs GCC and Clang Linux builds with
the mirror, ctest, and ABI gates, a macOS build with ctest and both Python
suites, a Clang sanitizer build, a Windows build with ctest and the DLL ABI
gate, Python-packaging jobs on 3.9 and 3.12, and an R job that runs testthat
plus `R CMD check --as-cran` with vignettes.

## Solver architecture

### Scalar-response stack

- `ActGDSolver` performs active-set coordinate descent for Gaussian
  models. `type.gaussian="auto"` / `type_gaussian="auto"` resolves to
  residual-based naive or lazy-covariance updates. Gaussian MCP/SCAD is direct
  nonconvex coordinate descent and does not use LLA.
- `ActNewtonSolver` handles binomial, Poisson, and square-root-lasso
  weighted-L1 subproblems. Binomial and Poisson use Proximal Newton/IRLS;
  square-root-lasso uses a global quadratic majorizer with active-set
  coordinate updates. It shares the driver but is not a raw Hessian/Newton
  method.
- MCP/SCAD non-Gaussian fits use adaptive LLA. The minimum and default maximum
  are three total stages (one lasso master plus two weighted-L1 updates).
  A larger public stage budget permits continuation to target stationarity.
- Binomial and Poisson objectives carry link-scale offsets (`double *offset`
  in the C API; may be nullptr).

Scalar C APIs are cumulative per family: legacy void entry points remain
ABI-compatible; V2 adds per-lambda residual mean squares (Gaussian) or
status plus LLA diagnostics (binomial, Poisson, square-root-lasso); V3 adds
the final smooth objective, and for Gaussian also explicit path-termination
status. R calls V3 for every scalar family; Python prefers V3 and falls back
to V2 or legacy symbols for older binaries.

### Multinomial stack

- `MultinomialObjective` implements stable softmax negative
  log-likelihood, gradient, and Hessian-vector products.
- `MultinomialActNewtonSolver` applies strong screening, solves the
  restricted class-coupled IRLS quadratic to active-set convergence, performs
  full KKT scans, expands the set, and repeats with Armijo line search.
- `MultinomialLlaSolver` wraps weighted-L1 subproblems for MCP/SCAD.
  The lasso master, rather than the final nonconvex point, warm-starts the next
  lambda.
- C APIs are cumulative: legacy; V2 diagnostics/status; V3 user stage budget;
  V4 generated-path saturation control; V5 native smooth NLL. R calls V5.
  Python prefers V5, then V4/V3/V2/legacy.

Output layout is
`beta[lambda * K * d + class * d + feature]` and
`intercept[lambda * K + class]`. Lambda commits are transactional:
`num_fit` is the usable prefix and failed suffix storage stays untouched.

Generated multinomial paths may stop after deviance saturation. Explicit
paths disable only that rule; `dfmax` or a hard failure can still truncate
them. Scalar solvers have their own normal path stopping. Never document
`dfmax` as a hard upper bound: the crossing model is retained.

## Language layers

The R entry point `picasso()` dispatches to family wrappers.
`picasso_R.cpp` bridges to the versioned C API; its `R_CallMethodDef`
registrations carry fixed argument counts that must match the R-side
`.Call` sites. Non-Gaussian fits surface status, failures, objectives,
KKT/stationarity, timing, and family-specific iteration counts.
`assess.picasso()`, `confusion.picasso()`, and `cv.picasso()` provide
evaluation helpers.

Python `pycasso.Solver` mirrors those features. Scalar coefficient paths
have shape `(L, d)`; multinomial paths have shape `(L, K, d)`.
Python path indices are zero-based, while R indices are one-based. A
two-element non-NumPy Python sequence means `(count, ratio)`; NumPy
arrays always mean explicit paths.

## Cross-language synchronization

`cmake/CheckMirrors.cmake` (also run as the `source_mirrors` ctest and in CI)
requires shared files under `src/`, `src/internal/`, and `include/picasso/`
to be byte-identical to their `R-package/src/` mirrors, validates
`cmake/PicassoSources.cmake` against the on-disk source tree, and requires
both unity builds (`amalgamation/picasso-all0.cpp` and
`R-package/src/picasso-all0.cpp`) to include the same sources in the same
order. There are no mirror exceptions:
`R-package/src/include/picasso/objective.hpp` is byte-identical to the root
header (its formerly R-only Eigen diagnostic pragmas now live in both copies).

When changing shared C++, update both copies; register new translation units
in `cmake/PicassoSources.cmake` and both unity builds; and keep C
declarations, implementations, the R bridge, Python ctypes signatures, and
the `EXPECTED_SYMBOLS` list in `tests/verify_c_api_exports.py` synchronized —
that script gates both header declarations and dynamic exports.

## Conventions

- Match nearby code; no repository-wide formatter or linter is configured.
- C++: C++11, two-space indentation, K&R braces, `PascalCase` types,
  `snake_case` functions, and `m_` members.
- Python: four spaces, PEP 8 naming, exact public docstrings, deterministic
  assertions in `python-package/test_pycasso.py`.
- R: generally two spaces and dotted S3 names; avoid unrelated formatting of
  legacy files. Add behavior tests under `R-package/tests/testthat/`.
- No coverage threshold is declared. Add deterministic Python-facing
  assertions to `python-package/test_pycasso.py`, R-facing cases under
  `R-package/tests/testthat/`, and focused native cases under `tests/`. Core
  or C API changes should pass CTest, both interface suites, and
  `R CMD check`.
- Commits use concise imperative subjects (`Add v1.6 features: ...`). Pull
  requests should summarize affected layers, list validation commands, link
  relevant issues, and call out API or documentation changes. Include
  screenshots only for rendered documentation or other visual changes.
- Performance reports are snapshots tied to their hardware, precision, shape,
  and source state. Do not turn them into universal speed claims. The
  comprehensive fast-mode report predates the current Gaussian automatic
  backend.
