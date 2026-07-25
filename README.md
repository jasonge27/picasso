# PICASSO

PICASSO is a C++ sparse-learning library with R and Python interfaces. It fits
regularization paths for Gaussian, binomial, Poisson, square-root-lasso, and
multinomial models using lasso, MCP, or SCAD penalties. The implementation
combines warm starts, screening, active sets, and family-specific coordinate,
majorization, or Newton updates.

## Supported solvers

| Family | Native optimization | MCP/SCAD strategy |
|---|---|---|
| Gaussian | Active-set coordinate descent with residual or lazy-covariance updates | Direct nonconvex coordinate updates |
| Binomial | Active-set Proximal Newton/IRLS | Adaptive LLA |
| Poisson | Active-set Proximal Newton/IRLS | Adaptive LLA |
| Square-root-lasso | Active-set quadratic-MM coordinate updates | Adaptive LLA |
| Multinomial | Class-coupled active-set Proximal Newton/IRLS | Adaptive LLA |

The multinomial solver first screens a strong working set, solves the
restricted IRLS quadratic to convergence on its active coordinates, performs a
full KKT scan, expands the set when necessary, and repeats. Lambda-path warm
starts carry the master solution forward.

## Current defaults

- High precision is the default: `fast.mode = FALSE` in R and
  `fast_mode=False` default to `prec=1e-7` and accept a custom
  positive tolerance. Fast mode uses
  calibrated stopping/KKT tolerances of `4e-4` for Poisson and `1e-4`
  for binomial, square-root-lasso, and multinomial; Gaussian remains at
  `1e-7`. These are achieved-accuracy presets, not glmnet's literal
  `thresh` value.
- Gaussian fits default to `auto`. It chooses lazy covariance updates only
  when path reuse and the design shape can amortize a bounded cache; wide,
  short-path, or more-than-1024-feature problems use residual updates.
- Non-Gaussian MCP/SCAD fits use a minimum and default maximum of three LLA
  stages: one lasso master and two weighted-lasso updates. A larger stage
  budget permits adaptive continuation to the requested stationarity.
- Generated paths request 100 lambdas down to a nominal
  `0.05 * lambda_max` by default. Normal saturation stopping, `dfmax`,
  or a solver failure can return a shorter committed prefix.

## R interface

The checkout contains R package version 2.0.0. A CRAN installation may be an
earlier published release; build this checkout to use the interfaces described
here:

```bash
R CMD build R-package
R CMD INSTALL picasso_2.0.0.tar.gz
```

```r
library(picasso)
set.seed(1)
X <- matrix(rnorm(200 * 40), 200, 40)
y <- 0.5 + X[, 1] - 0.7 * X[, 2] + rnorm(200)

fit <- picasso(X, y, family = "gaussian", nlambda = 30)
fit$type.gaussian                 # resolved "naive" or "covariance"
predict(fit, X[1:5, ], lambda.idx = fit$nlambda)
cv <- cv.picasso(X, y, family = "gaussian", nfolds = 5, nlambda = 30)
```

Use `family = "multinomial"` for three or more numeric, character, or
factor classes. Binomial and Poisson models accept link-scale `offset`
vectors. See `?picasso` and the installed *PICASSO 2.0.0 User Guide* for
complete return values, diagnostics, prediction, assessment, and CV behavior.

## Python interface

The checkout contains `pycasso` 2.0.0. The latest PyPI release can lag the
development source, so use a staged native build for the exact API documented
in this repository:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target stage_picasso
PICASSO_NATIVE_LIBRARY="$PWD/build/stage/libpicasso.dylib" \
  python -m pip install ./python-package
```

Use `libpicasso.so` on Linux or `picasso.dll` on Windows.

```python
import numpy as np
import pycasso

rng = np.random.default_rng(1)
X = rng.normal(size=(200, 40))
y = 0.5 + X[:, 0] - 0.7 * X[:, 1] + rng.normal(size=200)

model = pycasso.Solver(
    X, y, family="gaussian", lambdas=(30, 0.05),
    type_gaussian="auto")
model.train()
prediction = model.predict(X[:5], lambdidx=model.nlambda - 1)
cv = model.cross_validate(nfolds=5)
# Opt in only when folds are expensive; cap BLAS threads separately.
cv_parallel = model.cross_validate(nfolds=5, n_jobs=4)
```

Python cross-validation is serial by default (`n_jobs=1`). Larger values use
at most one thread per fold and preserve fold-order aggregation. Concurrent
fold solvers consume additional memory, and BLAS should be limited to one
thread to avoid oversubscription.

Python lambda indices are zero-based. A two-element non-NumPy sequence means
`(count, ratio)`; NumPy arrays always represent explicit paths, including
arrays of length one or two. See
[the Python guide](python-package/README.rst) and
[current tutorial](tutorials/tutorial.py).

## R/Python option mapping

| Purpose | R | Python |
|---|---|---|
| Penalty | `method` | `penalty` |
| Gaussian backend | `type.gaussian` | `type_gaussian` |
| Fast precision preset | `fast.mode` | `fast_mode` |
| LLA stage budget | `lla.max.stages` | `lla_max_stages` |
| Iteration budget | `max.ite` | `max_ite` |
| Lambda-value prediction | `s` | `lam` |
| New-data offset | `newoffset` | `newoffset` |

## Build and validation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
cmake --build build --target check_mirrors

python python-package/test_pycasso.py
R CMD check --as-cran picasso_2.0.0.tar.gz
```

The native and R-package C++ trees are intentionally mirrored; the CMake
mirror check prevents them from drifting. Contributor workflow and conventions
are in [AGENTS.md](AGENTS.md).

## Performance evidence

Benchmarks and raw aggregates live in [`profiling/`](profiling/). The
[fast-mode comparison](profiling/fast_mode_glmnet_comparison_report.md) is a
single-threaded dense L1 snapshot on arm64 macOS and should not be generalized
to other hardware, penalties, sparse inputs, or shapes. Its Gaussian
`public_default` rows predate the current automatic backend; the matched
naive/covariance rows remain historical reference points. PICASSO does not
claim to beat glmnet for every family and matrix shape.

## Repository layout

- [`src/objective/`](src/objective/) contains Gaussian, GLM,
  square-root, and multinomial objectives.
- [`src/solver/`](src/solver/) contains scalar and multinomial
  active-set solvers; [`src/c_api/`](src/c_api/) exposes the native ABI.
- [`include/picasso/`](include/picasso/) contains public C++ headers.
- [`tests/`](tests/) contains native objective, solver, C API, and
  stability tests.
- [`R-package/`](R-package/) contains the R interface, Rd help,
  vignette, testthat suite, and native-source mirrors.
- [`python-package/`](python-package/) contains the Python
  `ctypes` wrapper and Sphinx documentation.
- [`tutorials/PICASSO.pdf`](tutorials/PICASSO.pdf) is the historical
  paper-era document, not the current API guide.

## Citation

```bibtex
@article{ge2019picasso,
  title   = {Picasso: A Sparse Learning Library for High Dimensional Data Analysis in R and Python},
  author  = {Ge, Jason and Li, Xingguo and Jiang, Haoming and Liu, Han and Zhang, Tong and Wang, Mengdi and Zhao, Tuo},
  journal = {Journal of Machine Learning Research},
  volume  = {20},
  number  = {44},
  pages   = {1--5},
  year    = {2019}
}
```

The [original JMLR paper](https://www.jmlr.org/papers/v20/17-722.html)
describes the initial pathwise framework. Newer interfaces and algorithms are
documented in this repository.

PICASSO is distributed under GPL-3.0; see [LICENSE](LICENSE).
