# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**picasso** is a sparse learning library for GLMs with L1 (lasso), SCAD, and MCP penalties via pathwise coordinate optimization with warm starts, active-set updates, and screening rules. The core solvers are in C++ (using Eigen) and are shared by both an R package (CRAN) and a Python package (pycasso).

Reference paper: https://arxiv.org/abs/2006.15261

## Repository Structure

```
picasso-master/
├── include/picasso/       # Shared C++ headers (objective.hpp, c_api.hpp, solvers, etc.)
├── src/                   # Shared C++ sources
│   ├── objective/         #   gaussian_naive_update, gaussian_cov_update, glm, sqrtmse
│   ├── solver/            #   actgd, actnewton, solver_params
│   └── c_api/             #   c_api.cpp (5 exported C functions)
├── amalgamation/          # Unity build: picasso-all0.cpp #includes all src/*.cpp
├── R-package/             # R CRAN package (picasso v1.5)
│   ├── R/                 #   R source files
│   ├── src/               #   picasso_R.cpp (R bridge), picasso-all0.cpp (unity build)
│   ├── man/               #   Rd documentation
│   └── vignettes/         #   Package vignette
├── python-package/        # Python package (pycasso)
│   └── pycasso/           #   core.py (ctypes FFI), libpath.py, lib/libpicasso.so
├── CMakeLists.txt         # Builds libpicasso.so for Python (GLOB_RECURSE on src/)
└── lib/                   # Built shared library output
```

## Build Commands

### R Package

```bash
# Build and check
R CMD build R-package
R CMD check --as-cran picasso_1.5.tar.gz

# Install locally for testing
R CMD INSTALL R-package --library=.Rlib
# In R: library(picasso, lib.loc=".Rlib")
```

### Python Package (libpicasso.so)

```bash
# With CMake:
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make

# Without CMake (unity build):
g++ -shared -fPIC -O2 -std=c++11 \
    -I include -I R-package/src/include/eigen3 \
    -o lib/libpicasso.so amalgamation/picasso-all0.cpp

# Copy to Python package
cp lib/libpicasso.so python-package/pycasso/lib/
```

## Architecture

### C++ Core (shared by R and Python)

- **`include/picasso/c_api.hpp`** — 5 exported C functions: `SolveLinearRegressionNaiveUpdate`, `SolveLinearRegressionCovUpdate`, `SolveLogisticRegression`, `SolvePoissonRegression`, `SolveSqrtLinearRegression`. Each takes `dfmax` (early stopping), `num_fit` (output), `usePython` (row-major flag).
- **`include/picasso/objective.hpp`** — `ObjFunction` base → `GaussianNaiveUpdateObjective`, `GLMObjective` → `LogisticObjective`/`PoissonObjective`, `SqrtMSEObjective`. Also `RegFunction` hierarchy (L1/SCAD/MCP thresholding).
- **Solvers**: `ActGDSolver` (gaussian), `ActNewtonSolver` (logit/poisson/sqrtlasso).
- **Penalty flags**: L1=1, MCP=2, SCAD=3.

### R Layer (`R-package/`)

- `picasso.R` dispatches to family-specific functions (gaussian/binomial/poisson/sqrtlasso).
- `*_solver.R` files call C++ via `.Call()` through `picasso_R.cpp`.
- `picasso_utils.R` handles standardization, lambda path, rescaling, S3 methods.
- Eigen3 headers bundled in `R-package/src/include/eigen3/`.

### Python Layer (`python-package/`)

- `pycasso/core.py` — `Solver` class loads `libpicasso.so` via `ctypes` and calls C API directly.
- Supports all families, penalties, intercept, and `dfmax` early stopping.

### Data Flow

1. **R path**: `picasso()` → family R function → `.Call()` → `picasso_R.cpp` → C API → solver
2. **Python path**: `Solver.__init__()` → `ctypes` → C API → solver

## Key Design Notes

- The R package unity build (`R-package/src/picasso-all0.cpp`) uses relative paths `objective/xxx.cpp` etc., while `amalgamation/picasso-all0.cpp` uses `../src/objective/xxx.cpp`.
- When modifying C++ code, edit files under `src/` and `include/`, then sync to `R-package/src/` for the R build.
- The R package was archived on CRAN (2025-02-09) and is being resubmitted. See `R-package/cran-comments.md`.
