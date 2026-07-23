# Repository Guidelines

## Project Structure & Module Organization

PICASSO shares a C++11 core across two language packages. Implementations live in `src/` (`objective/`, `solver/`, and `c_api/`), with headers in `include/picasso/`. `R-package/` contains the R API, native mirrors, manuals, and tests. `python-package/pycasso/` contains the Python `ctypes` wrapper. Benchmarks live under `profiling/`.

## Build, Test, and Development Commands

- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build` builds the native library; `--target stage_picasso` stages one packaging artifact.
- `cmake --build build --target check_mirrors` verifies that shared native files match their R-package copies.
- `cd python-package && PICASSO_NATIVE_LIBRARY=../build/stage/libpicasso.so python -m pip install .` installs the Python wrapper from the staged Linux library. Use `libpicasso.dylib` on macOS or `picasso.dll` on Windows.
- `python python-package/test_pycasso.py` runs Python feature assertions.
- `ctest --test-dir build --output-on-failure` runs the native C++ unit and C API suite after a CMake build.
- `R CMD build R-package` builds the current R package tarball.
- `R CMD check --as-cran picasso_1.6.tar.gz` checks examples, documentation, and vignettes.

## Coding Style & Naming Conventions

Match nearby code; no repository-wide formatter or linter is configured. C++ uses two-space indentation, K&R braces, C++11, `PascalCase` types, `snake_case` functions, and `m_` private members. Python uses four spaces, PEP 8-style `snake_case`, `PascalCase` classes, and docstrings for public interfaces. R code generally uses two spaces and preserves dotted S3 names such as `print.cv.picasso`; legacy files vary, so avoid unrelated reformatting.

## Testing Guidelines

No coverage threshold is declared. Add deterministic Python-facing assertions to `test_pycasso.py`, R-facing cases under `R-package/tests/testthat/`, and focused native cases under `tests/`. Core or C API changes should pass CTest, both interface suites, and `R CMD check`.

## Cross-Language Changes

When changing shared C++, update the corresponding copies under `R-package/src/`; the mirror check requires every shared source and header — `objective.hpp` included — to be byte-identical to its R-package copy (the Eigen diagnostic pragmas live in both). Keep C API declarations, implementations, the R bridge, Python `ctypes` signatures, and the `EXPECTED_SYMBOLS` list in `tests/verify_c_api_exports.py` synchronized. Register new `.cpp` files in `cmake/PicassoSources.cmake` and both unity builds, then run `check_mirrors`. Treat the bundled Eigen tree under `R-package/src/include/eigen3/` as vendored code.

## Commit & Pull Request Guidelines

The available history is limited, but its subject uses a concise imperative form (`Add v1.6 features: ...`). Follow that style with focused commits, for example `Fix offset handling in cross-validation`. Pull requests should summarize affected layers, list validation commands, link relevant issues, and call out API or documentation changes. Include screenshots only for rendered documentation or other visual changes.
