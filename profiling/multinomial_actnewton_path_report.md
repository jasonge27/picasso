# Multinomial ActNewton Path Comparison

## Scope and Backup

The pre-replacement repository is preserved at
`/private/tmp/picasso-multinomial-actnewton-pre-replacement-20260715-223659`.
Key solver files were verified byte-for-byte before implementation. Baseline
binaries were built from that snapshot, while candidate binaries were built
from the current tree with identical compiler flags.

The baseline warm-started independent Proximal Newton/IRLS solves and used
full inactive-coordinate scans inside every quadratic subproblem. The new
solver mirrors Logistic `ActNewtonSolver`: retain a path-level strong set,
solve each fixed-IRLS quadratic on that set, update probabilities, scan the
true inactive KKT conditions, and commit warm state only after convergence.

## L1 Path Results

Times are medians of six serial ABBA runs. All 24 path points converged and
passed independent full-KKT and objective checks.

| Case | Old time (s) | New time (s) | Change | Old/new coordinates | Old/new inner full scans |
|---|---:|---:|---:|---:|---:|
| sparse K=3 | 0.286 | 0.244 | 14.8% faster | 12,372 / 12,750 | 46 / 0 |
| wide K=4 | 0.318 | 0.221 | 30.5% faster | 14,724 / 17,884 | 46 / 0 |
| high K=12 | 0.291 | 0.217 | 25.2% faster | 22,296 / 25,032 | 46 / 0 |
| dense control | 0.320 | 0.311 | 2.8% faster | 462,008 / 457,560 | 73 / 0 |

Objective, probability, support, and KKT digests were identical in the three
sparse cases. Dense differences were numerical roundoff (at most
`2.3e-13`). Median peak RSS changed by +0.6% to +3.0% (64--220 KiB), so this
change provides speed rather than a material memory reduction.

## MCP/SCAD LLA Results

Non-dense entries are two-run serial medians; dense entries are paired serial
runs. Every path passed target-stationarity, majorization, descent, support,
and finiteness checks.

| Case | MCP old/new (s) | MCP change | SCAD old/new (s) | SCAD change |
|---|---:|---:|---:|---:|
| sparse K=3 | 0.819 / 0.621 | 24.2% faster | 0.628 / 0.487 | 22.4% faster |
| wide K=4 | 1.212 / 0.838 | 30.9% faster | 0.936 / 0.665 | 28.9% faster |
| high K=12 | 1.432 / 0.984 | 31.3% faster | 1.079 / 0.753 | 30.3% faster |
| dense control | 24.98 / 24.58 | 1.6% faster | 20.72 / 20.67 | 0.2% faster |

An intermediate candidate restored expensive full quadratic inactive scans.
It recovered dense iteration counts but slowed sparse MCP from 0.628 s to
0.829 s. That candidate was rolled back. The retained implementation instead
uses the already-computed true gradient for an `O(dK)` inactive scan after
each accepted PN/IRLS step. It preserves the sparse gain and restores dense
MCP outer iterations from 527 to the baseline value of 486.

## Validation

- Four regular C++ suites and their ASan/UBSan builds pass.
- The Python integration suite passes, including L1/MCP/SCAD paths.
- R installation succeeds and all 125 `testthat` assertions pass.
- In that recorded run, `R CMD check --as-cran` completed with no ERROR
  or NOTE; its only WARNING came from the R 4.5.2 system header under Apple
  Clang 21.
- All 19 root/R source mirror pairs are byte-identical.
