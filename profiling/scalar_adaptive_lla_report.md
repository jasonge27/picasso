# Scalar Adaptive-LLA Benchmark

## Setup

The benchmark compares the backed-up fixed-three-stage library with the
current scalar `ActNewtonSolver` at maximum LLA stage budgets 3 and 25. It uses
one deterministic standardized design (`seed=20260716`, `n=600`, `d=120`), 24
identical lambdas from `lambda_max` to `0.05 * lambda_max`, `gamma=3.5`, and
`precision=1e-7`. Each number is the median of five fresh-process native calls
after one warm-up, with numerical-library threads fixed to one.

- Old SHA-256: `c3b76d4f7eb0d80764f89f3449932592c11db326bee4277124610302f0636d99`
- New SHA-256: `7e75c2833226fd0d96130b439db3bd557dc8e7d6838bd8878d2d61bb20d04714`
- Platform: Darwin 25.5.0 arm64, Python 3.14.3, NumPy 2.4.2

Reproduce with:

```bash
python profiling/scalar_adaptive_lla_benchmark.py \
  --old-library /private/tmp/picasso-pre-adaptive-lla-all-20260716/lib/libpicasso.so \
  --new-library lib/libpicasso.so --repeats 5 \
  --output /tmp/scalar-adaptive-lla.json
```

## Runtime and Memory

| Family / penalty | Old fixed-3 | Cap=3 before optimization | Cap=3 optimized | vs old | Cap=25 optimized | vs cap=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binomial MCP | 34.11 ms | 83.18 ms | 43.84 ms | 1.29x | 115.42 ms | 2.63x |
| Binomial SCAD | 33.83 ms | 81.61 ms | 42.21 ms | 1.25x | 102.05 ms | 2.42x |
| Poisson MCP | 31.76 ms | 91.25 ms | 50.90 ms | 1.60x | 231.41 ms | 4.55x |
| Poisson SCAD | 30.12 ms | 86.45 ms | 49.38 ms | 1.64x | 241.67 ms | 4.89x |
| Sqrt-Lasso MCP | 9.60 ms | 43.27 ms | 27.94 ms | 2.91x | 63.66 ms | 2.28x |
| Sqrt-Lasso SCAD | 8.85 ms | 39.45 ms | 25.26 ms | 2.85x | 53.91 ms | 2.13x |

The optimized solver is 1.55--1.93x faster than its initial strict-KKT
implementation at cap=3 and 1.56--1.79x faster at cap=25. The changes move
full finite-model validation from every coordinate to sweep boundaries, cache
fixed IRLS/Sqrt-Lasso column summaries, reuse gradient workspaces and unchanged
LLA anchor state, and use a standard inexact-Newton forcing threshold while
retaining the final full-KKT `precision` gate.

Median process peak RSS stayed within 40.16--40.53 MiB; mode-to-mode ratios
were within 0.5%. The high-water mark was already reached before each native
call, so measured RSS increments were zero. For this 24-lambda path, the new
core diagnostic arrays add approximately 780 bytes plus 672 bytes of V2 caller
outputs; this is negligible relative to the model and design matrix.

## Independent Accuracy Check

Objectives and stationarity below are recomputed from returned coefficients,
not copied from native diagnostics. “Objective” is the true MCP/SCAD objective
at the final lambda; “stationarity” is the maximum target-penalty residual over
the complete fitted path. Every mode fit all 24 lambdas and was byte-stable
across repeats. Native versus independent maximum absolute discrepancies were
`3.94e-8` for objective and `4.58e-7` for stationarity, so certification counts
in this table deliberately use the independent calculation.

| Family / penalty | Final objective: old / cap=3 / cap=25 | Max stationarity: old / cap=3 / cap=25 | Cap=25 certified |
| --- | ---: | ---: | ---: |
| Binomial MCP | .507603 / .507601 / .507518 | 3.744e-2 / 3.744e-2 / 6.939e-3 | 23/24, limit |
| Binomial SCAD | .508759 / .508757 / .508609 | 2.511e-2 / 2.510e-2 / 9.547e-8 | 24/24, completed |
| Poisson MCP | .823510 / .823508 / .823466 | 6.056e-3 / 6.045e-3 / 3.222e-6 | 2/24, limit |
| Poisson SCAD | .826137 / .826135 / .826070 | 9.043e-3 / 9.021e-3 / 1.117e-5 | 6/24, limit |
| Sqrt-Lasso MCP | .439637 / .439635 / .439634 | 7.708e-2 / 7.449e-2 / 2.663e-3 | 12/24, limit |
| Sqrt-Lasso SCAD | .443847 / .443847 / .443847 | 9.921e-2 / 9.140e-2 / 7.791e-3 | 7/24, limit |

Cap=25 never increased the target objective at any lambda and reduced the
worst path stationarity by 5.4x to approximately 260,000x relative to cap=3.
It nevertheless exhausted 25 stages in five of six workloads, so a raised cap
is a best-effort accuracy control rather than a certification guarantee.

## Decision

Retain the adaptive LLA mechanism, these optimizations, diagnostics, and the
three-stage default. The
optional higher cap provides real monotone objective improvement and often
large stationarity gains with negligible memory cost; cap=3 still gives nearly
the same final-lambda objective at much lower cost than cap=25. Do **not** claim
speed parity with the old unchecked solver: strict weighted-L1 certification
still costs 1.25--1.64x for GLMs and 2.85--2.91x for Sqrt-Lasso, while target
stationarity after three stages is mostly unchanged. Sqrt-Lasso's remaining
gap is accompanied by about 2.7x as many recorded inner sweeps, not repeated scans;
removing it would require a different certified subproblem algorithm or a
deliberately weaker KKT contract. Keep the correctness contract rather than
silently rolling it back, but treat a faster certified Sqrt-Lasso subproblem
as follow-up work for speed-sensitive use.
