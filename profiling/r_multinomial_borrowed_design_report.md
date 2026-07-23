# R Multinomial Borrowed-Design Benchmark

This July 20, 2026 A/B test measures the retained C-API change that borrows an
aligned, column-major R design matrix for the duration of a synchronous
multinomial solve. Python row-major input remains owning, and misaligned C
input safely falls back to an owning copy.

The fresh-process fixture in `r_multinomial_borrowed_design_benchmark.R` uses
`n=100000`, `d=100`, `K=3`, one high lambda, `standardize=FALSE`, and
`fast.mode=TRUE`. Run it under `/usr/bin/time -l`, selecting the baseline or
candidate installation with `R_LIBS_USER`.

| implementation | median fit (s) | max RSS (bytes) | peak footprint (bytes) |
| --- | ---: | ---: | ---: |
| owning C++ design | 0.107 | 447,578,112 | 370,001,024 |
| borrowed R design | 0.091 | 388,104,192 | 296,485,920 |

The candidate reduced max RSS by 56.7 MiB and macOS peak footprint by
70.1 MiB, while improving the three-run median by 15.0%. Serialized algorithmic
outputs were byte-identical after excluding the intentionally variable runtime
diagnostic. The input matrix checksum was unchanged.

Native validation additionally covered L1, MCP, and SCAD; fast and strict
precision; intercept on/off; aligned borrowing; misaligned fallback; and
transactional NaN/Inf rejection. ASan/UBSan, the complete native runner,
Python tests, R tests, old-client ABI loading, and hidden-symbol checks passed.

These measurements are local evidence, not universal performance guarantees.
An AVX build may require stronger alignment than an R allocation provides; in
that case correctness is preserved by the owning fallback, but the memory
benefit is not expected.
