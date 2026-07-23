# Multinomial LLA Stage-Budget Validation

The production MCP/SCAD path uses adaptive target-stationarity checks with a
default budget of three total stages (one L1 master and two weighted-L1
updates). If this budget is exhausted, the last majorization-checked model is
returned with `lla_stationarity_limit`; it is usable but not certified at the
requested tolerance. Raising `lla_max_stages`/`lla.max.stages` lets the same
adaptive algorithm continue. The historical fixed-stage status behavior
remains available through `MultinomialLlaOptions::fixed_stage_compatibility()`.

## Correctness Gate

On the deterministic Python fixture (`n=72`, `d=6`, `K=3`, MCP,
`tolerance=1e-6`), the three-stage model had a maximum stationarity residual
of `0.130856`. With the budget raised to 25, adaptive LLA used stages
`[3, 5, 3, 3]` and reduced the maximum residual to `4.74e-8`.

For the raised-budget runs, all benchmark cases require every path point to
finish, every weighted-L1 stage to have an outer-KKT certificate, target
stationarity to meet tolerance, and the LLA majorization/descent chain to
hold. Release and ASan/UBSan CTest both passed 5/5 when these measurements
were recorded.

## Runtime and Memory Tradeoff

Timings compare the same active-set solver under fixed-three-stage and
stationarity-certified modes. They are wall-clock measurements from fresh
processes on the development Mac and should be interpreted as regression
guards rather than portable performance claims.

| Case | Max 3 | Max 25 | Stationarity at 3 | Stationarity at 25 |
| --- | ---: | ---: | ---: | ---: |
| sparse K=3 MCP | 0.507 s | 0.605 s | 6.50e-2 | 2.61e-8 |
| wide K=4 MCP | 0.617 s | 1.062 s | 5.95e-2 | 9.61e-8 |
| high K=12 MCP | 0.729 s | 1.046 s | 3.70e-2 | 4.26e-8 |
| dense control MCP | 6.806 s | 19.317 s | nonstationary | 1.98e-7 |
| dense control SCAD | 6.213 s | 16.257 s | nonstationary | 1.99e-7 |

Peak RSS was effectively unchanged. The dense case originally exhausted
4,000 inner sweeps while solving an intermediate surrogate 100 times more
accurately than either outer certificate could observe. Capping the adaptive
surrogate inner tolerance at
`min(outer_kkt_tolerance, stationarity_tolerance)` made MCP and SCAD complete
18/18 path points without changing `max_ite`, ordinary L1, or fixed-stage
compatibility behavior.

### Current default smoke check

On 2026-07-16, a fresh-process ABBA check used the deterministic Python MCP
fixture (`n=72`, `d=6`, `K=3`, four lambdas) and 20 repeated native fits per
process. These machine-specific numbers verify the new public default and V3
override rather than serving as portable performance claims.

| Stage budget | Status | Median native call | Maximum stationarity | Peak RSS range |
| --- | --- | ---: | ---: | ---: |
| 3 | `lla_stationarity_limit` | 0.979 ms | 1.31e-1 | 35.9--36.0 MiB |
| 25 | `completed` | 1.317 ms | 7.76e-7 | 36.0--36.4 MiB |

The three-stage default was about 25.7% faster in this small case. Peak RSS
was effectively unchanged at process resolution; the higher budget traded
additional work for a strict `1e-6` stationarity certificate.

These measurements support a three-stage default for fast model selection and
prediction: it saves substantial work and does not materially change peak
memory. Users who need an independently checked first-order certificate can
raise the budget; in these cases, the residual improved by five to seven
orders of magnitude at the runtime cost shown above.
