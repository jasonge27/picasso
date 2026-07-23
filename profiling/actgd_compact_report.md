# Dense ActGD Compact Working-Set Report

## Change and rollback rule

The experiment replaced `d`-wide coordinate sweeps with compact strong and
ever-active index lists. Intercepts are updated after every coordinate pass,
and `ite_lamb` now records actual coordinate passes. The change would be
reverted unless complete paths remained numerically equivalent and at least
two dense, sparse-solution cases showed repeatable improvement.

MCP and SCAD always sweep a feature-sorted compact strong set. This preserves
the historical coordinate order because changing it selected different local
minima. Convex L1 uses active-first iteration only when the strong-inactive set
exceeds `max(64, 2 * active_size)`; otherwise it uses the lower-overhead compact
strong sweep. A newly nonzero coordinate always forces active-set
reconvergence before KKT certification.

## Benchmark

Both libraries used the final Gaussian objectives. The baseline was compiled
with `/tmp/actgd.cpp.before-compact-workset`; the candidate used the new
solver. Each measurement ran in a fresh subprocess. Medians below are from 15
alternating runs of the native naive-update C API at `prec=1e-10`.

| Case | Shape and path | Maximum df | Baseline | Compact | Speedup |
|---|---:|---:|---:|---:|---:|
| Wide sparse | `n=220, d=8000, L=45` | 59 | 16.662 ms | 16.151 ms | 1.032x |
| Moderate sparse | `n=600, d=3500, L=50` | 68 | 23.346 ms | 22.372 ms | 1.044x |
| Correlated screening | `n=300, d=6000, L=55` | 37 | 71.335 ms | 36.646 ms | 1.947x |

The correlated case has 40 latent feature groups with within-group correlation
0.97. It is the intended active-first regime: strong screening admits many
variables while the fitted model stays sparse. Against an otherwise identical
compact-strong-only implementation, the hybrid was 1.863x faster there. On the
two ordinary cases the adaptive branch did not activate, so paths were exactly
identical to compact-strong-only and timing differences were scheduler noise.

## Numerical comparison

All old/new paths had identical lengths and intercepts. For wide and moderate
cases, active sizes were identical; maximum coefficient differences were
`2.06e-5` and `1.77e-5`, maximum prediction differences were `1.27e-4` and
`8.89e-5`, and maximum objective differences were below `1.83e-5`.

In the deliberately ill-conditioned correlated case, the maximum coefficient,
prediction, and objective differences were `1.34e-3`, `1.88e-3`, and
`5.74e-5`; active size differed by at most one threshold-level coefficient.
Maximum L1 KKT residual improved from `6.36e-5` to `5.90e-5`. These differences
are stopping-order effects in a nearly collinear model, not degradation of the
optimized objective or KKT accuracy.

Screening flags shrink from two `int[d]` arrays to two byte arrays. With the
compact strong list and a sparse dynamic active list, persistent screening
metadata is approximately `6d + 4a` bytes instead of `8d + 4a` bytes.

## Reproduction and verification

```sh
python3 profiling/actgd_compact_benchmark.py \
  --baseline /tmp/libpicasso_actgd_before.dylib \
  --candidate /tmp/libpicasso_actgd_compact.dylib --repeats 15
```

The standalone old-loop path oracle passes for naive/covariance objectives,
with and without intercepts, and for L1/MCP/SCAD. The same test passes under
AddressSanitizer and UndefinedBehaviorSanitizer. The rebuilt R package and all
R `testthat` files also pass.
