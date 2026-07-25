# Gaussian Covariance Evaluation Report

`GaussianCovUpdateObjective::eval()` previously formed every prediction with a
column-major matrix row dot product.  The retained change evaluates one
cache-friendly GEMV into the objective's existing `Xb` buffer, then reduces the
residual squares.  It adds no persistent or temporary `n`-vector allocation and
does not materialize a Gram column.

Release binaries from the complete pre-change backup and candidate tree were
run in an ABBA sequence.  Each entry is the median of six fresh processes for a
six-point L1 path.

| Case | Construct before | Construct after | Solve before | Solve after | Solve speedup |
|---|---:|---:|---:|---:|---:|
| n=20000, d=120 | 5.904 ms | 5.353 ms | 10.926 ms | 3.881 ms | 2.816x |
| n=128, d=3000 | 0.833 ms | 0.551 ms | 2.159 ms | 0.649 ms | 3.325x |
| dense n=1000, d=600 | 1.234 ms | 0.816 ms | 38.921 ms | 35.761 ms | 1.088x |

All path checksums are exactly equal.  Median peak RSS is unchanged or lower.
The direct residual oracle covers arbitrary externally supplied coefficients
and intercepts, both intercept modes, and verifies that `eval()` leaves the
lazy covariance-column count unchanged.  The targeted native and ASan/UBSan
tests plus the complete R suite pass.

An `O(d)` identity based on the incrementally maintained gradient was not mixed
into this step.  The GEMV change already removes the observed hotspot without
introducing cancellation or external-model synchronization risk; the formula
can be evaluated later as a separate experiment with a direct-residual
fallback.

The complete baseline is
`/private/tmp/picasso-master-baseline-20260716-preincremental`; direct source
backups are `/private/tmp/picasso-step5-*.before`.
