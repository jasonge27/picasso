# Gaussian Extreme Standardization Report

The Gaussian wrapper previously used `colMeans()` before its max-scaled norm
calculation.  A fully finite `18 x 4` design with values near `1e308` made the
column mean overflow and the public Gaussian fit fail.  The centered Gaussian
branch now reuses the native max-scaled, compensated standardizer already used
by the other families.  Unstandardized and no-intercept branches are unchanged.

Both Gaussian objectives now complete the extreme finite fixture with finite
coefficients, intercepts, and deviance ratios.  The targeted stability test and
the complete R `testthat` suite pass.

The isolated preprocessing benchmark used a `3000 x 2000` ordinary dense
Gaussian matrix.  Each timing ran in a fresh R process; the table reports the
median of three runs.  Vcell high-water memory was reset after allocating the
input matrix.

| Version | Median time | Peak incremental Vcells |
|---|---:|---:|
| Before | 0.171 s | 138.681 MiB |
| After | 0.093 s | 137.403 MiB |

The retained change is 1.84x faster in this preprocessing benchmark and lowers
the measured peak by 1.278 MiB.  The complete pre-change working tree is backed
up at `/private/tmp/picasso-master-baseline-20260716-preincremental`; the two
directly changed files also have `picasso-step1-*.before` backups there.
