# Native Loss Path Benchmark

## Goal

Avoid rebuilding training-set predictions in R solely to populate
`dev.ratio`. Each native solver already evaluates its unpenalized loss at the
accepted model for every lambda. The new versioned C APIs expose that path;
the older ABIs remain unchanged.

## Method

Run `r_native_loss_benchmark.R` against the pre-change and candidate R
libraries. The fixture uses `n = 4000`, `p = 120`, 45 explicit lambdas, one
thread, pre-standardized dense input, `fast.mode = TRUE`, one warm-up, and five
measured fits per process. The table pools three independent processes (15
fits per family) and reports medians.

| Family | Before (s) | After (s) | Speedup | Post-fit time saved |
|---|---:|---:|---:|---:|
| Gaussian | 0.016 | 0.015 | 1.07x | 1.8 ms |
| Logistic | 0.042 | 0.035 | 1.20x | 5.0 ms |
| Poisson | 0.048 | 0.040 | 1.20x | 6.6 ms |
| Sqrt-Lasso | 0.032 | 0.030 | 1.07x | 1.9 ms |
| Multinomial | 0.409 | 0.248 | 1.65x | 159.1 ms |

The scalar families previously allocated dense `n x nlambda` predictors plus
intercept/offset temporaries. Multinomial repeatedly allocated and evaluated
`n x K` logits and probabilities for every lambda. The candidate reduces this
post-fit work to `O(nlambda)` scalar transforms; Poisson additionally computes
one `O(n)` response-only constant.

## Numerical Validation

- Existing R testthat suite: passed.
- New explicit-prediction oracle tests cover L1, MCP, and SCAD for every
  family, both Gaussian update modes, standardization, and offsets.
- New native C API tests compare the exposed losses with losses recomputed
  from returned coefficients and verify transactional `NaN` suffixes.
- The benchmark's `dev.ratio` checksums were unchanged for Gaussian and
  multinomial. Maximum aggregate differences for Logistic, Poisson, and
  Sqrt-Lasso were respectively `9.8e-15`, `2.1e-8`, and `3.2e-9`, consistent
  with native stable-loss and incremental-residual rounding.

Raw CSV files and the pre-change source archive are stored under
`/private/tmp/picasso-native-loss-backup-20260717/` for this validation run.

## Tall Multinomial Versus glmnet

The existing glmnet comparison was rerun without changing its data, lambda
path, one-thread policy, tolerance calibration, or timing order. Removing the
R post-fit replay reduced the matched-accuracy PICASSO median from 0.506 s to
0.365 s versus glmnet's 0.353 s (1.03x gap). In the fast track, PICASSO fell
from 0.361 s to 0.210 s versus glmnet's 0.136 s, reducing glmnet's advantage
from 2.69x to 1.54x. PICASSO retained the tighter external KKT residual in
both tracks and its fitted object was smaller (about 0.072 MB versus 0.119 MB).

The rerun artifacts are under
`/private/tmp/picasso-native-loss-glmnet-benchmark-20260717/`.
