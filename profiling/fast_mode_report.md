# Fast Mode Calibration and Validation

> Calibration snapshot from July 2026. The precision policy below is current;
> runtime figures remain specific to the recorded hardware, data, and source
> state.

## Interface and Precision Policy

Fast mode is opt-in: `fast.mode = FALSE` in R and `fast_mode=False` in
Python preserve the existing `prec=1e-7` behavior. When enabled, the effective
precision is calibrated by solver family:

| Family | Fast precision | Reason |
|---|---:|---|
| Gaussian | `1e-7` | Its scaled objective-change test already matches glmnet's convention. |
| Binomial, sqrt-lasso, multinomial | `1e-4` | Their approximately absolute KKT tolerance needs a looser numeric value to match glmnet-like achieved accuracy. |
| Poisson | `4e-4` | Calibration required a wider PICASSO tolerance to match glmnet-like achieved KKT accuracy. |

The fitted object and cross-validation result record the mode and effective
precision. Custom `prec` values remain available only with fast mode disabled.
No native ABI or additional working allocation was introduced.

## Why Gaussian Keeps `1e-7`

A candidate global `1e-4` preset was benchmarked and rejected. Across tall,
balanced, and wide Gaussian problems, its maximum external relative KKT was
`0.050`--`0.235`, versus `0.00016`--`0.00773` for glmnet. In a held-out smoke
test it increased MSE from `1.2701` to `1.2879` (1.4%). Retaining `1e-7`
makes Gaussian fast/high fits identical and avoids an accuracy regression.

## Speed and Model Quality

Measurements used R 4.5.2 on arm64 macOS, one BLAS thread, identical explicit
lambda paths, warm-up, and five alternating-order repetitions. On a common
`1000 x 300`, 30-lambda public-interface workload, median speedups were:

| Family | Speedup |
|---|---:|
| Binomial | 1.87x |
| Poisson | 1.71x |
| sqrt-lasso | 1.63x |
| Multinomial | 1.53x |

The non-Gaussian geometric mean was 1.68x. A larger multinomial calibration
with 45 lambdas gave 1.71x (tall), 2.39x (balanced), and 3.46x (wide), or
2.42x by geometric mean.

All five held-out smoke problems selected the same validation lambda in fast
and high-precision modes. Test loss changes were zero for Gaussian,
`6.88e-5` MSE for sqrt-lasso, `8.42e-5` log-loss for binomial, `1.43e-5`
Poisson deviance, and `8.70e-5` log-loss for multinomial. Binomial and
multinomial classification errors were unchanged.

## glmnet Accuracy Calibration

Independent full-path KKT checks used the same pre-standardized data and the
same explicit 45-lambda multinomial paths. PICASSO fast mode versus glmnet
default produced maximum relative KKT values of `0.0314` versus `0.0325`
(balanced), `0.0108` versus `0.0107` (tall), and `0.00964` versus `0.0141`
(wide). Maximum absolute PICASSO KKT was `9.91e-5`.

All three held-out comparisons selected the same lambda as glmnet. Maximum
test log-loss difference was `1.28e-5`; classification-error difference was
at most one observation in 1600 (`0.000625`). Thus the family presets
represent glmnet-like achieved optimization accuracy, not glmnet's literal
`thresh` value.
