# Gaussian R Preprocessing Report

The public Gaussian wrapper previously centered every intercept fit in R and
then copied the full design again while subsetting zero-norm columns. Native
naive and covariance objectives now profile the intercept and safely map
zero-curvature columns to zero, so unstandardized dense designs can pass
through unchanged. Standardized fits still perform the required centering and
scaling.

The isolated benchmark used a `3000 x 2000` dense double matrix and
`standardize = FALSE, intercept = FALSE`. Each timing ran in a fresh R process;
the table reports the median of three preprocessing calls. Vcell high-water
memory was reset after allocating the input and measured after preprocessing.

| Wrapper | Median time | Peak Vcell memory |
|---|---:|---:|
| Before | 0.120 s | 142.65 MiB |
| After | 0.033 s | 101.12 MiB |

This is a 3.64x preprocessing speedup and a 29.1% reduction in the measured R
vector-heap high-water mark. The optimization is retained because it removes
only redundant dense copies; the full `standardize x intercept x
naive/covariance` correctness matrix is covered by
`tests/testthat/test-gaussian-interface-semantics.R`.

The before-image is
`/tmp/picasso.gaussian.R.before-native-centering-memory-fix`; the repository-wide
backup remains under
`/private/tmp/picasso-pre-nonsparse-improvements-20260716`.
