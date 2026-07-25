# Python Scalar CV Batched-Scoring Benchmark

Scalar Python cross-validation formerly multiplied the test design by one
coefficient vector per lambda. The retained evaluator performs blocked GEMM
for only the requested loss, using a CV-specific 1 MiB predictor budget per
active worker. Multinomial remains one lambda at a time because its end-to-end
prototype did not improve and added memory.

`python_cv_batched_scoring_benchmark.py` uses five fixed folds, 100 lambdas,
fast mode, and one BLAS thread. Source-tree A/B runs used the same native
library. Representative serial medians were:

| family/measure | tall change | wide change |
| --- | ---: | ---: |
| Gaussian deviance | -34.2% | -10.5% |
| sqrt-lasso deviance | -4.3% | +1.0% |
| Poisson deviance | -3.7% | -3.8% |
| binomial deviance | -3.8% | -6.3% |

RSS changed by at most about 1.4% in the final comparisons. Four-worker tall
Gaussian also improved by 18.3%. Continuous CV metrics differed by at most
`9e-16`, and selected lambdas were unchanged.

Binomial class loss deliberately retains the prior per-lambda GEMV arithmetic.
A cancellation fixture demonstrated that BLAS-3 can round an exact zero link
to roughly `1e-17`, which would change the strict `eta > 0` decision. The
class branch is therefore byte-compatible and structurally tested not to call
the batched predictor. Multinomial batching was rejected: representative
end-to-end changes ranged from 0.16% faster to 7.1% slower while adding up to
about 8 MiB per worker.
