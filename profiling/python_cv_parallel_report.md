# Python Cross-Validation Threading Benchmark

Python fold fitting is serial by default. The retained public
`cross_validate(..., n_jobs=1)` argument enables a thread pool only when the
caller requests more than one worker. Native solver calls release the GIL;
each fold owns its solver and loss buffer, and the parent consumes results and
errors in fold order.

`python_cv_parallel_benchmark.py` uses five fixed folds, fast mode, 20 lambdas
(12 for wide square-root lasso), and one BLAS thread. Tall problems are
`20000 x 120`; wide problems are `300 x 2500`. The table reports serial time
divided by the four-worker time.

| setting | Gaussian | binomial | Poisson | sqrt-lasso | multinomial |
| --- | ---: | ---: | ---: | ---: | ---: |
| tall | 2.29x | 1.97x | 1.45x | 2.11x | 1.69x |
| wide | 2.14x | 1.98x | 2.41x | 2.00x | 2.17x |

The omitted argument and explicit `n_jobs=1` retained bitwise-identical output
for every supported family and measure. A source-level old/new comparison of
15 family/measure combinations also produced identical hashes; across ten
timing fixtures, the median serial ratio was 1.006, with no reproducible
regression.

Parallelism deliberately trades memory for elapsed time. Representative
multinomial max RSS increased by about 16%–17% from one to four workers. The
documentation therefore keeps one worker as the default, warns that each
active fold retains training data and solver outputs, and recommends limiting
BLAS to one thread to prevent oversubscription.

Tests additionally prove actual worker overlap, cap workers at the fold
count, preserve the lowest-fold exception, validate `n_jobs` before training
or RNG use, and retain certified early-stop prefixes.
