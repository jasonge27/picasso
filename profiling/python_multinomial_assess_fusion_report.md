# Python Multinomial Assessment Fusion

## Decision

**KEEP.** Multinomial `assess()` now forms one logits matrix per fitted lambda
and reuses it for deviance and class error. It retains
`argmax(softmax(logits))` exactly: changing this to `argmax(logits)` altered
historical results for logits separated by fractions of one floating-point
ulp. Probabilities and predictions are released after each lambda instead of
retaining an `n x nlambda` prediction array.

The scope was limited to assessment. Cross-validation, fitting, coefficient
scaling, input ownership, packaging, and scalar-family code were unchanged.

## Correctness

`python-package/test_pycasso.py` contains:

- an old-implementation oracle for subset and string-label assessment;
- exact-tie, extreme-logit, and 0.125/0.25/0.5-ulp near-tie probes;
- identical nonfinite-logit error checks and multinomial offset rejection;
- call counting requiring exactly `nlambda` logits, NLL, and softmax calls.

Both commands passed against the same explicit Release library:

```sh
PICASSO_NATIVE_LIBRARY=/private/tmp/picasso-actnewton-sink-final-validated-20260719/libpicasso.dylib PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 python-package/test_pycasso.py
PICASSO_NATIVE_LIBRARY=/private/tmp/picasso-actnewton-sink-final-validated-20260719/libpicasso.dylib PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 python-package/test_reproducibility.py
```

Native SHA-256:
`52a2ce02ba53d51b3eb8ee8a879ed021b99040dff9dd95eb91262813eb6e7014`.

## Interleaved A/B Benchmark

The benchmark used `n=20000`, `d=50`, four classes, 80 lambdas, seven
alternating fresh-process repetitions, three timed assessments per runtime
worker, and one BLAS thread. Old and new Python sources loaded the native
library above. Memory ran in separate fresh processes; `tracemalloc` records
Python/NumPy allocation peak and peak-RSS delta records the process high-water
increase after fixture construction.

| Metric (median) | Old source | Fused source | Change |
|---|---:|---:|---:|
| Runtime | 0.279095 s | 0.236245 s | 1.181x; -15.35% |
| `tracemalloc` peak | 26.85 MB | 4.19 MB | -22.66 MB; -84.41% |
| Peak-RSS delta | 37.55 MB | 12.48 MB | -25.07 MB; -66.75% |

All runtime and memory workers produced checksum
`62597a4abd85480d9074b40dfd80465d0d75907bba1473420abc2398539afe72`
for `lambda`, `deviance`, and `class_error`.

Reproduce with:

```sh
python3 profiling/python_multinomial_assess_fusion_benchmark.py \
  --baseline-root /private/tmp/picasso-python-mn-assess-fusion-before-20260719 \
  --candidate-root /Users/tourzhao/Desktop/picasso-master \
  --native-library /private/tmp/picasso-actnewton-sink-final-validated-20260719/libpicasso.dylib \
  --repeats 7 --inner-repeats 3 --n 20000 --d 50 --classes 4 \
  --nlambda 80 --seed 20260719 \
  --output profiling/python_multinomial_assess_fusion_results.json
```

Raw results are in `python_multinomial_assess_fusion_results.json`. Baseline
`core.py` SHA-256 is
`8c777a3b84dff3003e26879f575ae09f72593d8c595b913834518a344a27ea11`;
fused `core.py` SHA-256 is
`8bca15c666e082e1488fdcbc6032fdd56805910ec76d21fa4b78a898a561757c`.
The pre-phase snapshot is
`/private/tmp/picasso-python-mn-assess-fusion-before-20260719`.
