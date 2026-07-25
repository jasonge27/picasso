# Python Multinomial Rescaling Matrix-Vector Kernel

## Decision

**KEEP.** `_rescale_multinomial_solution_in_place()` now computes each blocked
intercept adjustment with matrix-vector multiplication instead of materializing
`beta_block * xm` and reducing it by row. The existing block policy and
in-place coefficient/intercept ownership are unchanged.

## Correctness

`python-package/test_pycasso.py` compares the new kernel with the former
formula for ordinary, truncated, single-coefficient, empty-path, zero-feature,
wide, and extreme-finite inputs. Coefficients remain bit-identical. Intercepts
must satisfy a scale-aware fixture acceptance tolerance because BLAS and NumPy
row sums can use different reduction orders. Buffer identity is also asserted.

Both full Python suites passed against native library SHA-256
`ff2ed89678a841fee9a0d6321beb65abf91e496f9e545c8459bc9043fdfaf515`:

```sh
PICASSO_NATIVE_LIBRARY=/private/tmp/picasso-python-mn-rescale-matvec-before-20260719/libpicasso-fixed.dylib PYTHONDONTWRITEBYTECODE=1 python3 python-package/test_pycasso.py
PICASSO_NATIVE_LIBRARY=/private/tmp/picasso-python-mn-rescale-matvec-before-20260719/libpicasso-fixed.dylib PYTHONDONTWRITEBYTECODE=1 python3 python-package/test_reproducibility.py
```

## Fresh-Process A/B Benchmark

The benchmark used 100 lambdas, eight classes, 20,000 features, one BLAS
thread, and 15 interleaved fresh processes per source and measurement mode.
Every runtime worker made 15 timed calls from the same intercept state; input
resetting occurred outside the timed region. Memory used separate one-call
workers.

| Median metric | Former row sum | Matrix-vector | Change |
|---|---:|---:|---:|
| Kernel runtime | 9.338 ms | 4.683 ms | 1.994x faster |
| `tracemalloc` peak | 8,323,504 B | 2,512 B | -8,320,992 B |
| Peak-RSS delta | 11,599,872 B | 32,768 B | -11,567,104 B |

Every worker hashed the complete coefficient and intercept arrays. The
old-formula oracle checksum was consistently
`5bef9689eae1f9acff38210e35632a7709ab94bf44286177a0fbf650a788dc7e`;
the coefficient checksum was identical in both arms. The largest candidate
intercept difference was `1.055e-15`, versus a `4.735e-13` fixture acceptance
tolerance. Its maximum ULP count was 32,768 at a near-zero value, so the absolute
error is the meaningful scale there.

Reproduce with:

```sh
python3 profiling/python_multinomial_rescale_benchmark.py \
  --baseline-root /private/tmp/picasso-python-mn-rescale-matvec-before-20260719 \
  --candidate-root /Users/tourzhao/Desktop/picasso-master \
  --native-library /private/tmp/picasso-python-mn-rescale-matvec-before-20260719/libpicasso-fixed.dylib \
  --repeats 15 --inner-repeats 15 --nlambda 100 --classes 8 --d 20000 \
  --seed 20260719 \
  --output profiling/python_multinomial_rescale_results.json
```

Raw per-process results, source/native hashes, benchmark-script hash, CPU,
Python, NumPy, Accelerate BLAS, and thread configuration are stored in
`python_multinomial_rescale_results.json`. The before snapshot is
`/private/tmp/picasso-python-mn-rescale-matvec-before-20260719`.
