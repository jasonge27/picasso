# R Multinomial Streaming Benchmark

Seven-run minimum: 7 alternating fresh-process runs per implementation and case.
`Rprofmem` is cumulative allocation, not peak memory. Maximum RSS is the fresh-process high-water mark reported by macOS `/usr/bin/time -l`; it includes the R runtime, inputs, and fitted object in both implementations.

| Case | Old time (s) | New time (s) | Time | Old allocation | New allocation | Allocation | Old max RSS | New max RSS | Max RSS | Exact output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| assess | 6.2240 | 6.1080 | -1.9% | 1279.9 MiB | 1279.9 MiB | -0.0% | 456.8 MiB | 398.0 MiB | -12.9% | yes |
| confusion | 0.4450 | 0.4530 | +1.8% | 145.5 MiB | 145.5 MiB | -0.0% | 303.5 MiB | 288.8 MiB | -4.8% | yes |
| predict_class | 1.4930 | 0.4340 | -70.9% | 147.0 MiB | 117.3 MiB | -20.2% | 293.6 MiB | 279.3 MiB | -4.9% | yes |
| cv_class | 0.4330 | 0.4470 | +3.2% | 121.2 MiB | 121.2 MiB | -0.0% | 282.5 MiB | 270.7 MiB | -4.2% | yes |
| cv_deviance | 0.8690 | 0.4450 | -48.8% | 217.6 MiB | 144.3 MiB | -33.7% | 292.4 MiB | 281.4 MiB | -3.8% | yes |

The retained-path change is clearer in the exact core-array live payload (logits plus the simultaneously used probability matrix where applicable):

| Case | Old live payload | New live payload | Change |
|---|---:|---:|---:|
| assess | 83.8 MiB | 2.7 MiB | -96.7% |
| confusion | 14.2 MiB | 0.9 MiB | -93.5% |
| predict_class | 0.9 MiB | 0.9 MiB | +0.0% |
| cv_class | 4.8 MiB | 0.4 MiB | -92.3% |
| cv_deviance | 4.8 MiB | 0.2 MiB | -96.2% |

Decision: KEEP (output equivalence: TRUE; no median runtime regression above 5%: TRUE).

## Compatibility audit

Classification still applies the historical softmax to each streamed logits matrix before first-tie selection. For logits `[0, f * .Machine$double.eps, -2]`, direct logits select class 2 for `f = 0.125, 0.25, 0.5, 1, 2`, while softmax-first selects classes `1, 1, 2, 2, 2`; tests lock this near-tie boundary.
NaN and positive-infinite logits retain the old controlled softmax error on class-scoring paths. CV deviance now skips that unused softmax and reports the NLL finite-logits validation instead. An isolated negative-infinite class logit remains accepted as zero probability by softmax (the legacy behavior), while multinomial NLL assessment continues to require finite logits.

Before timing, a class-confounded fold fixture correctly aborted with `Training fold 1 is missing multinomial class(es)`. The final driver assigns folds within each class, and its isolated CV child smoke test passed; the missing-training-class regression remains covered by `test-multinomial.R`.

## Reproducibility

- Script MD5: `650b2b594282d4f082a60b6d5c5ff090`
- Old sources: `picasso_utils.R=ccc0f6df895a9dc134a33e2ca1d625bb;picasso.multinomial.R=895b5293f5224fb67fdf6d80238bf85e;assess.picasso.R=d99c54574ff2506317c1e2baaf808b70;cv.picasso.R=bcd1a95a3352e0b20986c06b3e9b5f70`
- New sources: `picasso_utils.R=ccc0f6df895a9dc134a33e2ca1d625bb;picasso.multinomial.R=a32f759c161a59448b5dcc7ad4723642;assess.picasso.R=db436a2d678aa687f2f015cf48f28266;cv.picasso.R=487d5ea98debebcd934bd82436ebc6f6`
- R: R version 4.5.2 (2025-10-31) (aarch64-apple-darwin20)
- Matrix: 1.7.4
- BLAS: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib`
- LAPACK: `/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libRlapack.dylib`
- CPU: Apple M3 Pro
- Thread environment: `OMP_NUM_THREADS=<unset>;OPENBLAS_NUM_THREADS=<unset>;VECLIB_MAXIMUM_THREADS=<unset>;MKL_NUM_THREADS=<unset>;BLIS_NUM_THREADS=<unset>`
