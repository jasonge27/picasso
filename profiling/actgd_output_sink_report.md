# ActGD Direct Output-Sink Report

## Change and acceptance rule

The Gaussian C API previously retained every `ModelParam` in the C++ solver
and copied the completed path into caller-owned arrays afterwards.  The new
C-API adapter commits each model directly to those arrays while the public
C++ `solve()` method keeps its historical retained-path behavior.  This
removes one `L * d` coefficient-path allocation without changing the C ABI or
the solver iteration order.

The change is retained only if old/new outputs are byte-identical, sanitizer
and interface suites pass, peak memory falls by approximately the removed
path size, and median runtime does not regress by more than 3%.

## Isolated A/B benchmark

The driver loads each library in a fresh process and allocates/touches all
inputs and public output arrays before sampling baseline peak RSS.  A
strictly decreasing, high-lambda path keeps all coefficients at zero and
commits all 100 models, isolating output retention from optimization-path
differences.  Fifteen runs were interleaved on Apple arm64 with Apple Clang
21.  The RSS values are incremental process high-water marks during the
native call, not total solver residency.

| Fixture | Removed path | Old peak-RSS delta | Sink peak-RSS delta | Reduction | Runtime speedup |
|---|---:|---:|---:|---:|---:|
| `n=8, d=40,000, L=100` | 32.0 MB | 36.41 MB | 3.47 MB | 32.93 MB | 1.147x |
| `n=8, d=100,000, L=100` | 80.0 MB | 90.88 MB | 10.26 MB | 80.63 MB | 1.246x |

Both fixtures produced identical SHA-256 checksums across all runs, including
coefficients, intercepts, iteration counts, active sizes, runtimes, and smooth
objectives.  Each run fitted 100 models.  The timing result describes this
output-heavy microbenchmark; it is not a general Gaussian training-speed
claim.

## Verification and reproduction

- Release CTest: 15/15 passed.
- AddressSanitizer and UndefinedBehaviorSanitizer: 15/15 passed.  Leak
  detection was disabled because this macOS ASan runtime does not support it.
- The root/R mirror check and the 20-symbol C export/runtime verifier passed.
- Retained-versus-sink tests cover naive/covariance objectives, C API V1/V2,
  L1/MCP/SCAD, intercept on/off, dfmax crossing, null optional outputs, and
  injected failure at every native allocation.

```sh
python3 profiling/actgd_output_sink_benchmark.py \
  --baseline /path/to/before/libpicasso.dylib \
  --candidate /path/to/after/libpicasso.dylib \
  --repeats 15 --output profiling/actgd_output_sink_results.json
```

Raw measurements and complete source/runtime hashes are in
`actgd_output_sink_results.json`.
