# Python Scalar Finalization Buffer Benchmark

## Scope and isolation

This benchmark measures only Python result allocation and finalization; it is
not a solver-runtime benchmark. The driver stages the before and candidate
wrappers in separate temporary trees, excludes bundled native sources,
libraries, caches, and bytecode, then injects the same native library into
both. Every measurement runs in a fresh process with BLAS/OpenMP thread counts
set to one. Each worker verifies the loaded library path and SHA-256 before a
deterministic fake Gaussian V2 entry point page-touches the full native output
buffer. The fake retains no NumPy references. Current RSS is sampled before
peak RSS, and the worker rejects any nonempty measurement for which peak RSS
is smaller than current RSS.

Command:

```sh
python3 profiling/python_scalar_finalize_buffer_benchmark.py \
  --baseline-root /private/tmp/picasso-scalar-finalize-before-20260719 \
  --candidate-root . \
  --native-library /private/tmp/picasso-refactor-build-phase1/libpicasso.dylib \
  --repeats 5 \
  --output profiling/python_scalar_finalize_buffer_results.json
```

The fixture uses `n=8`, `d=100000`, 100 requested lambdas, and five fitted
lambdas for the partial-path case. The full coefficient buffer is 80,000,000
bytes. The native SHA-256 is
`0a084616f75b0994a6fbe2a64782703b23e8ff06e74924b563d06b89a374edca`.
The benchmark-script SHA-256 is
`c76ef13f8af97d25287a0951b0d79f45fbd1dd5d760b42a85565bbf8955756d2`.
The raw JSON also records source paths and hashes, Python 3.14.3, NumPy 2.4.2,
`sys.platform=darwin`, and
`macOS-26.5.2-arm64-arm-64bit-Mach-O`.

## Results

Values below are `(current RSS bytes, peak RSS bytes, elapsed seconds)`.

| Mode | Source | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|---|---|---:|---:|---:|---:|---:|
| Full, standardized | Before | 264683520, 264683520, 0.05202429200289771 | 264830976, 264830976, 0.036636625009123236 | 264781824, 264781824, 0.017858417006209493 | 264650752, 264650752, 0.024029749969486147 | 264503296, 264503296, 0.017574625031556934 |
| Full, standardized | Candidate | 184877056, 184877056, 0.01390287495451048 | 184860672, 184860672, 0.013634375005494803 | 184909824, 184909824, 0.013957290968392044 | 184778752, 184778752, 0.014499999990221113 | 184614912, 184614912, 0.013861624989658594 |
| Partial, unstandardized | Before | 137641984, 137641984, 0.004402416001539677 | 137740288, 137740288, 0.004557165957521647 | 137953280, 137953280, 0.004442208970431238 | 138100736, 138100736, 0.004772290994878858 | 138002432, 138002432, 0.0045191670069471 |
| Partial, unstandardized | Candidate | 142245888, 142245888, 0.004923874977976084 | 142213120, 142213120, 0.004674749972764403 | 142163968, 142163968, 0.004593290970660746 | 142016512, 142016512, 0.004941624996718019 | 142114816, 142114816, 0.004700957972090691 |

For the full standardized path, median RSS fell from 264,683,520 to
184,860,672 bytes (79,822,848 bytes, or 30.16%) and median finalization time
fell from 0.024029750 to 0.013902875 seconds (42.14%). The before timings show
warm-up variance, so the timing change should be treated as a local
microbenchmark result rather than a general training-speed claim.

For the partial unstandardized path, the public `beta` changed from a view
whose backing owner remained 80,000,000 bytes to an owning 4,000,000-byte
array, a 95% reduction in retained backing storage. Median elapsed time rose
by 0.000181791 seconds (4.02%), the cost of copying the fitted prefix. Median
current and peak RSS each rose by 4,210,688 bytes (3.05%).

Both implementations produced identical checksums on every repetition:

- Full standardized:
  `cbac04379b8acf6177fd25676e214a2dca0097c9e03e504a7b7435453f3f34ff`
- Partial unstandardized:
  `ffcb5f38857929f2e6ceacf5a5d0f439891e65588477e68a93d8d2ebedc7fadc`

## Decision

Retain the change. It removes an avoidable full-path allocation for the full
standardized case and prevents a short fitted prefix from keeping the entire
native coefficient path alive. In the partial case, process RSS rose by about
4.2 MB even though the retained NumPy backing owner shrank by 76 MB. The
allocator need not immediately return freed pages to macOS, so same-process
RSS is not a reliable retained-object measure here; ownership and backing
size directly test the lifetime problem. The small absolute copy cost is an
acceptable tradeoff for releasing the oversized owner.
