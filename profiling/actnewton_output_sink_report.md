# ActNewton Direct Output-Sink Report

## Change and acceptance rule

The scalar logistic, Poisson, and square-root-loss C APIs previously retained
every fitted `ModelParam` and then copied the path into caller-owned arrays.
The new adapter commits each fully validated final LLA model directly to those
arrays. Public C++ `solve()` calls still retain their historical path, and the
old private `solve_impl(bool)` symbol remains as a forwarding shim.

The change is accepted only if old/new algorithmic outputs are byte-identical,
the interface and sanitizer suites pass, incremental peak RSS falls by roughly
the removed `L * d` path, and median runtime does not regress by more than 3%.

## Isolated A/B benchmark

Each library runs in a fresh process with BLAS/OpenMP thread controls set to
one. Inputs and caller-owned outputs are allocated and touched before the RSS
baseline. A high-lambda logistic fixture keeps every coefficient at zero and
commits all 100 models, isolating path retention from optimizer behavior.
Fifteen old/new runs were interleaved on Apple arm64. RSS is the incremental
process high-water mark during the native call; sizes below use decimal MB.

| Fixture | Removed path | Old RSS delta | Sink RSS delta | Reduction | Runtime speedup |
|---|---:|---:|---:|---:|---:|
| `n=4, d=40,000, L=100` | 32.0 MB | 37.83 MB | 4.70 MB | 33.13 MB | 1.150x |
| `n=4, d=100,000, L=100` | 80.0 MB | 93.41 MB | 12.34 MB | 81.07 MB | 1.158x |

Both fixtures produced one identical checksum across all old/new runs. Timing
is specific to this output-heavy microbenchmark and is not a general training
speed claim.

## Correctness and reproduction

The shared-library oracle compared 18 configurations: all three scalar
families, L1/MCP/SCAD, intercept on/off, and nonzero GLM offsets. Every path
committed all four requested models; all algorithmic arrays, diagnostics,
statuses, and iteration counts were byte-identical. Separate tests cover
adaptive stages, hard failures, nullable outputs, dfmax, and allocation
failure before and after committed prefixes.

- Release CTest: 16/16 passed.
- ASan/UBSan CTest: 16/16 passed; leak detection is disabled because the local
  macOS ASan runtime does not support it.
- Root/R mirrors, all 20 C exports, and the old exported C++ symbol set passed.

```sh
python3 profiling/actnewton_output_sink_benchmark.py \
  --baseline /path/to/before/libpicasso.dylib \
  --candidate /path/to/after/libpicasso.dylib \
  --repeats 15 \
  --output profiling/actnewton_output_sink_results.json
```

Raw measurements and source/library hashes are recorded in
`actnewton_output_sink_results.json`. The recorded script SHA-256 is
`354da12788ecb907482b4c603b606b7be403801237aeb8fc8ba6b682eac6a97e`;
the isolated old/new library hashes are `4f7e67961b07a263ea6cd299d938d846aec9c302de5a4b5af4be630385f2b490`
and `52a2ce02ba53d51b3eb8ee8a879ed021b99040dff9dd95eb91262813eb6e7014`.
