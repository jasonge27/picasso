# Python Multinomial Output Buffers

## Change

The Python multinomial wrapper previously allocated outputs in
`_reset_result_for_training()`, allocated a second complete set inside the
native wrapper, and copied the native results back. It now passes the reset
buffers directly. A full fit retains those arrays; a truncated fit copies only
its committed prefix so it does not keep the requested-path owner alive.

## Isolated-process A/B

The fixture used `n=24`, `d=30000`, four classes, 80 explicit lambdas above
`lambda_max`, and five fresh processes per cell. Both source trees loaded the
same native library (SHA-256 `0a084616...74edca`). Peak RSS was sampled before
the zero-copy output checksum. The baseline is the complete validated Phase-1
checkpoint, including its original `core.py`, `libpath.py`, `__init__.py`, and
`VERSION`; it is not a mixed source tree.

| Mode | Phase-1 RSS | Direct-buffer RSS | RSS change | Phase-1 time | Direct-buffer time |
|---|---:|---:|---:|---:|---:|
| Normal allocation | 141.38 MiB | 141.30 MiB | -0.06% | 0.1273 s | 0.1240 s |
| Reset buffers force-touched | 214.64 MiB | 141.39 MiB | -34.13% | 0.1375 s | 0.1265 s |

Every run produced checksum
`b92542211bf3e377bda080b325f7db0e92107e8b159aa5562dbf6a02464a8db9`.
The normal RSS difference is deliberately reported as negligible: lazy pages
can hide the unused reset allocation. Force-touching makes the structural
saving visible. For this fixture the removed staging arrays total 76,808,004
bytes, including a 76.8 MB coefficient path. Normal runtime changed by about
2.6%, within the noise of this short benchmark; no speedup is claimed.

Reproduce with:

```sh
python profiling/python_multinomial_output_buffer_benchmark.py \
  --baseline-root /path/to/phase1-checkpoint \
  --candidate-root . \
  --native-library /path/to/libpicasso.dylib \
  --repeats 5 \
  --output profiling/python_multinomial_output_buffer_results.json
```

The driver is
[`python_multinomial_output_buffer_benchmark.py`](python_multinomial_output_buffer_benchmark.py).
It emits the raw per-process records together with medians and rejects unequal
baseline/candidate checksums. The retained measurements are in
[`python_multinomial_output_buffer_results.json`](python_multinomial_output_buffer_results.json).
For each original source, the controller copies `pycasso` into an isolated
temporary tree while excluding `lib`, `src`, and `__pycache__`, then injects
the requested native build under the single compatible package-local name;
orphaned `*.pyc` files are excluded as well.
This also isolates historical loaders that ignore `PICASSO_NATIVE_LIBRARY`.
Workers disable bytecode writes and report the actual `ctypes.CDLL` path and
SHA-256; the controller rejects any mismatch. The JSON preserves each original
source path and hashes `core.py`, `libpath.py`, `__init__.py`, and `VERSION`.
