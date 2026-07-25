# Python Default-Data Validation Benchmark

`Solver` owns a finite, C-contiguous copy of its training design. Prediction
and assessment previously rescanned all `n*d` entries when the data argument
was omitted. The retained implementation skips only that redundant scan;
explicit matrices, including an explicitly passed `solver._x_orig`, still
receive full validation.

`python_default_data_scan_benchmark.py` compares omitted data with the same
matrix passed explicitly. It uses a three-lambda Gaussian path, one BLAS
thread, and seven median repetitions:

| shape | predict reduction | assess reduction | avoided Boolean temporary |
| --- | ---: | ---: | ---: |
| `1,000,000 x 20` | 26.6% | 10.7% | about 20 MB |
| `100,000 x 100` | 39.5% | 32.8% | about 10 MB |
| `5,000 x 2,000` | 64.0% | 43.5% | about 10 MB |

All five families produce identical default and explicit-data results. Tests
also prove that caller mutation cannot affect the owned design, explicit NaN
inputs still fail for scalar and multinomial models, and multinomial logits
retain their post-computation finite check. Deliberately corrupting the private
`_x_orig` attribute is outside the supported API.
