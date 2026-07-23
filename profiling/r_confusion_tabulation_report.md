# R Confusion-Table Counting Benchmark

Both classification families formerly rebuilt factors and called `table()`
for every lambda. The retained helpers count integer class pairs with
`tabulate()` and then attach exactly the same dimensions, labels, and `table`
class. Prediction, tie handling, and lambda streaming are unchanged.

For binomial confusion, the `100000 x 20`, 100-lambda blocking fixture improved
from 0.371 to 0.183 seconds (2.03x) over seven repetitions. Max RSS changed by
1.23% and peak footprint by 2.53%, both below the 5% retention threshold. An
independent simultaneous run measured 0.463 versus 0.247 seconds. Results were
serialization-identical, including absent classes and exact-zero logits.

For multinomial confusion, the reproducible fixture in
`r_multinomial_confusion_tabulate_benchmark.R` uses 100,000 observations, 20
features, eight classes, and 100 lambdas. With one BLAS thread, an independent
seven-run comparison measured:

| counting implementation | median (s) | relative time |
| --- | ---: | ---: |
| factor plus table | 2.125 | 1.000 |
| integer tabulation | 1.678 | 0.790 |

The 21.0% gain was accompanied by no stable memory regression: repeated RSS
changes stayed within 5%, while two controlled runs measured +1.4% to +1.8%
RSS and -1.2% to -2.6% peak footprint. All 100 `8 x 8` tables and their RDS
serializations were byte-identical.

Tests cover missing cells, repeated and reordered lambdas, first-class tie
resolution, custom and named labels, Unicode, and class counts from one to
100. Named label vectors are normalized with `as.character()` because the old
factor path discarded their names.
