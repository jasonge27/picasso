# Multinomial Curvature Report

The fixed-IRLS quadratic previously computed coordinate curvatures for every
feature even when a restricted path solve could visit only a small active set.
The retained hybrid computes only active-feature curvatures when at most half
of the features are active.  Denser and unrestricted solves keep the original
kernel, so inner reactivation remains unchanged.  The implementation stores no
`X^2` matrix and adds no solver allocation.

Release binaries from the complete pre-change backup and the candidate tree
were measured in an ABBA sequence.  Each table entry is the median of six fresh
processes and covers a 24-point L1 path.

| Case | Before | After | Speedup | RSS before | RSS after |
|---|---:|---:|---:|---:|---:|
| sparse K=3 | 0.181910 s | 0.131270 s | 1.386x | 9.66 MiB | 9.70 MiB |
| wide K=4 | 0.165156 s | 0.083061 s | 1.988x | 9.11 MiB | 9.11 MiB |
| high K=12 | 0.158076 s | 0.076658 s | 2.062x | 5.27 MiB | 5.26 MiB |
| dense control | 0.224552 s | 0.220880 s | 1.017x | 2.14 MiB | 2.17 MiB |

The first active-only prototype was rolled forward with the 50% density gate
because it slowed the dense control by 3.2%.  The gated version removes that
regression.  All outer iterations, inner sweeps, coordinate visits, active-set
events, statuses, support, and probability digests match the backup.  Reported
KKT differences are at floating-point reduction noise scale.  Native
multinomial tests, the targeted ASan/UBSan test, all R tests, and the Python
feature suite pass.

The complete baseline is
`/private/tmp/picasso-master-baseline-20260716-preincremental`; direct source
backups are `/private/tmp/picasso-step3-*.before`.
