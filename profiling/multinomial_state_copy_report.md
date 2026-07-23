# Multinomial Path-State Copy Elimination

## Decision

**KEEP.** The nonconvex C-API path now lets the transactional path solver
update its call-local L1 master directly.  A later LLA or diagnostic failure
always returns from the C call, so the former pre-copy and post-commit copy
could not provide observable rollback.  The path solver also move-assigns its
fully validated local state.  This removes three deep `d x K` copies per
lambda without changing the public C++ path solver's failure atomicity.

## Correctness

Release and ASan/UBSan CTest suites both pass all 16 tests.  The existing path
state failure/retry test still proves that a failed proximal-Newton solve
leaves every public state field and its hidden smooth cache unchanged.  C-API
tests cover L1/MCP/SCAD, both layouts, adaptive LLA, failure prefixes, dfmax,
and diagnostic atomicity.  Every A/B workload below produced byte-identical
coefficients, intercepts, iteration counters, objectives, KKT diagnostics,
stationarity diagnostics, and smooth NLL values.

## Fresh-Process A/B Results

The benchmark used one BLAS thread and alternated old/new libraries across
fresh processes.  Caller-owned output buffers were allocated before the RSS
baseline.  The wide sparse cases use `n=8`, `d=100000`, `K=4`, 30 lambdas and
three LLA stages.  They deliberately produce zero coordinate updates, so they
isolate path-state management rather than represent general training speed.
The signal case uses `n=80`, `d=2000`, ten lambdas and 176,438 coordinate
updates.

| Workload | Repeats | Old | New | Speedup | Peak-RSS reduction |
|---|---:|---:|---:|---:|---:|
| L1 sparse path | 7 | 0.1589 s | 0.1594 s | 0.997x | 0.03 MB |
| MCP sparse path | 7 | 0.9589 s | 0.6781 s | 1.414x | 10.49 MB |
| SCAD sparse path | 7 | 0.9596 s | 0.6716 s | 1.429x | 10.49 MB |
| MCP nonzero path | 7 | 0.05162 s | 0.03288 s | 1.570x | 0.18 MB |

The L1-only final move is below measurement resolution in this workload.  The
large nonconvex gain comes from removing the two additional C-API master-state
copies at every lambda.  The nonzero MCP run returned status 10 in both arms
(the configured three-stage LLA stationarity limit), with the same fully
populated path and diagnostics.  Its 0.18 MB RSS difference is page-level
noise, so only the runtime result is interpreted.  These microbenchmarks do
not imply the same speedup for every dataset.

## Reproduction

Use `multinomial_state_copy_benchmark.py` with the before and after libraries
recorded in each JSON file.  Raw results are in:

- `multinomial_state_copy_results.json`
- `multinomial_state_copy_scad_results.json`
- `multinomial_state_copy_signal_results.json`
- `multinomial_state_copy_l1_results.json`

The script SHA-256 recorded by every run is
`498af533fadbacfa1b5cdc89668f8905ee5a4d009f5894518a1420121cff97ab`.
The before and validated-after snapshots are
`/private/tmp/picasso-multinomial-state-copy-before-20260719` and
`/private/tmp/picasso-multinomial-state-copy-after-validated-20260719`.
Their libraries have SHA-256 digests `ff2ed89678a841fee9a0d6321beb65abf91e496f9e545c8459bc9043fdfaf515`
and `a910807411ec4724cef43bbc1f930163cc800d400a871e7d799e26a0b8e2e5b8`,
respectively.  The benchmark also asserts identical status, fitted-path
length, failed-lambda/stage indices, and full diagnostic checksums.
