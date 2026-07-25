# Performance and Profiling Reports

Files in this directory are experiment snapshots, not unconditional
performance guarantees. Interpret every result with its recorded matrix shape,
penalty, precision, compiler, BLAS, hardware, and source state. Reports named
`before`, `after`, `current`, or `final` refer to the local A/B
comparison described by that report, not necessarily the present checkout.

## Current entry points

- [Fast-mode policy](fast_mode_report.md) documents the current
  family-specific precision presets and their calibration.
- [PICASSO versus glmnet](fast_mode_glmnet_comparison_report.md) is the most
  comprehensive repository dense-L1 comparison. It is a July 2026 snapshot and
  predates the current Gaussian automatic backend; its matched naive and
  covariance rows remain useful historical controls. Its archived
  `object_mb` values are invalid because glmnet retained the training matrix
  in its captured call; do not use them for memory claims.
- [Fast-mode benchmark driver](fast_mode_glmnet_benchmark.R) now uses the
  current Gaussian automatic backend within a fast-mode explicit-path
  protocol, strips captured calls before measuring returned objects, and
  records source/environment identifiers plus the resolved Gaussian backend.
  [Aggregate results](fast_mode_glmnet_results/) preserve the older snapshot
  and are not expected to match a new run.

## Optimization records

The focused reports document one retained or rejected implementation change:

- Scalar kernels: [weighted-norm cache](glm_weight_cache_report.md),
  [compact active sets](actgd_compact_report.md), and
  [square-root local-change cache](sqrt_local_change_report.md). The
  [R scalar borrowed-design view](r_scalar_borrowed_design_report.md) removes
  one `8*n*d` native copy for synchronous column-major R calls while retaining
  the owning Python path. The
  [R design-coercion pass](r_design_coercion_report.md) avoids another full
  copy when an input matrix is already double, while preserving the required
  integer-to-double conversion. The
  [ActGD direct output sink](actgd_output_sink_report.md) removes the retained
  `L * d` Gaussian C-API coefficient copy and verifies the change with
  fresh-process peak-RSS measurements and byte-identical output checks.
- Gaussian: [lazy covariance cache](gaussian_covariance_cache_report.md),
  [covariance objective evaluation](gaussian_covariance_eval_report.md), and
  [extreme standardization](gaussian_extreme_standardization_report.md).
- Nonconvex scalar paths:
  [adaptive LLA](scalar_adaptive_lla_report.md). The
  [ActNewton direct output sink](actnewton_output_sink_report.md) removes the
  retained `L * d` scalar C-API coefficient copy while preserving the public
  retained-path C++ behavior and old/new byte-identical outputs.
- Wrapper/path evaluation:
  [native R path loss](r_native_loss_report.md),
  [Python path evaluation](python_path_evaluation_report.md), and
  [Gaussian R preprocessing](gaussian_r_preprocessing_report.md). The
  [Gaussian R finalization refactor](r_gaussian_finalization_report.md)
  records a serialization-identical wrapper cleanup with an alternating,
  allocation-profiled A/B benchmark. The
  [Python scalar finalization buffer benchmark](python_scalar_finalize_buffer_report.md)
  isolates full-path in-place scaling and partial-path compact ownership with
  fresh processes, deterministic checksums, an explicitly verified native
  library, and recorded harness/runtime provenance. The
  [Python multinomial output-buffer benchmark](python_multinomial_output_buffer_report.md)
  runs each RSS measurement in a fresh process and can compare two Python
  source trees against the same explicitly selected native library. It stages
  each original source without bundled binaries or bytecode, injects exactly
  one native library, verifies the loaded path and SHA-256 in every worker,
  and records hashes for the four Python package identity files.
  The [Python multinomial assessment fusion](python_multinomial_assess_fusion_report.md)
  reuses one logits matrix for deviance and class error while retaining legacy
  softmax near-tie semantics; its interleaved A/B benchmark records exact
  output checksums, runtime, allocation peak, and fresh-process RSS delta.
  The [Python multinomial rescaling kernel](python_multinomial_rescale_report.md)
  replaces an 8 MiB blocked multiply temporary with BLAS matrix-vector
  products; full-array old-formula oracles check its last-bit reduction-order
  differences with a scale-aware fixture tolerance, and fresh-process
  measurements record runtime and memory.
  The [Python cross-validation threading benchmark](python_cv_parallel_report.md)
  records the opt-in fold pool across every family in tall and wide settings,
  including its memory tradeoff and unchanged serial default.
  The [Python scalar CV batched-scoring benchmark](python_cv_batched_scoring_report.md)
  records the separate 1 MiB predictor blocks and the exact binomial-class
  fallback. The
  [Python default-data validation benchmark](python_default_data_scan_report.md)
  measures removal of redundant scans of the owned training design.
  The [R multinomial streaming benchmark](r_multinomial_streaming_report.md)
  evaluates one lambda at a time in assessment, confusion matrices, and
  cross-validation while retaining legacy softmax tie behavior; it records
  serialization checksums, cumulative allocation, and fresh-process max RSS.
  The [R multinomial borrowed-design benchmark](r_multinomial_borrowed_design_report.md)
  measures the aligned column-major C-API view and its owning safety fallback.
  The [R scalar CV lambda-blocking benchmark](r_cv_lambda_blocking_report.md)
  records the conditional 8 MiB predictor path and its end-to-end RSS/runtime
  gate. The
  [R binomial confusion lambda-blocking benchmark](r_binomial_confusion_blocking_report.md)
  records the corresponding conditional path for confusion matrices, with
  serialization-identical tables and fresh-process memory measurements. The
  [R confusion-table counting benchmark](r_confusion_tabulation_report.md)
  isolates the exact integer-tabulation follow-up for both classification
  families.
- Multinomial kernels and solver:
  [ActNewton path](multinomial_actnewton_path_report.md),
  [path-state copy elimination](multinomial_state_copy_report.md),
  [curvature](multinomial_curvature_report.md),
  [exact KKT scans](multinomial_exact_kkt_scan_report.md),
  [compact active state](multinomial_compact_active_report.md),
  [adaptive LLA](multinomial_adaptive_lla_report.md), and
  [glmnet-oriented kernels](multinomial_glmnet_kernel_report.md).

The [round-two report](performance_round2_report.md) and
[earlier multinomial/glmnet comparison](multinomial_glmnet_performance_report.md)
use different fixtures, tolerances, and source states. They are historical
optimization records and must not be compared row-for-row with the fast-mode
report.

Use the source and raw result files named in each report when reproducing an
experiment. Temporary-file references are provenance notes, not durable
artifacts. A new publication-quality run should record the Git revision and
dirty state, native-library digest, CPU, BLAS, and thread environment. Do not
extrapolate L1 findings to MCP/SCAD or dense findings to sparse inputs without
a new benchmark.
