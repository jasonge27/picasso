pycasso changelog
=================

1.1.0
-----

* Added multinomial lasso, MCP, and SCAD paths with a class-coupled active-set
  Proximal Newton/IRLS solver, strong screening, full KKT checks, and retained
  class labels.
* Added adaptive LLA stage budgets for MCP/SCAD binomial, Poisson,
  square-root-lasso, and multinomial fits. Three total stages remain the
  default maximum.
* Added opt-in ``fast_mode`` precision presets and automatic Gaussian
  residual/covariance backend selection.
* Added offsets, path assessment, confusion matrices, stratified
  cross-validation, ``dfmax``, lambda-value interpolation, and expanded
  prediction types.
* Added versioned termination and per-lambda diagnostics, including explicit
  Gaussian iteration-limit reporting with an atomic converged path prefix;
  also added native path-loss reuse, bounded temporary storage, stricter input
  validation, and cross-language source-mirror checks.
* Unified categorical classification on finite link-scale scores and aligned
  confusion matrices with R/glmnet: predicted classes are rows and observed
  classes are columns.
* Solver now owns its design input, rejects lossy complex/string coercion for
  numeric arrays, and validates Poisson counts as exact integers.
* Training and cross-validation now serialize per Solver instance, preventing
  concurrent native calls from replacing each other's output buffers while
  preserving parallelism across independent Solver instances.
* Cross-validation can now fit folds concurrently with the opt-in
  ``n_jobs`` argument. Serial execution remains the default; threaded workers
  preserve fold-order results and errors and are capped at the fold count.
* Python standardization now uses the final design as its workspace and a
  no-temporary square-norm reduction, reducing both peak memory and setup time
  for tall and wide dense inputs.
* Result dictionaries now retain the requested ``lla_max_stages`` value.
* Lambda-value support queries now select the nearest fitted path point,
  matching R and avoiding artificial support created by coefficient
  interpolation.
* Prediction, assessment, and confusion entry points now reject empty design
  matrices instead of returning empty predictions, NaN metrics, or all-zero
  confusion matrices.
* Binomial models now accept any two homogeneous numeric, string, or Boolean
  response levels, retain their class map, and accept either original labels
  or encoded 0/1 values in assessment and confusion operations.
* Plotting now requires a trained model and strictly validates its Boolean and
  feature-count controls; passing an existing axes does not import the
  optional Matplotlib dependency. Confusion matrices reject empty lambda
  selections.
* Categorical sequence validation now preserves Python scalar types before
  NumPy can stringify mixed missing, numeric, or complex values. Byte-string
  labels are rejected consistently, and multinomial retraining restores its
  private class map to public result metadata.
* Reduced the mandatory runtime dependency to NumPy. Matplotlib is now an
  optional ``plot`` extra, and the obsolete pickle-backed example data is no
  longer shipped.
* Packaging checks now reject stale native artifacts and verify every exported
  C API symbol from the native library installed in the built wheel.
* Added a runtime ``PICASSO_NATIVE_LIBRARY`` override and reproducible
  isolated-process multinomial output-buffer profiling. The profiler stages
  complete source checkpoints, injects one verified native build, and records
  original source-file hashes.

1.0.1 and earlier
-----------------

Initial Python interface for scalar Gaussian, binomial, Poisson, and
square-root-lasso paths.
