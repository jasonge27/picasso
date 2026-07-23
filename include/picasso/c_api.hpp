#ifndef PICASSO_C_API_H
#define PICASSO_C_API_H

#if defined(PICASSO_BUILDING_SHARED) && defined(PICASSO_USING_SHARED)
#  error "PICASSO_BUILDING_SHARED and PICASSO_USING_SHARED are mutually exclusive"
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(PICASSO_BUILDING_SHARED)
#    define PICASSO_C_API __declspec(dllexport)
#  elif defined(PICASSO_USING_SHARED)
#    define PICASSO_C_API __declspec(dllimport)
#  else
#    define PICASSO_C_API
#  endif
#elif defined(PICASSO_BUILDING_SHARED) && \
    (defined(__GNUC__) || defined(__clang__))
#  define PICASSO_C_API __attribute__((visibility("default")))
#else
#  define PICASSO_C_API
#endif

// Common termination status for versioned adaptive-LLA entry points.  The
// numeric values intentionally match the established multinomial ABI.
enum PicassoLlaPathStatus {
  PICASSO_LLA_COMPLETED = 0,
  PICASSO_LLA_DFMAX_REACHED = 1,
  PICASSO_LLA_INVALID_INPUT = 2,
  PICASSO_LLA_SUBPROBLEM_FAILED = 3,
  PICASSO_LLA_INNER_ITERATION_LIMIT = 4,
  PICASSO_LLA_LINE_SEARCH_FAILED = 5,
  PICASSO_LLA_NO_DESCENT_DIRECTION = 6,
  PICASSO_LLA_NUMERICAL_FAILURE = 7,
  PICASSO_LLA_MAJORIZATION_FAILED = 8,
  PICASSO_LLA_EXCEPTION = 9,
  PICASSO_LLA_STATIONARITY_LIMIT = 10
};

extern "C" PICASSO_C_API const char *PicassoLlaPathStatusString(int status);

// Scalar X layout and lifetime contract: usePython=false means an nn-by-dd
// column-major buffer; usePython=true means a row-major buffer. Calls are
// synchronous and never retain input pointers after returning. Callers must
// keep X readable and unchanged for the duration of the call.

extern "C" PICASSO_C_API void SolveLogisticRegression(
    double *Y,       // input: 0/1 model response
    double *X,       // input: model covariates
    int nn,          // input: number of samples
    int dd,          // input: dimension
    double *lambda,  // input: regularization parameter
    int nnlambda,    // input: number of lambda on the regularization path
    double gamma,    // input: gamma for SCAD or MCP penalty
    int mmax_ite,    // input: max number of interations
    double pprec,    // input: optimization precision
    int reg_type,    // input: type of regularization
    bool intercept,  // input: to have intercept term or not
    int dfmax,       // input: max nonzero coefficients for early stopping (-1 = no limit)
    double *offset,  // input: per-observation offset (length nn, may be nullptr)
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    int *num_fit,    // output: number of lambdas actually fit
    // default settings
    bool usePython = false
    );

extern "C" PICASSO_C_API void SolvePoissonRegression(
    double *Y,       // input: count model response
    double *X,       // input: model covariates
    int nn,          // input: number of samples
    int dd,          // input: dimension
    double *lambda,  // input: regularization parameter
    int nnlambda,    // input: number of lambda on the regularization path
    double gamma,    // input: gamma for SCAD or MCP penalty
    int mmax_ite,    // input: max number of interations
    double pprec,    // input: optimization precision
    int reg_type,    // input: type of regularization
    bool intercept,  // input: to have intercept term or not
    int dfmax,       // input: max nonzero coefficients for early stopping (-1 = no limit)
    double *offset,  // input: per-observation offset (length nn, may be nullptr)
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    int *num_fit,    // output: number of lambdas actually fit
    // default settings
    bool usePython = false
    );

extern "C" PICASSO_C_API void SolveSqrtLinearRegression(
    double *Y,       // input: 0/1 model response
    double *X,       // input: model covariates
    int nn,          // input: number of samples
    int dd,          // input: dimension
    double *lambda,  // input: regularization parameter
    int nnlambda,    // input: number of lambda on the regularization path
    double gamma,    // input: gamma for SCAD or MCP penalty
    int mmax_ite,    // input: max number of interations
    double pprec,    // input: optimization precision
    int reg_type,    // input: type of regularization
    bool intercept,  // input: to have intercept term or not
    int dfmax,       // input: max nonzero coefficients for early stopping (-1 = no limit)
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    int *num_fit,    // output: number of lambdas actually fit
    // default settings
    bool usePython = false
    );

// Versioned scalar adaptive-LLA APIs.  The original void entry points above
// remain ABI-compatible and use the default maximum of three total stages.
// lla_max_stages includes the initial L1 master and must be at least three.
// The per-lambda diagnostics have length nlambda.  A stationarity-limit status
// means every returned model is usable, although at least one nonconvex fit
// was not certified before exhausting the requested stage budget.
extern "C" PICASSO_C_API int SolveLogisticRegressionV2(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *offset, double *beta, double *intcpt, int *ite_lamb,
    int *size_act, double *runt, int *num_fit, bool usePython,
    int lla_max_stages,
    int *failed_lambda,   // scalar: zero-based first hard failure, else -1
    int *failed_stage,    // scalar: zero-based failed LLA stage, else -1
    int *lla_stages,      // completed total stages per lambda (includes L1)
    double *objective,    // final L1 or MCP/SCAD target objective
    double *kkt,          // final weighted-L1 subproblem KKT residual
    double *stationarity  // final target-penalty stationarity residual
    );

extern "C" PICASSO_C_API int SolvePoissonRegressionV2(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *offset, double *beta, double *intcpt, int *ite_lamb,
    int *size_act, double *runt, int *num_fit, bool usePython,
    int lla_max_stages, int *failed_lambda, int *failed_stage,
    int *lla_stages, double *objective, double *kkt, double *stationarity);

extern "C" PICASSO_C_API int SolveSqrtLinearRegressionV2(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, int lla_max_stages,
    int *failed_lambda, int *failed_stage, int *lla_stages,
    double *objective, double *kkt, double *stationarity);

// V3 scalar ActNewton APIs add the final unpenalized objective for every
// committed lambda while preserving the V1/V2 ABI.
extern "C" PICASSO_C_API int SolveLogisticRegressionV3(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *offset, double *beta, double *intcpt, int *ite_lamb,
    int *size_act, double *runt, int *num_fit, bool usePython,
    int lla_max_stages, int *failed_lambda, int *failed_stage,
    int *lla_stages, double *objective, double *kkt, double *stationarity,
    double *smooth_objective);

extern "C" PICASSO_C_API int SolvePoissonRegressionV3(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *offset, double *beta, double *intcpt, int *ite_lamb,
    int *size_act, double *runt, int *num_fit, bool usePython,
    int lla_max_stages, int *failed_lambda, int *failed_stage,
    int *lla_stages, double *objective, double *kkt, double *stationarity,
    double *smooth_objective);

extern "C" PICASSO_C_API int SolveSqrtLinearRegressionV3(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, int lla_max_stages,
    int *failed_lambda, int *failed_stage, int *lla_stages,
    double *objective, double *kkt, double *stationarity,
    double *smooth_objective);

extern "C" PICASSO_C_API void SolveLinearRegressionNaiveUpdate(
    double *Y,       // input: 0/1 model response
    double *X,       // input: model covariates
    int nn,          // input: number of samples
    int dd,          // input: dimension
    double *lambda,  // input: regularization parameter
    int nnlambda,    // input: number of lambda on the regularization path
    double gamma,    // input: gamma for SCAD or MCP penalty
    int mmax_ite,    // input: max number of interations
    double pprec,    // input: optimization precision
    int reg_type,    // input: type of regularization
    bool intercept,  // input: to have intercept term or not
    int dfmax,       // input: max nonzero coefficients for early stopping (-1 = no limit)
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    int *num_fit,    // output: number of lambdas actually fit
    // default settings
    bool usePython = false
    );

extern "C" PICASSO_C_API void SolveLinearRegressionCovUpdate(
    double *Y,       // input: model response
    double *X,       // input: model covariates
    int nn,          // input: number of samples
    int dd,          // input: dimension
    double *lambda,  // input: regularization parameter
    int nnlambda,    // input: number of lambda on the regularization path
    double gamma,    // input: gamma for SCAD or MCP penalty
    int mmax_ite,    // input: max number of interations
    double pprec,    // input: optimization precision
    int reg_type,    // input: type of regularization
    bool intercept,  // input: to have intercept term or not
    int dfmax,       // input: max nonzero coefficients for early stopping (-1 = no limit)
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    int *num_fit,    // output: number of lambdas actually fit
    // default settings
    bool usePython = false
    );

// Versioned Gaussian APIs return the per-lambda residual mean square already
// evaluated by ActGD. The original entry points remain ABI-compatible.
extern "C" PICASSO_C_API void SolveLinearRegressionNaiveUpdateV2(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, double *smooth_objective);

extern "C" PICASSO_C_API void SolveLinearRegressionCovUpdateV2(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, double *smooth_objective);

// V3 reports whether ActGD completed normally, crossed dfmax, or exhausted
// its existing iteration budget.  A failed lambda is never committed; the
// zero-based index is returned through failed_lambda, or -1 on usable exits.
// Status values reuse PicassoLlaPathStatus so all path solvers share one
// stable termination vocabulary.  V1/V2 signatures remain unchanged.
extern "C" PICASSO_C_API int SolveLinearRegressionNaiveUpdateV3(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, double *smooth_objective,
    int *failed_lambda);

extern "C" PICASSO_C_API int SolveLinearRegressionCovUpdateV3(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, double *smooth_objective,
    int *failed_lambda);


// Multinomial X layout and lifetime contract: usePython=false expects an
// n-by-d column-major buffer and may borrow it for the synchronous call;
// usePython=true expects a row-major buffer and converts it to owning storage.
// No input pointer is retained. Keep X readable and unchanged, without
// concurrent mutation, until the call returns.
extern "C" PICASSO_C_API void SolveMultinomialRegression(
    double *Y_int,   // input: finite integer class labels 0..K-1, length n
    double *X,       // input: model covariates (layout selected by usePython)
    int nn,          // input: number of samples
    int dd,          // input: dimension
    int num_classes, // input: number of classes K
    double *lambda,  // input: finite nonnegative regularization parameters
    int nnlambda,    // input: number of lambdas
    double gamma,    // input: MCP gamma > 1 or SCAD gamma > 2
    int mmax_ite,    // input: outer-Newton and inner-sweep limit
    double pprec,    // input: positive outer KKT tolerance
    int reg_type,    // input: 1=L1, 2=MCP, 3=SCAD
    bool intercept,  // input: include intercept
    int dfmax,       // input: max nonzero; crossing fit is retained
    double *beta,    // output: d * K * nlambda (for each lambda: K*d, col-major per class)
    double *intcpt,  // output: K * nlambda
    int *ite_lamb,   // output: iterations per lambda
    int *size_act,   // output: total nonzero per lambda
    double *runt,    // output: reserved per-lambda runtime (currently zero)
    int *num_fit,    // output: successfully fitted lambda prefix length
    bool usePython = false
    );

// Termination status returned by versioned multinomial entry points.  The
// original void ABI above is retained unchanged for existing R/Python clients.
enum PicassoMultinomialPathStatus {
  PICASSO_MULTINOMIAL_COMPLETED = 0,
  PICASSO_MULTINOMIAL_DFMAX_REACHED = 1,
  PICASSO_MULTINOMIAL_INVALID_INPUT = 2,
  PICASSO_MULTINOMIAL_OUTER_ITERATION_LIMIT = 3,
  PICASSO_MULTINOMIAL_INNER_ITERATION_LIMIT = 4,
  PICASSO_MULTINOMIAL_LINE_SEARCH_FAILED = 5,
  PICASSO_MULTINOMIAL_NO_DESCENT_DIRECTION = 6,
  PICASSO_MULTINOMIAL_NUMERICAL_FAILURE = 7,
  PICASSO_MULTINOMIAL_LLA_MAJORIZATION_FAILED = 8,
  PICASSO_MULTINOMIAL_EXCEPTION = 9,
  PICASSO_MULTINOMIAL_LLA_STATIONARITY_LIMIT = 10
};

extern "C" PICASSO_C_API const char *PicassoMultinomialPathStatusString(
    int status);

// Versioned multinomial ABI with explicit termination and per-lambda
// diagnostics.  The first 20 arguments and coefficient layout are identical
// to SolveMultinomialRegression.  Successful lambda points are still
// committed atomically; diagnostics may additionally describe the first
// failed point at index *failed_lambda.
extern "C" PICASSO_C_API int SolveMultinomialRegressionV2(
    double *Y_int, double *X, int nn, int dd, int num_classes,
    double *lambda, int nnlambda, double gamma, int mmax_ite,
    double pprec, int reg_type, bool intercept, int dfmax,
    double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython,
    int *failed_lambda,       // scalar output: zero-based failure index, else -1
    int *failed_stage,        // zero-based failed LLA stage, else -1
    int *outer_ite,           // length nlambda; includes a failed point
    long long *inner_sweeps,  // length nlambda; 64-bit diagnostic count
    long long *coordinate_updates,  // length nlambda
    double *objective,        // L1 composite or MCP/SCAD target objective
    double *kkt,              // final weighted-L1 subproblem KKT residual
    double *stationarity      // L1 KKT or MCP/SCAD target stationarity
    );

// V3 adds a public adaptive-LLA stage budget without changing the V2 ABI.
// lla_max_stages counts the L1 master as stage zero and must be at least 3.
// Status LLA_STATIONARITY_LIMIT means the full path contains valid models but
// at least one MCP/SCAD point exhausted this budget before certification.
extern "C" PICASSO_C_API int SolveMultinomialRegressionV3(
    double *Y_int, double *X, int nn, int dd, int num_classes,
    double *lambda, int nnlambda, double gamma, int mmax_ite,
    double pprec, int reg_type, bool intercept, int dfmax,
    double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, int lla_max_stages,
    int *failed_lambda, int *failed_stage, int *outer_ite,
    long long *inner_sweeps, long long *coordinate_updates,
    double *objective, double *kkt, double *stationarity);

// V4 makes glmnet-style multinomial path termination opt-in.  V1--V3 always
// fit the complete requested path (unless dfmax or a solver failure stops it),
// preserving their historical explicit-path contract.  When path_early_stop
// is true, V4 may return a successfully fitted prefix after at least five
// points if explained deviance exceeds 0.999 or its consecutive-point gain is
// below 1e-5.
extern "C" PICASSO_C_API int SolveMultinomialRegressionV4(
    double *Y_int, double *X, int nn, int dd, int num_classes,
    double *lambda, int nnlambda, double gamma, int mmax_ite,
    double pprec, int reg_type, bool intercept, int dfmax,
    double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, int lla_max_stages,
    bool path_early_stop, int *failed_lambda, int *failed_stage,
    int *outer_ite, long long *inner_sweeps,
    long long *coordinate_updates, double *objective, double *kkt,
    double *stationarity);

// V5 additionally exposes the final smooth multinomial objective (mean
// negative log-likelihood) for every committed lambda.  Earlier ABI versions
// remain unchanged; uncommitted path slots are initialized to NaN.
extern "C" PICASSO_C_API int SolveMultinomialRegressionV5(
    double *Y_int, double *X, int nn, int dd, int num_classes,
    double *lambda, int nnlambda, double gamma, int mmax_ite,
    double pprec, int reg_type, bool intercept, int dfmax,
    double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, int lla_max_stages,
    bool path_early_stop, int *failed_lambda, int *failed_stage,
    int *outer_ite, long long *inner_sweeps,
    long long *coordinate_updates, double *objective, double *kkt,
    double *stationarity, double *smooth_nll);

#endif  // PICASSO_C_API_H
