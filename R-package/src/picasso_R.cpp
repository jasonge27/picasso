#include <R.h>
#include <picasso/c_api.hpp>

extern "C" void picasso_logit_solver(
    double* Y,       // input: 0/1 model response
    double* X,       // input: model covariates
    int* nn,         // input: number of samples
    int* dd,         // input: dimension
    double* lambda,  // input: regularization parameter
    int* nnlambda,   // input: number of lambda on the regularization path
    double* gamma,   // input: gamma for SCAD or MCP penalty
    int* mmax_ite,   // input: max number of interations
    double* pprec,   // input: optimization precision
    int* reg_type,   // input: type of regularization
    int* include_intercept,  // input: to have intercept term or not
    double* beta,            // output: an nlambda * d dim matrix
                             //         saving the coefficients for each lambda
    double* intcpt,          // output: an nlambda dim array
    //         saving the model intercept for each lambda
    int* ite_lamb,  // output: number of iterations for each lambda
    int* size_act,  // output: an array of solution sparsity (model df)
    double* runt    // output: runtime
    ) {
  SolveLogisticRegression(Y, X, *nn, *dd, lambda, *nnlambda, *gamma, *mmax_ite,
                          *pprec, *reg_type, *include_intercept, beta, intcpt,
                          ite_lamb, size_act, runt);
}

extern "C" void picasso_sqrt_lasso_solver(
    double* Y,       // input: 0/1 model response
    double* X,       // input: model covariates
    int* nn,         // input: number of samples
    int* dd,         // input: dimension
    double* lambda,  // input: regularization parameter
    int* nnlambda,   // input: number of lambda on the regularization path
    double* gamma,   // input: gamma for SCAD or MCP penalty
    int* mmax_ite,   // input: max number of interations
    double* pprec,   // input: optimization precision
    int* reg_type,   // input: type of regularization
    int* include_intercept,  // input: to have intercept term or not
    double* beta,            // output: an nlambda * d dim matrix
                             //         saving the coefficients for each lambda
    double* intcpt,          // output: an nlambda dim array
    //         saving the model intercept for each lambda
    int* ite_lamb,  // output: number of iterations for each lambda
    int* size_act,  // output: an array of solution sparsity (model df)
    double* runt    // output: runtime
    ) {
  SolveSqrtLinearRegression(Y, X, *nn, *dd, lambda, *nnlambda, *gamma,
                            *mmax_ite, *pprec, *reg_type, *include_intercept,
                            beta, intcpt, ite_lamb, size_act, runt);
}

extern "C" void picasso_poisson_solver(
    double* Y,       // input: 0/1 model response
    double* X,       // input: model covariates
    int* nn,         // input: number of samples
    int* dd,         // input: dimension
    double* lambda,  // input: regularization parameter
    int* nnlambda,   // input: number of lambda on the regularization path
    double* gamma,   // input: gamma for SCAD or MCP penalty
    int* mmax_ite,   // input: max number of interations
    double* pprec,   // input: optimization precision
    int* reg_type,   // input: type of regularization
    int* include_intercept,  // input: to have intercept term or not
    double* beta,            // output: an nlambda * d dim matrix
                             //         saving the coefficients for each lambda
    double* intcpt,          // output: an nlambda dim array
    //         saving the model intercept for each lambda
    int* ite_lamb,  // output: number of iterations for each lambda
    int* size_act,  // output: an array of solution sparsity (model df)
    double* runt    // output: runtime
    ) {
  // call picasso c api
  SolvePoissonRegression(Y, X, *nn, *dd, lambda, *nnlambda, *gamma, *mmax_ite,
                         *pprec, *reg_type, *include_intercept, beta, intcpt,
                         ite_lamb, size_act, runt);
}

extern "C" void picasso_gaussian_cov(
    double* Y,       // input: 0/1 model response
    double* X,       // input: model covariates
    int* nn,         // input: number of samples
    int* dd,         // input: dimension
    double* lambda,  // input: regularization parameter
    int* nnlambda,   // input: number of lambda on the regularization path
    double* gamma,   // input: gamma for SCAD or MCP penalty
    int* mmax_ite,   // input: max number of interations
    double* pprec,   // input: optimization precision
    int* reg_type,   // input: type of regularization
    int* include_intercept,  // input: to have intercept term or not
    double* beta,            // output: an nlambda * d dim matrix
                             //         saving the coefficients for each lambda
    double* intcpt,          // output: an nlambda dim array
    //         saving the model intercept for each lambda
    int* ite_lamb,  // output: number of iterations for each lambda
    int* size_act,  // output: an array of solution sparsity (model df)
    double* runt    // output: runtime
    ) {
  SolveLinearRegressionCovUpdate(
      Y, X, *nn, *dd, lambda, *nnlambda, *gamma, *mmax_ite, *pprec, *reg_type,
      *include_intercept, beta, intcpt, ite_lamb, size_act, runt);
}

extern "C" void picasso_gaussian_naive(
    double* Y,       // input: 0/1 model response
    double* X,       // input: model covariates
    int* nn,         // input: number of samples
    int* dd,         // input: dimension
    double* lambda,  // input: regularization parameter
    int* nnlambda,   // input: number of lambda on the regularization path
    double* gamma,   // input: gamma for SCAD or MCP penalty
    int* mmax_ite,   // input: max number of interations
    double* pprec,   // input: optimization precision
    int* reg_type,   // input: type of regularization
    int* include_intercept,  // input: to have intercept term or not
    double* beta,            // output: an nlambda * d dim matrix
                             //         saving the coefficients for each lambda
    double* intcpt,          // output: an nlambda dim array
    //         saving the model intercept for each lambda
    int* ite_lamb,  // output: number of iterations for each lambda
    int* size_act,  // output: an array of solution sparsity (model df)
    double* runt    // output: runtime
    ) {
  SolveLinearRegressionNaiveUpdate(
      Y, X, *nn, *dd, lambda, *nnlambda, *gamma, *mmax_ite, *pprec, *reg_type,
      *include_intercept, beta, intcpt, ite_lamb, size_act, runt);
}

extern "C" void standardize_design(double* X, double* xx, double* xm,
                                   double* xinvc, int* nn, int* dd) {
  int i, j, jn, n, d;

  n = *nn;
  d = *dd;

  for (j = 0; j < d; j++) {
    // Center
    xm[j] = 0;
    jn = j * n;
    for (i = 0; i < n; i++) xm[j] += X[jn + i];

    xm[j] = xm[j] / n;
    for (i = 0; i < n; i++) xx[jn + i] = X[jn + i] - xm[j];

    // Scale
    xinvc[j] = 0;
    for (i = 0; i < n; i++) {
      xinvc[j] += xx[jn + i] * xx[jn + i];
    }

    if (xinvc[j] > 0) {
      xinvc[j] = 1 / sqrt(xinvc[j] / (n - 1));
      for (i = 0; i < n; i++) {
        xx[jn + i] = xx[jn + i] * xinvc[j];
      }
    }
  }
}
