#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <picasso/c_api.hpp>
#include <cmath>

// Helper: create a named list from components
static SEXP make_result_list(SEXP beta, SEXP intcpt, SEXP ite_lamb,
                             SEXP size_act, SEXP runt, SEXP num_fit) {
  const char *names[] = {"beta", "intcpt", "ite_lamb", "size_act", "runt",
                         "num_fit", ""};
  SEXP result = PROTECT(Rf_mkNamed(VECSXP, names));
  SET_VECTOR_ELT(result, 0, beta);
  SET_VECTOR_ELT(result, 1, intcpt);
  SET_VECTOR_ELT(result, 2, ite_lamb);
  SET_VECTOR_ELT(result, 3, size_act);
  SET_VECTOR_ELT(result, 4, runt);
  SET_VECTOR_ELT(result, 5, num_fit);
  UNPROTECT(1);
  return result;
}

extern "C" SEXP picasso_gaussian_naive_call(
    SEXP Y_sexp, SEXP X_sexp, SEXP n_sexp, SEXP d_sexp,
    SEXP lambda_sexp, SEXP nlambda_sexp, SEXP gamma_sexp,
    SEXP max_ite_sexp, SEXP prec_sexp, SEXP reg_type_sexp,
    SEXP intercept_sexp, SEXP dfmax_sexp) {
  int n = Rf_asInteger(n_sexp);
  int d = Rf_asInteger(d_sexp);
  int nlambda = Rf_asInteger(nlambda_sexp);

  // Allocate outputs — written into directly by C++, no copy back
  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)d * nlambda));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));

  // Zero-initialize outputs
  memset(REAL(beta_sexp), 0, sizeof(double) * (size_t)d * nlambda);
  memset(REAL(intcpt_sexp), 0, sizeof(double) * nlambda);
  memset(INTEGER(ite_sexp), 0, sizeof(int) * nlambda);
  memset(INTEGER(size_sexp), 0, sizeof(int) * nlambda);
  memset(REAL(runt_sexp), 0, sizeof(double) * nlambda);
  INTEGER(nfit_sexp)[0] = 0;

  // Call solver — REAL(X_sexp) is a direct pointer, no copy
  SolveLinearRegressionNaiveUpdate(
      REAL(Y_sexp), REAL(X_sexp), n, d,
      REAL(lambda_sexp), nlambda,
      Rf_asReal(gamma_sexp), Rf_asInteger(max_ite_sexp),
      Rf_asReal(prec_sexp), Rf_asInteger(reg_type_sexp),
      Rf_asInteger(intercept_sexp), Rf_asInteger(dfmax_sexp),
      REAL(beta_sexp), REAL(intcpt_sexp),
      INTEGER(ite_sexp), INTEGER(size_sexp),
      REAL(runt_sexp), INTEGER(nfit_sexp));

  SEXP result = make_result_list(beta_sexp, intcpt_sexp, ite_sexp,
                                 size_sexp, runt_sexp, nfit_sexp);
  UNPROTECT(6);
  return result;
}

extern "C" SEXP picasso_gaussian_cov_call(
    SEXP Y_sexp, SEXP X_sexp, SEXP n_sexp, SEXP d_sexp,
    SEXP lambda_sexp, SEXP nlambda_sexp, SEXP gamma_sexp,
    SEXP max_ite_sexp, SEXP prec_sexp, SEXP reg_type_sexp,
    SEXP intercept_sexp, SEXP dfmax_sexp) {
  int n = Rf_asInteger(n_sexp);
  int d = Rf_asInteger(d_sexp);
  int nlambda = Rf_asInteger(nlambda_sexp);

  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)d * nlambda));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));

  memset(REAL(beta_sexp), 0, sizeof(double) * (size_t)d * nlambda);
  memset(REAL(intcpt_sexp), 0, sizeof(double) * nlambda);
  memset(INTEGER(ite_sexp), 0, sizeof(int) * nlambda);
  memset(INTEGER(size_sexp), 0, sizeof(int) * nlambda);
  memset(REAL(runt_sexp), 0, sizeof(double) * nlambda);
  INTEGER(nfit_sexp)[0] = 0;

  SolveLinearRegressionCovUpdate(
      REAL(Y_sexp), REAL(X_sexp), n, d,
      REAL(lambda_sexp), nlambda,
      Rf_asReal(gamma_sexp), Rf_asInteger(max_ite_sexp),
      Rf_asReal(prec_sexp), Rf_asInteger(reg_type_sexp),
      Rf_asInteger(intercept_sexp), Rf_asInteger(dfmax_sexp),
      REAL(beta_sexp), REAL(intcpt_sexp),
      INTEGER(ite_sexp), INTEGER(size_sexp),
      REAL(runt_sexp), INTEGER(nfit_sexp));

  SEXP result = make_result_list(beta_sexp, intcpt_sexp, ite_sexp,
                                 size_sexp, runt_sexp, nfit_sexp);
  UNPROTECT(6);
  return result;
}

extern "C" SEXP picasso_logit_call(
    SEXP Y_sexp, SEXP X_sexp, SEXP n_sexp, SEXP d_sexp,
    SEXP lambda_sexp, SEXP nlambda_sexp, SEXP gamma_sexp,
    SEXP max_ite_sexp, SEXP prec_sexp, SEXP reg_type_sexp,
    SEXP intercept_sexp, SEXP dfmax_sexp, SEXP offset_sexp) {
  int n = Rf_asInteger(n_sexp);
  int d = Rf_asInteger(d_sexp);
  int nlambda = Rf_asInteger(nlambda_sexp);

  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)d * nlambda));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));

  memset(REAL(beta_sexp), 0, sizeof(double) * (size_t)d * nlambda);
  memset(REAL(intcpt_sexp), 0, sizeof(double) * nlambda);
  memset(INTEGER(ite_sexp), 0, sizeof(int) * nlambda);
  memset(INTEGER(size_sexp), 0, sizeof(int) * nlambda);
  memset(REAL(runt_sexp), 0, sizeof(double) * nlambda);
  INTEGER(nfit_sexp)[0] = 0;

  SolveLogisticRegression(
      REAL(Y_sexp), REAL(X_sexp), n, d,
      REAL(lambda_sexp), nlambda,
      Rf_asReal(gamma_sexp), Rf_asInteger(max_ite_sexp),
      Rf_asReal(prec_sexp), Rf_asInteger(reg_type_sexp),
      Rf_asInteger(intercept_sexp), Rf_asInteger(dfmax_sexp),
      REAL(offset_sexp),
      REAL(beta_sexp), REAL(intcpt_sexp),
      INTEGER(ite_sexp), INTEGER(size_sexp),
      REAL(runt_sexp), INTEGER(nfit_sexp));

  SEXP result = make_result_list(beta_sexp, intcpt_sexp, ite_sexp,
                                 size_sexp, runt_sexp, nfit_sexp);
  UNPROTECT(6);
  return result;
}

extern "C" SEXP picasso_poisson_call(
    SEXP Y_sexp, SEXP X_sexp, SEXP n_sexp, SEXP d_sexp,
    SEXP lambda_sexp, SEXP nlambda_sexp, SEXP gamma_sexp,
    SEXP max_ite_sexp, SEXP prec_sexp, SEXP reg_type_sexp,
    SEXP intercept_sexp, SEXP dfmax_sexp, SEXP offset_sexp) {
  int n = Rf_asInteger(n_sexp);
  int d = Rf_asInteger(d_sexp);
  int nlambda = Rf_asInteger(nlambda_sexp);

  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)d * nlambda));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));

  memset(REAL(beta_sexp), 0, sizeof(double) * (size_t)d * nlambda);
  memset(REAL(intcpt_sexp), 0, sizeof(double) * nlambda);
  memset(INTEGER(ite_sexp), 0, sizeof(int) * nlambda);
  memset(INTEGER(size_sexp), 0, sizeof(int) * nlambda);
  memset(REAL(runt_sexp), 0, sizeof(double) * nlambda);
  INTEGER(nfit_sexp)[0] = 0;

  SolvePoissonRegression(
      REAL(Y_sexp), REAL(X_sexp), n, d,
      REAL(lambda_sexp), nlambda,
      Rf_asReal(gamma_sexp), Rf_asInteger(max_ite_sexp),
      Rf_asReal(prec_sexp), Rf_asInteger(reg_type_sexp),
      Rf_asInteger(intercept_sexp), Rf_asInteger(dfmax_sexp),
      REAL(offset_sexp),
      REAL(beta_sexp), REAL(intcpt_sexp),
      INTEGER(ite_sexp), INTEGER(size_sexp),
      REAL(runt_sexp), INTEGER(nfit_sexp));

  SEXP result = make_result_list(beta_sexp, intcpt_sexp, ite_sexp,
                                 size_sexp, runt_sexp, nfit_sexp);
  UNPROTECT(6);
  return result;
}

extern "C" SEXP picasso_sqrtlasso_call(
    SEXP Y_sexp, SEXP X_sexp, SEXP n_sexp, SEXP d_sexp,
    SEXP lambda_sexp, SEXP nlambda_sexp, SEXP gamma_sexp,
    SEXP max_ite_sexp, SEXP prec_sexp, SEXP reg_type_sexp,
    SEXP intercept_sexp, SEXP dfmax_sexp) {
  int n = Rf_asInteger(n_sexp);
  int d = Rf_asInteger(d_sexp);
  int nlambda = Rf_asInteger(nlambda_sexp);

  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)d * nlambda));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));

  memset(REAL(beta_sexp), 0, sizeof(double) * (size_t)d * nlambda);
  memset(REAL(intcpt_sexp), 0, sizeof(double) * nlambda);
  memset(INTEGER(ite_sexp), 0, sizeof(int) * nlambda);
  memset(INTEGER(size_sexp), 0, sizeof(int) * nlambda);
  memset(REAL(runt_sexp), 0, sizeof(double) * nlambda);
  INTEGER(nfit_sexp)[0] = 0;

  SolveSqrtLinearRegression(
      REAL(Y_sexp), REAL(X_sexp), n, d,
      REAL(lambda_sexp), nlambda,
      Rf_asReal(gamma_sexp), Rf_asInteger(max_ite_sexp),
      Rf_asReal(prec_sexp), Rf_asInteger(reg_type_sexp),
      Rf_asInteger(intercept_sexp), Rf_asInteger(dfmax_sexp),
      REAL(beta_sexp), REAL(intcpt_sexp),
      INTEGER(ite_sexp), INTEGER(size_sexp),
      REAL(runt_sexp), INTEGER(nfit_sexp));

  SEXP result = make_result_list(beta_sexp, intcpt_sexp, ite_sexp,
                                 size_sexp, runt_sexp, nfit_sexp);
  UNPROTECT(6);
  return result;
}

extern "C" SEXP picasso_standardize_call(SEXP X_sexp, SEXP n_sexp,
                                         SEXP d_sexp) {
  int n = Rf_asInteger(n_sexp);
  int d = Rf_asInteger(d_sexp);

  SEXP xx_sexp = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)n * d));
  SEXP xm_sexp = PROTECT(Rf_allocVector(REALSXP, d));
  SEXP xinvc_sexp = PROTECT(Rf_allocVector(REALSXP, d));

  double *X = REAL(X_sexp);
  double *xx = REAL(xx_sexp);
  double *xm = REAL(xm_sexp);
  double *xinvc = REAL(xinvc_sexp);

  for (int j = 0; j < d; j++) {
    int jn = j * n;
    xm[j] = 0;
    for (int i = 0; i < n; i++) xm[j] += X[jn + i];
    xm[j] = xm[j] / n;
    for (int i = 0; i < n; i++) xx[jn + i] = X[jn + i] - xm[j];

    xinvc[j] = 0;
    for (int i = 0; i < n; i++) xinvc[j] += xx[jn + i] * xx[jn + i];

    if (xinvc[j] > 0) {
      xinvc[j] = 1.0 / sqrt(xinvc[j] / (n - 1));
      for (int i = 0; i < n; i++) xx[jn + i] = xx[jn + i] * xinvc[j];
    }
  }

  const char *names[] = {"xx", "xm", "xinvc", ""};
  SEXP result = PROTECT(Rf_mkNamed(VECSXP, names));
  SET_VECTOR_ELT(result, 0, xx_sexp);
  SET_VECTOR_ELT(result, 1, xm_sexp);
  SET_VECTOR_ELT(result, 2, xinvc_sexp);
  UNPROTECT(4);
  return result;
}

extern "C" SEXP picasso_multinomial_call(
    SEXP Y_sexp, SEXP X_sexp, SEXP n_sexp, SEXP d_sexp, SEXP K_sexp,
    SEXP lambda_sexp, SEXP nlambda_sexp, SEXP gamma_sexp,
    SEXP max_ite_sexp, SEXP prec_sexp, SEXP reg_type_sexp,
    SEXP intercept_sexp, SEXP dfmax_sexp) {
  int n       = Rf_asInteger(n_sexp);
  int d       = Rf_asInteger(d_sexp);
  int K       = Rf_asInteger(K_sexp);
  int nlambda = Rf_asInteger(nlambda_sexp);

  // beta: d * K * nlambda;  intcpt: K * nlambda
  SEXP beta_sexp   = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)d * K * nlambda));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)K * nlambda));
  SEXP ite_sexp    = PROTECT(Rf_allocVector(INTSXP,  nlambda));
  SEXP size_sexp   = PROTECT(Rf_allocVector(INTSXP,  nlambda));
  SEXP runt_sexp   = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp   = PROTECT(Rf_allocVector(INTSXP,  1));

  memset(REAL(beta_sexp),    0, sizeof(double) * (size_t)d * K * nlambda);
  memset(REAL(intcpt_sexp),  0, sizeof(double) * (size_t)K * nlambda);
  memset(INTEGER(ite_sexp),  0, sizeof(int)    * nlambda);
  memset(INTEGER(size_sexp), 0, sizeof(int)    * nlambda);
  memset(REAL(runt_sexp),    0, sizeof(double) * nlambda);
  INTEGER(nfit_sexp)[0] = 0;

  SolveMultinomialRegression(
      REAL(Y_sexp), REAL(X_sexp), n, d, K,
      REAL(lambda_sexp), nlambda,
      Rf_asReal(gamma_sexp), Rf_asInteger(max_ite_sexp),
      Rf_asReal(prec_sexp), Rf_asInteger(reg_type_sexp),
      Rf_asInteger(intercept_sexp), Rf_asInteger(dfmax_sexp),
      REAL(beta_sexp), REAL(intcpt_sexp),
      INTEGER(ite_sexp), INTEGER(size_sexp),
      REAL(runt_sexp), INTEGER(nfit_sexp));

  SEXP result = make_result_list(beta_sexp, intcpt_sexp, ite_sexp,
                                 size_sexp, runt_sexp, nfit_sexp);
  UNPROTECT(6);
  return result;
}

// Registration
static const R_CallMethodDef CallEntries[] = {
    {"picasso_gaussian_naive_call", (DL_FUNC)&picasso_gaussian_naive_call, 12},
    {"picasso_gaussian_cov_call", (DL_FUNC)&picasso_gaussian_cov_call, 12},
    {"picasso_logit_call", (DL_FUNC)&picasso_logit_call, 13},
    {"picasso_poisson_call", (DL_FUNC)&picasso_poisson_call, 13},
    {"picasso_sqrtlasso_call", (DL_FUNC)&picasso_sqrtlasso_call, 12},
    {"picasso_standardize_call", (DL_FUNC)&picasso_standardize_call, 3},
    {"picasso_multinomial_call", (DL_FUNC)&picasso_multinomial_call, 13},
    {NULL, NULL, 0}};

void R_init_picasso(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
