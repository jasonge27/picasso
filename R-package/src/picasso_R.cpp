#include <cmath>
#include <cstddef>
#include <limits>

#include <picasso/c_api.hpp>

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

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

static SEXP make_gaussian_result_list(
    SEXP beta, SEXP intcpt, SEXP ite_lamb, SEXP size_act, SEXP runt,
    SEXP num_fit, SEXP smooth_objective, SEXP status,
    SEXP failed_lambda) {
  const char *names[] = {"beta", "intcpt", "ite_lamb", "size_act", "runt",
                         "num_fit", "smooth_objective", "status",
                         "failed_lambda", ""};
  SEXP result = PROTECT(Rf_mkNamed(VECSXP, names));
  SET_VECTOR_ELT(result, 0, beta);
  SET_VECTOR_ELT(result, 1, intcpt);
  SET_VECTOR_ELT(result, 2, ite_lamb);
  SET_VECTOR_ELT(result, 3, size_act);
  SET_VECTOR_ELT(result, 4, runt);
  SET_VECTOR_ELT(result, 5, num_fit);
  SET_VECTOR_ELT(result, 6, smooth_objective);
  SET_VECTOR_ELT(result, 7, status);
  SET_VECTOR_ELT(result, 8, failed_lambda);
  UNPROTECT(1);
  return result;
}

static SEXP make_scalar_lla_result_list(
    SEXP beta, SEXP intcpt, SEXP ite_lamb, SEXP size_act, SEXP runt,
    SEXP num_fit, SEXP status, SEXP failed_lambda, SEXP failed_stage,
    SEXP lla_stages, SEXP objective, SEXP kkt, SEXP stationarity,
    SEXP smooth_objective) {
  const char *names[] = {
      "beta", "intcpt", "ite_lamb", "size_act", "runt", "num_fit",
      "status", "failed_lambda", "failed_stage", "lla_stages",
      "objective", "kkt", "stationarity", "smooth_objective", ""};
  SEXP result = PROTECT(Rf_mkNamed(VECSXP, names));
  SET_VECTOR_ELT(result, 0, beta);
  SET_VECTOR_ELT(result, 1, intcpt);
  SET_VECTOR_ELT(result, 2, ite_lamb);
  SET_VECTOR_ELT(result, 3, size_act);
  SET_VECTOR_ELT(result, 4, runt);
  SET_VECTOR_ELT(result, 5, num_fit);
  SET_VECTOR_ELT(result, 6, status);
  SET_VECTOR_ELT(result, 7, failed_lambda);
  SET_VECTOR_ELT(result, 8, failed_stage);
  SET_VECTOR_ELT(result, 9, lla_stages);
  SET_VECTOR_ELT(result, 10, objective);
  SET_VECTOR_ELT(result, 11, kkt);
  SET_VECTOR_ELT(result, 12, stationarity);
  SET_VECTOR_ELT(result, 13, smooth_objective);
  UNPROTECT(1);
  return result;
}

static SEXP make_multinomial_result_list(
    SEXP beta, SEXP intcpt, SEXP ite_lamb, SEXP size_act, SEXP runt,
    SEXP num_fit, SEXP status, SEXP failed_lambda, SEXP failed_stage,
    SEXP outer_ite, SEXP inner_sweeps, SEXP coordinate_updates,
    SEXP objective, SEXP kkt, SEXP stationarity, SEXP smooth_nll) {
  const char *names[] = {
      "beta", "intcpt", "ite_lamb", "size_act", "runt", "num_fit",
      "status", "failed_lambda", "failed_stage", "outer_ite",
      "inner_sweeps", "coordinate_updates", "objective", "kkt",
      "stationarity", "smooth_nll", ""};
  SEXP result = PROTECT(Rf_mkNamed(VECSXP, names));
  SET_VECTOR_ELT(result, 0, beta);
  SET_VECTOR_ELT(result, 1, intcpt);
  SET_VECTOR_ELT(result, 2, ite_lamb);
  SET_VECTOR_ELT(result, 3, size_act);
  SET_VECTOR_ELT(result, 4, runt);
  SET_VECTOR_ELT(result, 5, num_fit);
  SET_VECTOR_ELT(result, 6, status);
  SET_VECTOR_ELT(result, 7, failed_lambda);
  SET_VECTOR_ELT(result, 8, failed_stage);
  SET_VECTOR_ELT(result, 9, outer_ite);
  SET_VECTOR_ELT(result, 10, inner_sweeps);
  SET_VECTOR_ELT(result, 11, coordinate_updates);
  SET_VECTOR_ELT(result, 12, objective);
  SET_VECTOR_ELT(result, 13, kkt);
  SET_VECTOR_ELT(result, 14, stationarity);
  SET_VECTOR_ELT(result, 15, smooth_nll);
  UNPROTECT(1);
  return result;
}

static bool checked_r_length_product(R_xlen_t left, R_xlen_t right,
                                     R_xlen_t *product) {
  if (product == NULL || left < 0 || right < 0) return false;
  if (left != 0 && right > R_XLEN_T_MAX / left) return false;
  *product = left * right;
  return true;
}

static R_xlen_t validate_scalar_bridge_inputs(
    const char *family, SEXP Y_sexp, SEXP X_sexp, SEXP lambda_sexp,
    SEXP offset_sexp, bool needs_offset, int n, int d, int nlambda) {
  if (n <= 0 || d <= 0 || nlambda <= 0)
    Rf_error("invalid %s dimensions", family);

  R_xlen_t design_count = 0;
  R_xlen_t beta_count = 0;
  if (!checked_r_length_product(static_cast<R_xlen_t>(n),
                                static_cast<R_xlen_t>(d),
                                &design_count) ||
      !checked_r_length_product(static_cast<R_xlen_t>(d),
                                static_cast<R_xlen_t>(nlambda),
                                &beta_count))
    Rf_error("%s dimensions overflow R vector limits", family);

  const R_xlen_t native_index_max =
      static_cast<R_xlen_t>(std::numeric_limits<int>::max());
  if (design_count > native_index_max || beta_count > native_index_max)
    Rf_error("%s dimensions exceed native 32-bit indexing limits", family);
  if (TYPEOF(Y_sexp) != REALSXP ||
      XLENGTH(Y_sexp) < static_cast<R_xlen_t>(n))
    Rf_error("%s response must be a double vector of length n", family);
  if (TYPEOF(X_sexp) != REALSXP || XLENGTH(X_sexp) < design_count)
    Rf_error("%s design must be a double matrix with n*d values", family);
  if (TYPEOF(lambda_sexp) != REALSXP ||
      XLENGTH(lambda_sexp) < static_cast<R_xlen_t>(nlambda))
    Rf_error("%s lambda must be a double vector of length nlambda", family);
  if (needs_offset &&
      (TYPEOF(offset_sexp) != REALSXP ||
       XLENGTH(offset_sexp) < static_cast<R_xlen_t>(n)))
    Rf_error("%s offset must be a double vector of length n", family);
  return beta_count;
}

static R_xlen_t validate_scalar_lla_bridge_inputs(
    const char *family, SEXP Y_sexp, SEXP X_sexp, SEXP lambda_sexp,
    SEXP offset_sexp, bool needs_offset, int n, int d, int nlambda,
    int lla_max_stages) {
  if (lla_max_stages < 3)
    Rf_error("%s lla.max.stages must be at least 3", family);
  return validate_scalar_bridge_inputs(
      family, Y_sexp, X_sexp, lambda_sexp, offset_sexp, needs_offset,
      n, d, nlambda);
}

static R_xlen_t validate_standardize_bridge_inputs(SEXP X_sexp, int n,
                                                    int d) {
  if (n <= 0 || d <= 0) Rf_error("invalid standardization dimensions");
  R_xlen_t design_count = 0;
  if (!checked_r_length_product(static_cast<R_xlen_t>(n),
                                static_cast<R_xlen_t>(d),
                                &design_count))
    Rf_error("standardization dimensions overflow R vector limits");
  if (design_count >
      static_cast<R_xlen_t>(std::numeric_limits<int>::max()))
    Rf_error("standardization dimensions exceed native 32-bit indexing limits");
  if (TYPEOF(X_sexp) != REALSXP || XLENGTH(X_sexp) < design_count)
    Rf_error("standardization design must be a double matrix with n*d values");
  return design_count;
}

extern "C" SEXP picasso_gaussian_naive_call(
    SEXP Y_sexp, SEXP X_sexp, SEXP n_sexp, SEXP d_sexp,
    SEXP lambda_sexp, SEXP nlambda_sexp, SEXP gamma_sexp,
    SEXP max_ite_sexp, SEXP prec_sexp, SEXP reg_type_sexp,
    SEXP intercept_sexp, SEXP dfmax_sexp) {
  int n = Rf_asInteger(n_sexp);
  int d = Rf_asInteger(d_sexp);
  int nlambda = Rf_asInteger(nlambda_sexp);
  const R_xlen_t beta_count = validate_scalar_bridge_inputs(
      "gaussian", Y_sexp, X_sexp, lambda_sexp, R_NilValue, false,
      n, d, nlambda);

  // Allocate outputs — written into directly by C++, no copy back
  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, beta_count));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP smooth_objective_sexp =
      PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP status_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP failed_lambda_sexp = PROTECT(Rf_allocVector(INTSXP, 1));

  // Zero-initialize outputs
  memset(REAL(beta_sexp), 0, sizeof(double) * (size_t)beta_count);
  memset(REAL(intcpt_sexp), 0, sizeof(double) * nlambda);
  memset(INTEGER(ite_sexp), 0, sizeof(int) * nlambda);
  memset(INTEGER(size_sexp), 0, sizeof(int) * nlambda);
  memset(REAL(runt_sexp), 0, sizeof(double) * nlambda);
  INTEGER(nfit_sexp)[0] = 0;

  // Call solver — REAL(X_sexp) is a direct pointer, no copy
  INTEGER(status_sexp)[0] = SolveLinearRegressionNaiveUpdateV3(
      REAL(Y_sexp), REAL(X_sexp), n, d,
      REAL(lambda_sexp), nlambda,
      Rf_asReal(gamma_sexp), Rf_asInteger(max_ite_sexp),
      Rf_asReal(prec_sexp), Rf_asInteger(reg_type_sexp),
      Rf_asInteger(intercept_sexp), Rf_asInteger(dfmax_sexp),
      REAL(beta_sexp), REAL(intcpt_sexp),
      INTEGER(ite_sexp), INTEGER(size_sexp),
      REAL(runt_sexp), INTEGER(nfit_sexp), false,
      REAL(smooth_objective_sexp), INTEGER(failed_lambda_sexp));

  SEXP result = make_gaussian_result_list(
      beta_sexp, intcpt_sexp, ite_sexp, size_sexp, runt_sexp, nfit_sexp,
      smooth_objective_sexp, status_sexp, failed_lambda_sexp);
  UNPROTECT(9);
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
  const R_xlen_t beta_count = validate_scalar_bridge_inputs(
      "gaussian", Y_sexp, X_sexp, lambda_sexp, R_NilValue, false,
      n, d, nlambda);

  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, beta_count));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP smooth_objective_sexp =
      PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP status_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP failed_lambda_sexp = PROTECT(Rf_allocVector(INTSXP, 1));

  memset(REAL(beta_sexp), 0, sizeof(double) * (size_t)beta_count);
  memset(REAL(intcpt_sexp), 0, sizeof(double) * nlambda);
  memset(INTEGER(ite_sexp), 0, sizeof(int) * nlambda);
  memset(INTEGER(size_sexp), 0, sizeof(int) * nlambda);
  memset(REAL(runt_sexp), 0, sizeof(double) * nlambda);
  INTEGER(nfit_sexp)[0] = 0;

  INTEGER(status_sexp)[0] = SolveLinearRegressionCovUpdateV3(
      REAL(Y_sexp), REAL(X_sexp), n, d,
      REAL(lambda_sexp), nlambda,
      Rf_asReal(gamma_sexp), Rf_asInteger(max_ite_sexp),
      Rf_asReal(prec_sexp), Rf_asInteger(reg_type_sexp),
      Rf_asInteger(intercept_sexp), Rf_asInteger(dfmax_sexp),
      REAL(beta_sexp), REAL(intcpt_sexp),
      INTEGER(ite_sexp), INTEGER(size_sexp),
      REAL(runt_sexp), INTEGER(nfit_sexp), false,
      REAL(smooth_objective_sexp), INTEGER(failed_lambda_sexp));

  SEXP result = make_gaussian_result_list(
      beta_sexp, intcpt_sexp, ite_sexp, size_sexp, runt_sexp, nfit_sexp,
      smooth_objective_sexp, status_sexp, failed_lambda_sexp);
  UNPROTECT(9);
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
  const R_xlen_t beta_count = validate_scalar_bridge_inputs(
      "binomial", Y_sexp, X_sexp, lambda_sexp, offset_sexp, true,
      n, d, nlambda);

  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, beta_count));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));

  memset(REAL(beta_sexp), 0, sizeof(double) * (size_t)beta_count);
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

extern "C" SEXP picasso_logit_lla_call(
    SEXP Y_sexp, SEXP X_sexp, SEXP n_sexp, SEXP d_sexp,
    SEXP lambda_sexp, SEXP nlambda_sexp, SEXP gamma_sexp,
    SEXP max_ite_sexp, SEXP prec_sexp, SEXP reg_type_sexp,
    SEXP intercept_sexp, SEXP dfmax_sexp, SEXP offset_sexp,
    SEXP lla_max_stages_sexp) {
  const int n = Rf_asInteger(n_sexp);
  const int d = Rf_asInteger(d_sexp);
  const int nlambda = Rf_asInteger(nlambda_sexp);
  const int lla_max_stages = Rf_asInteger(lla_max_stages_sexp);
  const R_xlen_t beta_count = validate_scalar_lla_bridge_inputs(
      "binomial", Y_sexp, X_sexp, lambda_sexp, offset_sexp, true,
      n, d, nlambda, lla_max_stages);

  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, beta_count));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP status_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP failed_lambda_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP failed_stage_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP stages_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP objective_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP kkt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP stationarity_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP smooth_objective_sexp =
      PROTECT(Rf_allocVector(REALSXP, nlambda));

  INTEGER(status_sexp)[0] = SolveLogisticRegressionV3(
      REAL(Y_sexp), REAL(X_sexp), n, d, REAL(lambda_sexp), nlambda,
      Rf_asReal(gamma_sexp), Rf_asInteger(max_ite_sexp),
      Rf_asReal(prec_sexp), Rf_asInteger(reg_type_sexp),
      Rf_asInteger(intercept_sexp), Rf_asInteger(dfmax_sexp),
      REAL(offset_sexp), REAL(beta_sexp), REAL(intcpt_sexp),
      INTEGER(ite_sexp), INTEGER(size_sexp), REAL(runt_sexp),
      INTEGER(nfit_sexp), false, lla_max_stages,
      INTEGER(failed_lambda_sexp), INTEGER(failed_stage_sexp),
      INTEGER(stages_sexp), REAL(objective_sexp), REAL(kkt_sexp),
      REAL(stationarity_sexp), REAL(smooth_objective_sexp));

  SEXP result = make_scalar_lla_result_list(
      beta_sexp, intcpt_sexp, ite_sexp, size_sexp, runt_sexp, nfit_sexp,
      status_sexp, failed_lambda_sexp, failed_stage_sexp, stages_sexp,
      objective_sexp, kkt_sexp, stationarity_sexp,
      smooth_objective_sexp);
  UNPROTECT(14);
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
  const R_xlen_t beta_count = validate_scalar_bridge_inputs(
      "poisson", Y_sexp, X_sexp, lambda_sexp, offset_sexp, true,
      n, d, nlambda);

  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, beta_count));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));

  memset(REAL(beta_sexp), 0, sizeof(double) * (size_t)beta_count);
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

extern "C" SEXP picasso_poisson_lla_call(
    SEXP Y_sexp, SEXP X_sexp, SEXP n_sexp, SEXP d_sexp,
    SEXP lambda_sexp, SEXP nlambda_sexp, SEXP gamma_sexp,
    SEXP max_ite_sexp, SEXP prec_sexp, SEXP reg_type_sexp,
    SEXP intercept_sexp, SEXP dfmax_sexp, SEXP offset_sexp,
    SEXP lla_max_stages_sexp) {
  const int n = Rf_asInteger(n_sexp);
  const int d = Rf_asInteger(d_sexp);
  const int nlambda = Rf_asInteger(nlambda_sexp);
  const int lla_max_stages = Rf_asInteger(lla_max_stages_sexp);
  const R_xlen_t beta_count = validate_scalar_lla_bridge_inputs(
      "poisson", Y_sexp, X_sexp, lambda_sexp, offset_sexp, true,
      n, d, nlambda, lla_max_stages);

  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, beta_count));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP status_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP failed_lambda_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP failed_stage_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP stages_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP objective_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP kkt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP stationarity_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP smooth_objective_sexp =
      PROTECT(Rf_allocVector(REALSXP, nlambda));

  INTEGER(status_sexp)[0] = SolvePoissonRegressionV3(
      REAL(Y_sexp), REAL(X_sexp), n, d, REAL(lambda_sexp), nlambda,
      Rf_asReal(gamma_sexp), Rf_asInteger(max_ite_sexp),
      Rf_asReal(prec_sexp), Rf_asInteger(reg_type_sexp),
      Rf_asInteger(intercept_sexp), Rf_asInteger(dfmax_sexp),
      REAL(offset_sexp), REAL(beta_sexp), REAL(intcpt_sexp),
      INTEGER(ite_sexp), INTEGER(size_sexp), REAL(runt_sexp),
      INTEGER(nfit_sexp), false, lla_max_stages,
      INTEGER(failed_lambda_sexp), INTEGER(failed_stage_sexp),
      INTEGER(stages_sexp), REAL(objective_sexp), REAL(kkt_sexp),
      REAL(stationarity_sexp), REAL(smooth_objective_sexp));

  SEXP result = make_scalar_lla_result_list(
      beta_sexp, intcpt_sexp, ite_sexp, size_sexp, runt_sexp, nfit_sexp,
      status_sexp, failed_lambda_sexp, failed_stage_sexp, stages_sexp,
      objective_sexp, kkt_sexp, stationarity_sexp,
      smooth_objective_sexp);
  UNPROTECT(14);
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
  const R_xlen_t beta_count = validate_scalar_bridge_inputs(
      "sqrt-lasso", Y_sexp, X_sexp, lambda_sexp, R_NilValue, false,
      n, d, nlambda);

  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, beta_count));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));

  memset(REAL(beta_sexp), 0, sizeof(double) * (size_t)beta_count);
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

extern "C" SEXP picasso_sqrtlasso_lla_call(
    SEXP Y_sexp, SEXP X_sexp, SEXP n_sexp, SEXP d_sexp,
    SEXP lambda_sexp, SEXP nlambda_sexp, SEXP gamma_sexp,
    SEXP max_ite_sexp, SEXP prec_sexp, SEXP reg_type_sexp,
    SEXP intercept_sexp, SEXP dfmax_sexp, SEXP lla_max_stages_sexp) {
  const int n = Rf_asInteger(n_sexp);
  const int d = Rf_asInteger(d_sexp);
  const int nlambda = Rf_asInteger(nlambda_sexp);
  const int lla_max_stages = Rf_asInteger(lla_max_stages_sexp);
  const R_xlen_t beta_count = validate_scalar_lla_bridge_inputs(
      "sqrt-lasso", Y_sexp, X_sexp, lambda_sexp, R_NilValue, false,
      n, d, nlambda, lla_max_stages);

  SEXP beta_sexp = PROTECT(Rf_allocVector(REALSXP, beta_count));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP ite_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP size_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP runt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP status_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP failed_lambda_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP failed_stage_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP stages_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP objective_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP kkt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP stationarity_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP smooth_objective_sexp =
      PROTECT(Rf_allocVector(REALSXP, nlambda));

  INTEGER(status_sexp)[0] = SolveSqrtLinearRegressionV3(
      REAL(Y_sexp), REAL(X_sexp), n, d, REAL(lambda_sexp), nlambda,
      Rf_asReal(gamma_sexp), Rf_asInteger(max_ite_sexp),
      Rf_asReal(prec_sexp), Rf_asInteger(reg_type_sexp),
      Rf_asInteger(intercept_sexp), Rf_asInteger(dfmax_sexp),
      REAL(beta_sexp), REAL(intcpt_sexp), INTEGER(ite_sexp),
      INTEGER(size_sexp), REAL(runt_sexp), INTEGER(nfit_sexp), false,
      lla_max_stages, INTEGER(failed_lambda_sexp),
      INTEGER(failed_stage_sexp), INTEGER(stages_sexp),
      REAL(objective_sexp), REAL(kkt_sexp), REAL(stationarity_sexp),
      REAL(smooth_objective_sexp));

  SEXP result = make_scalar_lla_result_list(
      beta_sexp, intcpt_sexp, ite_sexp, size_sexp, runt_sexp, nfit_sexp,
      status_sexp, failed_lambda_sexp, failed_stage_sexp, stages_sexp,
      objective_sexp, kkt_sexp, stationarity_sexp,
      smooth_objective_sexp);
  UNPROTECT(14);
  return result;
}

extern "C" SEXP picasso_standardize_call(SEXP X_sexp, SEXP n_sexp,
                                         SEXP d_sexp) {
  int n = Rf_asInteger(n_sexp);
  int d = Rf_asInteger(d_sexp);
  validate_standardize_bridge_inputs(X_sexp, n, d);

  SEXP xx_sexp = PROTECT(Rf_allocMatrix(REALSXP, n, d));
  SEXP xm_sexp = PROTECT(Rf_allocMatrix(REALSXP, 1, d));
  SEXP xinvc_sexp = PROTECT(Rf_allocVector(REALSXP, d));

  double *X = REAL(X_sexp);
  double *xx = REAL(xx_sexp);
  double *xm = REAL(xm_sexp);
  double *xinvc = REAL(xinvc_sexp);

  for (int j = 0; j < d; j++) {
    int jn = j * n;
    double column_scale = 0.0;
    for (int i = 0; i < n; i++)
      column_scale = std::fmax(column_scale, std::fabs(X[jn + i]));

    if (column_scale == 0.0) {
      xm[j] = 0.0;
      xinvc[j] = 0.0;
      for (int i = 0; i < n; i++) xx[jn + i] = 0.0;
      continue;
    }

    // Work relative to the largest magnitude in the column.  This keeps both
    // the mean and centered sum of squares finite even when the unscaled sum,
    // a centered difference, or a square would overflow.  Compensated
    // summation preserves the usual double-precision result on ordinary data.
    double scaled_sum = 0.0;
    double compensation = 0.0;
    for (int i = 0; i < n; i++) {
      double value = X[jn + i] / column_scale;
      double updated = scaled_sum + value;
      if (std::fabs(scaled_sum) >= std::fabs(value))
        compensation += (scaled_sum - updated) + value;
      else
        compensation += (value - updated) + scaled_sum;
      scaled_sum = updated;
    }
    double scaled_mean = (scaled_sum + compensation) / n;
    // The exact scaled mean lies in [-1, 1].  Clamp a possible last-bit
    // overshoot so reconstructing the original-scale mean cannot overflow.
    scaled_mean = std::fmax(-1.0, std::fmin(1.0, scaled_mean));
    xm[j] = column_scale * scaled_mean;

    double centered_scale = 0.0;
    for (int i = 0; i < n; i++) {
      double centered = X[jn + i] / column_scale - scaled_mean;
      xx[jn + i] = centered;
      centered_scale = std::fmax(centered_scale, std::fabs(centered));
    }

    xinvc[j] = 0.0;
    if (centered_scale > 0.0 && n > 1) {
      double relative_sum_squares = 0.0;
      for (int i = 0; i < n; i++) {
        double relative_centered = xx[jn + i] / centered_scale;
        relative_sum_squares += relative_centered * relative_centered;
      }
      double normalized_scale =
          std::sqrt((n - 1.0) / relative_sum_squares);
      xinvc[j] = (normalized_scale / column_scale) / centered_scale;
      for (int i = 0; i < n; i++)
        xx[jn + i] = (xx[jn + i] / centered_scale) * normalized_scale;
    } else {
      for (int i = 0; i < n; i++) xx[jn + i] = 0.0;
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
    SEXP intercept_sexp, SEXP dfmax_sexp, SEXP lla_max_stages_sexp,
    SEXP path_early_stop_sexp) {
  int n       = Rf_asInteger(n_sexp);
  int d       = Rf_asInteger(d_sexp);
  int K       = Rf_asInteger(K_sexp);
  int nlambda = Rf_asInteger(nlambda_sexp);
  int lla_max_stages = Rf_asInteger(lla_max_stages_sexp);
  const int path_early_stop = Rf_asLogical(path_early_stop_sexp);

  if (n <= 0 || d <= 0 || K < 2 || nlambda <= 0)
    Rf_error("invalid multinomial dimensions");
  if (lla_max_stages < 3)
    Rf_error("multinomial lla.max.stages must be at least 3");
  if (path_early_stop == NA_LOGICAL)
    Rf_error("multinomial path.early.stop must be TRUE or FALSE");
  if (TYPEOF(Y_sexp) != REALSXP || XLENGTH(Y_sexp) < n)
    Rf_error("multinomial labels must be a double vector of length n");
  R_xlen_t design_count = 0;
  R_xlen_t probability_count = 0;
  R_xlen_t coefficient_count = 0;
  R_xlen_t beta_count = 0;
  R_xlen_t intercept_count = 0;
  if (!checked_r_length_product(static_cast<R_xlen_t>(n),
                                static_cast<R_xlen_t>(d),
                                &design_count) ||
      !checked_r_length_product(static_cast<R_xlen_t>(d),
                                static_cast<R_xlen_t>(K),
                                &coefficient_count) ||
      !checked_r_length_product(static_cast<R_xlen_t>(n),
                                static_cast<R_xlen_t>(K),
                                &probability_count) ||
      !checked_r_length_product(coefficient_count,
                                static_cast<R_xlen_t>(nlambda),
                                &beta_count) ||
      !checked_r_length_product(static_cast<R_xlen_t>(K),
                                static_cast<R_xlen_t>(nlambda),
                                &intercept_count))
    Rf_error("multinomial output dimensions overflow R vector limits");
  const R_xlen_t native_index_max =
      static_cast<R_xlen_t>(std::numeric_limits<int>::max());
  if (design_count > native_index_max ||
      probability_count > native_index_max ||
      coefficient_count > native_index_max ||
      beta_count > native_index_max ||
      intercept_count > native_index_max)
    Rf_error("multinomial dimensions exceed native 32-bit indexing limits");
  if (TYPEOF(X_sexp) != REALSXP || XLENGTH(X_sexp) < design_count)
    Rf_error("multinomial design must be a double matrix with n*d values");
  if (TYPEOF(lambda_sexp) != REALSXP || XLENGTH(lambda_sexp) < nlambda)
    Rf_error("multinomial lambda must be a double vector of length nlambda");

  // beta: d * K * nlambda;  intcpt: K * nlambda
  SEXP beta_sexp   = PROTECT(Rf_allocVector(REALSXP, beta_count));
  SEXP intcpt_sexp = PROTECT(Rf_allocVector(REALSXP, intercept_count));
  SEXP ite_sexp    = PROTECT(Rf_allocVector(INTSXP,  nlambda));
  SEXP size_sexp   = PROTECT(Rf_allocVector(INTSXP,  nlambda));
  SEXP runt_sexp   = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP nfit_sexp   = PROTECT(Rf_allocVector(INTSXP,  1));
  SEXP status_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP failed_lambda_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP failed_stage_sexp = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP outer_sexp = PROTECT(Rf_allocVector(INTSXP, nlambda));
  SEXP inner_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP updates_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP objective_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP kkt_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP stationarity_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP smooth_nll_sexp = PROTECT(Rf_allocVector(REALSXP, nlambda));

  long long *inner_sweeps = reinterpret_cast<long long *>(
      R_alloc(static_cast<std::size_t>(nlambda), sizeof(long long)));
  long long *coordinate_updates = reinterpret_cast<long long *>(
      R_alloc(static_cast<std::size_t>(nlambda), sizeof(long long)));

  INTEGER(status_sexp)[0] = SolveMultinomialRegressionV5(
      REAL(Y_sexp), REAL(X_sexp), n, d, K,
      REAL(lambda_sexp), nlambda,
      Rf_asReal(gamma_sexp), Rf_asInteger(max_ite_sexp),
      Rf_asReal(prec_sexp), Rf_asInteger(reg_type_sexp),
      Rf_asInteger(intercept_sexp), Rf_asInteger(dfmax_sexp),
      REAL(beta_sexp), REAL(intcpt_sexp),
      INTEGER(ite_sexp), INTEGER(size_sexp),
      REAL(runt_sexp), INTEGER(nfit_sexp), false,
      lla_max_stages, path_early_stop != 0,
      INTEGER(failed_lambda_sexp), INTEGER(failed_stage_sexp),
      INTEGER(outer_sexp), inner_sweeps, coordinate_updates,
      REAL(objective_sexp), REAL(kkt_sexp), REAL(stationarity_sexp),
      REAL(smooth_nll_sexp));
  for (int index = 0; index < nlambda; ++index) {
    REAL(inner_sexp)[index] =
        static_cast<double>(inner_sweeps[static_cast<std::size_t>(index)]);
    REAL(updates_sexp)[index] = static_cast<double>(
        coordinate_updates[static_cast<std::size_t>(index)]);
  }

  SEXP result = make_multinomial_result_list(
      beta_sexp, intcpt_sexp, ite_sexp, size_sexp, runt_sexp, nfit_sexp,
      status_sexp, failed_lambda_sexp, failed_stage_sexp, outer_sexp,
      inner_sexp, updates_sexp, objective_sexp, kkt_sexp,
      stationarity_sexp, smooth_nll_sexp);
  UNPROTECT(16);
  return result;
}

// Registration
static const R_CallMethodDef CallEntries[] = {
    {"picasso_gaussian_naive_call", (DL_FUNC)&picasso_gaussian_naive_call, 12},
    {"picasso_gaussian_cov_call", (DL_FUNC)&picasso_gaussian_cov_call, 12},
    {"picasso_logit_call", (DL_FUNC)&picasso_logit_call, 13},
    {"picasso_logit_lla_call", (DL_FUNC)&picasso_logit_lla_call, 14},
    {"picasso_poisson_call", (DL_FUNC)&picasso_poisson_call, 13},
    {"picasso_poisson_lla_call", (DL_FUNC)&picasso_poisson_lla_call, 14},
    {"picasso_sqrtlasso_call", (DL_FUNC)&picasso_sqrtlasso_call, 12},
    {"picasso_sqrtlasso_lla_call", (DL_FUNC)&picasso_sqrtlasso_lla_call, 13},
    {"picasso_standardize_call", (DL_FUNC)&picasso_standardize_call, 3},
    {"picasso_multinomial_call", (DL_FUNC)&picasso_multinomial_call, 15},
    {NULL, NULL, 0}};

extern "C" void attribute_visible R_init_picasso(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
