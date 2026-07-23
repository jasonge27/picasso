#include <picasso/actgd.hpp>
#include <picasso/actnewton.hpp>
#include <picasso/c_api.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
picasso::detail::DesignStorage c_api_design_storage(bool use_python) {
  return use_python ? picasso::detail::DesignStorage::kOwned
                    : picasso::detail::DesignStorage::kBorrowedColumnMajor;
}

void zero_solver_outputs(int d, int nlambda, double *beta, double *intcpt,
                         int *ite_lamb, int *size_act, double *runt) {
  const int safe_d = (d > 0) ? d : 0;
  const int safe_nlambda = (nlambda > 0) ? nlambda : 0;

  if (safe_nlambda == 0) return;

  if (beta != nullptr && safe_d > 0) {
    std::fill_n(beta, static_cast<std::size_t>(safe_d) * safe_nlambda, 0.0);
  }
  if (intcpt != nullptr) std::fill_n(intcpt, safe_nlambda, 0.0);
  if (ite_lamb != nullptr) std::fill_n(ite_lamb, safe_nlambda, 0);
  if (size_act != nullptr) std::fill_n(size_act, safe_nlambda, 0);
  if (runt != nullptr) std::fill_n(runt, safe_nlambda, 0.0);
}

bool safe_solver_output_dimensions(int d, int nlambda) {
  return d > 0 && nlambda > 0 &&
         d <= std::numeric_limits<int>::max() / nlambda;
}

void initialize_smooth_objective_path(int nlambda,
                                      double *smooth_objective) {
  if (smooth_objective == nullptr || nlambda <= 0) return;
  std::fill_n(smooth_objective, nlambda,
              std::numeric_limits<double>::quiet_NaN());
}

bool invalid_problem_inputs(double *Y, double *X, int n, int d) {
  return Y == nullptr || X == nullptr || n <= 0 || d <= 0;
}

bool invalid_lambda_path(const double *lambda, int nlambda) {
  if (lambda == nullptr || nlambda <= 0) return true;
  for (int index = 0; index < nlambda; ++index) {
    if (!(lambda[index] >= 0.0) || !std::isfinite(lambda[index]))
      return true;
    if (index > 0 && !(lambda[index] < lambda[index - 1]))
      return true;
  }
  return false;
}

bool invalid_offset_inputs(const double *offset, int n) {
  if (offset == nullptr) return false;
  for (int index = 0; index < n; ++index) {
    if (!std::isfinite(offset[index])) return true;
  }
  return false;
}

void initialize_lla_diagnostics(int nlambda, int *failed_lambda,
                                int *failed_stage, int *lla_stages,
                                double *objective, double *kkt,
                                double *stationarity,
                                double *smooth_objective) {
  if (failed_lambda != nullptr) *failed_lambda = -1;
  if (failed_stage != nullptr) *failed_stage = -1;
  if (nlambda <= 0) return;

  if (lla_stages != nullptr) std::fill_n(lla_stages, nlambda, 0);
  const double missing = std::numeric_limits<double>::quiet_NaN();
  if (objective != nullptr) std::fill_n(objective, nlambda, missing);
  if (kkt != nullptr) std::fill_n(kkt, nlambda, missing);
  if (stationarity != nullptr)
    std::fill_n(stationarity, nlambda, missing);
  if (smooth_objective != nullptr)
    std::fill_n(smooth_objective, nlambda, missing);
}

bool invalid_lla_inputs(double *Y, double *X, double *lambda, int n, int d,
                        int nlambda, double gamma, int max_iter,
                        double precision, int reg_type, int dfmax,
                        int lla_max_stages) {
  if (invalid_problem_inputs(Y, X, n, d) ||
      invalid_lambda_path(lambda, nlambda) ||
      max_iter <= 0 || !(precision > 0.0) ||
      !std::isfinite(precision) || reg_type < 1 || reg_type > 3 ||
      dfmax < -1 || lla_max_stages < 3 ||
      d > std::numeric_limits<int>::max() / n ||
      d > std::numeric_limits<int>::max() / nlambda)
    return true;
  if ((reg_type == 2 && (!(gamma > 1.0) || !std::isfinite(gamma))) ||
      (reg_type == 3 && (!(gamma > 2.0) || !std::isfinite(gamma))))
    return true;
  return false;
}

picasso::solver::PicassoSolverParams make_params(
    double *lambda, int nlambda, double gamma, int max_ite, double prec,
    int reg_type, bool intercept, int dfmax,
    int num_relaxation_round = 3) {
  picasso::solver::PicassoSolverParams param;
  param.set_lambdas(lambda, nlambda);
  param.gamma = gamma;
  if (reg_type == 1)
    param.reg_type = picasso::solver::L1;
  else if (reg_type == 2)
    param.reg_type = picasso::solver::MCP;
  else
    param.reg_type = picasso::solver::SCAD;
  param.include_intercept = intercept;
  param.prec = prec;
  param.max_iter = max_ite;
  param.num_relaxation_round = num_relaxation_round;
  param.dfmax = dfmax;
  return param;
}

bool invalid_gaussian_inputs(double *Y, double *X, double *lambda, int n,
                             int d, int nlambda, double gamma, int max_iter,
                             double precision, int reg_type, int dfmax) {
  if (invalid_problem_inputs(Y, X, n, d)) return true;
  if (!safe_solver_output_dimensions(d, nlambda) ||
      d > std::numeric_limits<int>::max() / n || max_iter <= 0 ||
      !(precision > 0.0) || !std::isfinite(precision) || reg_type < 1 ||
      reg_type > 3 || dfmax < -1)
    return true;
  if ((reg_type == 2 && (!(gamma > 1.0) || !std::isfinite(gamma))) ||
      (reg_type == 3 && (!(gamma > 2.0) || !std::isfinite(gamma))))
    return true;
  return invalid_lambda_path(lambda, nlambda);
}

void initialize_gaussian_outputs(int d, int nlambda, double *beta,
                                 double *intcpt, int *ite_lamb,
                                 int *size_act, double *runt, int *num_fit,
                                 double *smooth_objective) {
  if (num_fit != nullptr) *num_fit = 0;
  if (!safe_solver_output_dimensions(d, nlambda)) return;
  zero_solver_outputs(d, nlambda, beta, intcpt, ite_lamb, size_act, runt);
  initialize_smooth_objective_path(nlambda, smooth_objective);
}

int actgd_status_to_c(picasso::solver::ActGDPathStatus status) {
  using picasso::solver::ActGDPathStatus;
  switch (status) {
    case ActGDPathStatus::kCompleted:
      return PICASSO_LLA_COMPLETED;
    case ActGDPathStatus::kDfmaxReached:
      return PICASSO_LLA_DFMAX_REACHED;
    case ActGDPathStatus::kIterationLimit:
      return PICASSO_LLA_INNER_ITERATION_LIMIT;
  }
  return PICASSO_LLA_NUMERICAL_FAILURE;
}

template <typename ObjectiveType>
int solve_gaussian_c_api(
    double *Y, double *X, int n, int d, double *lambda, int nlambda,
    double gamma, int max_iter, double precision, int reg_type,
    bool intercept, int dfmax, double *beta, double *intcpt, int *ite_lamb,
    int *size_act, double *runt, int *num_fit, bool use_python,
    double *smooth_objective, int *failed_lambda) {
  if (num_fit != nullptr) *num_fit = 0;
  if (failed_lambda != nullptr) *failed_lambda = -1;
  // Preserve the V2 contract even when d is invalid: this path has length
  // nlambda and does not depend on the coefficient-output dimensions.
  initialize_smooth_objective_path(nlambda, smooth_objective);
  const bool safe_outputs = safe_solver_output_dimensions(d, nlambda);
  if (invalid_gaussian_inputs(Y, X, lambda, n, d, nlambda, gamma, max_iter,
                              precision, reg_type, dfmax)) {
    if (safe_outputs)
      zero_solver_outputs(d, nlambda, beta, intcpt, ite_lamb, size_act, runt);
    return PICASSO_LLA_INVALID_INPUT;
  }

  try {
    ObjectiveType objective(X, Y, n, d, intercept, use_python,
                            c_api_design_storage(use_python));
    const auto params = make_params(lambda, nlambda, gamma, max_iter,
                                    precision, reg_type, intercept, dfmax);
    picasso::solver::ActGDSolver solver(&objective, params);
    const int actual_fit = solver.solve_to_buffers(
        beta, intcpt, ite_lamb, size_act, runt, smooth_objective);
    if (num_fit != nullptr) *num_fit = actual_fit;
    if (failed_lambda != nullptr)
      *failed_lambda = solver.get_failed_lambda();
    return actgd_status_to_c(solver.get_status());
  } catch (...) {
    // These legacy void APIs cannot return an error code. Preserve their ABI
    // while making failure explicit through an empty, initialized path.
    initialize_gaussian_outputs(d, nlambda, beta, intcpt, ite_lamb, size_act,
                                runt, num_fit, smooth_objective);
    if (failed_lambda != nullptr) *failed_lambda = 0;
    return PICASSO_LLA_EXCEPTION;
  }
}

int actnewton_status_to_c(picasso::solver::ActNewtonLlaStatus status) {
  using picasso::solver::ActNewtonLlaStatus;
  switch (status) {
    case ActNewtonLlaStatus::kCompleted:
      return PICASSO_LLA_COMPLETED;
    case ActNewtonLlaStatus::kStationarityLimit:
      return PICASSO_LLA_STATIONARITY_LIMIT;
    case ActNewtonLlaStatus::kSubproblemFailed:
      return PICASSO_LLA_SUBPROBLEM_FAILED;
    case ActNewtonLlaStatus::kMajorizationFailed:
      return PICASSO_LLA_MAJORIZATION_FAILED;
    case ActNewtonLlaStatus::kNumericalFailure:
      return PICASSO_LLA_NUMERICAL_FAILURE;
    case ActNewtonLlaStatus::kNotRun:
      return PICASSO_LLA_NUMERICAL_FAILURE;
  }
  return PICASSO_LLA_NUMERICAL_FAILURE;
}

template <typename Value>
void copy_diagnostic_path(const std::vector<Value> &source, int nlambda,
                          Value *destination) {
  if (destination == nullptr || nlambda <= 0) return;
  const int count =
      std::min(nlambda, static_cast<int>(source.size()));
  std::copy_n(source.begin(), count, destination);
}

void copy_actnewton_diagnostics(
    const picasso::solver::ActNewtonSolver &solver, int nlambda,
    int *lla_stages, double *objective, double *kkt,
    double *stationarity, double *smooth_objective) {
  copy_diagnostic_path(solver.get_lla_stages_path(), nlambda, lla_stages);
  copy_diagnostic_path(solver.get_objective_path(), nlambda, objective);
  copy_diagnostic_path(solver.get_kkt_path(), nlambda, kkt);
  copy_diagnostic_path(solver.get_stationarity_path(), nlambda,
                       stationarity);
  copy_diagnostic_path(solver.get_smooth_objective_path(), nlambda,
                       smooth_objective);
}

int run_actnewton_v2(
    picasso::ObjFunction *obj, double *lambda, int nlambda,
    double gamma, int max_ite, double prec, int reg_type, bool intercept,
    int dfmax, int lla_max_stages, double *beta, double *intcpt,
    int *ite_lamb, int *size_act, double *runt, int *num_fit,
    int *failed_lambda, int *failed_stage, int *lla_stages,
    double *objective, double *kkt, double *stationarity,
    double *smooth_objective) {
  auto param = make_params(lambda, nlambda, gamma, max_ite, prec, reg_type,
                           intercept, dfmax, lla_max_stages);
  picasso::solver::ActNewtonSolver solver(obj, param);
  bool caught_exception = false;
  int actual_fit = 0;
  int last_nonzero_count = 0;
  try {
    (void)solver.solve_preinitialized_to_buffers(
        beta, intcpt, ite_lamb, size_act, runt, smooth_objective,
        &actual_fit, &last_nonzero_count);
  } catch (...) {
    caught_exception = true;
  }

  if (num_fit != nullptr) *num_fit = actual_fit;
  copy_actnewton_diagnostics(solver, nlambda, lla_stages, objective, kkt,
                             stationarity, smooth_objective);

  if (caught_exception) {
    if (failed_lambda != nullptr)
      *failed_lambda = (actual_fit < nlambda) ? actual_fit : -1;
    if (failed_stage != nullptr) *failed_stage = solver.get_failed_stage();
    return PICASSO_LLA_EXCEPTION;
  }

  const int status = actnewton_status_to_c(solver.get_lla_path_status());
  if (status == PICASSO_LLA_SUBPROBLEM_FAILED ||
      status == PICASSO_LLA_NUMERICAL_FAILURE ||
      status == PICASSO_LLA_MAJORIZATION_FAILED) {
    if (failed_lambda != nullptr) {
      const int core_failed_lambda = solver.get_failed_lambda();
      *failed_lambda = core_failed_lambda >= 0 ? core_failed_lambda
                                               : actual_fit;
    }
    if (failed_stage != nullptr) *failed_stage = solver.get_failed_stage();
  }

  if (status == PICASSO_LLA_COMPLETED && actual_fit < nlambda &&
      dfmax >= 0 && last_nonzero_count > dfmax)
    return PICASSO_LLA_DFMAX_REACHED;
  return status;
}
}  // namespace

extern "C" const char *PicassoLlaPathStatusString(int status) {
  switch (status) {
    case PICASSO_LLA_COMPLETED:
      return "completed";
    case PICASSO_LLA_DFMAX_REACHED:
      return "dfmax_reached";
    case PICASSO_LLA_INVALID_INPUT:
      return "invalid_input";
    case PICASSO_LLA_SUBPROBLEM_FAILED:
      return "subproblem_failed";
    case PICASSO_LLA_INNER_ITERATION_LIMIT:
      return "inner_iteration_limit";
    case PICASSO_LLA_LINE_SEARCH_FAILED:
      return "line_search_failed";
    case PICASSO_LLA_NO_DESCENT_DIRECTION:
      return "no_descent_direction";
    case PICASSO_LLA_NUMERICAL_FAILURE:
      return "numerical_failure";
    case PICASSO_LLA_MAJORIZATION_FAILED:
      return "lla_majorization_failed";
    case PICASSO_LLA_EXCEPTION:
      return "exception";
    case PICASSO_LLA_STATIONARITY_LIMIT:
      return "lla_stationarity_limit";
    default:
      return "unknown";
  }
}

extern "C" void SolveLogisticRegression(
    double *Y, double *X, int n, int d, double *lambda, int nlambda,
    double gamma, int max_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *offset,
    double *beta, double *intcpt, int *ite_lamb, int *size_act, double *runt,
    int *num_fit, bool usePython) {
  (void)SolveLogisticRegressionV2(
      Y, X, n, d, lambda, nlambda, gamma, max_ite, pprec, reg_type,
      intercept, dfmax, offset, beta, intcpt, ite_lamb, size_act, runt,
      num_fit, usePython, 3, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr);
}

extern "C" void SolvePoissonRegression(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *offset,
    double *beta, double *intcpt, int *ite_lamb, int *size_act, double *runt,
    int *num_fit, bool usePython) {
  (void)SolvePoissonRegressionV2(
      Y, X, nn, dd, lambda, nnlambda, gamma, mmax_ite, pprec, reg_type,
      intercept, dfmax, offset, beta, intcpt, ite_lamb, size_act, runt,
      num_fit, usePython, 3, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr);
}

extern "C" void SolveSqrtLinearRegression(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax,
    double *beta, double *intcpt, int *ite_lamb, int *size_act, double *runt,
    int *num_fit, bool usePython) {
  (void)SolveSqrtLinearRegressionV2(
      Y, X, nn, dd, lambda, nnlambda, gamma, mmax_ite, pprec, reg_type,
      intercept, dfmax, beta, intcpt, ite_lamb, size_act, runt, num_fit,
      usePython, 3, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

extern "C" int SolveLogisticRegressionV2(
    double *Y, double *X, int n, int d, double *lambda, int nlambda,
    double gamma, int max_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *offset, double *beta, double *intcpt, int *ite_lamb,
    int *size_act, double *runt, int *num_fit, bool usePython,
    int lla_max_stages, int *failed_lambda, int *failed_stage,
    int *lla_stages, double *objective, double *kkt, double *stationarity) {
  if (num_fit != nullptr) *num_fit = 0;
  initialize_lla_diagnostics(nlambda, failed_lambda, failed_stage, lla_stages,
                             objective, kkt, stationarity, nullptr);
  if (safe_solver_output_dimensions(d, nlambda))
    zero_solver_outputs(d, nlambda, beta, intcpt, ite_lamb, size_act, runt);
  if (invalid_lla_inputs(Y, X, lambda, n, d, nlambda, gamma, max_ite,
                         pprec, reg_type, dfmax, lla_max_stages) ||
      invalid_offset_inputs(offset, n))
    return PICASSO_LLA_INVALID_INPUT;

  try {
    picasso::LogisticObjective obj(
        X, Y, n, d, intercept, usePython, c_api_design_storage(usePython));
    if (offset != nullptr && !obj.set_offset(offset, n))
      return PICASSO_LLA_INVALID_INPUT;
    return run_actnewton_v2(
        &obj, lambda, nlambda, gamma, max_ite, pprec, reg_type, intercept,
        dfmax, lla_max_stages, beta, intcpt, ite_lamb, size_act, runt,
        num_fit, failed_lambda, failed_stage, lla_stages, objective, kkt,
        stationarity, nullptr);
  } catch (...) {
    if (failed_lambda != nullptr) *failed_lambda = 0;
    return PICASSO_LLA_EXCEPTION;
  }
}

extern "C" int SolvePoissonRegressionV2(
    double *Y, double *X, int n, int d, double *lambda, int nlambda,
    double gamma, int max_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *offset, double *beta, double *intcpt, int *ite_lamb,
    int *size_act, double *runt, int *num_fit, bool usePython,
    int lla_max_stages, int *failed_lambda, int *failed_stage,
    int *lla_stages, double *objective, double *kkt, double *stationarity) {
  if (num_fit != nullptr) *num_fit = 0;
  initialize_lla_diagnostics(nlambda, failed_lambda, failed_stage, lla_stages,
                             objective, kkt, stationarity, nullptr);
  if (safe_solver_output_dimensions(d, nlambda))
    zero_solver_outputs(d, nlambda, beta, intcpt, ite_lamb, size_act, runt);
  if (invalid_lla_inputs(Y, X, lambda, n, d, nlambda, gamma, max_ite,
                         pprec, reg_type, dfmax, lla_max_stages) ||
      invalid_offset_inputs(offset, n))
    return PICASSO_LLA_INVALID_INPUT;

  try {
    picasso::PoissonObjective obj(
        X, Y, n, d, intercept, usePython, c_api_design_storage(usePython));
    if (offset != nullptr && !obj.set_offset(offset, n))
      return PICASSO_LLA_INVALID_INPUT;
    return run_actnewton_v2(
        &obj, lambda, nlambda, gamma, max_ite, pprec, reg_type, intercept,
        dfmax, lla_max_stages, beta, intcpt, ite_lamb, size_act, runt,
        num_fit, failed_lambda, failed_stage, lla_stages, objective, kkt,
        stationarity, nullptr);
  } catch (...) {
    if (failed_lambda != nullptr) *failed_lambda = 0;
    return PICASSO_LLA_EXCEPTION;
  }
}

extern "C" int SolveSqrtLinearRegressionV2(
    double *Y, double *X, int n, int d, double *lambda, int nlambda,
    double gamma, int max_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, int lla_max_stages,
    int *failed_lambda, int *failed_stage, int *lla_stages,
    double *objective, double *kkt, double *stationarity) {
  if (num_fit != nullptr) *num_fit = 0;
  initialize_lla_diagnostics(nlambda, failed_lambda, failed_stage, lla_stages,
                             objective, kkt, stationarity, nullptr);
  if (safe_solver_output_dimensions(d, nlambda))
    zero_solver_outputs(d, nlambda, beta, intcpt, ite_lamb, size_act, runt);
  if (invalid_lla_inputs(Y, X, lambda, n, d, nlambda, gamma, max_ite,
                         pprec, reg_type, dfmax, lla_max_stages))
    return PICASSO_LLA_INVALID_INPUT;

  try {
    picasso::SqrtMSEObjective obj(
        X, Y, n, d, intercept, usePython, c_api_design_storage(usePython));
    return run_actnewton_v2(
        &obj, lambda, nlambda, gamma, max_ite, pprec, reg_type, intercept,
        dfmax, lla_max_stages, beta, intcpt, ite_lamb, size_act, runt,
        num_fit, failed_lambda, failed_stage, lla_stages, objective, kkt,
        stationarity, nullptr);
  } catch (...) {
    if (failed_lambda != nullptr) *failed_lambda = 0;
    return PICASSO_LLA_EXCEPTION;
  }
}

extern "C" int SolveLogisticRegressionV3(
    double *Y, double *X, int n, int d, double *lambda, int nlambda,
    double gamma, int max_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *offset, double *beta, double *intcpt, int *ite_lamb,
    int *size_act, double *runt, int *num_fit, bool usePython,
    int lla_max_stages, int *failed_lambda, int *failed_stage,
    int *lla_stages, double *objective, double *kkt, double *stationarity,
    double *smooth_objective) {
  if (num_fit != nullptr) *num_fit = 0;
  initialize_lla_diagnostics(nlambda, failed_lambda, failed_stage, lla_stages,
                             objective, kkt, stationarity,
                             smooth_objective);
  if (safe_solver_output_dimensions(d, nlambda))
    zero_solver_outputs(d, nlambda, beta, intcpt, ite_lamb, size_act, runt);
  if (invalid_lla_inputs(Y, X, lambda, n, d, nlambda, gamma, max_ite,
                         pprec, reg_type, dfmax, lla_max_stages) ||
      invalid_offset_inputs(offset, n))
    return PICASSO_LLA_INVALID_INPUT;

  try {
    picasso::LogisticObjective obj(
        X, Y, n, d, intercept, usePython, c_api_design_storage(usePython));
    if (offset != nullptr && !obj.set_offset(offset, n))
      return PICASSO_LLA_INVALID_INPUT;
    return run_actnewton_v2(
        &obj, lambda, nlambda, gamma, max_ite, pprec, reg_type, intercept,
        dfmax, lla_max_stages, beta, intcpt, ite_lamb, size_act, runt,
        num_fit, failed_lambda, failed_stage, lla_stages, objective, kkt,
        stationarity, smooth_objective);
  } catch (...) {
    if (failed_lambda != nullptr) *failed_lambda = 0;
    return PICASSO_LLA_EXCEPTION;
  }
}

extern "C" int SolvePoissonRegressionV3(
    double *Y, double *X, int n, int d, double *lambda, int nlambda,
    double gamma, int max_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *offset, double *beta, double *intcpt, int *ite_lamb,
    int *size_act, double *runt, int *num_fit, bool usePython,
    int lla_max_stages, int *failed_lambda, int *failed_stage,
    int *lla_stages, double *objective, double *kkt, double *stationarity,
    double *smooth_objective) {
  if (num_fit != nullptr) *num_fit = 0;
  initialize_lla_diagnostics(nlambda, failed_lambda, failed_stage, lla_stages,
                             objective, kkt, stationarity,
                             smooth_objective);
  if (safe_solver_output_dimensions(d, nlambda))
    zero_solver_outputs(d, nlambda, beta, intcpt, ite_lamb, size_act, runt);
  if (invalid_lla_inputs(Y, X, lambda, n, d, nlambda, gamma, max_ite,
                         pprec, reg_type, dfmax, lla_max_stages) ||
      invalid_offset_inputs(offset, n))
    return PICASSO_LLA_INVALID_INPUT;

  try {
    picasso::PoissonObjective obj(
        X, Y, n, d, intercept, usePython, c_api_design_storage(usePython));
    if (offset != nullptr && !obj.set_offset(offset, n))
      return PICASSO_LLA_INVALID_INPUT;
    return run_actnewton_v2(
        &obj, lambda, nlambda, gamma, max_ite, pprec, reg_type, intercept,
        dfmax, lla_max_stages, beta, intcpt, ite_lamb, size_act, runt,
        num_fit, failed_lambda, failed_stage, lla_stages, objective, kkt,
        stationarity, smooth_objective);
  } catch (...) {
    if (failed_lambda != nullptr) *failed_lambda = 0;
    return PICASSO_LLA_EXCEPTION;
  }
}

extern "C" int SolveSqrtLinearRegressionV3(
    double *Y, double *X, int n, int d, double *lambda, int nlambda,
    double gamma, int max_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, int lla_max_stages,
    int *failed_lambda, int *failed_stage, int *lla_stages,
    double *objective, double *kkt, double *stationarity,
    double *smooth_objective) {
  if (num_fit != nullptr) *num_fit = 0;
  initialize_lla_diagnostics(nlambda, failed_lambda, failed_stage, lla_stages,
                             objective, kkt, stationarity,
                             smooth_objective);
  if (safe_solver_output_dimensions(d, nlambda))
    zero_solver_outputs(d, nlambda, beta, intcpt, ite_lamb, size_act, runt);
  if (invalid_lla_inputs(Y, X, lambda, n, d, nlambda, gamma, max_ite,
                         pprec, reg_type, dfmax, lla_max_stages))
    return PICASSO_LLA_INVALID_INPUT;

  try {
    picasso::SqrtMSEObjective obj(
        X, Y, n, d, intercept, usePython, c_api_design_storage(usePython));
    return run_actnewton_v2(
        &obj, lambda, nlambda, gamma, max_ite, pprec, reg_type, intercept,
        dfmax, lla_max_stages, beta, intcpt, ite_lamb, size_act, runt,
        num_fit, failed_lambda, failed_stage, lla_stages, objective, kkt,
        stationarity, smooth_objective);
  } catch (...) {
    if (failed_lambda != nullptr) *failed_lambda = 0;
    return PICASSO_LLA_EXCEPTION;
  }
}

extern "C" void SolveLinearRegressionNaiveUpdate(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax,
    double *beta, double *intcpt, int *ite_lamb, int *size_act, double *runt,
    int *num_fit, bool usePython) {
  solve_gaussian_c_api<picasso::GaussianNaiveUpdateObjective>(
      Y, X, nn, dd, lambda, nnlambda, gamma, mmax_ite, pprec, reg_type,
      intercept, dfmax, beta, intcpt, ite_lamb, size_act, runt, num_fit,
      usePython, nullptr, nullptr);
}

extern "C" void SolveLinearRegressionCovUpdate(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax,
    double *beta, double *intcpt, int *ite_lamb, int *size_act, double *runt,
    int *num_fit, bool usePython) {
  solve_gaussian_c_api<picasso::GaussianCovUpdateObjective>(
      Y, X, nn, dd, lambda, nnlambda, gamma, mmax_ite, pprec, reg_type,
      intercept, dfmax, beta, intcpt, ite_lamb, size_act, runt, num_fit,
      usePython, nullptr, nullptr);
}

extern "C" void SolveLinearRegressionNaiveUpdateV2(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython,
    double *smooth_objective) {
  solve_gaussian_c_api<picasso::GaussianNaiveUpdateObjective>(
      Y, X, nn, dd, lambda, nnlambda, gamma, mmax_ite, pprec, reg_type,
      intercept, dfmax, beta, intcpt, ite_lamb, size_act, runt, num_fit,
      usePython, smooth_objective, nullptr);
}

extern "C" void SolveLinearRegressionCovUpdateV2(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython,
    double *smooth_objective) {
  solve_gaussian_c_api<picasso::GaussianCovUpdateObjective>(
      Y, X, nn, dd, lambda, nnlambda, gamma, mmax_ite, pprec, reg_type,
      intercept, dfmax, beta, intcpt, ite_lamb, size_act, runt, num_fit,
      usePython, smooth_objective, nullptr);
}

extern "C" int SolveLinearRegressionNaiveUpdateV3(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, double *smooth_objective,
    int *failed_lambda) {
  return solve_gaussian_c_api<picasso::GaussianNaiveUpdateObjective>(
      Y, X, nn, dd, lambda, nnlambda, gamma, mmax_ite, pprec, reg_type,
      intercept, dfmax, beta, intcpt, ite_lamb, size_act, runt, num_fit,
      usePython, smooth_objective, failed_lambda);
}

extern "C" int SolveLinearRegressionCovUpdateV3(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax, double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, int *num_fit, bool usePython, double *smooth_objective,
    int *failed_lambda) {
  return solve_gaussian_c_api<picasso::GaussianCovUpdateObjective>(
      Y, X, nn, dd, lambda, nnlambda, gamma, mmax_ite, pprec, reg_type,
      intercept, dfmax, beta, intcpt, ite_lamb, size_act, runt, num_fit,
      usePython, smooth_objective, failed_lambda);
}
