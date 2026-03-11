#include <picasso/actgd.hpp>
#include <picasso/actnewton.hpp>
#include <picasso/c_api.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {
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

bool invalid_problem_inputs(double *Y, double *X, int n, int d) {
  return Y == nullptr || X == nullptr || n <= 0 || d <= 0;
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

template <typename SolverType>
void extract_results(SolverType &solver, int d, int nlambda, double *beta,
                     double *intcpt, int *ite_lamb, int *size_act,
                     double *runt, int *num_fit) {
  int actual_fit = solver.get_num_lambdas_fit();
  if (num_fit != nullptr) *num_fit = actual_fit;

  const auto &itercnt_path = solver.get_itercnt_path();
  const auto &runtime_path = solver.get_runtime_path();
  for (int i = 0; i < actual_fit; i++) {
    const picasso::ModelParam &model_param = solver.get_model_param(i);
    ite_lamb[i] = itercnt_path[i];
    size_act[i] = 0;
    for (int j = 0; j < d; j++) {
      beta[i * d + j] = model_param.beta[j];
      if (fabs(beta[i * d + j]) > 1e-8) size_act[i]++;
    }
    intcpt[i] = model_param.intercept;
    runt[i] = runtime_path[i];
  }
}
}  // namespace

extern "C" void SolveLogisticRegression(
    double *Y, double *X, int n, int d, double *lambda, int nlambda,
    double gamma, int max_ite, double pprec, int reg_type, bool intercept,
    int dfmax,
    double *beta, double *intcpt, int *ite_lamb, int *size_act, double *runt,
    int *num_fit, bool usePython) {
  if (invalid_problem_inputs(Y, X, n, d)) {
    zero_solver_outputs(d, nlambda, beta, intcpt, ite_lamb, size_act, runt);
    if (num_fit != nullptr) *num_fit = 0;
    return;
  }

  picasso::LogisticObjective obj(X, Y, n, d, intercept, usePython);
  auto param = make_params(lambda, nlambda, gamma, max_ite, pprec, reg_type,
                           intercept, dfmax);
  picasso::solver::ActNewtonSolver solver(&obj, param);
  solver.solve();
  extract_results(solver, d, nlambda, beta, intcpt, ite_lamb, size_act, runt,
                  num_fit);
}

extern "C" void SolvePoissonRegression(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax,
    double *beta, double *intcpt, int *ite_lamb, int *size_act, double *runt,
    int *num_fit, bool usePython) {
  if (invalid_problem_inputs(Y, X, nn, dd)) {
    zero_solver_outputs(dd, nnlambda, beta, intcpt, ite_lamb, size_act, runt);
    if (num_fit != nullptr) *num_fit = 0;
    return;
  }

  picasso::PoissonObjective obj(X, Y, nn, dd, intercept, usePython);
  auto param = make_params(lambda, nnlambda, gamma, mmax_ite, pprec, reg_type,
                           intercept, dfmax);
  picasso::solver::ActNewtonSolver solver(&obj, param);
  solver.solve();
  extract_results(solver, dd, nnlambda, beta, intcpt, ite_lamb, size_act,
                  runt, num_fit);
}

extern "C" void SolveSqrtLinearRegression(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax,
    double *beta, double *intcpt, int *ite_lamb, int *size_act, double *runt,
    int *num_fit, bool usePython) {
  if (invalid_problem_inputs(Y, X, nn, dd)) {
    zero_solver_outputs(dd, nnlambda, beta, intcpt, ite_lamb, size_act, runt);
    if (num_fit != nullptr) *num_fit = 0;
    return;
  }

  picasso::SqrtMSEObjective obj(X, Y, nn, dd, intercept, usePython);
  auto param = make_params(lambda, nnlambda, gamma, mmax_ite, pprec, reg_type,
                           intercept, dfmax);
  picasso::solver::ActNewtonSolver solver(&obj, param);
  solver.solve();
  extract_results(solver, dd, nnlambda, beta, intcpt, ite_lamb, size_act,
                  runt, num_fit);
}

extern "C" void SolveLinearRegressionNaiveUpdate(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax,
    double *beta, double *intcpt, int *ite_lamb, int *size_act, double *runt,
    int *num_fit, bool usePython) {
  if (invalid_problem_inputs(Y, X, nn, dd)) {
    zero_solver_outputs(dd, nnlambda, beta, intcpt, ite_lamb, size_act, runt);
    if (num_fit != nullptr) *num_fit = 0;
    return;
  }

  picasso::GaussianNaiveUpdateObjective obj(X, Y, nn, dd, intercept, usePython);
  auto param = make_params(lambda, nnlambda, gamma, mmax_ite, pprec, reg_type,
                           intercept, dfmax);
  picasso::solver::ActGDSolver solver(&obj, param);
  solver.solve();
  extract_results(solver, dd, nnlambda, beta, intcpt, ite_lamb, size_act,
                  runt, num_fit);
}

extern "C" void SolveLinearRegressionCovUpdate(
    double *Y, double *X, int nn, int dd, double *lambda, int nnlambda,
    double gamma, int mmax_ite, double pprec, int reg_type, bool intercept,
    int dfmax,
    double *beta, double *intcpt, int *ite_lamb, int *size_act, double *runt,
    int *num_fit, bool usePython) {
  if (invalid_problem_inputs(Y, X, nn, dd)) {
    zero_solver_outputs(dd, nnlambda, beta, intcpt, ite_lamb, size_act, runt);
    if (num_fit != nullptr) *num_fit = 0;
    return;
  }

  picasso::GaussianCovUpdateObjective obj(X, Y, nn, dd, intercept, usePython);
  auto param = make_params(lambda, nnlambda, gamma, mmax_ite, pprec, reg_type,
                           intercept, dfmax);
  picasso::solver::ActGDSolver solver(&obj, param);
  solver.solve();
  extract_results(solver, dd, nnlambda, beta, intcpt, ite_lamb, size_act,
                  runt, num_fit);
}
