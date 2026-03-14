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
    int dfmax, double *offset,
    double *beta, double *intcpt, int *ite_lamb, int *size_act, double *runt,
    int *num_fit, bool usePython) {
  if (invalid_problem_inputs(Y, X, n, d)) {
    zero_solver_outputs(d, nlambda, beta, intcpt, ite_lamb, size_act, runt);
    if (num_fit != nullptr) *num_fit = 0;
    return;
  }

  picasso::LogisticObjective obj(X, Y, n, d, intercept, usePython);
  if (offset != nullptr) obj.set_offset(offset, n);
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
    int dfmax, double *offset,
    double *beta, double *intcpt, int *ite_lamb, int *size_act, double *runt,
    int *num_fit, bool usePython) {
  if (invalid_problem_inputs(Y, X, nn, dd)) {
    zero_solver_outputs(dd, nnlambda, beta, intcpt, ite_lamb, size_act, runt);
    if (num_fit != nullptr) *num_fit = 0;
    return;
  }

  picasso::PoissonObjective obj(X, Y, nn, dd, intercept, usePython);
  if (offset != nullptr) obj.set_offset(offset, nn);
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

// ---- Multinomial (standalone coordinate descent solver) ----

namespace {
inline double soft_threshold_mn(double x, double thr) {
  if (x > thr)  return x - thr;
  if (x < -thr) return x + thr;
  return 0.0;
}

inline double threshold_mcp_mn(double x, double lam, double gam) {
  if (std::fabs(x) > gam * lam) return x;
  return soft_threshold_mn(x, lam) / (1.0 - 1.0 / gam);
}

inline double threshold_scad_mn(double x, double lam, double gam) {
  if (std::fabs(x) > gam * lam) return x;
  if (std::fabs(x) > 2.0 * lam)
    return soft_threshold_mn(x, gam * lam / (gam - 1.0)) /
           (1.0 - 1.0 / (gam - 1.0));
  return soft_threshold_mn(x, lam);
}
}  // namespace (multinomial helpers)

extern "C" void SolveMultinomialRegression(
    double *Y_int, double *X, int n, int d, int K,
    double *lambda, int nlambda, double gamma,
    int max_ite, double pprec, int reg_type,
    bool intercept_flag, int dfmax,
    double *beta_out, double *intcpt_out,
    int *ite_lamb, int *size_act, double *runt, int *num_fit,
    bool usePython) {

  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using Eigen::VectorXi;

  if (Y_int == nullptr || X == nullptr || n <= 0 || d <= 0 || K < 2) {
    if (num_fit) *num_fit = 0;
    return;
  }

  // Design matrix (column-major in R: X[j*n + i] = X(i,j))
  MatrixXd Xmat(n, d);
  if (!usePython) {
    for (int j = 0; j < d; j++)
      for (int i = 0; i < n; i++) Xmat(i, j) = X[j * n + i];
  } else {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < d; j++) Xmat(i, j) = X[i * d + j];
  }

  VectorXi Yclass(n);
  for (int i = 0; i < n; i++) Yclass(i) = static_cast<int>(Y_int[i]);

  MatrixXd Y_oh = MatrixXd::Zero(n, K);
  for (int i = 0; i < n; i++) Y_oh(i, Yclass(i)) = 1.0;

  MatrixXd beta_mat   = MatrixXd::Zero(d, K);
  VectorXd intcpt_vec = VectorXd::Zero(K);
  MatrixXd Xb         = MatrixXd::Zero(n, K);
  MatrixXd P(n, K), W(n, K), R(n, K);

  if (intercept_flag) {
    for (int k = 0; k < K; k++) {
      double nk = 0;
      for (int i = 0; i < n; i++) if (Yclass(i) == k) nk += 1.0;
      intcpt_vec(k) = std::log(std::max(nk / n, 1e-8));
    }
  }

  auto update_softmax = [&]() {
    for (int i = 0; i < n; i++) {
      double max_lp = -1e300;
      for (int k = 0; k < K; k++) {
        double lp = intcpt_vec(k) + Xb(i, k);
        if (lp > max_lp) max_lp = lp;
      }
      double sum_exp = 0.0;
      for (int k = 0; k < K; k++) {
        P(i, k) = std::exp(intcpt_vec(k) + Xb(i, k) - max_lp);
        sum_exp += P(i, k);
      }
      for (int k = 0; k < K; k++) {
        P(i, k) /= sum_exp;
        W(i, k) = P(i, k) * (1.0 - P(i, k));
        R(i, k) = Y_oh(i, k) - P(i, k);
      }
    }
  };

  auto eval_loss = [&]() -> double {
    double v = 0.0;
    for (int i = 0; i < n; i++)
      v -= std::log(std::max(P(i, Yclass(i)), 1e-15));
    return v / n;
  };

  auto grad_kj = [&](int k, int j) -> double {
    double g = 0.0;
    for (int i = 0; i < n; i++) g += R(i, k) * Xmat(i, j);
    return g / n;
  };

  auto cd_step = [&](int k, int j, double stage_lam) {
    double a = 0.0, rsum = 0.0;
    for (int i = 0; i < n; i++) {
      double xij = Xmat(i, j);
      a    += W(i, k) * xij * xij;
      rsum += R(i, k) * xij;
    }
    a /= n;
    if (a < 1e-15) return;

    double g = beta_mat(j, k) * a + rsum / n;

    double new_beta;
    if (reg_type == 1)
      new_beta = soft_threshold_mn(g, stage_lam) / a;
    else if (reg_type == 2)
      new_beta = threshold_mcp_mn(g, stage_lam, gamma) / a;
    else
      new_beta = threshold_scad_mn(g, stage_lam, gamma) / a;

    double delta = new_beta - beta_mat(j, k);
    if (std::fabs(delta) > 1e-15) {
      for (int i = 0; i < n; i++) {
        double xij = Xmat(i, j);
        Xb(i, k)  += delta * xij;
        R(i, k)   -= delta * W(i, k) * xij;
      }
      beta_mat(j, k) = new_beta;
    }
  };

  auto local_change_kj = [&](int k, int j, double old_b) -> double {
    double a = 0.0;
    for (int i = 0; i < n; i++) {
      double xij = Xmat(i, j);
      a += W(i, k) * xij * xij;
    }
    a /= n;
    double diff = old_b - beta_mat(j, k);
    return a * diff * diff / 2.0;
  };

  auto intercept_update_all = [&]() {
    if (!intercept_flag) return;
    for (int k = 0; k < K; k++) {
      double sum_r = R.col(k).sum();
      double sum_w = W.col(k).sum();
      if (std::fabs(sum_w) < 1e-15) continue;
      double delta = sum_r / sum_w;
      intcpt_vec(k) += delta;
      R.col(k) -= delta * W.col(k);
    }
  };

  update_softmax();
  double null_dev = eval_loss();
  double dev_thr  = null_dev * pprec;

  int total_dim = K * d;
  std::vector<double> stage_lambdas(total_dim);
  std::vector<double> grad_mag(total_dim, 0.0);
  for (int k = 0; k < K; k++)
    for (int j = 0; j < d; j++)
      grad_mag[k * d + j] = std::fabs(grad_kj(k, j));

  int actual_fit = 0;

  for (int li = 0; li < nlambda; li++) {
    double lam = lambda[li];

    for (int idx = 0; idx < total_dim; idx++) stage_lambdas[idx] = lam;

    std::vector<int> actset(total_dim, 0);
    double threshold = (li > 0) ? 2.0 * lam - lambda[li - 1] : 2.0 * lam;
    for (int idx = 0; idx < total_dim; idx++)
      if (grad_mag[idx] > threshold) actset[idx] = 1;

    update_softmax();

    int num_relaxation = (reg_type == 1) ? 1 : 3;
    int total_iter = 0;

    for (int relax = 0; relax < num_relaxation; relax++) {
      for (int iter1 = 0; iter1 < max_ite; iter1++) {
        bool terminate_l1 = true;

        std::vector<double> old_beta(total_dim);
        for (int idx = 0; idx < total_dim; idx++) {
          int k = idx / d, j = idx % d;
          old_beta[idx] = beta_mat(j, k);
        }

        std::vector<int> actset_idx;
        for (int idx = 0; idx < total_dim; idx++) {
          if (!actset[idx]) continue;
          int k = idx / d, j = idx % d;
          cd_step(k, j, stage_lambdas[idx]);
          if (std::fabs(beta_mat(j, k)) > 1e-8) actset_idx.push_back(idx);
        }

        for (int iter2 = 0; iter2 < max_ite; iter2++) {
          bool cvg2 = true;
          for (int idx : actset_idx) {
            int k = idx / d, j = idx % d;
            double old_b = beta_mat(j, k);
            cd_step(k, j, stage_lambdas[idx]);
            if (local_change_kj(k, j, old_b) > dev_thr) cvg2 = false;
          }
          intercept_update_all();
          total_iter++;
          if (cvg2) break;
        }

        for (int idx : actset_idx) {
          int k = idx / d, j = idx % d;
          if (local_change_kj(k, j, old_beta[idx]) > dev_thr)
            terminate_l1 = false;
        }

        update_softmax();

        bool new_active = false;
        for (int idx = 0; idx < total_dim; idx++) {
          if (actset[idx]) continue;
          int k = idx / d, j = idx % d;
          grad_mag[idx] = std::fabs(grad_kj(k, j));
          if (grad_mag[idx] > stage_lambdas[idx]) {
            actset[idx] = 1;
            new_active  = true;
          }
        }

        if (!new_active && terminate_l1) break;
      }

      if (reg_type == 1) break;
      update_softmax();
      for (int idx = 0; idx < total_dim; idx++) {
        int k = idx / d, j = idx % d;
        double b = std::fabs(beta_mat(j, k));
        if (reg_type == 2)
          stage_lambdas[idx] = (b > lam * gamma) ? 0.0 : lam - b / gamma;
        else
          stage_lambdas[idx] = (b > lam * gamma) ? 0.0 :
                               (b > lam)          ? (lam * gamma - b) / (gamma - 1.0)
                                                  : lam;
      }
    }

    int nnz = 0;
    for (int k = 0; k < K; k++)
      for (int j = 0; j < d; j++)
        if (std::fabs(beta_mat(j, k)) > 1e-8) nnz++;

    int base_b = li * K * d;
    int base_i = li * K;
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < d; j++)
        beta_out[base_b + k * d + j] = beta_mat(j, k);
      intcpt_out[base_i + k] = intcpt_vec(k);
    }
    ite_lamb[li] = total_iter;
    size_act[li] = nnz;
    runt[li]     = 0.0;
    actual_fit   = li + 1;

    for (int k = 0; k < K; k++)
      for (int j = 0; j < d; j++)
        grad_mag[k * d + j] = std::fabs(grad_kj(k, j));

    if (dfmax >= 0 && nnz > dfmax) break;
  }

  if (num_fit) *num_fit = actual_fit;
}
