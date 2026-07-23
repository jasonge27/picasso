#include <picasso/actgd.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void fail(const std::string &message) {
  std::cerr << "gaussian_covariance_cache_test: " << message << std::endl;
  std::exit(1);
}

void expect_true(bool condition, const std::string &message) {
  if (!condition) fail(message);
}

void expect_near(double actual, double expected, double tolerance,
                 const std::string &message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) > tolerance) {
    std::cerr << message << ": actual=" << actual
              << " expected=" << expected
              << " tolerance=" << tolerance << std::endl;
    fail("numeric comparison failed");
  }
}

// Frozen full-Gram implementation used before the lazy-column change. Keeping
// this oracle in the test makes every returned path coefficient comparable to
// the previous covariance algorithm without retaining its O(d^2) production
// allocation.
class FullGramCovarianceReference : public picasso::ObjFunction {
 private:
  Eigen::ArrayXd XX;
  Eigen::MatrixXd C;
  Eigen::ArrayXd Xy;
  Eigen::ArrayXd Xmean;
  double Ymean;
  bool include_intercept;

 public:
  FullGramCovarianceReference(const double *xmat, const double *y, int n,
                              int d, bool intercept)
      : ObjFunction(xmat, y, n, d, false),
        Ymean(0.0),
        include_intercept(intercept) {
    XX.resize(d);
    Xy.resize(d);
    Xmean.resize(d);
    for (int j = 0; j < d; ++j) Xmean[j] = X.col(j).sum() / n;
    Ymean = Y.sum() / n;

    if (include_intercept) {
      C.resize(d, d);
      for (int j = 0; j < d; ++j) {
        for (int k = 0; k <= j; ++k) {
          const double value =
              ((X.col(j) - Xmean[j]) * (X.col(k) - Xmean[k])).sum() / n;
          C(j, k) = value;
          C(k, j) = value;
        }
      }
    } else {
      C.noalias() = X.matrix().transpose() * X.matrix();
      C /= n;
    }
    for (int j = 0; j < d; ++j) {
      XX[j] = std::max(0.0, C(j, j));
      C(j, j) = XX[j];
    }

    if (include_intercept) {
      const Eigen::ArrayXd centered_y = Y - Ymean;
      for (int j = 0; j < d; ++j)
        Xy[j] = ((X.col(j) - Xmean[j]) * centered_y).sum() / n;
    } else {
      Xy = (X.matrix().transpose() * Y.matrix()).array() / n;
    }
    gr = Xy;
    intercept_update();
    deviance = std::fabs(eval());
  }

  double coordinate_descent(picasso::RegFunction *regularizer, int idx) {
    const double beta_old = model_param.beta[idx];
    const double linear_term = gr[idx] + beta_old * XX[idx];
    model_param.beta[idx] =
        XX[idx] > 0.0
            ? regularizer->coordinate_minimize(linear_term, XX[idx])
            : 0.0;
    const double delta = model_param.beta[idx] - beta_old;
    if (delta != 0.0) gr -= delta * C.col(idx).array();
    return model_param.beta[idx];
  }

  void intercept_update() {
    model_param.intercept =
        include_intercept ? Ymean - (Xmean * model_param.beta).sum() : 0.0;
  }

  void update_auxiliary() {
    gr = Xy - (C * model_param.beta.matrix()).array();
  }

  void update_gradient(int) {}

  double get_local_change(double old, int idx) {
    const double difference = old - model_param.beta[idx];
    return difference * difference * XX[idx];
  }

  double eval() {
    double value = 0.0;
    for (int i = 0; i < n; ++i) {
      const double prediction = model_param.intercept +
          model_param.beta.matrix().dot(X.row(i).matrix());
      const double residual = Y[i] - prediction;
      value += residual * residual;
    }
    return value / n;
  }
};

double uniform_signed(std::uint64_t &state) {
  state = state * UINT64_C(6364136223846793005) +
          UINT64_C(1442695040888963407);
  const double unit = static_cast<double>(state >> 11) *
                      (1.0 / 9007199254740992.0);
  return 2.0 * unit - 1.0;
}

void make_fixture(int n, int d, std::vector<double> *x,
                  std::vector<double> *y) {
  x->resize(static_cast<std::size_t>(n) * d);
  y->assign(n, 0.35);
  std::vector<double> shared(n);
  std::uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
  for (int i = 0; i < n; ++i) {
    shared[i] = uniform_signed(state);
    (*y)[i] += 0.08 * uniform_signed(state);
  }
  const double signal[] = {1.4, -1.1, 0.8, -0.65, 0.5, -0.4};
  const int signal_size =
      static_cast<int>(sizeof(signal) / sizeof(signal[0]));
  for (int j = 0; j < d; ++j) {
    for (int i = 0; i < n; ++i) {
      const double value =
          0.92 * uniform_signed(state) + 0.08 * shared[i];
      (*x)[static_cast<std::size_t>(j) * n + i] = value;
      if (j < signal_size) (*y)[i] += signal[j] * value;
    }
  }
}

double direct_residual_objective(const std::vector<double> &x,
                                 const std::vector<double> &y, int n, int d,
                                 const picasso::ModelParam &model) {
  long double squared_residual = 0.0L;
  for (int i = 0; i < n; ++i) {
    long double prediction = model.intercept;
    for (int j = 0; j < d; ++j)
      prediction += static_cast<long double>(
                        x[static_cast<std::size_t>(j) * n + i]) *
                    model.beta[j];
    const long double residual =
        static_cast<long double>(y[static_cast<std::size_t>(i)]) - prediction;
    squared_residual += residual * residual;
  }
  return static_cast<double>(squared_residual / n);
}

void check_eval_matches_direct_residual(bool include_intercept) {
  const int n = 73;
  const int d = 11;
  std::vector<double> x;
  std::vector<double> y;
  make_fixture(n, d, &x, &y);
  picasso::GaussianCovUpdateObjective objective(
      x.data(), y.data(), n, d, include_intercept, false);

  picasso::ModelParam model(d);
  model.intercept = -0.37;
  for (int j = 0; j < d; ++j)
    model.beta[j] = j < 7 ? 0.03 * (j + 1) * (j % 2 == 0 ? 1.0 : -1.0)
                          : 0.0;
  objective.set_model_param(model);
  objective.update_auxiliary();
  const int cached_before = objective.get_cached_covariance_column_count();
  expect_near(objective.eval(),
              direct_residual_objective(x, y, n, d, model), 3e-14,
              "external-model residual objective");
  expect_true(objective.get_cached_covariance_column_count() == cached_before,
              "objective evaluation materialized a Gram column");

  objective.intercept_update();
  model.intercept = objective.get_model_coef(-1);
  expect_near(objective.eval(),
              direct_residual_objective(x, y, n, d, model), 3e-14,
              "profiled residual objective");
  const Eigen::ArrayXd predictor_sentinel =
      Eigen::ArrayXd::Constant(n, -777.0);
  objective.set_model_Xb(predictor_sentinel);
  expect_near(objective.path_eval(),
              direct_residual_objective(x, y, n, d, model), 3e-13,
              "profiled covariance identity objective");
  expect_true((objective.get_model_Xb_ref() == predictor_sentinel).all(),
              "well-conditioned path objective used the residual fallback");
  expect_true(objective.get_cached_covariance_column_count() == cached_before,
              "profiled objective materialized a Gram column");
}

void check_path_eval_cancellation_fallback() {
  const int n = 8;
  const int d = 2;
  const double x[n * d] = {
      -2.0, -1.0, 0.0, 1.0, 2.0, -1.0, 1.0, 0.0,
       0.0,  1.0,-1.0, 2.0,-2.0,  1.0,-1.0, 0.0};
  double y[n];
  for (int i = 0; i < n; ++i)
    y[i] = 0.25 + 1.5 * x[i] - 0.9 * x[n + i];

  picasso::GaussianCovUpdateObjective objective(x, y, n, d, true, false);
  picasso::ModelParam model(d);
  model.beta[0] = 1.5;
  model.beta[1] = -0.9;
  objective.set_model_param(model);
  objective.update_auxiliary();
  objective.intercept_update();
  model.intercept = objective.get_model_coef(-1);

  const Eigen::ArrayXd predictor_sentinel =
      Eigen::ArrayXd::Constant(n, 123.0);
  objective.set_model_Xb(predictor_sentinel);
  const double direct = direct_residual_objective(
      std::vector<double>(x, x + n * d), std::vector<double>(y, y + n),
      n, d, model);
  expect_near(objective.path_eval(), direct, 1e-28,
              "near-interpolation fallback objective");
  expect_true(!(objective.get_model_Xb_ref() == predictor_sentinel).all(),
              "near-interpolation objective did not fall back to residuals");
}

void check_lazy_cache_lifecycle() {
  const int n = 8;
  const int d = 3;
  const double x[n * d] = {
      -2.0, -1.0, 0.0, 1.0, 2.0, -1.0, 1.0, 0.0,
       0.0,  1.0,-1.0, 2.0,-2.0,  1.0,-1.0, 0.0,
       1.0, -0.5, 0.4,-0.8, 0.7, -0.2, 0.3,-0.9};
  double y[n];
  for (int i = 0; i < n; ++i)
    y[i] = 0.2 + 1.5 * x[i] - 0.9 * x[n + i];

  picasso::GaussianCovUpdateObjective objective(x, y, n, d, true, false);
  expect_true(objective.get_cached_covariance_column_count() == 0,
              "constructor materialized a Gram column");

  picasso::ModelParam external(d);
  external.beta[0] = 0.4;
  objective.set_model_param(external);
  objective.update_auxiliary();
  expect_true(objective.get_cached_covariance_column_count() == 0,
              "gradient recomputation materialized a Gram column");

  picasso::ModelParam zero(d);
  objective.set_model_param(zero);
  objective.update_auxiliary();
  picasso::RegL1 l1;
  l1.set_param(1e6, 0.0);
  expect_true(objective.coordinate_descent(&l1, 0) == 0.0,
              "large lambda should keep the first coordinate at zero");
  expect_true(objective.get_cached_covariance_column_count() == 0,
              "zero coordinate update materialized a Gram column");

  l1.set_param(0.01, 0.0);
  expect_true(std::fabs(objective.coordinate_descent(&l1, 0)) > 0.0,
              "first signal coordinate did not update");
  expect_true(objective.get_cached_covariance_column_count() == 1,
              "first nonzero update did not materialize exactly one column");
  objective.coordinate_descent(&l1, 0);
  expect_true(objective.get_cached_covariance_column_count() == 1,
              "revisiting a coordinate duplicated its cached column");
  expect_true(std::fabs(objective.coordinate_descent(&l1, 1)) > 0.0,
              "second signal coordinate did not update");
  expect_true(objective.get_cached_covariance_column_count() == 2,
              "second nonzero coordinate did not add exactly one column");
}

void check_full_path_equivalence(bool include_intercept) {
  const int n = 160;
  const int d = 24;
  std::vector<double> x;
  std::vector<double> y;
  make_fixture(n, d, &x, &y);

  FullGramCovarianceReference reference(
      x.data(), y.data(), n, d, include_intercept);
  picasso::GaussianCovUpdateObjective lazy(
      x.data(), y.data(), n, d, include_intercept, false);

  double lambda_max = 0.0;
  for (int j = 0; j < d; ++j) {
    expect_near(lazy.get_grad(j), reference.get_grad(j), 2e-14,
                "initial gradient");
    lambda_max = std::max(lambda_max, std::fabs(reference.get_grad(j)));
  }
  const double ratios[] = {1.0, 0.82, 0.67, 0.54, 0.43, 0.34};
  const int nlambda = static_cast<int>(sizeof(ratios) / sizeof(ratios[0]));
  std::vector<double> lambdas(nlambda);
  for (int i = 0; i < nlambda; ++i)
    lambdas[i] = lambda_max * ratios[i];

  picasso::solver::PicassoSolverParams params;
  params.set_lambdas(lambdas.data(), nlambda);
  params.reg_type = picasso::solver::L1;
  params.include_intercept = include_intercept;
  params.prec = 1e-9;
  params.max_iter = 1000;
  params.min_lambda_count = nlambda + 1;

  picasso::solver::ActGDSolver reference_solver(&reference, params);
  picasso::solver::ActGDSolver lazy_solver(&lazy, params);
  reference_solver.solve();
  lazy_solver.solve();
  expect_true(reference_solver.get_num_lambdas_fit() == nlambda,
              "reference path was unexpectedly truncated");
  expect_true(lazy_solver.get_num_lambdas_fit() == nlambda,
              "lazy path was unexpectedly truncated");

  for (int path_index = 0; path_index < nlambda; ++path_index) {
    const picasso::ModelParam &expected =
        reference_solver.get_model_param(path_index);
    const picasso::ModelParam &actual =
        lazy_solver.get_model_param(path_index);
    expect_near(actual.intercept, expected.intercept, 2e-13,
                "path intercept");
    for (int j = 0; j < d; ++j)
      expect_near(actual.beta[j], expected.beta[j], 2e-13,
                  "path coefficient");
    const double direct = direct_residual_objective(
        x, y, n, d, actual);
    expect_near(lazy_solver.get_smooth_objective_path()[path_index], direct,
                5e-12 * std::max(1.0, direct),
                "solver covariance identity objective");
  }
  expect_true(lazy.get_cached_covariance_column_count() > 0,
              "sparse path did not materialize any covariance column");
  expect_true(lazy.get_cached_covariance_column_count() < d,
              "sparse path materialized the full Gram matrix");
}

}  // namespace

int main() {
  check_lazy_cache_lifecycle();
  check_eval_matches_direct_residual(true);
  check_eval_matches_direct_residual(false);
  check_path_eval_cancellation_fallback();
  check_full_path_equivalence(true);
  check_full_path_equivalence(false);
  std::cout << "gaussian_covariance_cache_test passed" << std::endl;
  return 0;
}
