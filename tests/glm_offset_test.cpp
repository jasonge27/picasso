#include <picasso/c_api.hpp>
#include <picasso/objective.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void fail(const std::string &message) {
  std::cerr << "glm_offset_test: " << message << std::endl;
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

double logistic_probability(double eta) {
  if (eta >= 0.0) {
    const double scaled = std::exp(-eta);
    return 1.0 / (1.0 + scaled);
  }
  const double scaled = std::exp(eta);
  return scaled / (1.0 + scaled);
}

void test_offset_validation_and_virtual_dispatch() {
  const int n = 5;
  const int d = 1;
  const double x[n] = {-2.0, -1.0, 0.0, 1.0, 2.0};
  const double y[n] = {0.0, 0.0, 1.0, 1.0, 1.0};
  picasso::LogisticObjective objective(x, y, n, d, true, false);
  picasso::ObjFunction *base = &objective;

  const double original_intercept = objective.get_model_coef(-1);
  const double original_deviance = objective.get_deviance();
  const double valid[n] = {-1.0, -0.5, 0.0, 0.5, 1.0};
  expect_true(!base->set_offset(valid, n - 1),
              "wrong offset length must be rejected");
  expect_near(objective.get_model_coef(-1), original_intercept, 0.0,
              "length failure changed intercept");
  expect_near(objective.get_deviance(), original_deviance, 0.0,
              "length failure changed deviance");

  double nonfinite[n] = {-1.0, 0.0, 1.0, 2.0, 3.0};
  nonfinite[2] = std::numeric_limits<double>::infinity();
  expect_true(!base->set_offset(nonfinite, n),
              "non-finite offset must be rejected");
  expect_near(objective.get_model_coef(-1), original_intercept, 0.0,
              "finite-value failure changed intercept");

  objective.set_model_coef(0.75, 0);
  expect_true(base->set_offset(valid, n),
              "valid offset failed through base virtual method");
  expect_true(objective.get_model_coef(0) == 0.0,
              "offset null reinitialization did not clear beta");
  expect_true(objective.get_model_Xb_ref().isZero(0.0),
              "offset null reinitialization did not clear X beta");
}

void test_logistic_null_refresh() {
  const int n = 7;
  const int d = 2;
  const double x[n * d] = {
      -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0,
       1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.5};
  const double y[n] = {0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0};
  const double offset[n] = {-6.0, -2.0, -0.5, 0.25, 1.5, 4.0, 8.0};
  picasso::LogisticObjective objective(x, y, n, d, true, false);
  expect_true(objective.set_offset(offset, n),
              "valid logistic offset was rejected");

  const double intercept = objective.get_model_coef(-1);
  double intercept_score = 0.0;
  double gradients[d] = {0.0, 0.0};
  for (int row = 0; row < n; ++row) {
    const double probability = logistic_probability(intercept + offset[row]);
    intercept_score += probability - y[row];
    for (int feature = 0; feature < d; ++feature) {
      gradients[feature] +=
          (y[row] - probability) * x[feature * n + row];
    }
  }
  intercept_score /= n;
  expect_near(objective.get_intercept_gradient(), intercept_score, 2e-15,
              "logistic intercept gradient state");
  expect_true(std::fabs(intercept_score) < 2e-14,
              "logistic null intercept did not solve its score equation");
  for (int feature = 0; feature < d; ++feature) {
    expect_near(objective.get_grad(feature), gradients[feature] / n, 2e-14,
                "logistic gradient was not refreshed");
  }
  expect_near(objective.get_deviance(), objective.eval(), 0.0,
              "logistic deviance was not refreshed");
}

void test_extreme_logistic_boundaries() {
  const int n = 4;
  const int d = 1;
  const double x[n] = {0.0, 0.0, 0.0, 0.0};
  const double all_zero[n] = {0.0, 0.0, 0.0, 0.0};
  const double all_one[n] = {1.0, 1.0, 1.0, 1.0};
  const double offset[n] = {-1000.0, -100.0, 100.0, 1000.0};

  picasso::LogisticObjective zero_objective(
      x, all_zero, n, d, true, false);
  picasso::LogisticObjective one_objective(
      x, all_one, n, d, true, false);
  expect_true(zero_objective.set_offset(offset, n),
              "extreme all-zero logistic offset was rejected");
  expect_true(one_objective.set_offset(offset, n),
              "extreme all-one logistic offset was rejected");

  picasso::LogisticObjective *objectives[] = {
      &zero_objective, &one_objective};
  for (int index = 0; index < 2; ++index) {
    expect_true(std::isfinite(objectives[index]->get_model_coef(-1)),
                "extreme logistic intercept is non-finite");
    expect_true(std::isfinite(objectives[index]->eval()),
                "extreme logistic loss is non-finite");
    expect_true(std::isfinite(objectives[index]->get_intercept_gradient()),
                "extreme logistic score is non-finite");
    expect_true(std::fabs(objectives[index]->get_intercept_gradient()) < 2e-12,
                "extreme logistic boundary score exceeded floor tolerance");
  }
}

void test_poisson_log_sum_exp_oracle() {
  const int n = 6;
  const int d = 1;
  const double x[n] = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
  const double y[n] = {0.0, 1.0, 4.0, 2.0, 0.0, 3.0};
  const double offset[n] = {-1000.0, -20.0, -1.0, 0.0, 20.0, 1000.0};
  picasso::PoissonObjective objective(x, y, n, d, true, false);
  expect_true(objective.set_offset(offset, n),
              "valid Poisson offset was rejected");

  long double response_sum = 0.0L;
  long double scaled_sum = 0.0L;
  const long double maximum = 1000.0L;
  for (int index = 0; index < n; ++index) {
    response_sum += y[index];
    scaled_sum += std::exp(static_cast<long double>(offset[index]) - maximum);
  }
  const long double expected =
      std::log(response_sum) - maximum - std::log(scaled_sum);
  expect_near(objective.get_model_coef(-1), static_cast<double>(expected),
              2e-13, "Poisson log-sum-exp intercept oracle");
  expect_true(std::fabs(objective.get_intercept_gradient()) < 2e-13,
              "Poisson null intercept did not solve its score equation");
  expect_near(objective.get_deviance(), objective.eval(), 0.0,
              "Poisson deviance was not refreshed");
  expect_true(std::isfinite(objective.get_grad(0)),
              "Poisson gradient was not refreshed to a finite value");

  const double all_zero[n] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  picasso::PoissonObjective zero_objective(
      x, all_zero, n, d, true, false);
  expect_true(zero_objective.set_offset(offset, n),
              "all-zero Poisson offset was rejected");
  expect_true(std::isfinite(zero_objective.get_model_coef(-1)) &&
                  std::isfinite(zero_objective.eval()),
              "all-zero offset-aware Poisson null model is non-finite");
}

void test_no_intercept_refresh() {
  const int n = 5;
  const int d = 1;
  const double x[n] = {-2.0, -1.0, 0.0, 1.0, 2.0};
  const double logistic_y[n] = {0.0, 1.0, 0.0, 1.0, 1.0};
  const double poisson_y[n] = {0.0, 1.0, 2.0, 1.0, 3.0};
  const double offset[n] = {-2.0, -1.0, 0.0, 1.0, 2.0};
  picasso::LogisticObjective logistic(
      x, logistic_y, n, d, false, false);
  picasso::PoissonObjective poisson(
      x, poisson_y, n, d, false, false);
  expect_true(logistic.set_offset(offset, n) && poisson.set_offset(offset, n),
              "no-intercept offset was rejected");
  expect_true(logistic.get_model_coef(-1) == 0.0 &&
                  poisson.get_model_coef(-1) == 0.0,
              "no-intercept offset changed an intercept");
  expect_near(logistic.get_deviance(), logistic.eval(), 0.0,
              "no-intercept logistic deviance was stale");
  expect_near(poisson.get_deviance(), poisson.eval(), 0.0,
              "no-intercept Poisson deviance was stale");
  expect_true(std::isfinite(logistic.get_grad(0)) &&
                  std::isfinite(poisson.get_grad(0)),
              "no-intercept gradients were not refreshed");
}

struct PathOutput {
  int status;
  int fitted;
  std::vector<double> beta;
  std::vector<double> intercept;
};

PathOutput logistic_path(const std::vector<double> &x,
                         const std::vector<double> &y, int n, int d,
                         double *offset) {
  double lambda[2] = {0.2, 0.08};
  PathOutput output;
  output.fitted = 0;
  output.beta.assign(2 * d, 0.0);
  output.intercept.assign(2, 0.0);
  output.status = SolveLogisticRegressionV2(
      const_cast<double *>(y.data()), const_cast<double *>(x.data()), n, d,
      lambda, 2, 3.0, 500, 1e-6, 1, true, -1, offset,
      output.beta.data(), output.intercept.data(), nullptr, nullptr, nullptr,
      &output.fitted, false, 3, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr);
  return output;
}

void test_constant_offset_path_invariance() {
  const int n = 80;
  const int d = 2;
  std::vector<double> x(n * d);
  std::vector<double> y(n);
  for (int row = 0; row < n; ++row) {
    x[row] = std::sin(0.17 * row);
    x[n + row] = std::cos(0.11 * row);
    const double eta = -0.3 + 1.2 * x[row] - 0.8 * x[n + row];
    y[row] = (static_cast<int>(row * 37) % 100) / 100.0 <
                     logistic_probability(eta)
                 ? 1.0
                 : 0.0;
  }
  std::vector<double> constant_offset(n, 2.75);
  PathOutput baseline = logistic_path(x, y, n, d, nullptr);
  PathOutput shifted =
      logistic_path(x, y, n, d, constant_offset.data());
  expect_true(baseline.status == PICASSO_LLA_COMPLETED &&
                  shifted.status == PICASSO_LLA_COMPLETED,
              "constant-offset comparison path did not complete");
  expect_true(baseline.fitted == 2 && shifted.fitted == 2,
              "constant-offset comparison returned an incomplete path");
  for (std::size_t index = 0; index < baseline.beta.size(); ++index) {
    expect_near(shifted.beta[index], baseline.beta[index], 2e-10,
                "constant offset changed a logistic coefficient");
  }
  for (int index = 0; index < 2; ++index) {
    expect_near(shifted.intercept[index],
                baseline.intercept[index] - 2.75, 2e-10,
                "constant offset did not translate the intercept");
  }
}

void test_c_api_rejects_nonfinite_offset() {
  const int n = 4;
  const int d = 1;
  double x[n] = {-1.0, 0.0, 1.0, 2.0};
  double y[n] = {0.0, 1.0, 0.0, 1.0};
  double lambda[1] = {0.2};
  double offset[n] = {0.0, 0.0,
                      std::numeric_limits<double>::quiet_NaN(), 0.0};
  double beta[1] = {7.0};
  double intercept[1] = {7.0};
  int fitted = 7;
  const int status = SolveLogisticRegressionV2(
      y, x, n, d, lambda, 1, 3.0, 100, 1e-6, 1, true, -1, offset,
      beta, intercept, nullptr, nullptr, nullptr, &fitted, false, 3,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  expect_true(status == PICASSO_LLA_INVALID_INPUT && fitted == 0,
              "C API did not reject a non-finite offset");
  expect_true(beta[0] == 0.0 && intercept[0] == 0.0,
              "C API invalid-offset outputs were not initialized");
}

void test_c_api_reports_unsafe_poisson_link() {
  const int n = 4;
  const int d = 1;
  double x[n] = {0.0, 0.0, 0.0, 0.0};
  double y[n] = {0.0, 1.0, 2.0, 3.0};
  double lambda[1] = {0.2};
  double offset[n] = {1000.0, 1000.0, 1000.0, 1000.0};
  double beta[1] = {0.0};
  double intercept[1] = {0.0};
  int fitted = 0;
  int failed_lambda = -1;
  const int status = SolvePoissonRegressionV2(
      y, x, n, d, lambda, 1, 3.0, 100, 1e-6, 1, false, -1, offset,
      beta, intercept, nullptr, nullptr, nullptr, &fitted, false, 3,
      &failed_lambda, nullptr, nullptr, nullptr, nullptr, nullptr);
  expect_true(status == PICASSO_LLA_NUMERICAL_FAILURE && fitted == 0 &&
                  failed_lambda == 0,
              "unsafe Poisson link was not reported as numerical failure");
}

}  // namespace

int main() {
  test_offset_validation_and_virtual_dispatch();
  test_logistic_null_refresh();
  test_extreme_logistic_boundaries();
  test_poisson_log_sum_exp_oracle();
  test_no_intercept_refresh();
  test_constant_offset_path_invariance();
  test_c_api_rejects_nonfinite_offset();
  test_c_api_reports_unsafe_poisson_link();
  std::cout << "glm_offset_test passed" << std::endl;
  return 0;
}
