#include <picasso/c_api.hpp>
#include <picasso/objective.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool require(bool condition, const std::string &message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

std::vector<double> make_design(int n, int d) {
  std::vector<double> x(static_cast<std::size_t>(n) * d);
  for (int row = 0; row < n; ++row) {
    for (int feature = 0; feature < d; ++feature) {
      x[static_cast<std::size_t>(row) * d + feature] =
          std::sin(0.17 * (row + 1.0) * (feature + 1.0)) +
          0.2 * std::cos(0.11 * (row + feature + 2.0));
    }
  }
  return x;
}

bool test_exact_fit_subgradient() {
  const int n = 7;
  const int d = 3;
  const double response = 0.125;
  std::vector<double> x = make_design(n, d);
  std::vector<double> y(n, response);
  picasso::SqrtMSEObjective objective(
      x.data(), y.data(), n, d, true, true);

  bool ok = true;
  ok &= require(objective.eval() == 0.0,
                "constant-response null fit must have zero loss");
  ok &= require(objective.get_model_coef(-1) == response,
                "constant-response intercept must equal the response exactly");
  objective.update_all_gradients();
  for (int feature = 0; feature < d; ++feature) {
    ok &= require(std::isfinite(objective.get_grad(feature)) &&
                      objective.get_grad(feature) == 0.0,
                  "zero-residual coefficient subgradient must be finite zero");
  }
  ok &= require(std::isfinite(objective.get_intercept_gradient()) &&
                    objective.get_intercept_gradient() == 0.0,
                "zero-residual intercept subgradient must be finite zero");

  picasso::RegL1 regularizer;
  regularizer.set_param(0.2, 0.0);
  const double old_coefficient = objective.get_model_coef(0);
  const double updated = objective.coordinate_descent(&regularizer, 0);
  ok &= require(std::isfinite(updated) && updated == old_coefficient,
                "an exact zero-residual solution must not enter a 1/L update");
  ok &= require(objective.get_local_change(old_coefficient, 0) == 0.0,
                "exact-fit no-op must report zero local change");
  return ok;
}

bool test_nonnull_exact_fit_is_not_certified_with_zero_gradient() {
  const int n = 3;
  const int d = 1;
  double x[n * d] = {1.0, 2.0, 3.0};
  double y[n] = {2.0, 4.0, 6.0};
  picasso::SqrtMSEObjective objective(x, y, n, d, false, true);
  Eigen::ArrayXd xb(n);
  for (int row = 0; row < n; ++row) xb[row] = y[row];
  objective.set_model_coef(2.0, 0);
  objective.set_model_Xb(xb);
  objective.update_auxiliary();
  objective.update_all_gradients();

  return require(objective.eval() == 0.0,
                 "nonnull interpolation fixture is not exact") &&
         require(!std::isfinite(objective.get_grad(0)),
                 "nonnull exact fit was incorrectly assigned a zero gradient");
}

bool test_versioned_path(int reg_type, bool include_intercept,
                         double response) {
  const int n = 19;
  const int d = 5;
  const int nlambda = 3;
  std::vector<double> x = make_design(n, d);
  std::vector<double> y(n, response);
  std::vector<double> lambda{0.4, 0.1, 0.0};
  std::vector<double> beta(static_cast<std::size_t>(nlambda) * d, 91.0);
  std::vector<double> intercept(nlambda, 91.0);
  std::vector<int> iterations(nlambda, 91);
  std::vector<int> active_size(nlambda, 91);
  std::vector<double> runtime(nlambda, 91.0);
  std::vector<int> stages(nlambda, 91);
  std::vector<double> objective(nlambda, 91.0);
  std::vector<double> kkt(nlambda, 91.0);
  std::vector<double> stationarity(nlambda, 91.0);
  int number_fit = 91;
  int failed_lambda = 91;
  int failed_stage = 91;

  const int status = SolveSqrtLinearRegressionV2(
      y.data(), x.data(), n, d, lambda.data(), nlambda, 3.5, 1000, 1e-8,
      reg_type, include_intercept, -1, beta.data(), intercept.data(),
      iterations.data(), active_size.data(), runtime.data(), &number_fit,
      true, 3, &failed_lambda, &failed_stage, stages.data(), objective.data(),
      kkt.data(), stationarity.data());

  const std::string label =
      std::string(reg_type == 1 ? "L1" : reg_type == 2 ? "MCP" : "SCAD") +
      (include_intercept ? " constant-response"
                         : " zero-response no-intercept");
  bool ok = true;
  ok &= require(status == PICASSO_LLA_COMPLETED,
                label + " path did not complete");
  ok &= require(number_fit == nlambda && failed_lambda == -1 &&
                    failed_stage == -1,
                label + " path was truncated");
  for (int index = 0; index < nlambda; ++index) {
    ok &= require(intercept[index] == response,
                  label + " intercept differs from the constant response");
    ok &= require(iterations[index] == 0 && active_size[index] == 0,
                  label + " exact solution performed unnecessary updates");
    ok &= require(stages[index] == (reg_type == 1 ? 1 : 3),
                  label + " reported an unexpected LLA stage count");
    ok &= require(objective[index] == 0.0 && kkt[index] == 0.0 &&
                      stationarity[index] == 0.0,
                  label + " exact-solution diagnostics are not zero");
    for (int feature = 0; feature < d; ++feature) {
      ok &= require(beta[static_cast<std::size_t>(index) * d + feature] == 0.0,
                    label + " slope is nonzero");
    }
  }
  return ok;
}

}  // namespace

int main() {
  bool ok = test_exact_fit_subgradient();
  ok &= test_nonnull_exact_fit_is_not_certified_with_zero_gradient();
  for (int reg_type = 1; reg_type <= 3; ++reg_type) {
    ok &= test_versioned_path(reg_type, true, 0.125);
    ok &= test_versioned_path(reg_type, false, 0.0);
  }
  return ok ? 0 : 1;
}
