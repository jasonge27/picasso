#include <picasso/c_api.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

const int kSampleCount = 8;
const int kFeatureCount = 3;
const int kPathSize = 2;

bool require(bool condition, const std::string &message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

struct Output {
  std::vector<double> beta;
  std::vector<double> intercept;
  std::vector<int> iterations;
  std::vector<int> active_size;
  std::vector<double> runtime;
  std::vector<double> smooth_objective;
  int num_fit;
  int failed_lambda;
  int status;

  Output()
      : beta(kPathSize * kFeatureCount, 0.0),
        intercept(kPathSize, 0.0),
        iterations(kPathSize, 0),
        active_size(kPathSize, 0),
        runtime(kPathSize, 0.0),
        smooth_objective(kPathSize, 0.0),
        num_fit(-99),
        failed_lambda(-99),
        status(-99) {}
};

void solve(bool covariance, int max_iterations, Output *output) {
  // Row-major storage exercises the same C API layout used by Python.
  double design[kSampleCount * kFeatureCount] = {
      1.0, 1.1, 0.9,  2.0, 2.1, 1.8,  3.0, 3.2, 2.7,
      4.0, 4.1, 3.7, -1.0,-1.2,-0.8, -2.0,-2.1,-1.7,
     -3.0,-3.1,-2.8, -4.0,-4.2,-3.6};
  double response[kSampleCount] = {
      3.0, 5.9, 9.2, 12.1, -3.2, -6.1, -9.0, -12.2};
  double lambda[kPathSize] = {50.0, 0.05};

  if (covariance) {
    output->status = SolveLinearRegressionCovUpdateV3(
        response, design, kSampleCount, kFeatureCount, lambda, kPathSize,
        3.0, max_iterations, 1e-7, 1, false, -1,
        output->beta.data(), output->intercept.data(),
        output->iterations.data(), output->active_size.data(),
        output->runtime.data(), &output->num_fit, true,
        output->smooth_objective.data(), &output->failed_lambda);
  } else {
    output->status = SolveLinearRegressionNaiveUpdateV3(
        response, design, kSampleCount, kFeatureCount, lambda, kPathSize,
        3.0, max_iterations, 1e-7, 1, false, -1,
        output->beta.data(), output->intercept.data(),
        output->iterations.data(), output->active_size.data(),
        output->runtime.data(), &output->num_fit, true,
        output->smooth_objective.data(), &output->failed_lambda);
  }
}

bool check_iteration_limit(bool covariance) {
  Output output;
  solve(covariance, 1, &output);
  const std::string label = covariance ? "covariance" : "naive";
  bool ok = true;
  ok &= require(output.status == PICASSO_LLA_INNER_ITERATION_LIMIT,
                label + " did not report its exhausted iteration budget");
  ok &= require(output.num_fit == 1,
                label + " did not retain exactly the converged prefix");
  ok &= require(output.failed_lambda == 1,
                label + " reported the wrong failed lambda");
  ok &= require(output.beta[0] == 0.0 && output.beta[1] == 0.0 &&
                    output.beta[2] == 0.0,
                label + " changed the converged null prefix");
  for (int index = kFeatureCount; index < kPathSize * kFeatureCount;
       ++index)
    ok &= require(output.beta[index] == 0.0,
                  label + " committed coefficients for the failed lambda");
  ok &= require(std::isfinite(output.smooth_objective[0]) &&
                    std::isnan(output.smooth_objective[1]),
                label + " did not keep diagnostics prefix-transactional");
  return ok;
}

bool check_completed_path(bool covariance) {
  Output output;
  solve(covariance, 1000, &output);
  const std::string label = covariance ? "covariance" : "naive";
  bool ok = true;
  ok &= require(output.status == PICASSO_LLA_COMPLETED,
                label + " rejected a normally converged path");
  ok &= require(output.num_fit == kPathSize && output.failed_lambda == -1,
                label + " truncated a normally converged path");
  ok &= require(std::isfinite(output.smooth_objective[0]) &&
                    std::isfinite(output.smooth_objective[1]),
                label + " lost completed smooth-objective diagnostics");
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= check_iteration_limit(false);
  ok &= check_iteration_limit(true);
  ok &= check_completed_path(false);
  ok &= check_completed_path(true);
  if (!ok) return 1;
  std::cout << "Gaussian iteration limits retain only converged prefixes.\n";
  return 0;
}
