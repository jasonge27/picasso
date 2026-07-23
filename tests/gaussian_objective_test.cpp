#include <picasso/objective.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void fail(const std::string &message) {
  std::cerr << "gaussian_objective_test: " << message << std::endl;
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

double soft_threshold(double value, double lambda) {
  if (value > lambda) return value - lambda;
  if (value < -lambda) return value + lambda;
  return 0.0;
}

double mcp_penalty(double magnitude, double lambda, double gamma) {
  if (magnitude <= gamma * lambda)
    return lambda * magnitude - magnitude * magnitude / (2.0 * gamma);
  return 0.5 * gamma * lambda * lambda;
}

double scad_penalty(double magnitude, double lambda, double gamma) {
  if (magnitude <= lambda) return lambda * magnitude;
  if (magnitude <= gamma * lambda) {
    return (-magnitude * magnitude + 2.0 * gamma * lambda * magnitude -
            lambda * lambda) /
           (2.0 * (gamma - 1.0));
  }
  return 0.5 * (gamma + 1.0) * lambda * lambda;
}

typedef double (*Penalty)(double, double, double);

double coordinate_value(double beta, double linear_term, double curvature,
                        double lambda, double gamma, Penalty penalty) {
  return 0.5 * curvature * beta * beta - linear_term * beta +
         penalty(std::fabs(beta), lambda, gamma);
}

void check_dense_grid_oracle(picasso::RegFunction *regularizer,
                             double lambda, double gamma,
                             const std::vector<double> &curvatures,
                             Penalty penalty, const std::string &name) {
  regularizer->set_param(lambda, gamma);
  const double linear_terms[] =
      {-3.0, -1.8, -0.9, -0.7, -0.3, 0.0,
        0.3,  0.7,  0.9,  1.8,  3.0};

  for (std::size_t c = 0; c < curvatures.size(); ++c) {
    const double curvature = curvatures[c];
    for (std::size_t k = 0;
         k < sizeof(linear_terms) / sizeof(linear_terms[0]); ++k) {
      const double linear_term = linear_terms[k];
      const double beta =
          regularizer->coordinate_minimize(linear_term, curvature);
      expect_true(std::isfinite(beta), name + " returned a non-finite value");

      const double knot = gamma * lambda;
      const double extent =
          std::max(1.5 * knot + 1.0,
                   1.25 * std::fabs(linear_term) / curvature + 1.0);
      const int grid_size = 60000;
      double grid_best = coordinate_value(0.0, linear_term, curvature,
                                          lambda, gamma, penalty);
      for (int i = 1; i <= grid_size; ++i) {
        const double magnitude = extent * i / grid_size;
        const double signed_beta =
            linear_term < 0.0 ? -magnitude : magnitude;
        grid_best = std::min(
            grid_best,
            coordinate_value(signed_beta, linear_term, curvature,
                             lambda, gamma, penalty));
      }

      const double actual = coordinate_value(beta, linear_term, curvature,
                                             lambda, gamma, penalty);
      const double tolerance = 1e-10 * (1.0 + std::fabs(grid_best));
      if (actual > grid_best + tolerance) {
        std::cerr << name << " grid failure: curvature=" << curvature
                  << " linear_term=" << linear_term << " beta=" << beta
                  << " value=" << actual << " grid_best=" << grid_best
                  << std::endl;
        fail(name + " failed dense-grid global-minimum oracle");
      }

      if (linear_term == 0.0)
        expect_true(beta == 0.0, name + " should choose zero at zero gradient");
      else
        expect_true(beta * linear_term >= 0.0,
                    name + " returned a coefficient with the wrong sign");
    }
  }
}

void test_curvature_aware_penalties() {
  picasso::RegL1 l1;
  l1.set_param(0.4, 0.0);
  const double l1_curvatures[] = {0.1, 0.7, 1.0, 3.2};
  const double l1_terms[] = {-1.3, -0.4, 0.0, 0.4, 1.3};
  for (std::size_t i = 0;
       i < sizeof(l1_curvatures) / sizeof(l1_curvatures[0]); ++i) {
    for (std::size_t j = 0;
         j < sizeof(l1_terms) / sizeof(l1_terms[0]); ++j) {
      const double expected = l1.threshold(l1_terms[j]) / l1_curvatures[i];
      expect_true(l1.coordinate_minimize(l1_terms[j], l1_curvatures[i]) ==
                      expected,
                  "L1 coordinate update changed");
    }
  }
  expect_true(l1.thresholded_coordinate_minimize(0.0, 0.0) == 0.0,
              "flat zero coordinate should select zero");
  expect_true(
      std::isnan(l1.thresholded_coordinate_minimize(0.2, 0.0)),
      "flat coordinate with a nonzero linear term should be rejected");
  expect_true(
      std::isnan(l1.thresholded_coordinate_minimize(0.0, -1.0)),
      "negative coordinate curvature should be rejected");
  expect_true(std::isnan(l1.thresholded_coordinate_minimize(
                  std::numeric_limits<double>::quiet_NaN(), 0.0)),
              "non-finite coordinate input should be rejected");

  const double mcp_gamma = 3.0;
  const double mcp_transition = 1.0 / mcp_gamma;
  const std::vector<double> mcp_curvatures = {
      0.12,
      std::nextafter(mcp_transition, 0.0),
      mcp_transition,
      std::nextafter(mcp_transition,
                     std::numeric_limits<double>::infinity()),
      0.8,
      2.4};
  picasso::RegMCP mcp;
  check_dense_grid_oracle(&mcp, 0.7, mcp_gamma, mcp_curvatures,
                          mcp_penalty, "MCP");

  const double scad_gamma = 3.7;
  const double scad_transition = 1.0 / (scad_gamma - 1.0);
  const std::vector<double> scad_curvatures = {
      0.12,
      std::nextafter(scad_transition, 0.0),
      scad_transition,
      std::nextafter(scad_transition,
                     std::numeric_limits<double>::infinity()),
      0.8,
      2.4};
  picasso::RegSCAD scad;
  check_dense_grid_oracle(&scad, 0.7, scad_gamma, scad_curvatures,
                          scad_penalty, "SCAD");

  // Unit curvature must agree with the legacy closed forms away from ties.
  const double compatibility_terms[] =
      {-4.0, -1.7, -0.8, -0.2, 0.2, 0.8, 1.7, 4.0};
  mcp.set_param(0.6, 3.0);
  scad.set_param(0.6, 3.7);
  for (std::size_t i = 0;
       i < sizeof(compatibility_terms) / sizeof(compatibility_terms[0]); ++i) {
    const double value = compatibility_terms[i];
    expect_near(mcp.coordinate_minimize(value, 1.0), mcp.threshold(value),
                1e-13, "unit-curvature MCP compatibility");
    expect_near(scad.coordinate_minimize(value, 1.0), scad.threshold(value),
                1e-13, "unit-curvature SCAD compatibility");
  }
}

template <typename Objective>
void check_one_column_l1(bool include_intercept, const std::string &name) {
  const int n = 6;
  const double x[n] = {2.0, 4.0, 5.0, 8.0, 11.0, 13.0};
  const double y[n] = {6.3, 9.6, 11.8, 16.1, 22.2, 24.6};
  const double lambda = 0.25;

  double x_mean = 0.0;
  double y_mean = 0.0;
  for (int i = 0; i < n; ++i) {
    x_mean += x[i];
    y_mean += y[i];
  }
  x_mean /= n;
  y_mean /= n;

  double curvature = 0.0;
  double linear_term = 0.0;
  for (int i = 0; i < n; ++i) {
    const double centered_x = include_intercept ? x[i] - x_mean : x[i];
    const double centered_y = include_intercept ? y[i] - y_mean : y[i];
    curvature += centered_x * centered_x;
    linear_term += centered_x * centered_y;
  }
  curvature /= n;
  linear_term /= n;

  Objective objective(x, y, n, 1, include_intercept, false);
  picasso::RegL1 l1;
  l1.set_param(lambda, 0.0);
  objective.update_gradient(0);
  const double beta = objective.coordinate_descent(&l1, 0);
  objective.intercept_update();

  const double expected_beta = soft_threshold(linear_term, lambda) / curvature;
  const double expected_intercept =
      include_intercept ? y_mean - x_mean * expected_beta : 0.0;
  expect_near(beta, expected_beta, 2e-14, name + " beta closed form");
  expect_near(objective.get_model_coef(-1), expected_intercept, 2e-14,
              name + " intercept closed form");

  double expected_rss = 0.0;
  for (int i = 0; i < n; ++i) {
    const double residual =
        y[i] - expected_intercept - x[i] * expected_beta;
    expected_rss += residual * residual;
  }
  expect_near(objective.eval(), expected_rss / n, 2e-13,
              name + " residual objective");
}

template <typename Objective>
void check_recomputed_profile_gradient(const std::string &name) {
  const int n = 6;
  const double x[n] = {2.0, 4.0, 5.0, 8.0, 11.0, 13.0};
  const double y[n] = {6.3, 9.6, 11.8, 16.1, 22.2, 24.6};
  picasso::ModelParam model(1);
  model.beta[0] = 0.43;
  model.intercept = -17.0;  // Profiled beta gradient must not depend on this.

  Objective objective(x, y, n, 1, true, false);
  objective.set_model_param(model);
  objective.update_auxiliary();
  objective.update_gradient(0);

  double x_mean = 0.0;
  double residual_mean = 0.0;
  double raw_gradient = 0.0;
  for (int i = 0; i < n; ++i) {
    x_mean += x[i];
    residual_mean += y[i] - x[i] * model.beta[0];
    raw_gradient += x[i] * (y[i] - x[i] * model.beta[0]);
  }
  x_mean /= n;
  residual_mean /= n;
  raw_gradient /= n;
  const double expected_gradient = raw_gradient - x_mean * residual_mean;
  expect_near(objective.get_grad(0), expected_gradient, 2e-13,
              name + " recomputed profiled gradient");

  objective.intercept_update();
  expect_near(objective.get_model_coef(-1), residual_mean, 2e-14,
              name + " recomputed intercept");
}

template <typename Objective>
void check_constant_column(const std::string &name) {
  const int n = 5;
  const double x[n] = {5.0, 5.0, 5.0, 5.0, 5.0};
  const double y[n] = {1.0, 3.0, 2.0, 6.0, 8.0};
  Objective objective(x, y, n, 1, true, false);
  picasso::RegL1 l1;
  l1.set_param(0.1, 0.0);
  objective.update_gradient(0);
  const double beta = objective.coordinate_descent(&l1, 0);
  objective.intercept_update();
  expect_true(beta == 0.0, name + " constant column should remain zero");
  expect_near(objective.get_model_coef(-1), 4.0, 1e-14,
              name + " constant-column intercept");
}

template <typename Objective>
void check_large_predictor_mean(const std::string &name) {
  const int n = 6;
  const double base = 1e9;
  const double offsets[n] = {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0};
  const double noise[n] = {0.2, -0.1, 0.05, -0.05, 0.1, -0.2};
  double x[n];
  double y[n];
  for (int i = 0; i < n; ++i) {
    x[i] = base + offsets[i];
    y[i] = 7.0 + 1.7 * offsets[i] + noise[i];
  }

  const double lambda = 0.2;
  long double curvature = 0.0L;
  long double linear_term = 0.0L;
  for (int i = 0; i < n; ++i) {
    curvature += static_cast<long double>(offsets[i]) * offsets[i];
    linear_term += static_cast<long double>(offsets[i]) * (y[i] - 7.0);
  }
  curvature /= n;
  linear_term /= n;
  const double expected_beta =
      soft_threshold(static_cast<double>(linear_term), lambda) /
      static_cast<double>(curvature);

  Objective objective(x, y, n, 1, true, false);
  objective.update_gradient(0);
  expect_near(objective.get_grad(0), static_cast<double>(linear_term), 2e-12,
              name + " shifted initial gradient");

  picasso::RegL1 l1;
  l1.set_param(lambda, 0.0);
  const double beta = objective.coordinate_descent(&l1, 0);
  objective.intercept_update();
  expect_near(beta, expected_beta, 2e-12,
              name + " shifted curvature update");
  expect_near(objective.get_model_coef(-1), 7.0 - base * beta, 2e-7,
              name + " shifted intercept");
}

template <typename Objective>
void check_nonconvex_gaussian_wiring(picasso::RegFunction *regularizer,
                                     double lambda, double gamma,
                                     double linear_term, Penalty penalty,
                                     const std::string &name) {
  const int n = 4;
  const double curvature = 0.2;
  const double scale = std::sqrt(curvature);
  const double x[n] = {scale, -scale, scale, -scale};
  double y[n];
  for (int i = 0; i < n; ++i)
    y[i] = (linear_term / curvature) * x[i];

  regularizer->set_param(lambda, gamma);
  Objective objective(x, y, n, 1, false, false);
  objective.update_gradient(0);
  const double beta = objective.coordinate_descent(regularizer, 0);
  const double expected =
      regularizer->coordinate_minimize(linear_term, curvature);
  expect_near(beta, expected, 2e-13, name + " curvature-aware wiring");

  const double actual_objective =
      0.5 * objective.eval() + penalty(std::fabs(beta), lambda, gamma);
  const double extent =
      std::max(1.5 * gamma * lambda + 1.0,
               1.25 * std::fabs(linear_term) / curvature + 1.0);
  double grid_best = std::numeric_limits<double>::infinity();
  for (int i = -60000; i <= 60000; ++i) {
    const double candidate = extent * i / 60000.0;
    double rss = 0.0;
    for (int row = 0; row < n; ++row) {
      const double residual = y[row] - x[row] * candidate;
      rss += residual * residual;
    }
    grid_best = std::min(
        grid_best,
        0.5 * rss / n + penalty(std::fabs(candidate), lambda, gamma));
  }
  expect_true(actual_objective <= grid_best + 1e-10,
              name + " failed Gaussian objective grid oracle");
}

void test_gaussian_objectives() {
  check_one_column_l1<picasso::GaussianNaiveUpdateObjective>(
      true, "naive centered");
  check_one_column_l1<picasso::GaussianCovUpdateObjective>(
      true, "covariance centered");
  check_one_column_l1<picasso::GaussianNaiveUpdateObjective>(
      false, "naive no-intercept");
  check_one_column_l1<picasso::GaussianCovUpdateObjective>(
      false, "covariance no-intercept");

  check_recomputed_profile_gradient<picasso::GaussianNaiveUpdateObjective>(
      "naive");
  check_recomputed_profile_gradient<picasso::GaussianCovUpdateObjective>(
      "covariance");
  check_constant_column<picasso::GaussianNaiveUpdateObjective>("naive");
  check_constant_column<picasso::GaussianCovUpdateObjective>("covariance");
  check_large_predictor_mean<picasso::GaussianNaiveUpdateObjective>("naive");
  check_large_predictor_mean<picasso::GaussianCovUpdateObjective>("covariance");

  picasso::RegMCP mcp;
  check_nonconvex_gaussian_wiring<picasso::GaussianNaiveUpdateObjective>(
      &mcp, 0.4, 3.0, 0.9, mcp_penalty, "naive MCP");
  check_nonconvex_gaussian_wiring<picasso::GaussianCovUpdateObjective>(
      &mcp, 0.4, 3.0, 0.9, mcp_penalty, "covariance MCP");

  picasso::RegSCAD scad;
  check_nonconvex_gaussian_wiring<picasso::GaussianNaiveUpdateObjective>(
      &scad, 0.4, 3.7, 0.9, scad_penalty, "naive SCAD");
  check_nonconvex_gaussian_wiring<picasso::GaussianCovUpdateObjective>(
      &scad, 0.4, 3.7, 0.9, scad_penalty, "covariance SCAD");
}

}  // namespace

int main() {
  test_curvature_aware_penalties();
  test_gaussian_objectives();
  std::cout << "gaussian_objective_test passed" << std::endl;
  return 0;
}
