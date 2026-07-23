#include <picasso/actgd.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>

// Standalone regression test for the compact ActGD working set. It embeds the
// pre-change full-scan algorithm as the path oracle, so every lambda can be
// compared without relying on a checked-in binary.
//
// Example:
//   clang++ -O3 -DNDEBUG -std=c++11 -Iinclude \
//     -IR-package/src/include/eigen3 \
//     tests/actgd_compact_path_test.cpp \
//     src/objective/gaussian_naive_update.cpp \
//     src/objective/gaussian_cov_update.cpp \
//     src/solver/actgd.cpp src/solver/solver_params.cpp \
//     -o /tmp/actgd_compact_path_test && /tmp/actgd_compact_path_test

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using picasso::GaussianCovUpdateObjective;
using picasso::GaussianNaiveUpdateObjective;
using picasso::ModelParam;
using picasso::ObjFunction;
using picasso::RegFunction;
using picasso::RegL1;
using picasso::RegMCP;
using picasso::RegSCAD;
using picasso::solver::ActGDSolver;
using picasso::solver::L1;
using picasso::solver::MCP;
using picasso::solver::PicassoSolverParams;
using picasso::solver::RegType;
using picasso::solver::SCAD;

bool require(bool condition, const std::string &message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

std::unique_ptr<RegFunction> make_penalty(RegType type) {
  if (type == MCP)
    return std::unique_ptr<RegFunction>(new RegMCP());
  if (type == SCAD)
    return std::unique_ptr<RegFunction>(new RegSCAD());
  return std::unique_ptr<RegFunction>(new RegL1());
}

// Exact copy of the full-d strong-set coordinate loop that preceded the
// compact implementation. Early stopping is deliberately disabled by the
// test parameters, leaving only the optimization path to compare.
std::vector<ModelParam> solve_full_scan_oracle(
    ObjFunction *objective, const PicassoSolverParams &parameters) {
  const int d = objective->get_dim();
  const std::vector<double> &lambdas = parameters.get_lambda_path();
  const double change_tolerance =
      objective->get_deviance() * parameters.prec;
  std::vector<int> strong_set(d, 0);
  std::vector<double> gradient(d, 0.0);
  for (int feature = 0; feature < d; ++feature)
    gradient[feature] = std::fabs(objective->get_grad(feature));

  std::unique_ptr<RegFunction> penalty = make_penalty(parameters.reg_type);
  std::vector<ModelParam> path;
  path.reserve(lambdas.size());
  for (std::size_t index = 0; index < lambdas.size(); ++index) {
    penalty->set_param(lambdas[index], parameters.gamma);
    const double strong_threshold =
        index == 0 ? 2.0 * lambdas[index]
                   : 2.0 * lambdas[index] - lambdas[index - 1];
    for (int feature = 0; feature < d; ++feature) {
      if (strong_set[feature] == 0 &&
          gradient[feature] > strong_threshold)
        strong_set[feature] = 1;
    }

    for (int outer = 0; outer < parameters.max_iter; ++outer) {
      for (int sweep = 0; sweep < parameters.max_iter; ++sweep) {
        bool converged = true;
        for (int feature = 0; feature < d; ++feature) {
          if (strong_set[feature] == 0) continue;
          const double old_value = objective->get_model_coef(feature);
          objective->update_gradient(feature);
          const double new_value =
              objective->coordinate_descent(penalty.get(), feature);
          if (new_value != old_value &&
              objective->get_local_change(old_value, feature) >
                  change_tolerance)
            converged = false;
        }
        if (converged) break;
      }

      bool violation = false;
      for (int feature = 0; feature < d; ++feature) {
        if (strong_set[feature] != 0) continue;
        objective->update_gradient(feature);
        gradient[feature] = std::fabs(objective->get_grad(feature));
        if (std::fabs(penalty->threshold(gradient[feature])) > 1e-8) {
          strong_set[feature] = 1;
          violation = true;
        }
      }
      if (!violation) break;
    }

    for (int feature = 0; feature < d; ++feature) {
      if (strong_set[feature] == 0) continue;
      objective->update_gradient(feature);
      gradient[feature] = std::fabs(objective->get_grad(feature));
    }
    if (parameters.include_intercept) objective->intercept_update();
    path.push_back(objective->get_model_param_ref());
  }
  return path;
}

struct Fixture {
  Eigen::MatrixXd x;
  Eigen::VectorXd y;
};

Fixture make_fixture() {
  const int n = 180;
  const int d = 48;
  Fixture fixture;
  fixture.x.resize(n, d);
  fixture.y.resize(n);

  for (int observation = 0; observation < n; ++observation) {
    const double row = static_cast<double>(observation + 1);
    for (int feature = 0; feature < d; ++feature) {
      const double column = static_cast<double>(feature + 1);
      fixture.x(observation, feature) =
          0.35 * std::sin(0.017 * row * column) +
          0.55 * std::cos(0.031 * row * (column + 2.0)) +
          0.08 * static_cast<double>((observation + 3 * feature) % 11 - 5) +
          0.015 * static_cast<double>(feature % 5);
    }
  }

  Eigen::VectorXd beta = Eigen::VectorXd::Zero(d);
  for (int feature = 0; feature < 9; ++feature) {
    const double sign = feature % 2 == 0 ? 1.0 : -1.0;
    beta[feature] = sign * (1.0 - 0.075 * feature);
  }
  fixture.y = fixture.x * beta;
  for (int observation = 0; observation < n; ++observation) {
    fixture.y[observation] +=
        1.7 + 0.18 * std::sin(0.29 * static_cast<double>(observation + 1));
  }
  return fixture;
}

std::vector<double> lambda_path(const Fixture &fixture, bool intercept) {
  Eigen::MatrixXd x = fixture.x;
  Eigen::VectorXd y = fixture.y;
  if (intercept) {
    x.rowwise() -= x.colwise().mean();
    y.array() -= y.mean();
  }
  const double lambda_max =
      (x.transpose() * y).cwiseAbs().maxCoeff() /
      static_cast<double>(x.rows());
  const int path_size = 16;
  std::vector<double> lambdas(path_size);
  for (int index = 0; index < path_size; ++index) {
    const double fraction = static_cast<double>(index) / (path_size - 1);
    lambdas[index] = lambda_max * std::pow(0.12, fraction);
  }
  return lambdas;
}

template <typename Objective>
bool compare_path(const Fixture &fixture, bool intercept, RegType type,
                  const std::string &label) {
  std::vector<double> lambdas = lambda_path(fixture, intercept);
  PicassoSolverParams parameters;
  parameters.set_lambdas(lambdas.data(), static_cast<int>(lambdas.size()));
  parameters.reg_type = type;
  parameters.gamma = 3.7;
  parameters.include_intercept = intercept;
  parameters.prec = 1e-10;
  parameters.max_iter = 10000;
  parameters.min_lambda_count = static_cast<int>(lambdas.size()) + 1;
  parameters.dfmax = -1;

  Objective oracle_objective(
      fixture.x.data(), fixture.y.data(), fixture.x.rows(), fixture.x.cols(),
      intercept, false);
  Objective compact_objective(
      fixture.x.data(), fixture.y.data(), fixture.x.rows(), fixture.x.cols(),
      intercept, false);
  const std::vector<ModelParam> oracle =
      solve_full_scan_oracle(&oracle_objective, parameters);
  ActGDSolver compact(&compact_objective, parameters);
  compact.solve();

  bool ok = true;
  ok &= require(compact.get_num_lambdas_fit() ==
                    static_cast<int>(oracle.size()),
                label + ": path length differs");
  const std::vector<int> &iterations = compact.get_itercnt_path();
  bool observed_work = false;
  for (std::size_t index = 0; index < oracle.size(); ++index) {
    const ModelParam &candidate = compact.get_model_param(index);
    const double beta_error =
        (candidate.beta - oracle[index].beta).abs().maxCoeff();
    const double intercept_error =
        std::fabs(candidate.intercept - oracle[index].intercept);
    const double tolerance = type == L1 ? 5e-5 : 8e-5;
    ok &= require(beta_error <= tolerance,
                  label + ": beta path differs at lambda " +
                      std::to_string(index) + " (max error " +
                      std::to_string(beta_error) + ")");
    ok &= require(intercept_error <= tolerance,
                  label + ": intercept path differs at lambda " +
                      std::to_string(index) + " (error " +
                      std::to_string(intercept_error) + ")");

    const double expected_intercept = intercept
        ? (fixture.y - fixture.x * candidate.beta.matrix()).mean()
        : 0.0;
    ok &= require(std::fabs(candidate.intercept - expected_intercept) <= 1e-10,
                  label + ": intercept is not conditionally optimal at lambda " +
                      std::to_string(index));
    ok &= require(iterations[index] >= 0,
                  label + ": negative iteration count");
    observed_work = observed_work || iterations[index] > 0;
  }
  ok &= require(observed_work, label + ": iteration counts remained zero");
  return ok;
}

template <typename Objective>
bool check_low_curvature_nonconvex_activation(RegType type,
                                              const std::string &label) {
  // curvature = X^T X / n = 0.1 and initial |gradient| = 0.9 < lambda.
  // Thus beta=0 passes the first-order zero KKT check. For both penalties the
  // exact coordinate objective is nevertheless lower at a distant nonzero
  // point, which the full screening scan must discover.
  const double x[] = {std::sqrt(0.1)};
  const double y[] = {0.9 / x[0]};
  const double lambda[] = {1.0};

  PicassoSolverParams parameters;
  parameters.set_lambdas(lambda, 1);
  parameters.reg_type = type;
  parameters.gamma = 3.7;
  parameters.include_intercept = false;
  parameters.prec = 1e-12;
  parameters.max_iter = 1000;
  parameters.min_lambda_count = 2;
  parameters.dfmax = -1;

  Objective objective(x, y, 1, 1, false, false);
  ActGDSolver solver(&objective, parameters);
  solver.solve();
  if (!require(solver.get_num_lambdas_fit() == 1,
               label + ": missing solution"))
    return false;

  const double coefficient = solver.get_model_param(0).beta[0];
  bool ok = require(std::isfinite(coefficient),
                    label + ": coefficient is not finite");
  ok &= require(std::fabs(coefficient) > parameters.gamma * lambda[0],
                label + ": exact nonconvex activation was screened out");
  return ok;
}

}  // namespace

int main() {
  const Fixture fixture = make_fixture();
  bool ok = true;
  const RegType penalties[] = {L1, MCP, SCAD};
  const char *penalty_names[] = {"l1", "mcp", "scad"};
  for (int penalty = 0; penalty < 3; ++penalty) {
    for (int intercept = 0; intercept < 2; ++intercept) {
      const std::string suffix =
          std::string(penalty_names[penalty]) +
          (intercept != 0 ? "-intercept" : "-no-intercept");
      ok &= compare_path<GaussianNaiveUpdateObjective>(
          fixture, intercept != 0, penalties[penalty], "naive-" + suffix);
      ok &= compare_path<GaussianCovUpdateObjective>(
          fixture, intercept != 0, penalties[penalty], "covariance-" + suffix);
    }
  }
  ok &= check_low_curvature_nonconvex_activation<
      GaussianNaiveUpdateObjective>(MCP, "naive-mcp-low-curvature");
  ok &= check_low_curvature_nonconvex_activation<
      GaussianNaiveUpdateObjective>(SCAD, "naive-scad-low-curvature");
  ok &= check_low_curvature_nonconvex_activation<
      GaussianCovUpdateObjective>(MCP, "covariance-mcp-low-curvature");
  ok &= check_low_curvature_nonconvex_activation<
      GaussianCovUpdateObjective>(SCAD, "covariance-scad-low-curvature");
  if (!ok) return 1;
  std::cout << "ActGD compact path matches the pre-change full-scan oracle.\n";
  return 0;
}
