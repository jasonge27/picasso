#include <picasso/multinomial_actnewton.hpp>
#include <picasso/multinomial_lla.hpp>
#include <picasso/multinomial_objective.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

typedef picasso::solver::MultinomialLlaPenalty Penalty;

bool require(bool condition, const std::string &message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

bool nearly_equal(double left, double right, double absolute_tolerance,
                  double relative_tolerance) {
  return std::fabs(left - right) <=
         absolute_tolerance +
             relative_tolerance * std::max(std::fabs(left), std::fabs(right));
}

double reference_penalty_value(Penalty penalty, double value, double lambda,
                               double gamma) {
  const double t = std::fabs(value);
  if (penalty == Penalty::kMCP) {
    if (t <= gamma * lambda) return lambda * t - t * t / (2.0 * gamma);
    return 0.5 * gamma * lambda * lambda;
  }
  if (t <= lambda) return lambda * t;
  if (t <= gamma * lambda) {
    return (-t * t + 2.0 * gamma * lambda * t - lambda * lambda) /
           (2.0 * (gamma - 1.0));
  }
  return 0.5 * (gamma + 1.0) * lambda * lambda;
}

double reference_penalty_derivative(Penalty penalty, double absolute_value,
                                    double lambda, double gamma) {
  if (penalty == Penalty::kMCP)
    return std::max(0.0, lambda - absolute_value / gamma);
  if (absolute_value <= lambda) return lambda;
  if (absolute_value <= gamma * lambda)
    return (gamma * lambda - absolute_value) / (gamma - 1.0);
  return 0.0;
}

double reference_target_objective(
    const picasso::MultinomialObjective &objective,
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    Penalty penalty, double lambda, double gamma) {
  double value = objective.negative_log_likelihood(beta, intercept);
  for (Eigen::Index j = 0; j < beta.rows(); ++j) {
    for (Eigen::Index k = 0; k < beta.cols(); ++k)
      value += reference_penalty_value(penalty, beta(j, k), lambda, gamma);
  }
  return value;
}

Eigen::MatrixXd reference_lla_weights(const Eigen::MatrixXd &anchor,
                                      Penalty penalty, double lambda,
                                      double gamma) {
  Eigen::MatrixXd weights(anchor.rows(), anchor.cols());
  for (Eigen::Index j = 0; j < anchor.rows(); ++j) {
    for (Eigen::Index k = 0; k < anchor.cols(); ++k) {
      weights(j, k) = reference_penalty_derivative(
          penalty, std::fabs(anchor(j, k)), lambda, gamma);
    }
  }
  return weights;
}

double weighted_l1(const Eigen::MatrixXd &beta,
                   const Eigen::MatrixXd &weights) {
  return (beta.cwiseAbs().array() * weights.array()).sum();
}

double reference_tangent_constant(const Eigen::MatrixXd &anchor,
                                  const Eigen::MatrixXd &weights,
                                  Penalty penalty, double lambda,
                                  double gamma) {
  double value = 0.0;
  for (Eigen::Index j = 0; j < anchor.rows(); ++j) {
    for (Eigen::Index k = 0; k < anchor.cols(); ++k) {
      value += reference_penalty_value(penalty, anchor(j, k), lambda, gamma) -
               weights(j, k) * std::fabs(anchor(j, k));
    }
  }
  return value;
}

double coefficient_stationarity(double coefficient, double gradient,
                                double penalty_derivative,
                                double zero_tolerance = 1e-10) {
  if (coefficient > zero_tolerance)
    return std::fabs(gradient + penalty_derivative);
  if (coefficient < -zero_tolerance)
    return std::fabs(gradient - penalty_derivative);
  return std::max(0.0, std::fabs(gradient) - penalty_derivative);
}

double reference_target_stationarity(
    const picasso::MultinomialObjective &objective,
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    Penalty penalty, double lambda, double gamma, bool include_intercept,
    double zero_tolerance = 1e-10) {
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  objective.smooth_gradient(beta, intercept, &beta_gradient,
                            &intercept_gradient);
  double residual = 0.0;
  for (Eigen::Index j = 0; j < beta.rows(); ++j) {
    for (Eigen::Index k = 0; k < beta.cols(); ++k) {
      residual = std::max(
          residual,
          coefficient_stationarity(
              beta(j, k), beta_gradient(j, k),
              reference_penalty_derivative(
                  penalty, std::fabs(beta(j, k)), lambda, gamma),
              zero_tolerance));
    }
  }
  if (include_intercept)
    residual = std::max(residual, intercept_gradient.cwiseAbs().maxCoeff());
  return residual;
}

void make_signal_fixture(Eigen::MatrixXd *x, Eigen::VectorXi *labels) {
  const int n = 96;
  const int d = 4;
  const int num_classes = 3;
  x->resize(n, d);
  labels->resize(n);
  Eigen::MatrixXd true_beta(d, num_classes);
  true_beta << 0.95, -0.55, -0.20,
              -0.35, 0.75, -0.30,
               0.20, -0.45, 0.35,
              -0.55, 0.15, 0.40;
  Eigen::VectorXd true_intercept(num_classes);
  true_intercept << 0.25, -0.05, -0.20;
  for (int i = 0; i < n; ++i) {
    (*x)(i, 0) = std::sin(0.087 * static_cast<double>(i + 1));
    (*x)(i, 1) = std::cos(0.131 * static_cast<double>(i + 2));
    (*x)(i, 2) = static_cast<double>((i * 7) % 19 - 9) / 9.0;
    (*x)(i, 3) = std::sin(0.037 * static_cast<double>((i + 3) * (i + 1)));
  }
  const Eigen::MatrixXd logits =
      (*x) * true_beta + true_intercept.transpose().replicate(n, 1);
  for (int i = 0; i < n; ++i) {
    const double maximum = logits.row(i).maxCoeff();
    Eigen::VectorXd probabilities(num_classes);
    for (int klass = 0; klass < num_classes; ++klass)
      probabilities[klass] = std::exp(logits(i, klass) - maximum);
    probabilities.array() /= probabilities.sum();
    const double draw =
        std::fmod(0.6180339887498949 * static_cast<double>(i + 1), 1.0);
    double cumulative = 0.0;
    (*labels)[i] = num_classes - 1;
    for (int klass = 0; klass < num_classes; ++klass) {
      cumulative += probabilities[klass];
      if (draw <= cumulative) {
        (*labels)[i] = klass;
        break;
      }
    }
  }
}

bool test_penalty_value_derivative_and_majorization() {
  bool ok = true;
  const double lambda = 0.7;
  const Penalty penalties[] = {Penalty::kMCP, Penalty::kSCAD};
  const double gammas[] = {3.4, 3.7};
  for (int kind = 0; kind < 2; ++kind) {
    const Penalty penalty = penalties[kind];
    const double gamma = gammas[kind];
    std::vector<double> points;
    points.push_back(0.0);
    points.push_back(lambda);
    points.push_back(gamma * lambda);
    const double epsilon = 1e-8 * lambda;
    points.push_back(std::max(0.0, lambda - epsilon));
    points.push_back(lambda + epsilon);
    points.push_back(gamma * lambda - epsilon);
    points.push_back(gamma * lambda + epsilon);
    for (std::size_t index = 0; index < points.size(); ++index) {
      const double t = points[index];
      const double value = picasso::solver::multinomial_lla_penalty_value(
          penalty, t, lambda, gamma);
      const double derivative =
          picasso::solver::multinomial_lla_penalty_derivative(
              penalty, t, lambda, gamma);
      ok &= require(std::isfinite(value) && std::isfinite(derivative),
                    "penalty value and derivative must be finite at all "
                    "pieces and boundaries");
      ok &= require(nearly_equal(
                        value,
                        reference_penalty_value(penalty, t, lambda, gamma),
                        2e-15, 2e-14) &&
                        nearly_equal(
                            derivative,
                            reference_penalty_derivative(penalty, t, lambda,
                                                         gamma),
                            2e-15, 2e-14),
                    "MCP/SCAD helper formulas must match an independent "
                    "piecewise oracle");
    }
    const double finite_difference_points[] = {
        0.4 * lambda, 1.4 * lambda, (gamma + 0.4) * lambda};
    for (std::size_t index = 0; index < 3; ++index) {
      const double t = finite_difference_points[index];
      const double step = 1e-6 * lambda;
      const double finite_difference =
          (picasso::solver::multinomial_lla_penalty_value(
               penalty, t + step, lambda, gamma) -
           picasso::solver::multinomial_lla_penalty_value(
               penalty, t - step, lambda, gamma)) /
          (2.0 * step);
      const double derivative =
          picasso::solver::multinomial_lla_penalty_derivative(
              penalty, t, lambda, gamma);
      ok &= require(nearly_equal(finite_difference, derivative, 2e-9, 2e-8),
                    "analytic MCP/SCAD derivative must match centered finite "
                    "differences away from kinks");
    }

    unsigned int state = 1729u + static_cast<unsigned int>(kind);
    for (int trial = 0; trial < 160; ++trial) {
      state = 1664525u * state + 1013904223u;
      const double unit_anchor =
          static_cast<double>(state & 0x00ffffffu) / 16777216.0;
      state = 1664525u * state + 1013904223u;
      const double unit_candidate =
          static_cast<double>(state & 0x00ffffffu) / 16777216.0;
      const double anchor = (gamma + 1.5) * lambda * unit_anchor;
      const double candidate = (gamma + 1.5) * lambda * unit_candidate;
      const double anchor_value =
          picasso::solver::multinomial_lla_penalty_value(
              penalty, anchor, lambda, gamma);
      const double weight =
          picasso::solver::multinomial_lla_penalty_derivative(
              penalty, anchor, lambda, gamma);
      const double tangent = anchor_value + weight * (candidate - anchor);
      const double candidate_value =
          picasso::solver::multinomial_lla_penalty_value(
              penalty, candidate, lambda, gamma);
      ok &= require(candidate_value <= tangent + 2e-14,
                    "every deterministic random MCP/SCAD tangent must "
                    "majorize the penalty");
      ok &= require(anchor_value == anchor_value + weight * (anchor - anchor),
                    "the LLA tangent must equal the target at its anchor");
    }
  }
  return ok;
}

bool helper_call_throws(Penalty penalty, double t, double lambda,
                        double gamma, bool derivative) {
  try {
    if (derivative) {
      picasso::solver::multinomial_lla_penalty_derivative(
          penalty, t, lambda, gamma);
    } else {
      picasso::solver::multinomial_lla_penalty_value(penalty, t, lambda,
                                                     gamma);
    }
  } catch (const std::invalid_argument &) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

bool solve_call_throws(const picasso::solver::MultinomialLlaSolver &solver,
                       Penalty penalty, double lambda, double gamma) {
  try {
    solver.solve(penalty, lambda, gamma);
  } catch (const std::invalid_argument &) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

bool test_invalid_penalty_and_solver_inputs() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  bool ok = true;
  const Penalty penalties[] = {Penalty::kMCP, Penalty::kSCAD};
  for (int kind = 0; kind < 2; ++kind) {
    const Penalty penalty = penalties[kind];
    const double valid_gamma = kind == 0 ? 3.0 : 3.7;
    ok &= require(helper_call_throws(penalty, -1e-12, 0.2, valid_gamma,
                                     false) &&
                      helper_call_throws(penalty, nan, 0.2, valid_gamma,
                                         true),
                  "penalty helpers must reject negative and NaN absolute "
                  "values");
    ok &= require(helper_call_throws(penalty, 0.1, -0.1, valid_gamma,
                                     false) &&
                      helper_call_throws(penalty, 0.1, infinity, valid_gamma,
                                         true),
                  "penalty helpers must reject invalid lambda values");
    ok &= require(helper_call_throws(penalty, 0.1, 0.2,
                                     kind == 0 ? 1.0 : 2.0, false) &&
                      helper_call_throws(penalty, 0.1, 0.2, nan, true) &&
                      helper_call_throws(penalty, 0.1, 0.2, infinity, false),
                  "penalty helpers must enforce finite MCP/SCAD gamma "
                  "domains");
  }

  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(9, 2);
  Eigen::VectorXi labels(9);
  labels << 0, 1, 2, 0, 1, 2, 0, 1, 2;
  picasso::MultinomialObjective objective(x, labels, 3);
  picasso::solver::MultinomialActNewtonOptions pn_options;

  picasso::solver::MultinomialLlaOptions invalid_options;
  ok &= require(
      invalid_options.stopping_rule ==
              picasso::solver::MultinomialLlaStoppingRule::
                  kTargetStationarity &&
          invalid_options.minimum_stages == 3 &&
          invalid_options.maximum_stages == 3,
      "default LLA must use adaptive stationarity with a three-stage cap");
  invalid_options.minimum_stages = 2;
  bool threw = false;
  try {
    picasso::solver::MultinomialLlaSolver invalid_solver(
        objective, pn_options, invalid_options);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  ok &= require(threw, "fewer than three minimum stages must be rejected");
  invalid_options = picasso::solver::MultinomialLlaOptions();
  invalid_options.maximum_stages = 2;
  threw = false;
  try {
    picasso::solver::MultinomialLlaSolver invalid_solver(
        objective, pn_options, invalid_options);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  ok &= require(threw, "maximum stages below the minimum must be rejected");
  invalid_options =
      picasso::solver::MultinomialLlaOptions::fixed_stage_compatibility();
  invalid_options.maximum_stages = 4;
  threw = false;
  try {
    picasso::solver::MultinomialLlaSolver invalid_solver(
        objective, pn_options, invalid_options);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  ok &= require(threw,
                "fixed-stage compatibility must have one exact stage count");
  invalid_options = picasso::solver::MultinomialLlaOptions();
  invalid_options.stationarity_tolerance = -1e-12;
  threw = false;
  try {
    picasso::solver::MultinomialLlaSolver invalid_solver(
        objective, pn_options, invalid_options);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  ok &= require(threw,
                "negative stationarity tolerance must be rejected");
  invalid_options.stationarity_tolerance = infinity;
  threw = false;
  try {
    picasso::solver::MultinomialLlaSolver invalid_solver(
        objective, pn_options, invalid_options);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  ok &= require(threw,
                "infinite stationarity tolerance must be rejected");
  invalid_options = picasso::solver::MultinomialLlaOptions();
  invalid_options.majorization_tolerance = -1e-12;
  threw = false;
  try {
    picasso::solver::MultinomialLlaSolver invalid_solver(
        objective, pn_options, invalid_options);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  ok &= require(threw, "negative majorization tolerance must be rejected");
  invalid_options.majorization_tolerance = nan;
  threw = false;
  try {
    picasso::solver::MultinomialLlaSolver invalid_solver(
        objective, pn_options, invalid_options);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  ok &= require(threw, "NaN majorization tolerance must be rejected");
  invalid_options.majorization_tolerance = infinity;
  threw = false;
  try {
    picasso::solver::MultinomialLlaSolver invalid_solver(
        objective, pn_options, invalid_options);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  ok &= require(threw, "infinite majorization tolerance must be rejected");

  picasso::solver::MultinomialLlaSolver solver(objective);
  ok &= require(solve_call_throws(solver, Penalty::kMCP, -0.1, 3.0) &&
                    solve_call_throws(solver, Penalty::kMCP, nan, 3.0) &&
                    solve_call_throws(solver, Penalty::kMCP, 0.1, 1.0) &&
                    solve_call_throws(solver, Penalty::kSCAD, 0.1, 2.0) &&
                    solve_call_throws(solver, Penalty::kSCAD, 0.1,
                                      infinity),
                "LLA solve must reject invalid lambda and gamma values");
  Eigen::MatrixXd beta = Eigen::MatrixXd::Zero(2, 3);
  Eigen::VectorXd intercept = Eigen::VectorXd::Zero(3);
  threw = false;
  try {
    solver.solve(Penalty::kMCP, 0.1, 3.0,
                 Eigen::MatrixXd::Zero(1, 3), intercept);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  ok &= require(threw, "LLA solve must reject an initial beta shape mismatch");
  beta(0, 0) = nan;
  threw = false;
  try {
    solver.solve(Penalty::kMCP, 0.1, 3.0, beta, intercept);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  ok &= require(threw, "LLA solve must reject nonfinite initial parameters");
  return ok;
}

bool check_manual_three_stage_equivalence(Penalty penalty, double gamma,
                                          const std::string &name) {
  Eigen::MatrixXd x;
  Eigen::VectorXi labels;
  make_signal_fixture(&x, &labels);
  picasso::MultinomialObjective objective(x, labels, 3);
  const double lambda = 0.018;
  picasso::solver::MultinomialActNewtonOptions pn_options;
  pn_options.max_outer_iterations = 120;
  pn_options.max_inner_sweeps = 5000;
  pn_options.outer_kkt_tolerance = 5e-7;
  pn_options.inner_kkt_tolerance = 1e-9;
  const picasso::solver::MultinomialLlaOptions lla_options =
      picasso::solver::MultinomialLlaOptions::fixed_stage_compatibility();
  picasso::solver::MultinomialLlaSolver driver(objective, pn_options,
                                               lla_options);
  const picasso::solver::MultinomialLlaResult result =
      driver.solve(penalty, lambda, gamma);

  picasso::solver::MultinomialActNewtonSolver pn_solver(objective,
                                                        pn_options);
  std::vector<picasso::solver::MultinomialActNewtonResult> manual(3);
  manual[0] = pn_solver.solve(lambda);
  for (int stage = 1; stage < 3; ++stage) {
    const Eigen::MatrixXd weights = reference_lla_weights(
        manual[stage - 1].beta, penalty, lambda, gamma);
    manual[stage] = pn_solver.solve(weights, manual[stage - 1].beta,
                                    manual[stage - 1].intercept);
  }

  bool ok = true;
  ok &= require(result.completed() && result.failed_stage == -1 &&
                    result.completed_stages == 3 &&
                    result.stages.size() == 3,
                name + " driver must complete one L1 and two LLA stages");
  for (int stage = 0; stage < 3; ++stage) {
    ok &= require(manual[stage].converged(),
                  name + " manual subproblem must converge at stage " +
                      std::to_string(stage));
  }
  ok &= require((result.l1_master_beta - manual[0].beta)
                            .cwiseAbs()
                            .maxCoeff() < 2e-13 &&
                    (result.l1_master_intercept - manual[0].intercept)
                            .cwiseAbs()
                            .maxCoeff() < 2e-13,
                name + " driver must preserve the independently solved L1 "
                "master");
  ok &= require((result.beta - manual[2].beta).cwiseAbs().maxCoeff() <
                        2e-13 &&
                    (result.intercept - manual[2].intercept)
                            .cwiseAbs()
                            .maxCoeff() < 2e-13,
                name + " driver final parameters must equal two explicit "
                "weighted-L1 solves");

  long long coordinate_updates = 0;
  int outer_iterations = 0;
  int inner_sweeps = 0;
  double previous_target = std::numeric_limits<double>::infinity();
  for (int stage = 0; stage < 3; ++stage) {
    const picasso::solver::MultinomialLlaStageRecord &record =
        result.stages[static_cast<std::size_t>(stage)];
    coordinate_updates += manual[stage].total_coordinate_updates;
    outer_iterations += manual[stage].outer_iterations;
    inner_sweeps += manual[stage].total_inner_sweeps;
    const double target = reference_target_objective(
        objective, manual[stage].beta, manual[stage].intercept, penalty,
        lambda, gamma);
    const double stationarity = reference_target_stationarity(
        objective, manual[stage].beta, manual[stage].intercept, penalty,
        lambda, gamma, true, pn_options.zero_tolerance);
    ok &= require(record.stage == stage &&
                      record.is_l1_master == (stage == 0) &&
                      record.subproblem_status ==
                          picasso::solver::MultinomialSolverStatus::kConverged &&
                      record.outer_iterations == manual[stage].outer_iterations &&
                      record.inner_sweeps == manual[stage].total_inner_sweeps &&
                      record.coordinate_updates ==
                          manual[stage].total_coordinate_updates,
                  name + " stage record must reproduce manual PN metadata");
    ok &= require(nearly_equal(record.surrogate_objective,
                               manual[stage].final_objective, 2e-13,
                               2e-12) &&
                      nearly_equal(record.target_objective, target, 2e-13,
                                   2e-12) &&
                      nearly_equal(record.target_stationarity, stationarity,
                                   2e-12, 2e-10),
                  name + " stage objectives/stationarity must be independently "
                  "reproducible");
    if (stage > 0) {
      const Eigen::MatrixXd weights = reference_lla_weights(
          manual[stage - 1].beta, penalty, lambda, gamma);
      const double tangent_constant = reference_tangent_constant(
          manual[stage - 1].beta, weights, penalty, lambda, gamma);
      const double majorizer_anchor =
          objective.negative_log_likelihood(manual[stage - 1].beta,
                                            manual[stage - 1].intercept) +
          weighted_l1(manual[stage - 1].beta, weights) + tangent_constant;
      const double majorizer_solution =
          objective.negative_log_likelihood(manual[stage].beta,
                                            manual[stage].intercept) +
          weighted_l1(manual[stage].beta, weights) + tangent_constant;
      ok &= require(nearly_equal(record.tangent_constant, tangent_constant,
                                 2e-13, 2e-12) &&
                        nearly_equal(record.majorizer_at_anchor,
                                     majorizer_anchor, 2e-13, 2e-12) &&
                        nearly_equal(record.majorizer_at_solution,
                                     majorizer_solution, 2e-13, 2e-12),
                    name + " LLA majorizer chain must match an independent "
                    "tangent reconstruction");
      ok &= require(target <= majorizer_solution + 2e-12 &&
                        majorizer_solution <= majorizer_anchor + 2e-12 &&
                        nearly_equal(majorizer_anchor, previous_target,
                                     2e-13, 2e-12) &&
                        target <= previous_target + 2e-12,
                    name + " accepted LLA stage must satisfy Q(new) <= "
                    "M(new) <= M(anchor) = Q(anchor)");
    }
    previous_target = target;
  }
  ok &= require(result.total_outer_iterations == outer_iterations &&
                    result.total_inner_sweeps == inner_sweeps &&
                    result.total_coordinate_updates == coordinate_updates,
                name + " aggregate work must equal the three manual solves");
  const double final_target = reference_target_objective(
      objective, result.beta, result.intercept, penalty, lambda, gamma);
  const double final_stationarity = reference_target_stationarity(
      objective, result.beta, result.intercept, penalty, lambda, gamma, true,
      pn_options.zero_tolerance);
  ok &= require(nearly_equal(result.final_target_objective, final_target,
                             2e-13, 2e-12) &&
                    nearly_equal(result.final_target_stationarity,
                                 final_stationarity, 2e-12, 2e-10),
                name + " final target objective and stationarity must be "
                "independently reproducible");
  Eigen::MatrixXd driver_probabilities;
  Eigen::MatrixXd manual_probabilities;
  objective.negative_log_likelihood(result.beta, result.intercept,
                                    &driver_probabilities);
  objective.negative_log_likelihood(manual[2].beta, manual[2].intercept,
                                    &manual_probabilities);
  ok &= require((driver_probabilities - manual_probabilities)
                        .cwiseAbs()
                        .maxCoeff() < 2e-13,
                name + " manual and driver fitted probabilities must match");
  return ok;
}

bool test_end_to_end_manual_equivalence() {
  bool ok = true;
  ok &= check_manual_three_stage_equivalence(Penalty::kMCP, 3.0, "MCP");
  ok &= check_manual_three_stage_equivalence(Penalty::kSCAD, 3.7, "SCAD");
  return ok;
}

bool test_adaptive_stationarity_and_fixed_compatibility() {
  Eigen::MatrixXd x;
  Eigen::VectorXi labels;
  make_signal_fixture(&x, &labels);
  picasso::MultinomialObjective objective(x, labels, 3);
  picasso::solver::MultinomialActNewtonOptions pn_options;
  pn_options.max_outer_iterations = 120;
  pn_options.max_inner_sweeps = 5000;
  pn_options.outer_kkt_tolerance = 5e-7;
  pn_options.inner_kkt_tolerance = 1e-9;
  const double lambda = 0.08;
  const double gamma = 3.7;

  const picasso::solver::MultinomialLlaOptions fixed_options =
      picasso::solver::MultinomialLlaOptions::fixed_stage_compatibility();
  picasso::solver::MultinomialLlaSolver fixed_solver(
      objective, pn_options, fixed_options);
  const picasso::solver::MultinomialLlaResult fixed = fixed_solver.solve(
      Penalty::kSCAD, lambda, gamma);

  picasso::solver::MultinomialLlaOptions adaptive_options;
  adaptive_options.stationarity_tolerance = 5e-7;
  adaptive_options.maximum_stages = 25;
  picasso::solver::MultinomialLlaSolver adaptive_solver(
      objective, pn_options, adaptive_options);
  const picasso::solver::MultinomialLlaResult adaptive =
      adaptive_solver.solve(Penalty::kSCAD, lambda, gamma);

  picasso::solver::MultinomialLlaOptions capped_options = adaptive_options;
  capped_options.maximum_stages = capped_options.minimum_stages;
  picasso::solver::MultinomialLlaSolver capped_solver(
      objective, pn_options, capped_options);
  const picasso::solver::MultinomialLlaResult capped =
      capped_solver.solve(Penalty::kSCAD, lambda, gamma);

  bool ok = true;
  ok &= require(fixed.completed() && fixed.completed_stages == 3 &&
                    fixed.stages.size() == 3 &&
                    fixed.final_target_stationarity >
                        adaptive_options.stationarity_tolerance,
                "explicit fixed-stage compatibility must retain the historical "
                "nonstationary three-stage result");
  ok &= require(adaptive.completed() && adaptive.failed_stage == -1 &&
                    adaptive.completed_stages > 3 &&
                    adaptive.completed_stages <=
                        adaptive_options.maximum_stages &&
                    adaptive.stages.size() ==
                        static_cast<std::size_t>(adaptive.completed_stages) &&
                    adaptive.final_target_stationarity <=
                        adaptive_options.stationarity_tolerance,
                "adaptive LLA must continue past three stages until target "
                "stationarity reaches tolerance");
  ok &= require(adaptive.final_target_objective <=
                        fixed.final_target_objective + 2e-12,
                "adaptive LLA must not increase the target objective beyond "
                "the compatible three-stage result");
  for (std::size_t stage = 1; stage < adaptive.stages.size(); ++stage) {
    const picasso::solver::MultinomialLlaStageRecord &record =
        adaptive.stages[stage];
    ok &= require(record.subproblem_status ==
                          picasso::solver::MultinomialSolverStatus::kConverged &&
                      record.subproblem_kkt_residual <=
                          pn_options.outer_kkt_tolerance,
                  "every adaptive weighted-L1 stage must retain an outer-KKT "
                  "certificate at the requested PN tolerance");
    ok &= require(record.target_objective <=
                          record.majorizer_at_solution + 2e-12 &&
                      record.majorizer_at_solution <=
                          record.majorizer_at_anchor + 2e-12 &&
                      record.target_objective <=
                          adaptive.stages[stage - 1].target_objective + 2e-12,
                  "every extra adaptive stage must preserve the majorization "
                  "and target-descent chain");
  }
  ok &= require(capped.status ==
                        picasso::solver::MultinomialLlaStatus::
                            kStationarityLimit &&
                    !capped.completed() && capped.completed_stages == 3 &&
                    capped.failed_stage == -1 &&
                    capped.has_valid_solution() &&
                    capped.final_target_stationarity >
                        capped_options.stationarity_tolerance &&
                    std::string(picasso::solver::multinomial_lla_status_string(
                        capped.status)) == "stationarity_limit",
                "an adaptive stage ceiling must retain a valid model while "
                "reporting a distinct stationarity-limit status");
  const double capped_target = reference_target_objective(
      objective, capped.beta, capped.intercept, Penalty::kSCAD, lambda, gamma);
  const double capped_stationarity = reference_target_stationarity(
      objective, capped.beta, capped.intercept, Penalty::kSCAD, lambda, gamma,
      true, pn_options.zero_tolerance);
  ok &= require(nearly_equal(capped.final_target_objective, capped_target,
                             2e-13, 2e-12) &&
                    nearly_equal(capped.final_target_stationarity,
                                 capped_stationarity, 2e-12, 2e-10) &&
                    capped.stages.back().subproblem_status ==
                        picasso::solver::MultinomialSolverStatus::kConverged,
                "a stationarity-limit result must expose an independently "
                "reproducible last fully accepted stage");
  return ok;
}

bool test_active_set_and_cache_ab_equivalence() {
  Eigen::MatrixXd x;
  Eigen::VectorXi labels;
  make_signal_fixture(&x, &labels);
  picasso::MultinomialObjective objective(x, labels, 3);
  const Penalty penalties[] = {Penalty::kMCP, Penalty::kSCAD};
  const double gammas[] = {3.0, 3.7};
  bool ok = true;
  for (int kind = 0; kind < 2; ++kind) {
    picasso::solver::MultinomialActNewtonOptions full_options;
    full_options.max_outer_iterations = 120;
    full_options.max_inner_sweeps = 5000;
    full_options.outer_kkt_tolerance = 5e-7;
    full_options.inner_kkt_tolerance = 1e-9;
    full_options.use_active_set = false;
    full_options.use_probability_dot_direction_cache = true;
    picasso::solver::MultinomialActNewtonOptions active_options =
        full_options;
    active_options.use_active_set = true;
    picasso::solver::MultinomialActNewtonOptions uncached_options =
        active_options;
    uncached_options.use_probability_dot_direction_cache = false;
    picasso::solver::MultinomialLlaOptions lla_options;
    lla_options.maximum_stages = 25;
    picasso::solver::MultinomialLlaSolver full_solver(objective,
                                                      full_options,
                                                      lla_options);
    picasso::solver::MultinomialLlaSolver active_solver(objective,
                                                        active_options,
                                                        lla_options);
    picasso::solver::MultinomialLlaSolver uncached_solver(
        objective, uncached_options, lla_options);
    const picasso::solver::MultinomialLlaResult full =
        full_solver.solve(penalties[kind], 0.018, gammas[kind]);
    const picasso::solver::MultinomialLlaResult active =
        active_solver.solve(penalties[kind], 0.018, gammas[kind]);
    const picasso::solver::MultinomialLlaResult uncached =
        uncached_solver.solve(penalties[kind], 0.018, gammas[kind]);
    ok &= require(full.completed() && active.completed() &&
                      uncached.completed(),
                  "LLA active/cache A/B solves must all complete");
    Eigen::MatrixXd full_probabilities;
    Eigen::MatrixXd active_probabilities;
    Eigen::MatrixXd uncached_probabilities;
    objective.negative_log_likelihood(full.beta, full.intercept,
                                      &full_probabilities);
    objective.negative_log_likelihood(active.beta, active.intercept,
                                      &active_probabilities);
    objective.negative_log_likelihood(uncached.beta, uncached.intercept,
                                      &uncached_probabilities);
    ok &= require(nearly_equal(full.final_target_objective,
                               active.final_target_objective, 2e-9, 2e-7) &&
                      (full_probabilities - active_probabilities)
                              .cwiseAbs()
                              .maxCoeff() < 2e-6,
                  "LLA active-set on/off must reach the same target fit");
    ok &= require(nearly_equal(active.final_target_objective,
                               uncached.final_target_objective, 2e-10,
                               2e-8) &&
                      (active_probabilities - uncached_probabilities)
                              .cwiseAbs()
                              .maxCoeff() < 2e-7,
                  "LLA probability cache on/off must reach the same target "
                  "fit");
  }
  return ok;
}

Eigen::VectorXd empirical_null_intercept(
    const picasso::MultinomialObjective &objective) {
  Eigen::VectorXd intercept = Eigen::VectorXd::Zero(objective.class_num());
  for (Eigen::Index i = 0; i < objective.labels().size(); ++i)
    intercept[objective.labels()[i]] += 1.0;
  intercept.array() /= static_cast<double>(objective.sample_num());
  for (Eigen::Index k = 0; k < intercept.size(); ++k)
    intercept[k] = std::log(std::max(intercept[k], 1e-8));
  intercept.array() -= intercept.mean();
  return intercept;
}

bool test_lambda_boundaries() {
  Eigen::MatrixXd x;
  Eigen::VectorXi labels;
  make_signal_fixture(&x, &labels);
  picasso::MultinomialObjective objective(x, labels, 3);
  const Eigen::MatrixXd zero_beta = Eigen::MatrixXd::Zero(4, 3);
  const Eigen::VectorXd null_intercept = empirical_null_intercept(objective);
  Eigen::MatrixXd gradient;
  Eigen::VectorXd intercept_gradient;
  objective.smooth_gradient(zero_beta, null_intercept, &gradient,
                            &intercept_gradient);
  const double lambda = 1.00000001 * gradient.cwiseAbs().maxCoeff();
  const Penalty penalties[] = {Penalty::kMCP, Penalty::kSCAD};
  const double gammas[] = {3.0, 3.7};
  bool ok = true;
  for (int kind = 0; kind < 2; ++kind) {
    picasso::solver::MultinomialLlaSolver solver(objective);
    const picasso::solver::MultinomialLlaResult result =
        solver.solve(penalties[kind], lambda, gammas[kind]);
    ok &= require(result.completed() &&
                      result.beta.cwiseAbs().maxCoeff() < 1e-14 &&
                      result.completed_stages == 3,
                  "lambda >= lambda_max must retain an all-zero slope in "
                  "every LLA stage");
    for (std::size_t stage = 0; stage < result.stages.size(); ++stage) {
      ok &= require(result.stages[stage].subproblem_status ==
                            picasso::solver::MultinomialSolverStatus::
                                kConverged &&
                        result.stages[stage].outer_iterations == 0,
                    "lambda_max stages must be immediate converged PN "
                    "subproblems");
    }
  }

  Eigen::MatrixXd zero_x = Eigen::MatrixXd::Zero(12, 2);
  Eigen::VectorXi balanced_labels(12);
  for (int i = 0; i < 12; ++i) balanced_labels[i] = i % 3;
  picasso::MultinomialObjective zero_objective(zero_x, balanced_labels, 3);
  picasso::solver::MultinomialActNewtonSolver zero_pn(zero_objective);
  const picasso::solver::MultinomialActNewtonResult unpenalized =
      zero_pn.solve(0.0);
  for (int kind = 0; kind < 2; ++kind) {
    picasso::solver::MultinomialLlaSolver zero_solver(zero_objective);
    const picasso::solver::MultinomialLlaResult result =
        zero_solver.solve(penalties[kind], 0.0, gammas[kind]);
    ok &= require(result.completed() &&
                      (result.beta - unpenalized.beta)
                              .cwiseAbs()
                              .maxCoeff() == 0.0 &&
                      (result.intercept - unpenalized.intercept)
                              .cwiseAbs()
                              .maxCoeff() == 0.0 &&
                      result.final_target_objective ==
                          unpenalized.final_objective,
                  "lambda=0 LLA must reduce exactly to the unpenalized PN "
                  "solution");
  }
  return ok;
}

bool test_failed_stage_is_atomic() {
  Eigen::MatrixXd x;
  Eigen::VectorXi labels;
  make_signal_fixture(&x, &labels);
  picasso::MultinomialObjective objective(x, labels, 3);
  const double lambda = 0.006;
  picasso::solver::MultinomialActNewtonOptions strict_options;
  strict_options.max_outer_iterations = 150;
  strict_options.max_inner_sweeps = 6000;
  strict_options.outer_kkt_tolerance = 1e-9;
  strict_options.inner_kkt_tolerance = 1e-11;
  picasso::solver::MultinomialActNewtonSolver master_solver(
      objective, strict_options);
  const picasso::solver::MultinomialActNewtonResult master =
      master_solver.solve(lambda);

  bool ok = true;
  ok &= require(master.converged() && master.beta.cwiseAbs().maxCoeff() > 1e-4,
                "atomic-failure fixture must have a converged nonzero L1 "
                "master");
  picasso::solver::MultinomialActNewtonOptions capped_options =
      strict_options;
  capped_options.max_outer_iterations = 1;
  capped_options.max_inner_sweeps = 1;
  picasso::solver::MultinomialLlaSolver capped_driver(objective,
                                                      capped_options);
  const picasso::solver::MultinomialLlaResult result = capped_driver.solve(
      Penalty::kMCP, lambda, 3.0, master.beta, master.intercept);
  ok &= require(result.status ==
                        picasso::solver::MultinomialLlaStatus::
                            kSubproblemFailed &&
                    result.failed_stage == 1 &&
                    result.completed_stages == 1 &&
                    result.stages.size() == 2 &&
                    result.stages.back().subproblem_status !=
                        picasso::solver::MultinomialSolverStatus::kConverged,
                "a capped stage-one PN solve must be reported as the failed "
                "stage");
  ok &= require((result.beta - result.l1_master_beta)
                            .cwiseAbs()
                            .maxCoeff() == 0.0 &&
                    (result.intercept - result.l1_master_intercept)
                            .cwiseAbs()
                            .maxCoeff() == 0.0 &&
                    (result.l1_master_beta - master.beta)
                            .cwiseAbs()
                            .maxCoeff() < 2e-14 &&
                    (result.l1_master_intercept - master.intercept)
                            .cwiseAbs()
                            .maxCoeff() < 2e-14,
                "a failed LLA subproblem must not propagate its partial "
                "iterate over the last accepted master");
  const double retained_target = reference_target_objective(
      objective, master.beta, master.intercept, Penalty::kMCP, lambda, 3.0);
  ok &= require(nearly_equal(result.final_target_objective, retained_target,
                             2e-14, 2e-14),
                "failure must report the retained accepted target objective");
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_penalty_value_derivative_and_majorization();
  ok &= test_invalid_penalty_and_solver_inputs();
  ok &= test_end_to_end_manual_equivalence();
  ok &= test_adaptive_stationarity_and_fixed_compatibility();
  ok &= test_active_set_and_cache_ab_equivalence();
  ok &= test_lambda_boundaries();
  ok &= test_failed_stage_is_atomic();
  if (!ok) return 1;
  std::cout << "multinomial_lla_test: PASS\n";
  return 0;
}
