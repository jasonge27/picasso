#include <picasso/multinomial_actnewton.hpp>
#include <picasso/multinomial_lla.hpp>
#include <picasso/multinomial_objective.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct LegacyMultinomialActNewtonOptionsLayout {
  int max_outer_iterations;
  int max_inner_sweeps;
  int max_line_search_steps;
  int exact_kkt_scan_interval;
  double outer_kkt_tolerance;
  double inner_kkt_tolerance;
  double armijo_constant;
  double backtracking_factor;
  double minimum_step_size;
  double hessian_damping;
  double zero_tolerance;
  bool include_intercept;
  bool use_probability_dot_direction_cache;
  bool use_active_set;
  bool canonicalize_feature_l1_gauge;
};

using CurrentMultinomialActNewtonOptions =
    picasso::solver::MultinomialActNewtonOptions;

static_assert(
    std::is_nothrow_move_assignable<
        picasso::solver::MultinomialActNewtonPathState>::value,
    "transactional path commits require non-throwing state moves");

static_assert(
    std::is_copy_constructible<picasso::MultinomialObjective>::value &&
        !std::is_copy_assignable<picasso::MultinomialObjective>::value &&
        !std::is_move_assignable<picasso::MultinomialObjective>::value,
    "multinomial objectives may be constructed but not replaced in place");

static_assert(
    std::is_standard_layout<LegacyMultinomialActNewtonOptionsLayout>::value &&
        std::is_standard_layout<CurrentMultinomialActNewtonOptions>::value,
    "multinomial option layouts must support portable offsetof checks");
static_assert(sizeof(CurrentMultinomialActNewtonOptions) ==
                  sizeof(LegacyMultinomialActNewtonOptionsLayout),
              "new multinomial kernel flags must fit in legacy tail padding");
static_assert(alignof(CurrentMultinomialActNewtonOptions) ==
                  alignof(LegacyMultinomialActNewtonOptionsLayout),
              "multinomial option alignment must remain ABI-compatible");
#define PICASSO_ASSERT_LEGACY_OPTION_OFFSET(field)                         \
  static_assert(offsetof(CurrentMultinomialActNewtonOptions, field) ==     \
                    offsetof(LegacyMultinomialActNewtonOptionsLayout,      \
                             field),                                      \
                "legacy multinomial option offset changed: " #field)
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(max_outer_iterations);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(max_inner_sweeps);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(max_line_search_steps);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(exact_kkt_scan_interval);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(outer_kkt_tolerance);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(inner_kkt_tolerance);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(armijo_constant);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(backtracking_factor);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(minimum_step_size);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(hessian_damping);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(zero_tolerance);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(include_intercept);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(use_probability_dot_direction_cache);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(use_active_set);
PICASSO_ASSERT_LEGACY_OPTION_OFFSET(canonicalize_feature_l1_gauge);
#undef PICASSO_ASSERT_LEGACY_OPTION_OFFSET
static_assert(
    offsetof(CurrentMultinomialActNewtonOptions,
             use_adaptive_inner_tolerance) >=
        offsetof(LegacyMultinomialActNewtonOptionsLayout,
                 canonicalize_feature_l1_gauge) +
            sizeof(bool),
    "new multinomial flags must follow every legacy field");
static_assert(
    offsetof(CurrentMultinomialActNewtonOptions,
             use_adaptive_inner_tolerance) +
            sizeof(bool) <=
        offsetof(CurrentMultinomialActNewtonOptions,
                 use_vectorized_coordinate_kernels) &&
        offsetof(CurrentMultinomialActNewtonOptions,
                 use_vectorized_coordinate_kernels) +
                sizeof(bool) <=
            offsetof(CurrentMultinomialActNewtonOptions,
                     reuse_line_search_probabilities) &&
        offsetof(CurrentMultinomialActNewtonOptions,
                 reuse_line_search_probabilities) +
                sizeof(bool) <=
            offsetof(CurrentMultinomialActNewtonOptions,
                     use_compact_inner_active_set) &&
        offsetof(CurrentMultinomialActNewtonOptions,
                 use_compact_inner_active_set) +
                sizeof(bool) <=
            sizeof(LegacyMultinomialActNewtonOptionsLayout),
    "new multinomial flags must remain entirely inside legacy tail padding");

bool require(bool condition, const std::string &message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

double coefficient_kkt(double coefficient, double gradient, double lambda,
                       double zero_tolerance = 1e-10) {
  if (coefficient > zero_tolerance)
    return std::fabs(gradient + lambda);
  if (coefficient < -zero_tolerance)
    return std::fabs(gradient - lambda);
  return std::max(0.0, std::fabs(gradient) - lambda);
}

double independent_outer_kkt(const picasso::MultinomialObjective &objective,
                             const Eigen::MatrixXd &beta,
                             const Eigen::VectorXd &intercept,
                             double lambda, bool include_intercept) {
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  objective.smooth_gradient(beta, intercept, &beta_gradient,
                            &intercept_gradient);
  double residual = 0.0;
  for (Eigen::Index j = 0; j < beta.rows(); ++j) {
    for (Eigen::Index k = 0; k < beta.cols(); ++k) {
      residual = std::max(
          residual,
          coefficient_kkt(beta(j, k), beta_gradient(j, k), lambda));
    }
  }
  if (include_intercept)
    residual = std::max(residual, intercept_gradient.cwiseAbs().maxCoeff());
  return residual;
}

double weighted_l1_penalty(const Eigen::MatrixXd &beta,
                           const Eigen::MatrixXd &penalties) {
  return (beta.cwiseAbs().array() * penalties.array()).sum();
}

double independent_weighted_outer_kkt(
    const picasso::MultinomialObjective &objective,
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    const Eigen::MatrixXd &penalties, bool include_intercept) {
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  objective.smooth_gradient(beta, intercept, &beta_gradient,
                            &intercept_gradient);
  double residual = 0.0;
  for (Eigen::Index j = 0; j < beta.rows(); ++j) {
    for (Eigen::Index k = 0; k < beta.cols(); ++k) {
      residual = std::max(
          residual, coefficient_kkt(beta(j, k), beta_gradient(j, k),
                                    penalties(j, k)));
    }
  }
  if (include_intercept)
    residual = std::max(residual, intercept_gradient.cwiseAbs().maxCoeff());
  return residual;
}

double independent_nonconvex_objective(
    const picasso::MultinomialObjective &objective,
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    picasso::solver::MultinomialLlaPenalty penalty, double lambda,
    double gamma) {
  double value = objective.negative_log_likelihood(beta, intercept);
  for (Eigen::Index j = 0; j < beta.rows(); ++j) {
    for (Eigen::Index k = 0; k < beta.cols(); ++k) {
      value += picasso::solver::multinomial_lla_penalty_value(
          penalty, std::fabs(beta(j, k)), lambda, gamma);
    }
  }
  return value;
}

double independent_nonconvex_stationarity(
    const picasso::MultinomialObjective &objective,
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    picasso::solver::MultinomialLlaPenalty penalty, double lambda,
    double gamma, bool include_intercept) {
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  objective.smooth_gradient(beta, intercept, &beta_gradient,
                            &intercept_gradient);
  double residual = 0.0;
  for (Eigen::Index j = 0; j < beta.rows(); ++j) {
    for (Eigen::Index k = 0; k < beta.cols(); ++k) {
      const double derivative =
          picasso::solver::multinomial_lla_penalty_derivative(
              penalty, std::fabs(beta(j, k)), lambda, gamma);
      residual = std::max(
          residual,
          coefficient_kkt(beta(j, k), beta_gradient(j, k), derivative));
    }
  }
  if (include_intercept)
    residual = std::max(residual, intercept_gradient.cwiseAbs().maxCoeff());
  return residual;
}

bool nearly_equal(double left, double right, double absolute_tolerance,
                  double relative_tolerance) {
  return std::fabs(left - right) <=
         absolute_tolerance +
             relative_tolerance * std::max(std::fabs(left), std::fabs(right));
}

bool has_same_support(const Eigen::MatrixXd &left,
                      const Eigen::MatrixXd &right,
                      double zero_tolerance = 1e-8) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) return false;
  for (Eigen::Index row = 0; row < left.rows(); ++row) {
    for (Eigen::Index column = 0; column < left.cols(); ++column) {
      if ((std::fabs(left(row, column)) > zero_tolerance) !=
          (std::fabs(right(row, column)) > zero_tolerance))
        return false;
    }
  }
  return true;
}

double matrix_l1_norm(const Eigen::MatrixXd &value) {
  return value.cwiseAbs().sum();
}

double row_median_center(const Eigen::MatrixXd &value, Eigen::Index row) {
  std::vector<double> entries(static_cast<std::size_t>(value.cols()));
  for (Eigen::Index column = 0; column < value.cols(); ++column)
    entries[static_cast<std::size_t>(column)] = value(row, column);
  std::sort(entries.begin(), entries.end());
  const std::size_t lower =
      static_cast<std::size_t>((value.cols() - 1) / 2);
  const std::size_t upper = static_cast<std::size_t>(value.cols() / 2);
  return 0.5 * entries[lower] + 0.5 * entries[upper];
}

bool has_canonical_feature_gauge(const Eigen::MatrixXd &beta,
                                 double tolerance = 1e-13) {
  for (Eigen::Index feature = 0; feature < beta.rows(); ++feature) {
    if (std::fabs(row_median_center(beta, feature)) > tolerance) return false;
  }
  return true;
}

double weighted_row_center(const Eigen::MatrixXd &beta,
                           const Eigen::MatrixXd &penalties,
                           Eigen::Index row) {
  std::vector<std::pair<double, double> > entries;
  double total_weight = 0.0;
  for (Eigen::Index klass = 0; klass < beta.cols(); ++klass) {
    const double weight = penalties(row, klass);
    if (weight > 0.0) {
      entries.push_back(std::make_pair(beta(row, klass), weight));
      total_weight += weight;
    }
  }
  if (entries.empty()) return row_median_center(beta, row);
  std::sort(entries.begin(), entries.end());
  const double half_weight = 0.5 * total_weight;
  double cumulative = 0.0;
  double lower = entries.back().first;
  double upper = entries.back().first;
  bool found_lower = false;
  bool found_upper = false;
  for (std::size_t index = 0; index < entries.size(); ++index) {
    cumulative += entries[index].second;
    if (!found_lower && cumulative >= half_weight) {
      lower = entries[index].first;
      found_lower = true;
    }
    if (!found_upper && cumulative > half_weight) {
      upper = entries[index].first;
      found_upper = true;
    }
  }
  return 0.5 * lower + 0.5 * upper;
}

bool has_canonical_weighted_feature_gauge(
    const Eigen::MatrixXd &beta, const Eigen::MatrixXd &penalties,
    double tolerance = 1e-12) {
  for (Eigen::Index row = 0; row < beta.rows(); ++row) {
    if (std::fabs(weighted_row_center(beta, penalties, row)) > tolerance)
      return false;
  }
  return true;
}

bool rejects_l1_penalties(
    const picasso::solver::MultinomialActNewtonSolver &solver,
    const Eigen::MatrixXd &penalties) {
  try {
    solver.solve(penalties);
  } catch (const std::invalid_argument &) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

bool equivalent_result_fields(
    const picasso::solver::MultinomialActNewtonResult &scalar,
    const picasso::solver::MultinomialActNewtonResult &weighted,
    const std::string &case_name) {
  bool ok = true;
  ok &= require(
      scalar.status == weighted.status &&
          scalar.outer_iterations == weighted.outer_iterations &&
          scalar.total_inner_sweeps == weighted.total_inner_sweeps &&
          scalar.total_coordinate_updates == weighted.total_coordinate_updates &&
          scalar.final_active_features == weighted.final_active_features &&
          scalar.total_reactivated_features ==
              weighted.total_reactivated_features &&
          scalar.total_subproblem_reactivated_features ==
              weighted.total_subproblem_reactivated_features &&
          scalar.total_outer_reactivated_features ==
              weighted.total_outer_reactivated_features &&
          scalar.total_full_subproblem_kkt_scans ==
              weighted.total_full_subproblem_kkt_scans,
      case_name + " result status and integer diagnostics must match");
  ok &= require(
      (scalar.beta - weighted.beta).cwiseAbs().maxCoeff() <= 2e-14 &&
          (scalar.intercept - weighted.intercept).cwiseAbs().maxCoeff() <=
              2e-14 &&
          nearly_equal(scalar.final_objective, weighted.final_objective,
                       2e-14, 2e-14) &&
          nearly_equal(scalar.final_kkt_residual,
                       weighted.final_kkt_residual, 2e-14, 2e-14),
      case_name + " final numerical fields must match");
  ok &= require(scalar.history.size() == weighted.history.size(),
                case_name + " history lengths must match");
  if (scalar.history.size() != weighted.history.size()) return false;
  for (std::size_t index = 0; index < scalar.history.size(); ++index) {
    const picasso::solver::MultinomialIterationRecord &left =
        scalar.history[index];
    const picasso::solver::MultinomialIterationRecord &right =
        weighted.history[index];
    ok &= require(
        left.outer_iteration == right.outer_iteration &&
            left.inner_sweeps == right.inner_sweeps &&
            left.line_search_steps == right.line_search_steps &&
            left.inner_converged == right.inner_converged &&
            left.active_features == right.active_features &&
            left.newly_activated_features == right.newly_activated_features &&
            left.subproblem_reactivated_features ==
                right.subproblem_reactivated_features &&
            left.outer_reactivated_features ==
                right.outer_reactivated_features &&
            left.full_subproblem_kkt_scans ==
                right.full_subproblem_kkt_scans,
        case_name + " history metadata must match");
    ok &= require(
        nearly_equal(left.objective, right.objective, 2e-14, 2e-14) &&
            nearly_equal(left.kkt_residual, right.kkt_residual, 2e-14,
                         2e-14) &&
            nearly_equal(left.inner_kkt_residual,
                         right.inner_kkt_residual, 2e-14, 2e-14) &&
            left.step_size == right.step_size &&
            nearly_equal(left.direction_norm, right.direction_norm, 2e-14,
                         2e-14) &&
            nearly_equal(left.composite_directional_derivative,
                         right.composite_directional_derivative, 2e-14,
                         2e-14),
        case_name + " history numerical fields must match");
  }
  return ok;
}

bool test_lambda_max_zero_solution_and_intercept_gauge() {
  Eigen::MatrixXd x(12, 2);
  for (int i = 0; i < x.rows(); ++i) {
    x(i, 0) = std::sin(0.7 * static_cast<double>(i + 1));
    x(i, 1) = std::cos(0.4 * static_cast<double>(i + 2));
  }
  Eigen::VectorXi labels(12);
  labels << 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2;
  picasso::MultinomialObjective objective(x, labels, 3);

  Eigen::MatrixXd zero_beta = Eigen::MatrixXd::Zero(2, 3);
  Eigen::VectorXd intercept(3);
  intercept << std::log(5.0 / 12.0), std::log(4.0 / 12.0),
      std::log(3.0 / 12.0);
  intercept.array() += 1000.0;

  Eigen::MatrixXd gradient;
  Eigen::VectorXd intercept_gradient;
  objective.smooth_gradient(zero_beta, intercept, &gradient,
                            &intercept_gradient);
  const double lambda_max = gradient.cwiseAbs().maxCoeff();

  picasso::solver::MultinomialActNewtonOptions options;
  options.outer_kkt_tolerance = 1e-9;
  picasso::solver::MultinomialActNewtonSolver solver(objective, options);
  const picasso::solver::MultinomialActNewtonResult result =
      solver.solve(lambda_max * (1.0 + 1e-8), zero_beta, intercept);
  const picasso::solver::MultinomialActNewtonResult default_result =
      solver.solve(lambda_max * (1.0 + 1e-8));
  const picasso::solver::MultinomialActNewtonResult explicit_zero_result =
      solver.solve(lambda_max * (1.0 + 1e-8), zero_beta,
                   Eigen::VectorXd::Zero(3));

  Eigen::VectorXd expected_intercept(3);
  expected_intercept << std::log(5.0 / 12.0), std::log(4.0 / 12.0),
      std::log(3.0 / 12.0);
  expected_intercept.array() -= expected_intercept.mean();

  bool ok = true;
  ok &= require(result.converged(),
                "lambda above lambda_max must converge at the zero slope");
  ok &= require(result.beta.cwiseAbs().maxCoeff() < 1e-14,
                "lambda above lambda_max must keep every slope at zero");
  ok &= require(std::fabs(result.intercept.mean()) < 1e-13,
                "a common-shift initial intercept must be mean centered");
  ok &= require(result.final_kkt_residual <= options.outer_kkt_tolerance,
                "zero-slope solution must pass the reported outer KKT test");
  ok &= require(default_result.converged() &&
                    default_result.outer_iterations == 0 &&
                    default_result.total_inner_sweeps == 0,
                "default empirical intercept must make lambda_max an "
                "immediate zero-slope solution");
  ok &= require((default_result.intercept - expected_intercept)
                            .cwiseAbs()
                            .maxCoeff() < 2e-14,
                "default intercept must equal centered empirical log "
                "class proportions");
  ok &= require(explicit_zero_result.converged() &&
                    nearly_equal(
                        explicit_zero_result.history.front().objective,
                        std::log(3.0), 2e-14, 2e-14),
                "the explicit-initial-state overload must continue to honor "
                "a caller-provided zero intercept");
  ok &= require(nearly_equal(default_result.final_objective,
                             explicit_zero_result.final_objective, 2e-12,
                             2e-10),
                "empirical and explicit-zero starts must reach the same "
                "lambda_max objective");
  return ok;
}

bool test_default_intercept_edge_cases() {
  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(8, 1);
  Eigen::VectorXi balanced_labels(8);
  balanced_labels << 0, 1, 2, 3, 0, 1, 2, 3;
  picasso::MultinomialObjective balanced_objective(x, balanced_labels, 4);
  picasso::solver::MultinomialActNewtonSolver balanced_solver(
      balanced_objective);
  const picasso::solver::MultinomialActNewtonResult balanced_result =
      balanced_solver.solve(1.0);

  bool ok = true;
  ok &= require(balanced_result.converged() &&
                    balanced_result.outer_iterations == 0 &&
                    balanced_result.intercept.isZero(0.0),
                "balanced classes must retain the exact zero default "
                "intercept");

  picasso::solver::MultinomialActNewtonOptions no_intercept_options;
  no_intercept_options.include_intercept = false;
  picasso::solver::MultinomialActNewtonSolver no_intercept_solver(
      balanced_objective, no_intercept_options);
  const picasso::solver::MultinomialActNewtonResult no_intercept_result =
      no_intercept_solver.solve(1.0);
  ok &= require(no_intercept_result.intercept.isZero(0.0),
                "include_intercept=false must bypass empirical "
                "initialization");

  Eigen::VectorXi missing_labels(8);
  missing_labels << 0, 0, 0, 0, 1, 1, 1, 2;
  picasso::MultinomialObjective missing_objective(x, missing_labels, 4);
  picasso::solver::MultinomialActNewtonSolver missing_solver(
      missing_objective);
  const picasso::solver::MultinomialActNewtonResult missing_result =
      missing_solver.solve(1.0);
  ok &= require(missing_result.intercept.allFinite() &&
                    std::fabs(missing_result.intercept.mean()) < 1e-13 &&
                    std::isfinite(missing_result.history.front().objective),
                "a low-level missing-class call must receive a finite, "
                "centered compatibility initialization");
  ok &= require(missing_result.intercept[3] <
                    missing_result.intercept.head(3).minCoeff(),
                "the missing class must receive the finite probability "
                "floor, not an observed-class intercept");
  return ok;
}

bool test_dense_quadratic_oracle() {
  Eigen::MatrixXd x(9, 2);
  x << -1.0, 0.2,
       -0.7, 0.8,
       -0.4, -0.6,
       -0.1, 1.0,
        0.2, -0.3,
        0.5, 0.7,
        0.8, -0.9,
        1.1, 0.4,
        1.4, -0.1;
  Eigen::VectorXi labels(9);
  labels << 0, 1, 2, 0, 2, 1, 0, 2, 1;
  picasso::MultinomialObjective objective(x, labels, 3);

  const int d = objective.feature_num();
  const int num_classes = objective.class_num();
  const int parameter_count = d * num_classes;
  Eigen::MatrixXd initial_beta = Eigen::MatrixXd::Zero(d, num_classes);
  Eigen::VectorXd ignored_intercept = Eigen::VectorXd::Constant(3, 7.0);
  Eigen::VectorXd zero_intercept = Eigen::VectorXd::Zero(3);
  Eigen::MatrixXd probabilities;
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  objective.smooth_gradient(initial_beta, zero_intercept, &beta_gradient,
                            &intercept_gradient, &probabilities);

  picasso::solver::MultinomialActNewtonOptions options;
  options.include_intercept = false;
  options.max_outer_iterations = 1;
  options.max_inner_sweeps = 10000;
  options.outer_kkt_tolerance = 1e-14;
  options.inner_kkt_tolerance = 1e-10;
  options.hessian_damping = 1e-8;
  options.use_active_set = false;
  const double lambda = 0.025;
  picasso::solver::MultinomialActNewtonSolver solver(objective, options);
  const picasso::solver::MultinomialActNewtonResult result =
      solver.solve(lambda, initial_beta, ignored_intercept);

  bool ok = true;
  ok &= require(result.history.size() == 2,
                "one outer iteration must produce one accepted history row");
  if (result.history.size() != 2) return false;
  ok &= require(result.status ==
                    picasso::solver::MultinomialSolverStatus::
                        kOuterIterationLimit,
                "one deliberately capped outer iteration must report the "
                "outer limit, not convergence");
  const double step_size = result.history.back().step_size;
  ok &= require(step_size > 0.0 && step_size <= 1.0,
                "dense-oracle step size must be valid");
  ok &= require(result.intercept.isZero(0.0),
                "include_intercept=false must force the intercept to zero");

  Eigen::MatrixXd dense_hessian =
      Eigen::MatrixXd::Zero(parameter_count, parameter_count);
  const double inverse_n = 1.0 / static_cast<double>(x.rows());
  for (int i = 0; i < x.rows(); ++i) {
    for (int left_feature = 0; left_feature < d; ++left_feature) {
      for (int right_feature = 0; right_feature < d; ++right_feature) {
        for (int left_class = 0; left_class < num_classes; ++left_class) {
          for (int right_class = 0; right_class < num_classes;
               ++right_class) {
            const double class_covariance =
                (left_class == right_class
                     ? probabilities(i, left_class)
                     : 0.0) -
                probabilities(i, left_class) *
                    probabilities(i, right_class);
            const int row = left_feature * num_classes + left_class;
            const int column = right_feature * num_classes + right_class;
            dense_hessian(row, column) +=
                inverse_n * x(i, left_feature) * x(i, right_feature) *
                class_covariance;
          }
        }
      }
    }
  }
  dense_hessian.diagonal().array() += options.hessian_damping;

  Eigen::VectorXd packed_gradient(parameter_count);
  Eigen::VectorXd packed_direction(parameter_count);
  Eigen::VectorXd packed_subproblem_coefficient(parameter_count);
  for (int j = 0; j < d; ++j) {
    for (int klass = 0; klass < num_classes; ++klass) {
      const int index = j * num_classes + klass;
      packed_gradient[index] = beta_gradient(j, klass);
      packed_direction[index] =
          (result.beta(j, klass) - initial_beta(j, klass)) / step_size;
      packed_subproblem_coefficient[index] =
          initial_beta(j, klass) + packed_direction[index];
    }
  }
  const Eigen::VectorXd dense_quadratic_gradient =
      packed_gradient + dense_hessian * packed_direction;
  double dense_kkt = 0.0;
  for (int index = 0; index < parameter_count; ++index) {
    dense_kkt = std::max(
        dense_kkt,
        coefficient_kkt(packed_subproblem_coefficient[index],
                        dense_quadratic_gradient[index], lambda));
  }
  ok &= require(dense_kkt < 5e-9,
                "inner direction must satisfy the independently materialized "
                "dense quadratic KKT system");
  ok &= require(result.history.back().inner_kkt_residual < 5e-9,
                "production inner KKT residual must agree with dense oracle");

  const double composite_delta =
      packed_gradient.dot(packed_direction) +
      lambda * (packed_subproblem_coefficient.cwiseAbs().sum() -
                initial_beta.cwiseAbs().sum());
  const double armijo_bound =
      result.history.front().objective +
      options.armijo_constant * step_size * composite_delta;
  ok &= require(composite_delta < 0.0,
                "the dense-oracle proximal Newton direction must be descent");
  ok &= require(std::fabs(composite_delta -
                          result.history.back()
                              .composite_directional_derivative) < 1e-12,
                "recorded composite slope must match an independent "
                "gradient-plus-L1 calculation");
  ok &= require(result.history.back().objective <= armijo_bound + 1e-13,
                "accepted step must satisfy the composite Armijo bound, "
                "including the L1 directional difference");

  picasso::solver::MultinomialActNewtonOptions limited_options = options;
  limited_options.max_inner_sweeps = 1;
  limited_options.max_outer_iterations = 2;
  limited_options.inner_kkt_tolerance = 1e-15;
  limited_options.use_active_set = true;
  limited_options.exact_kkt_scan_interval = 4;
  picasso::solver::MultinomialActNewtonSolver limited_solver(
      objective, limited_options);
  const picasso::solver::MultinomialActNewtonResult limited_result =
      limited_solver.solve(lambda, initial_beta, ignored_intercept);
  ok &= require(limited_result.status ==
                    picasso::solver::MultinomialSolverStatus::
                        kInnerIterationLimit,
                "an intentionally capped inner solve must report the inner "
                "limit, not silently converge");
  ok &= require(limited_result.total_full_subproblem_kkt_scans > 0,
                "the final allowed periodic sweep must receive an exact full "
                "KKT certification before reporting the inner limit");
  return ok;
}

bool test_final_kkt_and_monotone_armijo_history() {
  const int n = 72;
  const int d = 4;
  const int num_classes = 3;
  Eigen::MatrixXd x(n, d);
  Eigen::VectorXi labels(n);
  for (int i = 0; i < n; ++i) {
    x(i, 0) = std::sin(0.17 * static_cast<double>(i + 1));
    x(i, 1) = std::cos(0.31 * static_cast<double>(i + 2));
    x(i, 2) = static_cast<double>((i % 7) - 3) / 3.0;
    x(i, 3) = static_cast<double>((i * 5) % 11 - 5) / 5.0;
    labels[i] = (i * 7 + i / 4) % num_classes;
  }
  picasso::MultinomialObjective objective(x, labels, num_classes);

  picasso::solver::MultinomialActNewtonOptions options;
  options.max_outer_iterations = 80;
  options.max_inner_sweeps = 4000;
  options.outer_kkt_tolerance = 5e-7;
  options.inner_kkt_tolerance = 1e-9;
  const double lambda = 0.015;
  picasso::solver::MultinomialActNewtonSolver solver(objective, options);
  const picasso::solver::MultinomialActNewtonResult result =
      solver.solve(lambda);

  bool ok = true;
  ok &= require(result.converged(),
                std::string("full solve must converge, got status ") +
                    picasso::solver::multinomial_solver_status_string(
                        result.status));
  const double independent_kkt = independent_outer_kkt(
      objective, result.beta, result.intercept, lambda, true);
  ok &= require(independent_kkt <= 1.1 * options.outer_kkt_tolerance,
                "final coefficients must pass an independently recomputed KKT "
                "test");
  ok &= require(std::fabs(independent_kkt - result.final_kkt_residual) < 1e-12,
                "reported and independently recomputed KKT residuals must "
                "match");
  ok &= require(result.history.size() >= 2,
                "nontrivial solve must record accepted outer iterations");
  for (std::size_t i = 1; i < result.history.size(); ++i) {
    const double allowance =
        1e-12 * std::max(1.0, std::fabs(result.history[i - 1].objective));
    ok &= require(result.history[i].objective <=
                      result.history[i - 1].objective + allowance,
                  "Armijo history must be monotonically nonincreasing");
    const double armijo_bound =
        result.history[i - 1].objective +
        options.armijo_constant * result.history[i].step_size *
            result.history[i].composite_directional_derivative;
    ok &= require(result.history[i].objective <= armijo_bound + allowance,
                  "every accepted step must satisfy the recorded composite "
                  "Armijo bound");
    ok &= require(result.history[i].inner_converged,
                  "every accepted outer direction must pass inner KKT");
  }
  ok &= require(std::fabs(result.intercept.mean()) < 1e-13,
                "accepted intercept iterates must retain the zero-mean gauge");
  const picasso::solver::MultinomialActNewtonResult repeated_result =
      solver.solve(lambda, result.beta, result.intercept);
  ok &= require(repeated_result.converged() &&
                    repeated_result.outer_iterations == 0 &&
                    repeated_result.total_inner_sweeps == 0 &&
                    repeated_result.total_coordinate_updates == 0,
                "a repeated path lambda warm-started from its converged fit "
                "must require no Newton work");
  ok &= require(nearly_equal(repeated_result.final_objective,
                             result.final_objective, 2e-14, 2e-14) &&
                    repeated_result.final_kkt_residual ==
                        result.final_kkt_residual,
                "a repeated warm path point must preserve its objective and "
                "KKT residual");
  return ok;
}

bool test_probability_dot_direction_cache_ab_equivalence() {
  const int n = 96;
  const int d = 6;
  const int num_classes = 5;
  Eigen::MatrixXd x(n, d);
  Eigen::VectorXi labels(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      const double row = static_cast<double>(i + 1);
      const double column = static_cast<double>(j + 1);
      x(i, j) = std::sin(0.071 * row * column) +
                0.35 * std::cos(0.113 * row * (column + 1.0));
    }
    labels[i] = (i * 11 + i / 3 + 2) % num_classes;
  }
  picasso::MultinomialObjective objective(x, labels, num_classes);

  Eigen::MatrixXd initial_beta(d, num_classes);
  for (int j = 0; j < d; ++j) {
    for (int klass = 0; klass < num_classes; ++klass) {
      initial_beta(j, klass) =
          0.015 * std::sin(static_cast<double>((j + 2) * (klass + 1)));
    }
  }
  Eigen::VectorXd initial_intercept(num_classes);
  initial_intercept << 50.04, 49.97, 50.01, 49.95, 50.03;

  picasso::solver::MultinomialActNewtonOptions cached_options;
  bool ok = true;
  ok &= require(cached_options.use_probability_dot_direction_cache,
                "the p_i^T Deta_i cache must be enabled by default");
  cached_options.max_outer_iterations = 80;
  cached_options.max_inner_sweeps = 4000;
  cached_options.outer_kkt_tolerance = 5e-7;
  cached_options.inner_kkt_tolerance = 1e-9;
  picasso::solver::MultinomialActNewtonOptions naive_options =
      cached_options;
  naive_options.use_probability_dot_direction_cache = false;
  picasso::solver::MultinomialActNewtonOptions vectorized_options =
      cached_options;
  vectorized_options.use_vectorized_coordinate_kernels = true;
  picasso::solver::MultinomialActNewtonOptions reused_options =
      vectorized_options;
  reused_options.reuse_line_search_probabilities = true;
  picasso::solver::MultinomialActNewtonOptions production_options =
      reused_options;
  production_options.use_adaptive_inner_tolerance = true;
  production_options.use_compact_inner_active_set = true;
  picasso::solver::MultinomialActNewtonOptions fast_exact_options =
      production_options;
  fast_exact_options.outer_kkt_tolerance = 1e-4;
  fast_exact_options.inner_kkt_tolerance = 1e-6;
  fast_exact_options.reuse_line_search_probabilities = false;
  picasso::solver::MultinomialActNewtonOptions fast_incremental_options =
      fast_exact_options;
  fast_incremental_options.reuse_line_search_probabilities = true;
  ok &= require(!cached_options.use_adaptive_inner_tolerance &&
                    !cached_options.use_vectorized_coordinate_kernels &&
                    !cached_options.reuse_line_search_probabilities &&
                    !cached_options.use_compact_inner_active_set &&
                    production_options.use_adaptive_inner_tolerance &&
                    production_options.use_vectorized_coordinate_kernels &&
                    production_options.reuse_line_search_probabilities &&
                    production_options.use_compact_inner_active_set,
                "the production-kernel A/B fixture must compare all four "
                "switches against the strict scalar baseline");

  const double lambda = 0.018;
  picasso::solver::MultinomialActNewtonSolver cached_solver(
      objective, cached_options);
  picasso::solver::MultinomialActNewtonSolver naive_solver(
      objective, naive_options);
  picasso::solver::MultinomialActNewtonSolver vectorized_solver(
      objective, vectorized_options);
  picasso::solver::MultinomialActNewtonSolver reused_solver(
      objective, reused_options);
  picasso::solver::MultinomialActNewtonSolver production_solver(
      objective, production_options);
  picasso::solver::MultinomialActNewtonSolver fast_exact_solver(
      objective, fast_exact_options);
  picasso::solver::MultinomialActNewtonSolver fast_incremental_solver(
      objective, fast_incremental_options);
  const picasso::solver::MultinomialActNewtonResult cached_result =
      cached_solver.solve(lambda, initial_beta, initial_intercept);
  const picasso::solver::MultinomialActNewtonResult naive_result =
      naive_solver.solve(lambda, initial_beta, initial_intercept);
  const picasso::solver::MultinomialActNewtonResult vectorized_result =
      vectorized_solver.solve(lambda, initial_beta, initial_intercept);
  const picasso::solver::MultinomialActNewtonResult reused_result =
      reused_solver.solve(lambda, initial_beta, initial_intercept);
  const picasso::solver::MultinomialActNewtonResult production_result =
      production_solver.solve(lambda, initial_beta, initial_intercept);
  const picasso::solver::MultinomialActNewtonResult fast_exact_result =
      fast_exact_solver.solve(lambda, initial_beta, initial_intercept);
  const picasso::solver::MultinomialActNewtonResult fast_incremental_result =
      fast_incremental_solver.solve(lambda, initial_beta,
                                    initial_intercept);

  ok &= require(cached_result.status == naive_result.status,
                "cache on/off must return the same solver status");
  ok &= require(cached_result.converged() && naive_result.converged(),
                "both cache modes must converge in the A/B case");
  ok &= require(vectorized_result.converged(),
                "the vectorized coordinate kernel must converge in the A/B "
                "case");
  ok &= require(reused_result.converged(),
                "the line-search probability reuse path must converge in "
                "the A/B case");
  ok &= require(production_result.converged(),
                "the combined adaptive/vectorized/reuse L1 path must "
                "converge in the A/B case");
  ok &= require(
      fast_exact_result.converged() && fast_incremental_result.converged(),
      "both exact and incremental fast line-search paths must converge");
  ok &= require(nearly_equal(cached_result.final_objective,
                             naive_result.final_objective, 2e-12, 2e-11),
                "cache on/off must have the same final objective");
  ok &= require(nearly_equal(cached_result.final_kkt_residual,
                             naive_result.final_kkt_residual, 2e-11, 2e-9),
                "cache on/off must have the same reported final KKT");

  Eigen::MatrixXd cached_probabilities;
  Eigen::MatrixXd naive_probabilities;
  Eigen::MatrixXd vectorized_probabilities;
  Eigen::MatrixXd reused_probabilities;
  Eigen::MatrixXd production_probabilities;
  Eigen::MatrixXd fast_exact_probabilities;
  Eigen::MatrixXd fast_incremental_probabilities;
  objective.negative_log_likelihood(cached_result.beta,
                                    cached_result.intercept,
                                    &cached_probabilities);
  objective.negative_log_likelihood(naive_result.beta,
                                    naive_result.intercept,
                                    &naive_probabilities);
  objective.negative_log_likelihood(vectorized_result.beta,
                                    vectorized_result.intercept,
                                    &vectorized_probabilities);
  objective.negative_log_likelihood(reused_result.beta,
                                    reused_result.intercept,
                                    &reused_probabilities);
  const double production_nll = objective.negative_log_likelihood(
      production_result.beta, production_result.intercept,
      &production_probabilities);
  const double fast_exact_nll = objective.negative_log_likelihood(
      fast_exact_result.beta, fast_exact_result.intercept,
      &fast_exact_probabilities);
  const double fast_incremental_nll = objective.negative_log_likelihood(
      fast_incremental_result.beta, fast_incremental_result.intercept,
      &fast_incremental_probabilities);
  ok &= require((cached_probabilities - naive_probabilities)
                        .cwiseAbs()
                        .maxCoeff() < 2e-10,
                "cache on/off must produce the same fitted probabilities");
  ok &= require(
      nearly_equal(vectorized_result.final_objective,
                   cached_result.final_objective, 2e-12, 2e-11) &&
          (vectorized_probabilities - cached_probabilities)
                  .cwiseAbs()
                  .maxCoeff() < 2e-9,
      "vectorized and scalar coordinate kernels must preserve objective and "
      "fitted probabilities");
  ok &= require(
      nearly_equal(reused_result.final_objective,
                   vectorized_result.final_objective, 2e-12, 2e-11) &&
          (reused_probabilities - vectorized_probabilities)
                  .cwiseAbs()
                  .maxCoeff() < 2e-9,
      "reusing accepted line-search probabilities must preserve objective "
      "and fitted probabilities");
  const double production_objective =
      production_nll + lambda * matrix_l1_norm(production_result.beta);
  ok &= require(
      nearly_equal(production_result.final_objective,
                   production_objective, 2e-12, 2e-11) &&
          nearly_equal(production_result.final_objective,
                       cached_result.final_objective, 2e-10, 2e-8) &&
          (production_probabilities - cached_probabilities)
                  .cwiseAbs()
                  .maxCoeff() < 5e-6 &&
          has_same_support(production_result.beta, cached_result.beta),
      "the combined adaptive/vectorized/reuse L1 path must preserve the "
      "independent objective, probabilities, and support");
  const double fast_exact_objective =
      fast_exact_nll + lambda * matrix_l1_norm(fast_exact_result.beta);
  const double fast_incremental_objective =
      fast_incremental_nll +
      lambda * matrix_l1_norm(fast_incremental_result.beta);
  ok &= require(
      nearly_equal(fast_exact_result.final_objective,
                   fast_exact_objective, 2e-12, 2e-11) &&
          nearly_equal(fast_incremental_result.final_objective,
                       fast_incremental_objective, 2e-12, 2e-11) &&
          nearly_equal(fast_exact_objective, fast_incremental_objective,
                       2e-12, 2e-10) &&
          (fast_exact_probabilities - fast_incremental_probabilities)
                  .cwiseAbs()
                  .maxCoeff() < 1e-8 &&
          has_same_support(fast_exact_result.beta,
                           fast_incremental_result.beta),
      "incremental fast logits must preserve the independently evaluated "
      "objective, probabilities, and support");
  ok &= require(
      fast_exact_result.outer_iterations ==
              fast_incremental_result.outer_iterations &&
          fast_exact_result.total_inner_sweeps ==
              fast_incremental_result.total_inner_sweeps &&
          fast_exact_result.total_coordinate_updates ==
              fast_incremental_result.total_coordinate_updates,
      "incremental fast logits must preserve solver work diagnostics");

  const double cached_independent_kkt = independent_outer_kkt(
      objective, cached_result.beta, cached_result.intercept, lambda, true);
  const double naive_independent_kkt = independent_outer_kkt(
      objective, naive_result.beta, naive_result.intercept, lambda, true);
  const double vectorized_independent_kkt = independent_outer_kkt(
      objective, vectorized_result.beta, vectorized_result.intercept, lambda,
      true);
  const double reused_independent_kkt = independent_outer_kkt(
      objective, reused_result.beta, reused_result.intercept, lambda, true);
  const double production_independent_kkt = independent_outer_kkt(
      objective, production_result.beta, production_result.intercept, lambda,
      true);
  const double fast_exact_independent_kkt = independent_outer_kkt(
      objective, fast_exact_result.beta, fast_exact_result.intercept, lambda,
      true);
  const double fast_incremental_independent_kkt = independent_outer_kkt(
      objective, fast_incremental_result.beta,
      fast_incremental_result.intercept, lambda, true);
  ok &= require(nearly_equal(cached_result.final_kkt_residual,
                             cached_independent_kkt, 2e-12, 2e-10) &&
                    nearly_equal(naive_result.final_kkt_residual,
                                 naive_independent_kkt, 2e-12, 2e-10),
                "both cache modes must pass an independently recomputed KKT");
  ok &= require(
      vectorized_independent_kkt <=
              1.1 * vectorized_options.outer_kkt_tolerance &&
          nearly_equal(vectorized_result.final_kkt_residual,
                       vectorized_independent_kkt, 2e-12, 2e-10),
      "the vectorized coordinate kernel must pass an independently "
      "recomputed KKT");
  ok &= require(
      reused_independent_kkt <= 1.1 * reused_options.outer_kkt_tolerance &&
          nearly_equal(reused_result.final_kkt_residual,
                       reused_independent_kkt, 2e-12, 2e-10),
      "the line-search probability reuse path must pass an independently "
      "recomputed KKT");
  ok &= require(
      production_independent_kkt <=
              1.1 * production_options.outer_kkt_tolerance &&
          nearly_equal(production_result.final_kkt_residual,
                       production_independent_kkt, 2e-12, 2e-10),
      "the combined adaptive/vectorized/reuse L1 path must pass an "
      "independently recomputed KKT");
  ok &= require(
      fast_exact_independent_kkt <=
              1.1 * fast_exact_options.outer_kkt_tolerance &&
          fast_incremental_independent_kkt <=
              1.1 * fast_incremental_options.outer_kkt_tolerance &&
          nearly_equal(fast_incremental_result.final_kkt_residual,
                       fast_incremental_independent_kkt, 2e-12, 2e-10),
      "incremental fast logits must pass an independently recomputed KKT");
  for (std::size_t index = 1;
       index < fast_incremental_result.history.size(); ++index) {
    const double previous =
        fast_incremental_result.history[index - 1].objective;
    const double current = fast_incremental_result.history[index].objective;
    ok &= require(
        current <= previous +
                       32.0 * std::numeric_limits<double>::epsilon() *
                           std::max(1.0, std::fabs(previous)),
        "incremental fast logits must retain monotone Armijo history");
  }

  picasso::solver::MultinomialActNewtonPathSolver fast_path_solver(
      objective, fast_incremental_options);
  picasso::solver::MultinomialActNewtonPathState fast_path_state;
  const double fast_path_lambdas[] = {0.03, 0.022, 0.016};
  for (std::size_t point = 0;
       point < sizeof(fast_path_lambdas) / sizeof(fast_path_lambdas[0]);
       ++point) {
    const double path_lambda = fast_path_lambdas[point];
    const picasso::solver::MultinomialActNewtonPathResult path_point =
        fast_path_solver.solve(path_lambda, &fast_path_state);
    ok &= require(
        path_point.solution.converged() &&
            path_point.reused_initial_smooth_state == (point != 0),
        "fast paths must reuse normalized logits after their first lambda");
    if (!path_point.solution.converged()) continue;
    const double exact_path_objective =
        objective.negative_log_likelihood(path_point.solution.beta,
                                          path_point.solution.intercept) +
        path_lambda * matrix_l1_norm(path_point.solution.beta);
    const double exact_path_kkt = independent_outer_kkt(
        objective, path_point.solution.beta, path_point.solution.intercept,
        path_lambda, true);
    ok &= require(
        nearly_equal(path_point.solution.final_objective,
                     exact_path_objective, 2e-12, 2e-10) &&
            exact_path_kkt <=
                1.1 * fast_incremental_options.outer_kkt_tolerance,
        "cached fast logits must preserve the exact path objective and KKT");
  }

  ok &= require(cached_result.history.size() == naive_result.history.size(),
                "cache on/off must record histories of the same length");
  if (cached_result.history.size() == naive_result.history.size()) {
    for (std::size_t index = 0; index < cached_result.history.size(); ++index) {
      const picasso::solver::MultinomialIterationRecord &cached =
          cached_result.history[index];
      const picasso::solver::MultinomialIterationRecord &naive =
          naive_result.history[index];
      ok &= require(cached.outer_iteration == naive.outer_iteration &&
                        cached.inner_sweeps == naive.inner_sweeps &&
                        cached.line_search_steps == naive.line_search_steps &&
                        cached.inner_converged == naive.inner_converged,
                    "cache on/off history metadata must match");
      ok &= require(
          nearly_equal(cached.objective, naive.objective, 2e-12, 2e-11) &&
              nearly_equal(cached.kkt_residual, naive.kkt_residual, 2e-11,
                           2e-9) &&
              nearly_equal(cached.inner_kkt_residual,
                           naive.inner_kkt_residual, 2e-11, 2e-9) &&
              nearly_equal(cached.step_size, naive.step_size, 0.0, 0.0) &&
              nearly_equal(cached.direction_norm, naive.direction_norm,
                           2e-11, 2e-9) &&
              nearly_equal(cached.composite_directional_derivative,
                           naive.composite_directional_derivative, 2e-11,
                           2e-9),
          "cache on/off history numerics must match");
    }
  }

  const picasso::solver::MultinomialLlaOptions lla_options =
      picasso::solver::MultinomialLlaOptions::fixed_stage_compatibility(3);
  picasso::solver::MultinomialLlaSolver strict_lla_solver(
      objective, cached_options, lla_options);
  picasso::solver::MultinomialLlaSolver production_lla_solver(
      objective, production_options, lla_options);
  const picasso::solver::MultinomialLlaPenalty mcp =
      picasso::solver::MultinomialLlaPenalty::kMCP;
  const double gamma = 3.0;
  const picasso::solver::MultinomialLlaResult strict_mcp =
      strict_lla_solver.solve(mcp, lambda, gamma, initial_beta,
                              initial_intercept);
  const picasso::solver::MultinomialLlaResult production_mcp =
      production_lla_solver.solve(mcp, lambda, gamma, initial_beta,
                                  initial_intercept);
  ok &= require(strict_mcp.completed() && production_mcp.completed() &&
                    strict_mcp.stages.size() == 3 &&
                    production_mcp.stages.size() == 3,
                "strict and combined production-kernel MCP LLA solves must "
                "complete the same three validated stages");
  if (strict_mcp.completed() && production_mcp.completed()) {
    Eigen::MatrixXd strict_mcp_probabilities;
    Eigen::MatrixXd production_mcp_probabilities;
    objective.negative_log_likelihood(strict_mcp.beta, strict_mcp.intercept,
                                      &strict_mcp_probabilities);
    objective.negative_log_likelihood(
        production_mcp.beta, production_mcp.intercept,
        &production_mcp_probabilities);
    const double strict_mcp_objective = independent_nonconvex_objective(
        objective, strict_mcp.beta, strict_mcp.intercept, mcp, lambda, gamma);
    const double production_mcp_objective = independent_nonconvex_objective(
        objective, production_mcp.beta, production_mcp.intercept, mcp, lambda,
        gamma);
    const double strict_mcp_stationarity =
        independent_nonconvex_stationarity(
            objective, strict_mcp.beta, strict_mcp.intercept, mcp, lambda,
            gamma, true);
    const double production_mcp_stationarity =
        independent_nonconvex_stationarity(
            objective, production_mcp.beta, production_mcp.intercept, mcp,
            lambda, gamma, true);
    ok &= require(
        nearly_equal(strict_mcp.final_target_objective,
                     strict_mcp_objective, 2e-12, 2e-11) &&
            nearly_equal(production_mcp.final_target_objective,
                         production_mcp_objective, 2e-12, 2e-11) &&
            nearly_equal(production_mcp.final_target_objective,
                         strict_mcp.final_target_objective, 5e-9, 5e-7) &&
            (production_mcp_probabilities - strict_mcp_probabilities)
                    .cwiseAbs()
                    .maxCoeff() < 1e-5 &&
            has_same_support(production_mcp.beta, strict_mcp.beta),
        "the combined adaptive/vectorized/reuse MCP path must preserve the "
        "independent target objective, probabilities, and support");
    ok &= require(
        nearly_equal(strict_mcp.final_target_stationarity,
                     strict_mcp_stationarity, 2e-12, 2e-10) &&
            nearly_equal(production_mcp.final_target_stationarity,
                         production_mcp_stationarity, 2e-12, 2e-10) &&
            strict_mcp.stages.back().subproblem_kkt_residual <=
                1.1 * cached_options.outer_kkt_tolerance &&
            production_mcp.stages.back().subproblem_kkt_residual <=
                1.1 * production_options.outer_kkt_tolerance,
        "strict and combined production-kernel MCP paths must preserve "
        "independent target stationarity and final surrogate KKT "
        "certification");
  }
  return ok;
}

bool test_exact_feature_working_set_ab_equivalence() {
  const int n = 240;
  const int d = 96;
  const int num_classes = 4;
  const double pi = 3.14159265358979323846;
  Eigen::MatrixXd x(n, d);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      x(i, j) = std::sqrt(2.0) *
                std::cos(pi * (static_cast<double>(i) + 0.5) *
                         static_cast<double>(j + 1) / static_cast<double>(n));
    }
  }
  Eigen::MatrixXd true_beta = Eigen::MatrixXd::Zero(d, num_classes);
  true_beta(0, 0) = 1.1;
  true_beta(0, 1) = -0.6;
  true_beta(1, 1) = 0.9;
  true_beta(1, 2) = -0.7;
  true_beta(2, 2) = 0.8;
  true_beta(3, 3) = -0.9;
  Eigen::VectorXd true_intercept(num_classes);
  true_intercept << 0.35, 0.05, -0.10, -0.30;
  const Eigen::MatrixXd logits =
      x * true_beta + true_intercept.transpose().replicate(n, 1);
  Eigen::VectorXi labels(n);
  for (int i = 0; i < n; ++i) {
    const double maximum = logits.row(i).maxCoeff();
    Eigen::VectorXd weights(num_classes);
    double weight_sum = 0.0;
    for (int klass = 0; klass < num_classes; ++klass) {
      weights[klass] = std::exp(logits(i, klass) - maximum);
      weight_sum += weights[klass];
    }
    const double unit_draw =
        std::fmod(0.6180339887498949 * static_cast<double>(i + 1), 1.0);
    const double draw = unit_draw * weight_sum;
    double cumulative = 0.0;
    labels[i] = num_classes - 1;
    for (int klass = 0; klass < num_classes; ++klass) {
      cumulative += weights[klass];
      if (draw <= cumulative) {
        labels[i] = klass;
        break;
      }
    }
  }

  picasso::MultinomialObjective objective(x, labels, num_classes);
  Eigen::VectorXd null_intercept = Eigen::VectorXd::Zero(num_classes);
  for (int i = 0; i < n; ++i) null_intercept[labels[i]] += 1.0;
  null_intercept.array() /= static_cast<double>(n);
  null_intercept = null_intercept.array().log().matrix();
  null_intercept.array() -= null_intercept.mean();
  Eigen::MatrixXd null_gradient;
  Eigen::VectorXd ignored_intercept_gradient;
  objective.smooth_gradient(Eigen::MatrixXd::Zero(d, num_classes),
                            null_intercept, &null_gradient,
                            &ignored_intercept_gradient);
  const double lambda = 0.55 * null_gradient.cwiseAbs().maxCoeff();

  picasso::solver::MultinomialActNewtonOptions full_options;
  bool ok = true;
  ok &= require(full_options.use_active_set,
                "the exact feature working set must be enabled by default");
  full_options.max_outer_iterations = 100;
  full_options.max_inner_sweeps = 4000;
  full_options.outer_kkt_tolerance = 5e-7;
  full_options.inner_kkt_tolerance = 1e-9;
  full_options.use_active_set = false;
  full_options.exact_kkt_scan_interval = 1;
  picasso::solver::MultinomialActNewtonOptions working_options =
      full_options;
  working_options.use_active_set = true;
  working_options.exact_kkt_scan_interval = 1;
  picasso::solver::MultinomialActNewtonOptions working_no_cache_options =
      working_options;
  working_no_cache_options.use_probability_dot_direction_cache = false;
  picasso::solver::MultinomialActNewtonOptions interval_options =
      working_options;
  interval_options.exact_kkt_scan_interval = 4;

  picasso::solver::MultinomialActNewtonSolver full_solver(objective,
                                                          full_options);
  picasso::solver::MultinomialActNewtonSolver working_solver(
      objective, working_options);
  picasso::solver::MultinomialActNewtonSolver working_no_cache_solver(
      objective, working_no_cache_options);
  picasso::solver::MultinomialActNewtonSolver interval_solver(
      objective, interval_options);
  const picasso::solver::MultinomialActNewtonResult full_result =
      full_solver.solve(lambda);
  const picasso::solver::MultinomialActNewtonResult working_result =
      working_solver.solve(lambda);
  const picasso::solver::MultinomialActNewtonResult working_no_cache_result =
      working_no_cache_solver.solve(lambda);
  const picasso::solver::MultinomialActNewtonResult interval_result =
      interval_solver.solve(lambda);

  ok &= require(full_result.converged() && working_result.converged() &&
                    working_no_cache_result.converged() &&
                    interval_result.converged(),
                "full and exact working-set solves must all converge");
  Eigen::MatrixXd full_probabilities;
  Eigen::MatrixXd working_probabilities;
  Eigen::MatrixXd working_no_cache_probabilities;
  Eigen::MatrixXd interval_probabilities;
  objective.negative_log_likelihood(full_result.beta,
                                    full_result.intercept,
                                    &full_probabilities);
  objective.negative_log_likelihood(working_result.beta,
                                    working_result.intercept,
                                    &working_probabilities);
  objective.negative_log_likelihood(
      working_no_cache_result.beta, working_no_cache_result.intercept,
      &working_no_cache_probabilities);
  objective.negative_log_likelihood(interval_result.beta,
                                    interval_result.intercept,
                                    &interval_probabilities);
  ok &= require(nearly_equal(full_result.final_objective,
                             working_result.final_objective, 1e-9, 2e-8) &&
                    (full_probabilities - working_probabilities)
                            .cwiseAbs()
                            .maxCoeff() < 2e-6,
                "working-set on/off must reach the same objective and "
                "probabilities");
  ok &= require(nearly_equal(working_result.final_objective,
                             working_no_cache_result.final_objective, 1e-10,
                             2e-9) &&
                    (working_probabilities - working_no_cache_probabilities)
                            .cwiseAbs()
                            .maxCoeff() < 2e-7,
                "working-set and probability-cache switches must compose "
                "without changing the fit");
  ok &= require(
      interval_result.status == working_result.status &&
          nearly_equal(interval_result.final_objective,
                       working_result.final_objective, 1e-9, 2e-8) &&
          (interval_probabilities - working_probabilities)
                  .cwiseAbs()
                  .maxCoeff() < 2e-6 &&
          has_same_support(interval_result.beta, working_result.beta),
      "periodic exact KKT scans must preserve sparse status, objective, "
      "probabilities, and support");
  const double working_kkt = independent_outer_kkt(
      objective, working_result.beta, working_result.intercept, lambda, true);
  ok &= require(working_kkt <= 1.1 * working_options.outer_kkt_tolerance &&
                    std::fabs(working_kkt -
                              working_result.final_kkt_residual) < 1e-12,
                "working-set convergence must be certified by a full "
                "independent outer KKT scan");
  const double interval_kkt = independent_outer_kkt(
      objective, interval_result.beta, interval_result.intercept, lambda,
      true);
  ok &= require(
      interval_kkt <= 1.1 * interval_options.outer_kkt_tolerance &&
          std::fabs(interval_kkt - interval_result.final_kkt_residual) <
              1e-12,
      "periodic scan convergence must retain exact independent outer KKT "
      "certification");
  ok &= require(working_result.final_active_features < d &&
                    working_result.total_coordinate_updates <
                        full_result.total_coordinate_updates,
                "the sparse fixture must screen features and reduce "
                "coordinate visits");
  ok &= require(working_result.total_full_subproblem_kkt_scans > 0,
                "the exact working set must certify restricted quadratic "
                "solutions with full KKT scans");
  for (std::size_t index = 1; index < working_result.history.size(); ++index) {
    ok &= require(working_result.history[index].active_features <= d,
                  "working-set diagnostics must report a valid feature "
                  "count");
  }

  // glmnet-style two-tier working set: deliberately pass every feature as a
  // strong candidate, then verify that partial quadratic sweeps can use a
  // coefficient-resolution compact list without replacing or shrinking the
  // authoritative outer feature mask.
  const std::vector<unsigned char> oversized_strong_set(
      static_cast<std::size_t>(d), 1);
  picasso::solver::MultinomialActNewtonOptions legacy_restricted_options =
      working_options;
  legacy_restricted_options.use_compact_inner_active_set = false;
  picasso::solver::MultinomialActNewtonOptions compact_restricted_options =
      legacy_restricted_options;
  compact_restricted_options.use_compact_inner_active_set = true;
  picasso::solver::MultinomialActNewtonSolver legacy_restricted_solver(
      objective, legacy_restricted_options);
  picasso::solver::MultinomialActNewtonSolver compact_restricted_solver(
      objective, compact_restricted_options);
  const Eigen::MatrixXd zero_beta =
      Eigen::MatrixXd::Zero(d, num_classes);
  const picasso::solver::MultinomialActNewtonResult legacy_restricted_result =
      legacy_restricted_solver.solve(lambda, zero_beta, null_intercept,
                                     oversized_strong_set);
  const picasso::solver::MultinomialActNewtonResult compact_restricted_result =
      compact_restricted_solver.solve(lambda, zero_beta, null_intercept,
                                      oversized_strong_set);
  Eigen::MatrixXd legacy_restricted_probabilities;
  Eigen::MatrixXd compact_restricted_probabilities;
  objective.negative_log_likelihood(
      legacy_restricted_result.beta, legacy_restricted_result.intercept,
      &legacy_restricted_probabilities);
  objective.negative_log_likelihood(
      compact_restricted_result.beta, compact_restricted_result.intercept,
      &compact_restricted_probabilities);
  const long long legacy_full_mask_visits =
      static_cast<long long>(legacy_restricted_result.total_inner_sweeps) *
      static_cast<long long>(num_classes) *
      static_cast<long long>(d + 1);
  bool retained_zero_candidate = false;
  for (int feature = 0; feature < d; ++feature) {
    if (compact_restricted_result.active_features[
            static_cast<std::size_t>(feature)] != 0 &&
        compact_restricted_result.beta.row(feature).cwiseAbs().maxCoeff() <=
            compact_restricted_options.zero_tolerance) {
      retained_zero_candidate = true;
      break;
    }
  }
  const double compact_restricted_kkt = independent_outer_kkt(
      objective, compact_restricted_result.beta,
      compact_restricted_result.intercept, lambda, true);
  ok &= require(
      legacy_restricted_result.converged() &&
          compact_restricted_result.converged() &&
          nearly_equal(compact_restricted_result.final_objective,
                       legacy_restricted_result.final_objective, 1e-9,
                       2e-8) &&
          (compact_restricted_probabilities -
           legacy_restricted_probabilities)
                  .cwiseAbs()
                  .maxCoeff() < 2e-6 &&
          has_same_support(compact_restricted_result.beta,
                           legacy_restricted_result.beta) &&
          compact_restricted_kkt <=
              1.1 * compact_restricted_options.outer_kkt_tolerance,
      "compact restricted sweeps must preserve objective, probabilities, "
      "support, and independent full-model KKT");
  ok &= require(
      legacy_restricted_result.initial_active_features == d &&
          legacy_restricted_result.final_active_features == d &&
          compact_restricted_result.initial_active_features == d &&
          compact_restricted_result.final_active_features == d &&
          retained_zero_candidate,
      "the compact inner list must not replace or shrink the outer strong "
      "candidate mask");
  ok &= require(
      legacy_restricted_result.total_coordinate_updates ==
              legacy_full_mask_visits &&
          compact_restricted_result.total_coordinate_updates <
              legacy_restricted_result.total_coordinate_updates,
      "compact partial sweeps must reduce coordinate visits relative to the "
      "legacy full-mask restricted solve");

  // If at least 75% of candidate coordinates are already nonzero, compact
  // storage must immediately fall back to the legacy full sweep.  Starting
  // from a dense, noncanonicalized coefficient matrix makes this transition
  // deterministic and lets every integer and floating diagnostic be compared.
  Eigen::MatrixXd dense_initial_beta(d, num_classes);
  for (int feature = 0; feature < d; ++feature) {
    for (int klass = 0; klass < num_classes; ++klass) {
      dense_initial_beta(feature, klass) =
          0.003 * static_cast<double>(1 + feature * num_classes + klass);
    }
  }
  picasso::solver::MultinomialActNewtonOptions dense_legacy_options =
      legacy_restricted_options;
  dense_legacy_options.max_outer_iterations = 1;
  dense_legacy_options.canonicalize_feature_l1_gauge = false;
  picasso::solver::MultinomialActNewtonOptions dense_compact_options =
      dense_legacy_options;
  dense_compact_options.use_compact_inner_active_set = true;
  picasso::solver::MultinomialActNewtonSolver dense_legacy_solver(
      objective, dense_legacy_options);
  picasso::solver::MultinomialActNewtonSolver dense_compact_solver(
      objective, dense_compact_options);
  const picasso::solver::MultinomialActNewtonResult dense_legacy_result =
      dense_legacy_solver.solve(lambda, dense_initial_beta, null_intercept,
                                oversized_strong_set);
  const picasso::solver::MultinomialActNewtonResult dense_compact_result =
      dense_compact_solver.solve(lambda, dense_initial_beta, null_intercept,
                                 oversized_strong_set);
  ok &= require(
      dense_legacy_result.status ==
              picasso::solver::MultinomialSolverStatus::kOuterIterationLimit &&
          dense_compact_result.status ==
              picasso::solver::MultinomialSolverStatus::kOuterIterationLimit,
      "one-step dense compact-fallback A/B solves must reach the same outer "
      "iteration limit");
  ok &= equivalent_result_fields(
      dense_legacy_result, dense_compact_result,
      "dense 75-percent compact fallback");

  // The shape-adaptive schedule deliberately keeps feature-resolution
  // partial sweeps for small, low-class problems.  Exercise that branch with
  // a genuinely restricted solve (the ordinary solve may reactivate features
  // and therefore disables the compact inner tier).
  const int feature_compact_d = 24;
  const Eigen::MatrixXd feature_compact_x = x.leftCols(feature_compact_d);
  picasso::MultinomialObjective feature_compact_objective(
      feature_compact_x, labels, num_classes);
  Eigen::MatrixXd feature_compact_null_gradient;
  feature_compact_objective.smooth_gradient(
      Eigen::MatrixXd::Zero(feature_compact_d, num_classes), null_intercept,
      &feature_compact_null_gradient, &ignored_intercept_gradient);
  const double feature_compact_lambda =
      0.55 * feature_compact_null_gradient.cwiseAbs().maxCoeff();
  const std::vector<unsigned char> feature_compact_strong_set(
      static_cast<std::size_t>(feature_compact_d), 1);
  picasso::solver::MultinomialActNewtonOptions feature_legacy_options =
      legacy_restricted_options;
  feature_legacy_options.outer_kkt_tolerance = 1e-4;
  picasso::solver::MultinomialActNewtonOptions feature_compact_options =
      feature_legacy_options;
  feature_compact_options.use_compact_inner_active_set = true;
  picasso::solver::MultinomialActNewtonSolver feature_legacy_solver(
      feature_compact_objective, feature_legacy_options);
  picasso::solver::MultinomialActNewtonSolver feature_compact_solver(
      feature_compact_objective, feature_compact_options);
  const picasso::solver::MultinomialActNewtonResult feature_legacy_result =
      feature_legacy_solver.solve(
          feature_compact_lambda,
          Eigen::MatrixXd::Zero(feature_compact_d, num_classes),
          null_intercept, feature_compact_strong_set);
  const picasso::solver::MultinomialActNewtonResult feature_compact_result =
      feature_compact_solver.solve(
          feature_compact_lambda,
          Eigen::MatrixXd::Zero(feature_compact_d, num_classes),
          null_intercept, feature_compact_strong_set);
  Eigen::MatrixXd feature_legacy_probabilities;
  Eigen::MatrixXd feature_compact_probabilities;
  feature_compact_objective.negative_log_likelihood(
      feature_legacy_result.beta, feature_legacy_result.intercept,
      &feature_legacy_probabilities);
  feature_compact_objective.negative_log_likelihood(
      feature_compact_result.beta, feature_compact_result.intercept,
      &feature_compact_probabilities);
  const double feature_compact_kkt = independent_outer_kkt(
      feature_compact_objective, feature_compact_result.beta,
      feature_compact_result.intercept, feature_compact_lambda, true);
  ok &= require(
      feature_legacy_result.converged() && feature_compact_result.converged() &&
          nearly_equal(feature_compact_result.final_objective,
                       feature_legacy_result.final_objective, 1e-9, 2e-8) &&
          (feature_compact_probabilities - feature_legacy_probabilities)
                  .cwiseAbs()
                  .maxCoeff() < 2e-6 &&
          has_same_support(feature_compact_result.beta,
                           feature_legacy_result.beta) &&
          feature_compact_kkt <=
              1.1 * feature_compact_options.outer_kkt_tolerance &&
          feature_compact_result.total_coordinate_updates <
              feature_legacy_result.total_coordinate_updates,
      "small-problem feature-resolution compact sweeps must reduce work "
      "without changing the certified solution");

  Eigen::MatrixXd feature_dense_initial_beta(feature_compact_d, num_classes);
  for (int feature = 0; feature < feature_compact_d; ++feature) {
    for (int klass = 0; klass < num_classes; ++klass) {
      feature_dense_initial_beta(feature, klass) =
          0.003 * static_cast<double>(1 + feature * num_classes + klass);
    }
  }
  picasso::solver::MultinomialActNewtonOptions feature_dense_legacy_options =
      feature_legacy_options;
  feature_dense_legacy_options.max_outer_iterations = 1;
  feature_dense_legacy_options.canonicalize_feature_l1_gauge = false;
  picasso::solver::MultinomialActNewtonOptions feature_dense_compact_options =
      feature_dense_legacy_options;
  feature_dense_compact_options.use_compact_inner_active_set = true;
  picasso::solver::MultinomialActNewtonSolver feature_dense_legacy_solver(
      feature_compact_objective, feature_dense_legacy_options);
  picasso::solver::MultinomialActNewtonSolver feature_dense_compact_solver(
      feature_compact_objective, feature_dense_compact_options);
  const picasso::solver::MultinomialActNewtonResult
      feature_dense_legacy_result = feature_dense_legacy_solver.solve(
          feature_compact_lambda, feature_dense_initial_beta, null_intercept,
          feature_compact_strong_set);
  const picasso::solver::MultinomialActNewtonResult
      feature_dense_compact_result = feature_dense_compact_solver.solve(
          feature_compact_lambda, feature_dense_initial_beta, null_intercept,
          feature_compact_strong_set);
  ok &= require(
      feature_dense_legacy_result.status ==
              picasso::solver::MultinomialSolverStatus::kOuterIterationLimit &&
          feature_dense_compact_result.status ==
              picasso::solver::MultinomialSolverStatus::kOuterIterationLimit,
      "small-problem dense compact-fallback A/B solves must reach the same "
      "outer iteration limit");
  ok &= equivalent_result_fields(
      feature_dense_legacy_result, feature_dense_compact_result,
      "feature-resolution dense 75-percent compact fallback");
  return ok;
}

bool test_working_set_full_quadratic_kkt_reactivation() {
  Eigen::MatrixXd x(30, 5);
  x <<
      -0.9779232086869919, -0.025656837667261866, -1.2134871936234404,
      -0.87158512395346777, -0.18491072199097425,
      0.099932649793540787, -0.95899034777648851, 0.98478946878299101,
      -0.16228124774913436, -0.74663870591306281,
      -2.1472517733243168, 0.26754551406579757, -1.5749709828740606,
      -1.6255492898683404, 0.34268356399089822,
      0.67458568120315265, -0.13421817124380464, 0.70790103151892825,
      0.50668875889445864, -0.84292526238808962,
      0.88427517409325695, 1.5994371535193648, 0.57793605736241449,
      0.61688138158083461, 1.7193562549842865,
      1.3257743109841609, 0.95681645247541824, 0.59792764027580447,
      1.5592869869331392, 1.2545148556301768,
      0.031475954477511066, -2.6544280421292394, 1.0084591493440784,
      -0.26104880949053594, -1.9267645756970853,
      0.25537624823712485, 0.40013154396234346, -0.6700169355919583,
      -0.076085316216220653, 0.68351661050348267,
      0.17198058117152354, -0.02051345888087917, -0.0073002007453751589,
      0.066355830984144118, -0.11295043741272397,
      -0.19715286699964443, -0.8426266811774904, -0.57275794297151816,
      -0.0032119516751666255, -1.1706885114968406,
      1.2977555115701771, 1.0407589939834399, 0.56530654149093507,
      1.5952796946980614, 1.193013644966376,
      0.46295554662878308, 0.0081786962270255399, 0.95246308909311173,
      0.27129929166788069, -0.1659427796693296,
      -1.2334254234916699, -0.67001663883013884, -1.4783438172758068,
      -1.4285925882713379, -0.15921297318388408,
      -0.53303118386548987, 1.1343551569281038, -0.93980519688413244,
      -0.27107340375082595, 1.0091523528745421,
      -0.95282181161021895, -0.83615733449825691, -0.15768120933087482,
      -0.70610349339168166, -0.93957880855827369,
      0.15462304582711675, -0.36541631135674763, 0.45008288810128166,
      -0.047859231861231699, 0.049579678659674574,
      -2.3497655130237329, 0.3259332612679558, -1.3419157995548934,
      -2.1370104382385331, 0.46115712451919277,
      -0.034748091854872439, 1.2934612815177, -0.53536624726620907,
      -0.25868088077962098, 1.4051489353225382,
      -1.3930401284112002, 1.0053601210335401, -1.6730038130082245,
      -0.89691884993139326, 1.3260988229082722,
      1.1015041189089199, -0.81295102451071732, 2.0531449032874551,
      1.528468977154481, -1.1298996288658298,
      0.053327274701232046, -0.45032093636007131, 0.22868046891903793,
      -0.064268225150937316, -0.87658742429084147,
      1.6944919213534062, -1.8391902901151558, 1.5020368199715666,
      1.9309071029141702, -1.8767329389259177,
      1.032771582024012, 0.20345043440892435, 0.46081371591596371,
      0.78236698020965045, 0.29094362355082054,
      -0.96813486828899853, 0.018142048926980313, -0.81284695163352227,
      -1.3107772074515369, -0.053750933383294446,
      -0.37328209933025641, 1.6349398492487786, -1.2027765315409131,
      -0.65460459634530288, 1.4770676290517899,
      0.18393127838649037, -0.32044372557238054, -0.53421159932960516,
      0.19510075503573149, -0.59533911004408802,
      0.65709785567860635, 1.4132872110096497, 0.71286811551080409,
      0.89398102248043443, 1.137571739602816,
      1.2273321549309657, -1.4469115087056343, 1.4795906443496092,
      1.3297435088197456, -1.2255156793924697,
      -0.77468265666189751, 0.019946178039500256, -0.40075999052529632,
      -1.0231153359291976, 0.070238462278408825,
      0.62606873557931064, 0.056097412209744651, 0.83324387823184876,
      0.5224056986817327, -0.41260480763057206;
  Eigen::VectorXi labels(30);
  labels << 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 2, 0, 1, 2, 1,
      0, 1, 2, 2, 0, 2, 2, 0, 1, 2, 1, 2, 1, 2, 1;
  const int num_classes = 3;
  picasso::MultinomialObjective objective(x, labels, num_classes);
  Eigen::VectorXd null_intercept = Eigen::VectorXd::Zero(num_classes);
  for (int i = 0; i < labels.size(); ++i)
    null_intercept[labels[i]] += 1.0;
  null_intercept.array() /= static_cast<double>(labels.size());
  null_intercept = null_intercept.array().log().matrix();
  null_intercept.array() -= null_intercept.mean();
  Eigen::MatrixXd null_gradient;
  Eigen::VectorXd ignored_intercept_gradient;
  objective.smooth_gradient(Eigen::MatrixXd::Zero(5, num_classes),
                            null_intercept, &null_gradient,
                            &ignored_intercept_gradient);
  const double lambda = 0.59 * null_gradient.cwiseAbs().maxCoeff();

  picasso::solver::MultinomialActNewtonOptions full_options;
  full_options.max_inner_sweeps = 4000;
  full_options.outer_kkt_tolerance = 5e-7;
  full_options.inner_kkt_tolerance = 1e-9;
  full_options.use_active_set = false;
  full_options.exact_kkt_scan_interval = 1;
  picasso::solver::MultinomialActNewtonOptions working_options =
      full_options;
  working_options.use_active_set = true;
  working_options.exact_kkt_scan_interval = 1;
  picasso::solver::MultinomialActNewtonOptions interval_options =
      working_options;
  interval_options.exact_kkt_scan_interval = 4;
  picasso::solver::MultinomialActNewtonSolver full_solver(objective,
                                                          full_options);
  picasso::solver::MultinomialActNewtonSolver working_solver(
      objective, working_options);
  picasso::solver::MultinomialActNewtonSolver interval_solver(
      objective, interval_options);
  const picasso::solver::MultinomialActNewtonResult full_result =
      full_solver.solve(lambda);
  const picasso::solver::MultinomialActNewtonResult working_result =
      working_solver.solve(lambda);
  const picasso::solver::MultinomialActNewtonResult interval_result =
      interval_solver.solve(lambda);

  bool ok = true;
  ok &= require(full_result.converged() && working_result.converged() &&
                    interval_result.converged(),
                "reactivation fixture must converge with working set on/off");
  ok &= require(working_result.total_subproblem_reactivated_features > 0,
                "a full quadratic KKT scan must reactivate a feature that "
                "the initial screen missed");
  ok &= require(working_result.total_outer_reactivated_features == 0,
                "the quadratic repair fixture must reactivate before the "
                "outer-model KKT fallback");
  ok &= require(interval_result.total_subproblem_reactivated_features > 0 &&
                    interval_result.total_outer_reactivated_features == 0,
                "periodic scans must still perform exact in-subproblem "
                "inactive-feature reactivation");
  Eigen::MatrixXd full_probabilities;
  Eigen::MatrixXd working_probabilities;
  Eigen::MatrixXd interval_probabilities;
  objective.negative_log_likelihood(full_result.beta,
                                    full_result.intercept,
                                    &full_probabilities);
  objective.negative_log_likelihood(working_result.beta,
                                    working_result.intercept,
                                    &working_probabilities);
  objective.negative_log_likelihood(interval_result.beta,
                                    interval_result.intercept,
                                    &interval_probabilities);
  ok &= require(nearly_equal(full_result.final_objective,
                             working_result.final_objective, 1e-9, 2e-8) &&
                    (full_probabilities - working_probabilities)
                            .cwiseAbs()
                            .maxCoeff() < 2e-6,
                "quadratic reactivation must recover the full-solve fit");
  ok &= require(nearly_equal(full_result.final_objective,
                             interval_result.final_objective, 1e-9, 2e-8) &&
                    (full_probabilities - interval_probabilities)
                            .cwiseAbs()
                            .maxCoeff() < 2e-6 &&
                    has_same_support(full_result.beta, interval_result.beta),
                "periodic-scan reactivation must recover the full-solve "
                "objective, probabilities, and support");
  const double working_kkt = independent_outer_kkt(
      objective, working_result.beta, working_result.intercept, lambda, true);
  ok &= require(working_kkt <= 1.1 * working_options.outer_kkt_tolerance,
                "reactivated working-set fit must pass independent full KKT");
  const double interval_kkt = independent_outer_kkt(
      objective, interval_result.beta, interval_result.intercept, lambda,
      true);
  ok &= require(interval_kkt <= 1.1 * interval_options.outer_kkt_tolerance,
                "periodic-scan reactivated fit must pass independent full "
                "KKT");
  return ok;
}

bool check_entry_feature_gauge(const Eigen::MatrixXd &initial_beta,
                               const Eigen::MatrixXd &expected_beta,
                               const std::string &case_name) {
  const int num_classes = static_cast<int>(initial_beta.cols());
  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(2 * num_classes,
                                             initial_beta.rows());
  Eigen::VectorXi labels(2 * num_classes);
  for (int i = 0; i < labels.size(); ++i) labels[i] = i % num_classes;
  picasso::MultinomialObjective objective(x, labels, num_classes);
  Eigen::VectorXd initial_intercept = Eigen::VectorXd::Constant(
      num_classes, 17.0);

  picasso::solver::MultinomialActNewtonOptions gauge_options;
  bool ok = true;
  ok &= require(gauge_options.canonicalize_feature_l1_gauge,
                "feature L1 gauge canonicalization must be on by default");
  gauge_options.include_intercept = false;
  picasso::solver::MultinomialActNewtonSolver gauge_solver(
      objective, gauge_options);
  const picasso::solver::MultinomialActNewtonResult gauge_result =
      gauge_solver.solve(0.0, initial_beta, initial_intercept);
  ok &= require(gauge_result.converged(),
                case_name + " entry gauge case must converge immediately");
  ok &= require(gauge_result.outer_iterations == 0,
                case_name + " gauge must be applied before the first KKT");
  ok &= require((gauge_result.beta - expected_beta).cwiseAbs().maxCoeff() ==
                    0.0,
                case_name + " must subtract the symmetric median center");
  ok &= require(has_canonical_feature_gauge(gauge_result.beta, 0.0),
                case_name + " must center its median interval at zero");
  ok &= require(matrix_l1_norm(gauge_result.beta) <=
                    matrix_l1_norm(initial_beta),
                case_name + " canonicalization must not increase L1");

  picasso::solver::MultinomialActNewtonOptions legacy_options =
      gauge_options;
  legacy_options.canonicalize_feature_l1_gauge = false;
  picasso::solver::MultinomialActNewtonSolver legacy_solver(
      objective, legacy_options);
  const picasso::solver::MultinomialActNewtonResult legacy_result =
      legacy_solver.solve(0.0, initial_beta, initial_intercept);
  ok &= require(legacy_result.converged() &&
                    (legacy_result.beta - initial_beta)
                            .cwiseAbs()
                            .maxCoeff() == 0.0,
                case_name + " gauge=false must retain the entry A/B baseline");
  return ok;
}

bool test_feature_l1_gauge_odd_even_and_solver_ab() {
  Eigen::MatrixXd odd_initial(2, 3);
  odd_initial << 4.0, -1.0, 2.0,
                -5.0, -5.0, 8.0;
  Eigen::MatrixXd odd_expected(2, 3);
  odd_expected << 2.0, -3.0, 0.0,
                  0.0, 0.0, 13.0;
  bool ok = check_entry_feature_gauge(odd_initial, odd_expected, "odd-K");

  Eigen::MatrixXd even_initial(2, 4);
  even_initial << 5.0, -2.0, 1.0, 9.0,
                  7.0, 7.0, -3.0, 7.0;
  Eigen::MatrixXd even_expected(2, 4);
  even_expected << 2.0, -5.0, -2.0, 6.0,
                   0.0, 0.0, -10.0, 0.0;
  ok &= check_entry_feature_gauge(even_initial, even_expected, "even-K");
  ok &= check_entry_feature_gauge(-even_initial, -even_expected,
                                  "even-K sign equivariance");

  Eigen::MatrixXd even_permuted = even_initial;
  Eigen::MatrixXd even_expected_permuted = even_expected;
  even_permuted.col(0).swap(even_permuted.col(3));
  even_permuted.col(1).swap(even_permuted.col(2));
  even_expected_permuted.col(0).swap(even_expected_permuted.col(3));
  even_expected_permuted.col(1).swap(even_expected_permuted.col(2));
  ok &= check_entry_feature_gauge(even_permuted, even_expected_permuted,
                                  "even-K class permutation");
  ok &= check_entry_feature_gauge(even_expected, even_expected,
                                  "even-K idempotence");

  const int n = 84;
  const int d = 4;
  const int num_classes = 4;
  Eigen::MatrixXd x(n, d);
  Eigen::VectorXi labels(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j)
      x(i, j) = std::sin(0.037 * static_cast<double>((i + 2) * (j + 1))) +
                  0.4 * std::cos(0.091 * static_cast<double>(i + 3 + j));
    labels[i] = (i * 5 + i / 7) % num_classes;
  }
  picasso::MultinomialObjective objective(x, labels, num_classes);
  Eigen::MatrixXd base_beta(d, num_classes);
  for (int j = 0; j < d; ++j) {
    for (int klass = 0; klass < num_classes; ++klass)
      base_beta(j, klass) =
          0.03 * std::sin(static_cast<double>((j + 1) * (klass + 2)));
  }
  Eigen::VectorXd feature_shift(d);
  feature_shift << 12.0, -8.0, 0.75, 4.5;
  Eigen::MatrixXd shifted_beta = base_beta;
  for (int j = 0; j < d; ++j)
    shifted_beta.row(j).array() += feature_shift[j];
  Eigen::VectorXd initial_intercept(num_classes);
  initial_intercept << 3.1, 2.9, 3.05, 2.95;

  Eigen::MatrixXd base_probabilities;
  Eigen::MatrixXd shifted_probabilities;
  const double base_loss = objective.negative_log_likelihood(
      base_beta, initial_intercept, &base_probabilities);
  const double shifted_loss = objective.negative_log_likelihood(
      shifted_beta, initial_intercept, &shifted_probabilities);
  ok &= require(nearly_equal(base_loss, shifted_loss, 2e-14, 2e-14) &&
                    (base_probabilities - shifted_probabilities)
                            .cwiseAbs()
                            .maxCoeff() < 2e-14,
                "a common feature shift must leave loss and probabilities "
                "unchanged");

  picasso::solver::MultinomialActNewtonOptions gauge_options;
  gauge_options.max_outer_iterations = 100;
  gauge_options.max_inner_sweeps = 4000;
  gauge_options.outer_kkt_tolerance = 5e-7;
  gauge_options.inner_kkt_tolerance = 1e-9;
  picasso::solver::MultinomialActNewtonOptions legacy_options = gauge_options;
  legacy_options.canonicalize_feature_l1_gauge = false;
  const double lambda = 0.018;
  picasso::solver::MultinomialActNewtonSolver gauge_solver(
      objective, gauge_options);
  picasso::solver::MultinomialActNewtonSolver legacy_solver(
      objective, legacy_options);
  const picasso::solver::MultinomialActNewtonResult gauge_result =
      gauge_solver.solve(lambda, shifted_beta, initial_intercept);
  const picasso::solver::MultinomialActNewtonResult base_gauge_result =
      gauge_solver.solve(lambda, base_beta, initial_intercept);
  const picasso::solver::MultinomialActNewtonResult legacy_result =
      legacy_solver.solve(lambda, shifted_beta, initial_intercept);
  ok &= require(gauge_result.converged() && base_gauge_result.converged() &&
                    legacy_result.converged(),
                "feature-gauge on/off full solves must both converge");
  ok &= require((gauge_result.beta - base_gauge_result.beta)
                            .cwiseAbs()
                            .maxCoeff() < 2e-9 &&
                    (gauge_result.intercept - base_gauge_result.intercept)
                            .cwiseAbs()
                            .maxCoeff() < 2e-9,
                "warm starts differing only by feature gauge must reach the "
                "same canonical state");
  ok &= require(gauge_result.status == legacy_result.status,
                "feature-gauge on/off must return the same final status");
  ok &= require(has_canonical_feature_gauge(gauge_result.beta),
                "the final default solve must retain the feature L1 gauge");
  ok &= require(gauge_result.history.front().objective <=
                    legacy_result.history.front().objective,
                "entry canonicalization must not increase the recorded "
                "objective");

  Eigen::MatrixXd gauge_probabilities;
  Eigen::MatrixXd legacy_probabilities;
  objective.negative_log_likelihood(gauge_result.beta, gauge_result.intercept,
                                    &gauge_probabilities);
  objective.negative_log_likelihood(legacy_result.beta,
                                    legacy_result.intercept,
                                    &legacy_probabilities);
  ok &= require(nearly_equal(gauge_result.final_objective,
                             legacy_result.final_objective, 2e-10, 2e-8) &&
                    (gauge_probabilities - legacy_probabilities)
                            .cwiseAbs()
                            .maxCoeff() < 2e-7,
                "feature-gauge on/off must reach the same fitted objective "
                "and probabilities");
  const double gauge_kkt = independent_outer_kkt(
      objective, gauge_result.beta, gauge_result.intercept, lambda, true);
  const double legacy_kkt = independent_outer_kkt(
      objective, legacy_result.beta, legacy_result.intercept, lambda, true);
  ok &= require(nearly_equal(gauge_kkt, gauge_result.final_kkt_residual,
                             2e-12, 2e-10) &&
                    nearly_equal(legacy_kkt, legacy_result.final_kkt_residual,
                                 2e-12, 2e-10),
                "feature-gauge on/off final KKT reports must be independent");
  for (std::size_t index = 1; index < gauge_result.history.size(); ++index) {
    const double allowance =
        1e-12 * std::max(1.0, std::fabs(gauge_result.history[index - 1]
                                           .objective));
    ok &= require(gauge_result.history[index].objective <=
                      gauge_result.history[index - 1].objective + allowance,
                  "feature-gauge objective history must remain monotone");
  }
  return ok;
}

bool test_feature_gauge_twenty_seed_standardized_stress() {
  bool ok = true;
  int converged_count = 0;
  for (int seed = 0; seed < 20; ++seed) {
    const int n = 72;
    const int d = 7;
    const int num_classes = 3 + seed % 2;
    Eigen::MatrixXd x(n, d);
    Eigen::VectorXi labels(n);
    for (int i = 0; i < n; ++i) {
      const double row = static_cast<double>(i + 1);
      const double base =
          std::sin(0.071 * row * static_cast<double>(seed + 1)) +
          0.25 * std::cos(0.13 * row + static_cast<double>(seed));
      x(i, 0) = 0.0;
      x(i, 1) = base;
      x(i, 2) = base;
      // Match the package's default standardize=TRUE path while retaining a
      // zero column, exact duplicates, class imbalance, and correlated data.
      x(i, 3) = std::cos(0.017 * row * (seed + 2));
      x(i, 4) = std::sin(0.19 * row + seed);
      x(i, 5) = static_cast<double>((i + 3 * seed) % 9 - 4);
      x(i, 6) = std::cos(0.043 * row * row + 0.2 * seed);
      if ((i + seed) % 10 < 8) {
        labels[i] = 0;
      } else {
        labels[i] = 1 + ((i / 10 + seed) % (num_classes - 1));
      }
    }
    picasso::MultinomialObjective objective(x, labels, num_classes);
    picasso::solver::MultinomialActNewtonOptions options;
    picasso::solver::MultinomialActNewtonOptions legacy_options = options;
    legacy_options.canonicalize_feature_l1_gauge = false;
    picasso::solver::MultinomialActNewtonSolver solver(objective, options);
    picasso::solver::MultinomialActNewtonSolver legacy_solver(
        objective, legacy_options);
    const picasso::solver::MultinomialActNewtonResult result =
        solver.solve(0.03);
    const picasso::solver::MultinomialActNewtonResult legacy_result =
        legacy_solver.solve(0.03);
    if (result.converged()) ++converged_count;
    ok &= require(result.converged(),
                  "default damping must converge on standardized "
                  "feature-gauge stress "
                  "seed " + std::to_string(seed));
    ok &= require(result.status == legacy_result.status,
                  "feature gauge must not change the stress status for seed " +
                      std::to_string(seed));
    Eigen::MatrixXd probabilities;
    Eigen::MatrixXd legacy_probabilities;
    objective.negative_log_likelihood(result.beta, result.intercept,
                                      &probabilities);
    objective.negative_log_likelihood(
        legacy_result.beta, legacy_result.intercept, &legacy_probabilities);
    ok &= require(nearly_equal(result.final_objective,
                               legacy_result.final_objective, 2e-9, 2e-7) &&
                      (probabilities - legacy_probabilities)
                              .cwiseAbs()
                              .maxCoeff() < 2e-6,
                  "feature gauge must preserve the standardized stress fit "
                  "for seed " + std::to_string(seed));
    ok &= require(result.beta.allFinite() && result.intercept.allFinite() &&
                      std::isfinite(result.final_objective),
                  "feature-gauge stress state must remain finite");
    ok &= require(has_canonical_feature_gauge(result.beta),
                  "feature-gauge stress result must be canonical");
  }
  ok &= require(converged_count == 20,
                "all twenty default-damping stress cases must converge");
  return ok;
}

bool test_ill_scaled_feature_gauge_diagnostic() {
  bool ok = true;
  const int diagnostic_seeds[] = {0, 2, 3, 7, 9, 11};
  for (std::size_t seed_index = 0;
       seed_index < sizeof(diagnostic_seeds) / sizeof(diagnostic_seeds[0]);
       ++seed_index) {
    const int seed = diagnostic_seeds[seed_index];
    const int n = 72;
    const int d = 7;
    const int num_classes = 3 + seed % 2;
    Eigen::MatrixXd x(n, d);
    Eigen::VectorXi labels(n);
    for (int i = 0; i < n; ++i) {
      const double row = static_cast<double>(i + 1);
      const double base =
          std::sin(0.071 * row * static_cast<double>(seed + 1)) +
          0.25 * std::cos(0.13 * row + static_cast<double>(seed));
      x(i, 0) = 0.0;
      x(i, 1) = base;
      x(i, 2) = base;
      x(i, 3) = 1e4 * std::cos(0.017 * row * (seed + 2));
      x(i, 4) = 1e-4 * std::sin(0.19 * row + seed);
      x(i, 5) = static_cast<double>((i + 3 * seed) % 9 - 4);
      x(i, 6) = std::cos(0.043 * row * row + 0.2 * seed);
      if ((i + seed) % 10 < 8) {
        labels[i] = 0;
      } else {
        labels[i] = 1 + ((i / 10 + seed) % (num_classes - 1));
      }
    }

    picasso::MultinomialObjective objective(x, labels, num_classes);
    picasso::solver::MultinomialActNewtonSolver solver(objective);
    picasso::solver::MultinomialActNewtonOptions legacy_options;
    legacy_options.canonicalize_feature_l1_gauge = false;
    picasso::solver::MultinomialActNewtonSolver legacy_solver(
        objective, legacy_options);
    const picasso::solver::MultinomialActNewtonResult result =
        solver.solve(0.03);
    const picasso::solver::MultinomialActNewtonResult legacy_result =
        legacy_solver.solve(0.03);
    ok &= require(result.status ==
                          picasso::solver::MultinomialSolverStatus::kConverged ||
                      result.status == picasso::solver::
                                           MultinomialSolverStatus::
                                               kInnerIterationLimit,
                  "ill-scaled input must converge or report its inner cap "
                  "explicitly");
    ok &= require(result.status == legacy_result.status,
                  "symmetric feature gauge must preserve the ill-scaled "
                  "diagnostic status");
    ok &= require(result.beta.allFinite() && result.intercept.allFinite() &&
                      std::isfinite(result.final_objective) &&
                      std::isfinite(result.final_kkt_residual),
                  "ill-scaled diagnostic must preserve a finite outer state");
    ok &= require(has_canonical_feature_gauge(result.beta),
                  "ill-scaled diagnostic must retain the feature gauge");
    Eigen::MatrixXd probabilities;
    objective.negative_log_likelihood(result.beta, result.intercept,
                                      &probabilities);
    ok &= require(probabilities.allFinite() &&
                      (probabilities.rowwise().sum().array() - 1.0)
                              .abs()
                              .maxCoeff() < 2e-14,
                  "ill-scaled diagnostic probabilities must remain valid");
    if (seed == 3 || seed == 7 || seed == 9) {
      Eigen::MatrixXd legacy_probabilities;
      objective.negative_log_likelihood(
          legacy_result.beta, legacy_result.intercept,
          &legacy_probabilities);
      ok &= require(result.converged() && legacy_result.converged(),
                    "even-K symmetric gauge must avoid the endpoint-kink "
                    "inner slowdown");
      ok &= require(nearly_equal(result.final_objective,
                                 legacy_result.final_objective, 2e-9,
                                 2e-7) &&
                        (probabilities - legacy_probabilities)
                                .cwiseAbs()
                                .maxCoeff() < 2e-6,
                    "even-K symmetric gauge must preserve the ill-scaled "
                    "diagnostic fit");
    }
    for (std::size_t index = 1; index < result.history.size(); ++index) {
      const double allowance =
          1e-12 * std::max(1.0, std::fabs(result.history[index - 1]
                                              .objective));
      ok &= require(result.history[index].objective <=
                        result.history[index - 1].objective + allowance,
                    "ill-scaled diagnostic accepted objectives must remain "
                    "monotone");
    }
  }
  return ok;
}

bool test_uniform_matrix_penalty_exact_dispatch() {
  bool ok = true;
  for (int num_classes = 3; num_classes <= 4; ++num_classes) {
    const int n = 42;
    const int d = 3;
    Eigen::MatrixXd x(n, d);
    Eigen::VectorXi labels(n);
    for (int i = 0; i < n; ++i) {
      x(i, 0) = std::sin(0.17 * static_cast<double>(i + 1));
      x(i, 1) = std::cos(0.29 * static_cast<double>(i + 2));
      x(i, 2) = static_cast<double>((i * 5) % 13 - 6) / 6.0;
      labels[i] = (i * 7 + i / 5 + 1) % num_classes;
    }
    picasso::MultinomialObjective objective(x, labels, num_classes);
    Eigen::MatrixXd initial_beta(d, num_classes);
    for (int j = 0; j < d; ++j) {
      for (int klass = 0; klass < num_classes; ++klass) {
        initial_beta(j, klass) =
            0.02 * std::sin(static_cast<double>((j + 1) * (klass + 2)));
      }
    }
    Eigen::VectorXd initial_intercept(num_classes);
    for (int klass = 0; klass < num_classes; ++klass)
      initial_intercept[klass] = 0.03 * static_cast<double>(klass - 1);

    for (int mode = 0; mode < 2; ++mode) {
      picasso::solver::MultinomialActNewtonOptions options;
      options.max_outer_iterations = 100;
      options.max_inner_sweeps = 4000;
      options.outer_kkt_tolerance = 1e-6;
      options.inner_kkt_tolerance = 1e-9;
      options.use_active_set = mode != 0;
      options.use_probability_dot_direction_cache = mode != 0;
      picasso::solver::MultinomialActNewtonSolver solver(objective, options);
      const double lambdas[] = {0.0, 0.018};
      for (std::size_t lambda_index = 0; lambda_index < 2; ++lambda_index) {
        const double lambda = lambdas[lambda_index];
        const Eigen::MatrixXd penalties = Eigen::MatrixXd::Constant(
            d, num_classes, lambda);
        const picasso::solver::MultinomialActNewtonResult scalar =
            solver.solve(lambda, initial_beta, initial_intercept);
        const picasso::solver::MultinomialActNewtonResult weighted =
            solver.solve(penalties, initial_beta, initial_intercept);
        const std::string case_name =
            "uniform K=" + std::to_string(num_classes) +
            " lambda=" + std::to_string(lambda) +
            (mode == 0 ? " full/uncached" : " active/cached");
        ok &= equivalent_result_fields(scalar, weighted, case_name);
      }
    }
  }
  return ok;
}

bool test_matrix_penalty_validation() {
  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(9, 2);
  Eigen::VectorXi labels(9);
  labels << 0, 1, 2, 0, 1, 2, 0, 1, 2;
  picasso::MultinomialObjective objective(x, labels, 3);
  picasso::solver::MultinomialActNewtonSolver solver(objective);
  Eigen::MatrixXd valid = Eigen::MatrixXd::Constant(2, 3, 0.1);

  bool ok = true;
  ok &= require(
      picasso::solver::MultinomialActNewtonOptions()
              .exact_kkt_scan_interval == 4,
      "production options must enable the validated four-sweep exact KKT "
      "schedule by default");
  for (int invalid_interval = 0; invalid_interval >= -1;
       --invalid_interval) {
    picasso::solver::MultinomialActNewtonOptions invalid_options;
    invalid_options.exact_kkt_scan_interval = invalid_interval;
    bool rejected = false;
    try {
      picasso::solver::MultinomialActNewtonSolver invalid_solver(
          objective, invalid_options);
      (void)invalid_solver;
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    ok &= require(rejected,
                  "nonpositive exact KKT scan intervals must throw");
  }
  {
    picasso::solver::MultinomialActNewtonOptions invalid_options;
    invalid_options.use_adaptive_inner_tolerance = true;
    invalid_options.inner_kkt_tolerance =
        2.0 * invalid_options.outer_kkt_tolerance;
    bool rejected = false;
    try {
      picasso::solver::MultinomialActNewtonSolver invalid_solver(
          objective, invalid_options);
      (void)invalid_solver;
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    ok &= require(
        rejected,
        "adaptive inner tolerance floor must not exceed outer tolerance");
  }
  {
    typedef picasso::solver::MultinomialActNewtonOptions Options;
    double Options::*fields[] = {
        &Options::outer_kkt_tolerance, &Options::inner_kkt_tolerance,
        &Options::armijo_constant, &Options::backtracking_factor,
        &Options::minimum_step_size, &Options::hessian_damping,
        &Options::zero_tolerance};
    const char *names[] = {
        "outer KKT tolerance", "inner KKT tolerance",
        "Armijo constant", "backtracking factor", "minimum step size",
        "Hessian damping", "zero tolerance"};
    const double invalid_values[] = {
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    const char *value_names[] = {"NaN", "positive-infinite",
                                 "negative-infinite"};
    for (std::size_t index = 0;
         index < sizeof(fields) / sizeof(fields[0]); ++index) {
      for (std::size_t value_index = 0;
           value_index < sizeof(invalid_values) / sizeof(invalid_values[0]);
           ++value_index) {
        Options invalid_options;
        invalid_options.*fields[index] = invalid_values[value_index];
        bool rejected = false;
        try {
          picasso::solver::MultinomialActNewtonSolver invalid_solver(
              objective, invalid_options);
          (void)invalid_solver;
        } catch (const std::invalid_argument &) {
          rejected = true;
        }
        ok &= require(rejected,
                      std::string(value_names[value_index]) + " " +
                          names[index] + " must throw");
      }
    }
  }
  ok &= require(rejects_l1_penalties(solver, Eigen::MatrixXd::Zero(1, 3)),
                "matrix penalties with the wrong row count must throw");
  ok &= require(rejects_l1_penalties(solver, Eigen::MatrixXd::Zero(2, 4)),
                "matrix penalties with the wrong class count must throw");
  Eigen::MatrixXd invalid = valid;
  invalid(0, 1) = -1e-12;
  ok &= require(rejects_l1_penalties(solver, invalid),
                "negative matrix penalties must throw");
  invalid = valid;
  invalid(1, 2) = std::numeric_limits<double>::quiet_NaN();
  ok &= require(rejects_l1_penalties(solver, invalid),
                "NaN matrix penalties must throw");
  invalid = valid;
  invalid(0, 0) = std::numeric_limits<double>::infinity();
  ok &= require(rejects_l1_penalties(solver, invalid),
                "positive-infinite matrix penalties must throw");
  invalid = valid;
  invalid(0, 0) = -std::numeric_limits<double>::infinity();
  ok &= require(rejects_l1_penalties(solver, invalid),
                "negative-infinite matrix penalties must throw");
  return ok;
}

bool test_nonuniform_matrix_penalty_objective_kkt_and_active_ab() {
  const int n = 120;
  const int d = 3;
  const int num_classes = 3;
  Eigen::MatrixXd x(n, d);
  Eigen::MatrixXd true_beta(d, num_classes);
  true_beta << 0.8, -0.4, -0.2,
              -0.3, 0.7, -0.4,
               0.2, -0.5, 0.3;
  Eigen::VectorXi labels(n);
  for (int i = 0; i < n; ++i) {
    x(i, 0) = std::sin(0.11 * static_cast<double>(i + 1));
    x(i, 1) = std::cos(0.07 * static_cast<double>(i + 3));
    x(i, 2) = static_cast<double>((i * 7) % 17 - 8) / 8.0;
  }
  const Eigen::MatrixXd logits = x * true_beta;
  for (int i = 0; i < n; ++i) {
    const double maximum = logits.row(i).maxCoeff();
    Eigen::VectorXd probabilities(num_classes);
    for (int klass = 0; klass < num_classes; ++klass)
      probabilities[klass] = std::exp(logits(i, klass) - maximum);
    probabilities.array() /= probabilities.sum();
    const double draw =
        std::fmod(0.6180339887498949 * static_cast<double>(i + 1), 1.0);
    double cumulative = 0.0;
    labels[i] = num_classes - 1;
    for (int klass = 0; klass < num_classes; ++klass) {
      cumulative += probabilities[klass];
      if (draw <= cumulative) {
        labels[i] = klass;
        break;
      }
    }
  }
  picasso::MultinomialObjective objective(x, labels, num_classes);
  Eigen::MatrixXd penalties(d, num_classes);
  penalties << 0.012, 0.020, 0.035,
               0.025, 0.009, 0.018,
               0.016, 0.030, 0.000;

  picasso::solver::MultinomialActNewtonOptions full_options;
  full_options.max_outer_iterations = 120;
  full_options.max_inner_sweeps = 5000;
  full_options.outer_kkt_tolerance = 5e-7;
  full_options.inner_kkt_tolerance = 1e-9;
  full_options.use_active_set = false;
  full_options.exact_kkt_scan_interval = 1;
  picasso::solver::MultinomialActNewtonOptions active_options = full_options;
  active_options.use_active_set = true;
  active_options.use_compact_inner_active_set = true;
  active_options.exact_kkt_scan_interval = 1;
  picasso::solver::MultinomialActNewtonOptions interval_options =
      active_options;
  interval_options.exact_kkt_scan_interval = 4;
  picasso::solver::MultinomialActNewtonSolver full_solver(objective,
                                                          full_options);
  picasso::solver::MultinomialActNewtonSolver active_solver(objective,
                                                            active_options);
  picasso::solver::MultinomialActNewtonSolver interval_solver(
      objective, interval_options);
  const picasso::solver::MultinomialActNewtonResult full =
      full_solver.solve(penalties);
  const picasso::solver::MultinomialActNewtonResult active =
      active_solver.solve(penalties);
  const picasso::solver::MultinomialActNewtonResult interval =
      interval_solver.solve(penalties);

  bool ok = true;
  ok &= require(full.converged() && active.converged() &&
                    interval.converged(),
                "nonuniform weighted solves must converge with active set "
                "on and off");
  ok &= require(std::fabs(active.beta(2, 2)) >
                    active_options.zero_tolerance,
                "a zero-penalty coordinate with nonzero smooth gradient must "
                "be activated instead of screened out");
  const double independent_objective =
      objective.negative_log_likelihood(active.beta, active.intercept) +
      weighted_l1_penalty(active.beta, penalties);
  const double independent_kkt = independent_weighted_outer_kkt(
      objective, active.beta, active.intercept, penalties, true);
  ok &= require(nearly_equal(active.final_objective, independent_objective,
                             2e-13, 2e-12),
                "reported weighted objective must match an independent "
                "NLL-plus-elementwise-L1 calculation");
  ok &= require(nearly_equal(active.final_kkt_residual, independent_kkt,
                             2e-12, 2e-10) &&
                    independent_kkt <=
                        1.1 * active_options.outer_kkt_tolerance,
                "reported weighted KKT must match an independent "
                "per-coordinate calculation");
  Eigen::MatrixXd full_probabilities;
  Eigen::MatrixXd active_probabilities;
  Eigen::MatrixXd interval_probabilities;
  objective.negative_log_likelihood(full.beta, full.intercept,
                                    &full_probabilities);
  objective.negative_log_likelihood(active.beta, active.intercept,
                                    &active_probabilities);
  objective.negative_log_likelihood(interval.beta, interval.intercept,
                                    &interval_probabilities);
  ok &= require(nearly_equal(full.final_objective, active.final_objective,
                             2e-10, 2e-8) &&
                    (full_probabilities - active_probabilities)
                            .cwiseAbs()
                            .maxCoeff() < 2e-6,
                "weighted active-set on/off must reach the same fit");
  const double interval_kkt = independent_weighted_outer_kkt(
      objective, interval.beta, interval.intercept, penalties, true);
  ok &= require(
      interval.status == active.status &&
          nearly_equal(interval.final_objective, active.final_objective,
                       2e-10, 2e-8) &&
          (interval_probabilities - active_probabilities)
                  .cwiseAbs()
                  .maxCoeff() < 2e-6 &&
          has_same_support(interval.beta, active.beta) &&
          interval_kkt <= 1.1 * interval_options.outer_kkt_tolerance &&
          nearly_equal(interval.final_kkt_residual, interval_kkt, 2e-12,
                       2e-10),
      "periodic scans must preserve weighted-L1 status, objective, "
      "probabilities, support, and exact KKT certification");
  ok &= require(has_canonical_weighted_feature_gauge(active.beta,
                                                      penalties),
                "nonuniform solution must retain the weighted feature "
                "gauge");
  ok &= require(active_probabilities.allFinite() &&
                    (active_probabilities.rowwise().sum().array() - 1.0)
                            .abs()
                            .maxCoeff() < 2e-14,
                "weighted fitted probabilities must be finite and sum to "
                "one");
  return ok;
}

bool check_weighted_entry_gauge(const Eigen::MatrixXd &initial_beta,
                                const Eigen::MatrixXd &penalties,
                                const Eigen::MatrixXd &expected_beta,
                                const std::string &case_name) {
  const int num_classes = static_cast<int>(initial_beta.cols());
  const int n = 2 * num_classes;
  Eigen::MatrixXd x(n, initial_beta.rows());
  Eigen::VectorXi labels(n);
  for (int i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < initial_beta.rows(); ++j)
      x(i, j) = std::sin(0.19 * static_cast<double>((i + 1) * (j + 1)));
    labels[i] = i % num_classes;
  }
  picasso::MultinomialObjective objective(x, labels, num_classes);
  Eigen::VectorXd intercept = Eigen::VectorXd::Zero(num_classes);
  Eigen::MatrixXd before_probabilities;
  objective.negative_log_likelihood(initial_beta, intercept,
                                    &before_probabilities);
  const double before_penalty = weighted_l1_penalty(initial_beta, penalties);

  picasso::solver::MultinomialActNewtonOptions options;
  options.include_intercept = false;
  options.outer_kkt_tolerance = 1e6;
  picasso::solver::MultinomialActNewtonSolver solver(objective, options);
  const picasso::solver::MultinomialActNewtonResult result =
      solver.solve(penalties, initial_beta, intercept);
  Eigen::MatrixXd after_probabilities;
  objective.negative_log_likelihood(result.beta, result.intercept,
                                    &after_probabilities);

  bool ok = true;
  ok &= require(result.converged() && result.outer_iterations == 0,
                case_name + " weighted gauge must run before the first KKT");
  ok &= require((result.beta - expected_beta).cwiseAbs().maxCoeff() < 1e-14,
                case_name + " must select the expected weighted midpoint");
  ok &= require(has_canonical_weighted_feature_gauge(result.beta,
                                                      penalties, 1e-14),
                case_name + " must be weighted-gauge canonical");
  ok &= require((before_probabilities - after_probabilities)
                        .cwiseAbs()
                        .maxCoeff() < 2e-14,
                case_name + " common row shifts must preserve probabilities");
  ok &= require(weighted_l1_penalty(result.beta, penalties) <=
                    before_penalty + 1e-14,
                case_name + " gauge must not increase weighted L1");
  return ok;
}

bool test_weighted_feature_gauge_edge_cases() {
  Eigen::MatrixXd counter_beta(1, 3);
  counter_beta << 0.0, 1.0, 10.0;
  Eigen::MatrixXd counter_penalties(1, 3);
  counter_penalties << 100.0, 1.0, 1.0;
  bool ok = check_weighted_entry_gauge(counter_beta, counter_penalties,
                                       counter_beta,
                                       "heavy-weight counterexample");
  Eigen::MatrixXd unweighted_centered = counter_beta;
  unweighted_centered.row(0).array() -= row_median_center(counter_beta, 0);
  ok &= require(weighted_l1_penalty(counter_beta, counter_penalties) == 11.0 &&
                    weighted_l1_penalty(unweighted_centered,
                                        counter_penalties) == 109.0,
                "the [0,1,10] fixture must reject an unweighted-median "
                "gauge under [100,1,1]");

  Eigen::MatrixXd flat_beta(1, 4);
  flat_beta << 0.0, 1.0, 2.0, 3.0;
  Eigen::MatrixXd flat_penalties(1, 4);
  flat_penalties << 1.0, 2.0, 0.5, 3.5;
  Eigen::MatrixXd flat_expected(1, 4);
  flat_expected << -2.5, -1.5, -0.5, 0.5;
  ok &= check_weighted_entry_gauge(flat_beta, flat_penalties, flat_expected,
                                   "flat weighted-median interval");

  Eigen::MatrixXd partial_beta(1, 4);
  partial_beta << 0.0, 5.0, 10.0, 20.0;
  Eigen::MatrixXd partial_penalties(1, 4);
  partial_penalties << 0.0, 2.0, 0.0, 2.0;
  Eigen::MatrixXd partial_expected(1, 4);
  partial_expected << -12.5, -7.5, -2.5, 7.5;
  ok &= check_weighted_entry_gauge(partial_beta, partial_penalties,
                                   partial_expected,
                                   "partial-zero weighted row");

  Eigen::MatrixXd zero_row_beta(2, 3);
  zero_row_beta << 0.0, 1.0, 10.0,
                  -4.0, 2.0, 6.0;
  Eigen::MatrixXd zero_row_penalties(2, 3);
  zero_row_penalties << 0.0, 0.0, 0.0,
                        1.0, 5.0, 2.0;
  Eigen::MatrixXd zero_row_expected(2, 3);
  zero_row_expected << -1.0, 0.0, 9.0,
                       -6.0, 0.0, 4.0;
  ok &= check_weighted_entry_gauge(zero_row_beta, zero_row_penalties,
                                   zero_row_expected,
                                   "all-zero-row compatibility fallback");
  return ok;
}

bool test_extreme_finite_data_and_numerical_status() {
  Eigen::MatrixXd x(12, 1);
  Eigen::VectorXi labels(12);
  for (int i = 0; i < 12; ++i) {
    x(i, 0) = 5.0 * static_cast<double>(i - 6);
    labels[i] = i % 3;
  }
  picasso::MultinomialObjective objective(x, labels, 3);
  Eigen::MatrixXd initial_beta(1, 3);
  initial_beta << 2.0, -2.0, 0.0;
  Eigen::VectorXd initial_intercept(3);
  initial_intercept << 1e8 + 1.0, 1e8 - 1.0, 1e8;

  picasso::solver::MultinomialActNewtonOptions options;
  options.max_outer_iterations = 100;
  options.max_inner_sweeps = 5000;
  options.outer_kkt_tolerance = 1e-6;
  options.inner_kkt_tolerance = 1e-8;
  // A saturated starting softmax has almost no curvature.  Explicit damping
  // is the intended proximal-Newton safeguard for this stress case.
  options.hessian_damping = 1e-4;
  picasso::solver::MultinomialActNewtonSolver solver(objective, options);
  const picasso::solver::MultinomialActNewtonResult result =
      solver.solve(0.2, initial_beta, initial_intercept);

  bool ok = true;
  ok &= require(result.converged(),
                std::string("extreme finite logits must converge without a ") +
                    "numerical status, got " +
                    picasso::solver::multinomial_solver_status_string(
                        result.status));
  ok &= require(result.beta.allFinite() && result.intercept.allFinite() &&
                    std::isfinite(result.final_objective),
                "extreme finite data must retain finite solver state");
  for (std::size_t i = 1; i < result.history.size(); ++i)
    ok &= require(result.history[i].objective <=
                      result.history[i - 1].objective + 1e-10,
                  "extreme-data Armijo history must be monotone");

  Eigen::MatrixXd nonfinite_x = Eigen::MatrixXd::Zero(3, 1);
  nonfinite_x(1, 0) = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXi nonfinite_labels(3);
  nonfinite_labels << 0, 1, 2;
  picasso::MultinomialObjective nonfinite_objective(nonfinite_x,
                                                    nonfinite_labels, 3);
  picasso::solver::MultinomialActNewtonSolver nonfinite_solver(
      nonfinite_objective, options);
  bool threw = false;
  picasso::solver::MultinomialActNewtonResult nonfinite_result;
  try {
    nonfinite_result = nonfinite_solver.solve(0.2);
  } catch (...) {
    threw = true;
  }
  ok &= require(!threw,
                "nonfinite design data must become an explicit solver status");
  ok &= require(!threw &&
                    nonfinite_result.status ==
                        picasso::solver::MultinomialSolverStatus::
                            kNumericalFailure,
                "nonfinite design data must report numerical_failure");
  return ok;
}

bool test_logistic_style_path_strong_rule_and_kkt_correction() {
  const int n = 60;
  const int d = 12;
  const int num_classes = 3;
  Eigen::MatrixXd x(n, d);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      const double block = static_cast<double>((j % 4) + 1);
      x(i, j) =
          0.92 * std::sin(0.071 * static_cast<double>(i + 1) * block +
                          0.017 * block) +
          0.392 * std::cos(0.113 * static_cast<double>(i + 2) *
                               static_cast<double>(j + 1) +
                           0.029);
    }
  }
  for (int j = 0; j < d; ++j) {
    const double mean = x.col(j).mean();
    x.col(j).array() -= mean;
    const double scale = std::sqrt(x.col(j).squaredNorm() /
                                   static_cast<double>(n));
    x.col(j).array() /= scale;
  }
  const int label_values[n] = {
      0, 1, 0, 2, 2, 0, 2, 1, 2, 2, 0, 2, 0, 0, 1,
      0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2,
      0, 0, 2, 2, 0, 2, 0, 2, 2, 0, 2, 0, 2, 1, 0,
      1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 2, 1};
  Eigen::VectorXi labels(n);
  for (int i = 0; i < n; ++i) labels[i] = label_values[i];

  picasso::MultinomialObjective objective(x, labels, num_classes);
  Eigen::VectorXd null_intercept = Eigen::VectorXd::Zero(num_classes);
  for (int i = 0; i < n; ++i) null_intercept[labels[i]] += 1.0;
  null_intercept.array() /= static_cast<double>(n);
  null_intercept = null_intercept.array().log().matrix();
  null_intercept.array() -= null_intercept.mean();
  Eigen::MatrixXd null_gradient;
  Eigen::VectorXd ignored_intercept_gradient;
  objective.smooth_gradient(Eigen::MatrixXd::Zero(d, num_classes),
                            null_intercept, &null_gradient,
                            &ignored_intercept_gradient);
  const double lambda_max = null_gradient.cwiseAbs().maxCoeff();
  const double ratios[] = {
      1.05, 0.8266361950101179, 0.6507879989531481,
      0.51234753829798, 0.403357161506136,
      0.3175520278262036, 0.25};

  picasso::solver::MultinomialActNewtonOptions path_options;
  path_options.max_outer_iterations = 100;
  path_options.max_inner_sweeps = 4000;
  path_options.outer_kkt_tolerance = 2e-7;
  path_options.inner_kkt_tolerance = 2e-9;
  path_options.use_active_set = true;
  picasso::solver::MultinomialActNewtonOptions adaptive_options =
      path_options;
  adaptive_options.use_adaptive_inner_tolerance = true;
  picasso::solver::MultinomialActNewtonOptions oracle_options = path_options;
  oracle_options.use_active_set = false;
  picasso::solver::MultinomialActNewtonPathSolver path_solver(
      objective, path_options);
  picasso::solver::MultinomialActNewtonPathSolver adaptive_solver(
      objective, adaptive_options);
  picasso::solver::MultinomialActNewtonSolver oracle_solver(
      objective, oracle_options);
  picasso::solver::MultinomialActNewtonPathState state;
  picasso::solver::MultinomialActNewtonPathState adaptive_state;

  Eigen::MatrixXd oracle_beta;
  Eigen::VectorXd oracle_intercept;
  bool have_oracle = false;
  bool saw_screening = false;
  bool saw_relaxed_inner_certificate = false;
  long long path_coordinate_updates = 0;
  long long adaptive_coordinate_updates = 0;
  long long oracle_coordinate_updates = 0;
  bool ok = true;
  for (std::size_t point = 0;
       point < sizeof(ratios) / sizeof(ratios[0]); ++point) {
    const double lambda = ratios[point] * lambda_max;
    const picasso::solver::MultinomialActNewtonPathResult path =
        path_solver.solve(lambda, &state);
    const picasso::solver::MultinomialActNewtonPathResult adaptive =
        adaptive_solver.solve(lambda, &adaptive_state);
    const picasso::solver::MultinomialActNewtonResult oracle =
        have_oracle
            ? oracle_solver.solve(lambda, oracle_beta, oracle_intercept)
            : oracle_solver.solve(lambda);
    ok &= require(path.solution.converged() &&
                      adaptive.solution.converged() && oracle.converged(),
                  "fixed, adaptive, and unscreened path solves must "
                  "converge");
    if (!path.solution.converged() ||
        !adaptive.solution.converged() || !oracle.converged())
      return false;
    ok &= require(
        path.reused_initial_smooth_state == (point != 0) &&
            adaptive.reused_initial_smooth_state == (point != 0),
        "a path must compute its first smooth state once and reuse every "
        "subsequent committed smooth state");

    Eigen::MatrixXd committed_gradient;
    Eigen::VectorXd committed_intercept_gradient;
    objective.smooth_gradient(
        state.beta, state.intercept, &committed_gradient,
        &committed_intercept_gradient);
    double gradient_summary_error = 0.0;
    for (int feature = 0; feature < d; ++feature) {
      gradient_summary_error = std::max(
          gradient_summary_error,
          std::fabs(state.feature_gradient_max[feature] -
                    committed_gradient.row(feature)
                        .cwiseAbs()
                        .maxCoeff()));
    }
    ok &= require(
        committed_gradient.allFinite() &&
            committed_intercept_gradient.allFinite() &&
            gradient_summary_error <= 1e-12,
        "path state must retain the converged solver's exact gradient "
        "summary");
    oracle_beta = oracle.beta;
    oracle_intercept = oracle.intercept;
    have_oracle = true;
    path_coordinate_updates += path.solution.total_coordinate_updates;
    adaptive_coordinate_updates +=
        adaptive.solution.total_coordinate_updates;
    oracle_coordinate_updates += oracle.total_coordinate_updates;
    for (std::size_t history_index = 1;
         history_index < adaptive.solution.history.size(); ++history_index) {
      saw_relaxed_inner_certificate =
          saw_relaxed_inner_certificate ||
          adaptive.solution.history[history_index].inner_kkt_residual >
              1.01 * adaptive_options.inner_kkt_tolerance;
    }
    saw_screening =
        saw_screening || path.solution.initial_active_features < d;

    Eigen::MatrixXd path_probabilities;
    Eigen::MatrixXd adaptive_probabilities;
    Eigen::MatrixXd oracle_probabilities;
    objective.negative_log_likelihood(path.solution.beta,
                                      path.solution.intercept,
                                      &path_probabilities);
    objective.negative_log_likelihood(adaptive.solution.beta,
                                      adaptive.solution.intercept,
                                      &adaptive_probabilities);
    objective.negative_log_likelihood(oracle.beta, oracle.intercept,
                                      &oracle_probabilities);
    ok &= require(
        nearly_equal(path.solution.final_objective,
                     oracle.final_objective, 2e-9, 2e-7) &&
            (path_probabilities - oracle_probabilities)
                    .cwiseAbs()
                    .maxCoeff() < 2e-6 &&
            nearly_equal(adaptive.solution.final_objective,
                         oracle.final_objective, 2e-9, 2e-7) &&
            (adaptive_probabilities - oracle_probabilities)
                    .cwiseAbs()
                    .maxCoeff() < 5e-6 &&
            has_same_support(path.solution.beta, oracle.beta) &&
            has_same_support(adaptive.solution.beta, oracle.beta),
        "fixed and adaptive path screening must preserve objective, "
        "probabilities, and support");
    const double independent_kkt = independent_outer_kkt(
        objective, path.solution.beta, path.solution.intercept, lambda, true);
    ok &= require(
        independent_kkt <= 1.1 * path_options.outer_kkt_tolerance &&
            std::fabs(independent_kkt -
                      path.solution.final_kkt_residual) < 1e-12,
        "every screened path point must pass a full independent KKT scan");
    const double adaptive_independent_kkt = independent_outer_kkt(
        objective, adaptive.solution.beta, adaptive.solution.intercept,
        lambda, true);
    ok &= require(
        adaptive_independent_kkt <=
                1.1 * adaptive_options.outer_kkt_tolerance &&
            std::fabs(adaptive_independent_kkt -
                      adaptive.solution.final_kkt_residual) < 1e-12,
        "every adaptive path point must pass a full independent KKT scan");
  }
  ok &= require(saw_screening,
                "the deterministic path fixture must screen features");
  ok &= require(path_coordinate_updates < oracle_coordinate_updates,
                "path-level ActNewton screening must reduce coordinate "
                "updates versus an unscreened oracle");
  ok &= require(
      saw_relaxed_inner_certificate &&
          adaptive_coordinate_updates < path_coordinate_updates,
      "adaptive inexact-Newton must relax an early quadratic certificate "
      "and reduce total coordinate work");

  // Exercise the safe path correction independently of the heuristic's hit
  // rate.  Logistic's first-point threshold is 2*lambda; at lambda_max/2 it
  // screens every feature on the strict comparison, so a full true-KKT scan
  // must repair the deliberately incomplete strong set.
  const double correction_lambda = 0.5 * lambda_max;
  picasso::solver::MultinomialActNewtonPathState correction_state;
  const picasso::solver::MultinomialActNewtonPathResult corrected_path =
      path_solver.solve(correction_lambda, &correction_state);
  const picasso::solver::MultinomialActNewtonResult corrected =
      corrected_path.solution;
  const picasso::solver::MultinomialActNewtonResult correction_oracle =
      oracle_solver.solve(correction_lambda);
  Eigen::MatrixXd corrected_probabilities;
  Eigen::MatrixXd correction_oracle_probabilities;
  objective.negative_log_likelihood(corrected.beta, corrected.intercept,
                                    &corrected_probabilities);
  objective.negative_log_likelihood(
      correction_oracle.beta, correction_oracle.intercept,
      &correction_oracle_probabilities);
  ok &= require(
      corrected.converged() && correction_oracle.converged() &&
          corrected_path.initial_strong_features == 0 &&
          corrected_path.full_kkt_reactivated_features > 0 &&
          corrected.total_subproblem_reactivated_features == 0 &&
          corrected.total_full_subproblem_kkt_scans == 0 &&
          corrected.total_outer_reactivated_features > 0 &&
          nearly_equal(corrected.final_objective,
                       correction_oracle.final_objective, 2e-9, 2e-7) &&
          (corrected_probabilities - correction_oracle_probabilities)
                  .cwiseAbs()
                  .maxCoeff() < 2e-6,
      "restricted ActNewton must repair an incomplete seed with a full-KKT "
      "scan and recover the unscreened solution");

  // A failed lambda must not poison the warm path state.  The first point is
  // above lambda_max and therefore converges at the null model without an
  // iteration; the deliberately undersized inner budget then forces the
  // second point to fail.
  picasso::solver::MultinomialActNewtonOptions failure_options = path_options;
  failure_options.max_inner_sweeps = 1;
  picasso::solver::MultinomialActNewtonPathSolver failure_solver(
      objective, failure_options);
  picasso::solver::MultinomialActNewtonPathState failure_state;
  const picasso::solver::MultinomialActNewtonPathResult safe_point =
      failure_solver.solve(1.05 * lambda_max, &failure_state);
  const picasso::solver::MultinomialActNewtonPathState saved_state =
      failure_state;
  const picasso::solver::MultinomialActNewtonPathResult failed_point =
      failure_solver.solve(0.05 * lambda_max, &failure_state);
  ok &= require(
      safe_point.solution.converged() &&
          !safe_point.reused_initial_smooth_state &&
          !failed_point.solution.converged() &&
          failed_point.reused_initial_smooth_state &&
          failure_state.initialized == saved_state.initialized &&
          failure_state.previous_lambda == saved_state.previous_lambda &&
          failure_state.strong_set == saved_state.strong_set &&
          (failure_state.beta - saved_state.beta).cwiseAbs().maxCoeff() ==
              0.0 &&
          (failure_state.intercept - saved_state.intercept)
                  .cwiseAbs()
                  .maxCoeff() == 0.0 &&
          (failure_state.feature_gradient_max -
           saved_state.feature_gradient_max)
                  .cwiseAbs()
                  .maxCoeff() == 0.0,
      "a failed path point must leave every committed warm-state field "
      "unchanged");

  picasso::solver::MultinomialActNewtonPathState retry_after_failure =
      failure_state;
  picasso::solver::MultinomialActNewtonPathState retry_from_saved =
      saved_state;
  const picasso::solver::MultinomialActNewtonPathResult failed_state_retry =
      path_solver.solve(0.25 * lambda_max, &retry_after_failure);
  const picasso::solver::MultinomialActNewtonPathResult saved_state_retry =
      path_solver.solve(0.25 * lambda_max, &retry_from_saved);
  ok &= require(
      failed_state_retry.solution.converged() &&
          saved_state_retry.solution.converged() &&
          failed_state_retry.reused_initial_smooth_state &&
          saved_state_retry.reused_initial_smooth_state &&
          (failed_state_retry.solution.beta -
           saved_state_retry.solution.beta)
                  .cwiseAbs()
                  .maxCoeff() == 0.0 &&
          (failed_state_retry.solution.intercept -
           saved_state_retry.solution.intercept)
                  .cwiseAbs()
                  .maxCoeff() == 0.0,
      "a failed solve must preserve the hidden cache for an identical retry");

  // Force one accepted restricted outer step whose active KKT is still well
  // above tolerance.  The solver therefore holds only an active-row gradient
  // at loop exhaustion and must rebuild the committed probabilities/full
  // gradient before reporting the failed point.
  int seed_feature = 0;
  double seed_gradient = 0.0;
  for (int feature = 0; feature < d; ++feature) {
    const double row_gradient =
        null_gradient.row(feature).cwiseAbs().maxCoeff();
    if (row_gradient > seed_gradient) {
      seed_gradient = row_gradient;
      seed_feature = feature;
    }
  }
  std::vector<unsigned char> one_feature_mask(
      static_cast<std::size_t>(d), 0);
  one_feature_mask[static_cast<std::size_t>(seed_feature)] = 1;
  picasso::solver::MultinomialActNewtonOptions terminal_options =
      path_options;
  terminal_options.max_outer_iterations = 1;
  terminal_options.outer_kkt_tolerance = 1e-14;
  terminal_options.inner_kkt_tolerance = 1e-10;
  picasso::solver::MultinomialActNewtonSolver terminal_solver(
      objective, terminal_options);
  const double terminal_lambda = 0.25 * lambda_max;
  const picasso::solver::MultinomialActNewtonResult terminal_result =
      terminal_solver.solve(
          terminal_lambda, Eigen::MatrixXd::Zero(d, num_classes),
          null_intercept, one_feature_mask);
  Eigen::MatrixXd terminal_gradient;
  Eigen::VectorXd terminal_intercept_gradient;
  objective.smooth_gradient(
      terminal_result.beta, terminal_result.intercept, &terminal_gradient,
      &terminal_intercept_gradient);
  double terminal_active_kkt =
      terminal_intercept_gradient.cwiseAbs().maxCoeff();
  for (int klass = 0; klass < num_classes; ++klass) {
    terminal_active_kkt = std::max(
        terminal_active_kkt,
        coefficient_kkt(terminal_result.beta(seed_feature, klass),
                        terminal_gradient(seed_feature, klass),
                        terminal_lambda));
  }
  const double terminal_full_kkt = independent_outer_kkt(
      objective, terminal_result.beta, terminal_result.intercept,
      terminal_lambda, true);
  ok &= require(
      terminal_result.status ==
              picasso::solver::MultinomialSolverStatus::
                  kOuterIterationLimit &&
          terminal_result.history.size() == 2 &&
          terminal_result.initial_active_features == 1 &&
          terminal_active_kkt > terminal_options.outer_kkt_tolerance &&
          std::fabs(terminal_result.final_kkt_residual -
                    terminal_full_kkt) < 1e-12 &&
          std::fabs(terminal_result.history.back().kkt_residual -
                    terminal_full_kkt) < 1e-12,
      "restricted outer-limit diagnostics must refresh the exact committed "
      "full KKT after an active-only gradient step");

  // Cache validity is tied to the exact committed parameter snapshot.  A
  // caller mutation must safely fall back to recomputation, while reset must
  // behave exactly like a fresh state.
  picasso::solver::MultinomialActNewtonPathState reference_state = state;
  picasso::solver::MultinomialActNewtonPathState mutated_state = state;
  mutated_state.beta(0, 0) += 1e-8;
  const picasso::solver::MultinomialActNewtonPathResult reference_point =
      path_solver.solve(0.20 * lambda_max, &reference_state);
  const picasso::solver::MultinomialActNewtonPathResult mutated_point =
      path_solver.solve(0.20 * lambda_max, &mutated_state);
  ok &= require(
      reference_point.solution.converged() &&
          reference_point.reused_initial_smooth_state &&
          mutated_point.solution.converged() &&
          !mutated_point.reused_initial_smooth_state &&
          nearly_equal(reference_point.solution.final_objective,
                       mutated_point.solution.final_objective, 2e-9, 2e-7),
      "a mutated path state must invalidate the cache without changing the "
      "certified solution");

  picasso::solver::MultinomialActNewtonPathState reset_state = state;
  reset_state.reset();
  const picasso::solver::MultinomialActNewtonPathResult reset_point =
      path_solver.solve(1.05 * lambda_max, &reset_state);
  ok &= require(reset_point.solution.converged() &&
                    !reset_point.reused_initial_smooth_state,
                "reset must discard the private smooth-state cache");

  // The sequential strong rule is valid only for a nonincreasing path.
  // Increasing lambda must conservatively disable screening.
  const picasso::solver::MultinomialActNewtonPathResult increasing_point =
      path_solver.solve(0.75 * lambda_max, &correction_state);
  ok &= require(
      increasing_point.solution.converged() &&
          !increasing_point.used_strong_rule &&
          increasing_point.solution.initial_active_features == d,
      "an increasing lambda step must disable the sequential strong rule");
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_lambda_max_zero_solution_and_intercept_gauge();
  ok &= test_default_intercept_edge_cases();
  ok &= test_dense_quadratic_oracle();
  ok &= test_final_kkt_and_monotone_armijo_history();
  ok &= test_probability_dot_direction_cache_ab_equivalence();
  ok &= test_exact_feature_working_set_ab_equivalence();
  ok &= test_working_set_full_quadratic_kkt_reactivation();
  ok &= test_feature_l1_gauge_odd_even_and_solver_ab();
  ok &= test_feature_gauge_twenty_seed_standardized_stress();
  ok &= test_ill_scaled_feature_gauge_diagnostic();
  ok &= test_uniform_matrix_penalty_exact_dispatch();
  ok &= test_matrix_penalty_validation();
  ok &= test_nonuniform_matrix_penalty_objective_kkt_and_active_ab();
  ok &= test_weighted_feature_gauge_edge_cases();
  ok &= test_extreme_finite_data_and_numerical_status();
  ok &= test_logistic_style_path_strong_rule_and_kkt_correction();
  if (!ok) return 1;
  std::cout << "multinomial_actnewton_test: PASS\n";
  return 0;
}
