#include <picasso/c_api.hpp>
#include <picasso/multinomial_actnewton.hpp>
#include <picasso/multinomial_lla.hpp>
#include <picasso/multinomial_objective.hpp>
#include <picasso/solver_params.hpp>

#include "../internal/multinomial_solver_view.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

namespace {

struct MultinomialCapiDiagnostics {
  int *failed_lambda;
  int *failed_stage;
  int *outer_iterations;
  long long *inner_sweeps;
  long long *coordinate_updates;
  double *objective;
  double *kkt;
  double *stationarity;
  double *smooth_nll;
};

bool multinomial_c_api_checked_multiply(std::size_t left,
                                        std::size_t right,
                                        std::size_t *product) {
  if (product == nullptr) return false;
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  *product = left * right;
  return true;
}

bool multinomial_c_api_addressable(std::size_t count) {
  return count <= static_cast<std::size_t>(
                      std::numeric_limits<std::ptrdiff_t>::max());
}

bool multinomial_c_api_valid_solution(const Eigen::MatrixXd &beta,
                                      const Eigen::VectorXd &intercept,
                                      int d, int num_classes) {
  return beta.rows() == d && beta.cols() == num_classes &&
         intercept.size() == num_classes && beta.allFinite() &&
         intercept.allFinite();
}

int multinomial_c_api_nonzero_count(const Eigen::MatrixXd &beta) {
  int count = 0;
  for (Eigen::Index feature = 0; feature < beta.rows(); ++feature) {
    for (Eigen::Index klass = 0; klass < beta.cols(); ++klass) {
      if (std::fabs(beta(feature, klass)) > 1e-8) ++count;
    }
  }
  return count;
}

// Match glmnet's multinomial path-termination defaults and statistic.  Every
// returned point is still solved to its requested KKT tolerance; these checks
// only avoid an uninformative, saturated path tail when V4 opts in.
const int kMultinomialMinimumLambdaCount = 5;
const double kMultinomialMaximumDevianceRatio = 0.999;
const double kMultinomialMinimumDevianceRatioGain = 1e-5;

double multinomial_c_api_null_negative_log_likelihood(
    const Eigen::VectorXi &labels, int num_classes, bool include_intercept) {
  if (!include_intercept) return std::log(static_cast<double>(num_classes));

  Eigen::VectorXi counts = Eigen::VectorXi::Zero(num_classes);
  for (Eigen::Index observation = 0; observation < labels.size();
       ++observation)
    ++counts[labels[observation]];
  const double inverse_n = 1.0 / static_cast<double>(labels.size());
  double value = 0.0;
  for (int klass = 0; klass < num_classes; ++klass) {
    if (counts[klass] == 0) continue;
    const double proportion = static_cast<double>(counts[klass]) * inverse_n;
    value -= proportion * std::log(proportion);
  }
  return value;
}

bool multinomial_c_api_nonnegative_difference(double total_objective,
                                              double penalty,
                                              double *difference) {
  if (difference == nullptr) return false;
  *difference = std::numeric_limits<double>::quiet_NaN();
  if (!std::isfinite(total_objective) || !std::isfinite(penalty))
    return false;

  const double scale = std::max(
      1.0, std::max(std::fabs(total_objective), std::fabs(penalty)));
  const double allowance =
      64.0 * std::numeric_limits<double>::epsilon() *
      scale;
  if (total_objective < -allowance || penalty < -allowance)
    return false;

  const double value = total_objective - penalty;
  if (!std::isfinite(value) || value < -allowance) return false;
  *difference = std::max(0.0, value);
  return std::isfinite(*difference);
}

bool multinomial_c_api_l1_negative_log_likelihood(
    const picasso::solver::MultinomialActNewtonResult &result,
    double lambda, double *negative_log_likelihood) {
  const double absolute_sum = result.beta.array().abs().sum();
  if (!std::isfinite(absolute_sum)) return false;
  const double penalty = lambda * absolute_sum;
  return multinomial_c_api_nonnegative_difference(
      result.final_objective, penalty, negative_log_likelihood);
}

bool multinomial_c_api_lla_negative_log_likelihood(
    const picasso::solver::MultinomialLlaResult &result,
    picasso::solver::MultinomialLlaPenalty penalty, double lambda,
    double gamma, double *negative_log_likelihood) {
  double penalty_sum = 0.0;
  for (Eigen::Index feature = 0; feature < result.beta.rows(); ++feature) {
    for (Eigen::Index klass = 0; klass < result.beta.cols(); ++klass) {
      const double penalty_value =
          picasso::solver::multinomial_lla_penalty_value(
              penalty, std::fabs(result.beta(feature, klass)), lambda,
              gamma);
      if (!std::isfinite(penalty_value)) return false;
      penalty_sum += penalty_value;
      if (!std::isfinite(penalty_sum)) return false;
    }
  }
  return multinomial_c_api_nonnegative_difference(
      result.final_target_objective, penalty_sum,
      negative_log_likelihood);
}

bool multinomial_c_api_path_should_stop(
    double negative_log_likelihood, double null_negative_log_likelihood,
    int number_fit, double *previous_deviance_ratio) {
  if (previous_deviance_ratio == nullptr) return false;
  if (!std::isfinite(negative_log_likelihood) ||
      !(null_negative_log_likelihood > 0.0) ||
      !std::isfinite(null_negative_log_likelihood)) {
    *previous_deviance_ratio =
        std::numeric_limits<double>::quiet_NaN();
    return false;
  }

  const double deviance_ratio =
      1.0 - negative_log_likelihood / null_negative_log_likelihood;
  const double previous = *previous_deviance_ratio;
  *previous_deviance_ratio = deviance_ratio;
  if (number_fit < kMultinomialMinimumLambdaCount ||
      !std::isfinite(deviance_ratio))
    return false;
  if (deviance_ratio > kMultinomialMaximumDevianceRatio) return true;
  return std::isfinite(previous) &&
         deviance_ratio - previous < kMultinomialMinimumDevianceRatioGain;
}

bool multinomial_c_api_complete_diagnostics(
    const MultinomialCapiDiagnostics *diagnostics) {
  return diagnostics == nullptr ||
         (diagnostics->failed_lambda != nullptr &&
          diagnostics->failed_stage != nullptr &&
          diagnostics->outer_iterations != nullptr &&
          diagnostics->inner_sweeps != nullptr &&
          diagnostics->coordinate_updates != nullptr &&
          diagnostics->objective != nullptr && diagnostics->kkt != nullptr &&
          diagnostics->stationarity != nullptr);
}

void multinomial_c_api_initialize_diagnostics(
    std::size_t path_size, MultinomialCapiDiagnostics *diagnostics) {
  if (diagnostics == nullptr) return;
  if (diagnostics->failed_lambda != nullptr)
    *diagnostics->failed_lambda = -1;
  if (diagnostics->failed_stage != nullptr) *diagnostics->failed_stage = -1;
  const double missing = std::numeric_limits<double>::quiet_NaN();
  if (diagnostics->outer_iterations != nullptr)
    std::fill_n(diagnostics->outer_iterations, path_size, 0);
  if (diagnostics->inner_sweeps != nullptr)
    std::fill_n(diagnostics->inner_sweeps, path_size, 0LL);
  if (diagnostics->coordinate_updates != nullptr)
    std::fill_n(diagnostics->coordinate_updates, path_size, 0LL);
  if (diagnostics->objective != nullptr)
    std::fill_n(diagnostics->objective, path_size, missing);
  if (diagnostics->kkt != nullptr)
    std::fill_n(diagnostics->kkt, path_size, missing);
  if (diagnostics->stationarity != nullptr)
    std::fill_n(diagnostics->stationarity, path_size, missing);
  if (diagnostics->smooth_nll != nullptr)
    std::fill_n(diagnostics->smooth_nll, path_size, missing);
}

int multinomial_c_api_solver_status(
    picasso::solver::MultinomialSolverStatus status) {
  using picasso::solver::MultinomialSolverStatus;
  switch (status) {
    case MultinomialSolverStatus::kConverged:
      return PICASSO_MULTINOMIAL_COMPLETED;
    case MultinomialSolverStatus::kOuterIterationLimit:
      return PICASSO_MULTINOMIAL_OUTER_ITERATION_LIMIT;
    case MultinomialSolverStatus::kInnerIterationLimit:
      return PICASSO_MULTINOMIAL_INNER_ITERATION_LIMIT;
    case MultinomialSolverStatus::kLineSearchFailed:
      return PICASSO_MULTINOMIAL_LINE_SEARCH_FAILED;
    case MultinomialSolverStatus::kNoDescentDirection:
      return PICASSO_MULTINOMIAL_NO_DESCENT_DIRECTION;
    case MultinomialSolverStatus::kNumericalFailure:
      return PICASSO_MULTINOMIAL_NUMERICAL_FAILURE;
  }
  return PICASSO_MULTINOMIAL_NUMERICAL_FAILURE;
}

double multinomial_c_api_lla_kkt(
    const picasso::solver::MultinomialLlaResult &result) {
  if (result.stages.empty()) return std::numeric_limits<double>::quiet_NaN();
  return result.stages.back().subproblem_kkt_residual;
}

int multinomial_c_api_lla_status(
    const picasso::solver::MultinomialLlaResult &result) {
  using picasso::solver::MultinomialLlaStatus;
  if (result.status == MultinomialLlaStatus::kCompleted)
    return PICASSO_MULTINOMIAL_COMPLETED;
  if (result.status == MultinomialLlaStatus::kStationarityLimit)
    return PICASSO_MULTINOMIAL_LLA_STATIONARITY_LIMIT;
  if (result.status == MultinomialLlaStatus::kMajorizationFailed)
    return PICASSO_MULTINOMIAL_LLA_MAJORIZATION_FAILED;
  if (result.status == MultinomialLlaStatus::kNumericalFailure)
    return PICASSO_MULTINOMIAL_NUMERICAL_FAILURE;
  if (!result.stages.empty())
    return multinomial_c_api_solver_status(
        result.stages.back().subproblem_status);
  return PICASSO_MULTINOMIAL_NUMERICAL_FAILURE;
}

void multinomial_c_api_record_actnewton(
    int lambda_index,
    const picasso::solver::MultinomialActNewtonResult &result,
    MultinomialCapiDiagnostics *diagnostics) {
  if (diagnostics == nullptr) return;
  diagnostics->outer_iterations[lambda_index] = result.outer_iterations;
  diagnostics->inner_sweeps[lambda_index] =
      static_cast<long long>(result.total_inner_sweeps);
  diagnostics->coordinate_updates[lambda_index] =
      result.total_coordinate_updates;
  diagnostics->objective[lambda_index] = result.final_objective;
  diagnostics->kkt[lambda_index] = result.final_kkt_residual;
  diagnostics->stationarity[lambda_index] = result.final_kkt_residual;
}

void multinomial_c_api_record_lla(
    int lambda_index, const picasso::solver::MultinomialLlaResult &result,
    MultinomialCapiDiagnostics *diagnostics) {
  if (diagnostics == nullptr) return;
  diagnostics->outer_iterations[lambda_index] =
      result.total_outer_iterations;
  diagnostics->inner_sweeps[lambda_index] =
      static_cast<long long>(result.total_inner_sweeps);
  diagnostics->coordinate_updates[lambda_index] =
      result.total_coordinate_updates;
  diagnostics->objective[lambda_index] = result.final_target_objective;
  diagnostics->kkt[lambda_index] = multinomial_c_api_lla_kkt(result);
  diagnostics->stationarity[lambda_index] =
      result.final_target_stationarity;
}

void multinomial_c_api_write_solution(
    int lambda_index, int d, int num_classes,
    std::size_t coefficient_count, const Eigen::MatrixXd &beta,
    const Eigen::VectorXd &intercept, int inner_sweeps, int nonzero_count,
    double runtime_seconds, double *beta_out, double *intercept_out,
    int *iterations_out, int *active_size_out, double *runtime_out,
    int *num_fit) {
  const std::size_t beta_offset =
      static_cast<std::size_t>(lambda_index) * coefficient_count;
  const std::size_t intercept_offset =
      static_cast<std::size_t>(lambda_index) *
      static_cast<std::size_t>(num_classes);
  for (int klass = 0; klass < num_classes; ++klass) {
    for (int feature = 0; feature < d; ++feature) {
      beta_out[beta_offset + static_cast<std::size_t>(klass) *
                                   static_cast<std::size_t>(d) +
               static_cast<std::size_t>(feature)] = beta(feature, klass);
    }
    intercept_out[intercept_offset + static_cast<std::size_t>(klass)] =
        intercept[klass];
  }
  iterations_out[lambda_index] = inner_sweeps;
  active_size_out[lambda_index] = nonzero_count;
  runtime_out[lambda_index] = runtime_seconds;
  *num_fit = lambda_index + 1;
}

int solve_multinomial_regression_impl(
    double *Y_int, double *X, int n, int d, int K, double *lambda,
    int nlambda, double gamma, int max_ite, double pprec, int reg_type,
    bool intercept_flag, int dfmax, double *beta_out, double *intcpt_out,
    int *ite_lamb, int *size_act, double *runt, int *num_fit,
    bool use_python, int lla_max_stages, bool path_early_stop,
    MultinomialCapiDiagnostics *diagnostics) {
  if (num_fit != nullptr) *num_fit = 0;
  int current_lambda = -1;

  try {
    const bool path_size_safe = nlambda > 0;
    const std::size_t path_size =
        path_size_safe ? static_cast<std::size_t>(nlambda) : 0;
    multinomial_c_api_initialize_diagnostics(path_size, diagnostics);

    std::size_t coefficient_count = 0;
    const bool coefficient_count_safe =
        d > 0 && K > 0 &&
        multinomial_c_api_checked_multiply(
            static_cast<std::size_t>(d), static_cast<std::size_t>(K),
            &coefficient_count) &&
        multinomial_c_api_addressable(coefficient_count) &&
        coefficient_count <=
            static_cast<std::size_t>(std::numeric_limits<int>::max());

    std::size_t beta_output_count = 0;
    const bool beta_output_count_safe =
        coefficient_count_safe && path_size_safe &&
        multinomial_c_api_checked_multiply(
            coefficient_count, path_size, &beta_output_count) &&
        multinomial_c_api_addressable(beta_output_count) &&
        beta_output_count <=
            static_cast<std::size_t>(std::numeric_limits<int>::max());

    std::size_t intercept_output_count = 0;
    const bool intercept_output_count_safe =
        K > 0 && path_size_safe &&
        multinomial_c_api_checked_multiply(
            static_cast<std::size_t>(K), path_size,
            &intercept_output_count) &&
        multinomial_c_api_addressable(intercept_output_count) &&
        intercept_output_count <=
            static_cast<std::size_t>(std::numeric_limits<int>::max());

    if (beta_out != nullptr && beta_output_count_safe)
      std::fill_n(beta_out, beta_output_count, 0.0);
    if (intcpt_out != nullptr && intercept_output_count_safe)
      std::fill_n(intcpt_out, intercept_output_count, 0.0);
    if (path_size_safe) {
      if (ite_lamb != nullptr) std::fill_n(ite_lamb, path_size, 0);
      if (size_act != nullptr) std::fill_n(size_act, path_size, 0);
      if (runt != nullptr) std::fill_n(runt, path_size, 0.0);
    }

    if (Y_int == nullptr || X == nullptr || lambda == nullptr ||
        beta_out == nullptr || intcpt_out == nullptr ||
        ite_lamb == nullptr || size_act == nullptr || runt == nullptr ||
        num_fit == nullptr ||
        !multinomial_c_api_complete_diagnostics(diagnostics))
      return PICASSO_MULTINOMIAL_INVALID_INPUT;
    if (n <= 0 || d <= 0 || K < 2 || nlambda <= 0 || max_ite <= 0 ||
        lla_max_stages < 3)
      return PICASSO_MULTINOMIAL_INVALID_INPUT;
    if (dfmax < -1)
      return PICASSO_MULTINOMIAL_INVALID_INPUT;
    if (!std::isfinite(pprec) || !(pprec > 0.0))
      return PICASSO_MULTINOMIAL_INVALID_INPUT;
    if (reg_type != 1 && reg_type != 2 && reg_type != 3)
      return PICASSO_MULTINOMIAL_INVALID_INPUT;
    if (reg_type == 2 &&
        (!std::isfinite(gamma) || !(gamma > 1.0)))
      return PICASSO_MULTINOMIAL_INVALID_INPUT;
    if (reg_type == 3 &&
        (!std::isfinite(gamma) || !(gamma > 2.0)))
      return PICASSO_MULTINOMIAL_INVALID_INPUT;
    if (!coefficient_count_safe || !beta_output_count_safe ||
        !intercept_output_count_safe)
      return PICASSO_MULTINOMIAL_INVALID_INPUT;

    std::size_t design_count = 0;
    std::size_t probability_count = 0;
    if (!multinomial_c_api_checked_multiply(
            static_cast<std::size_t>(n), static_cast<std::size_t>(d),
            &design_count) ||
        !multinomial_c_api_checked_multiply(
            static_cast<std::size_t>(n), static_cast<std::size_t>(K),
            &probability_count) ||
        !multinomial_c_api_addressable(design_count) ||
        !multinomial_c_api_addressable(probability_count) ||
        design_count >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        probability_count >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        design_count >
            static_cast<std::size_t>(
                std::numeric_limits<Eigen::Index>::max()) ||
        probability_count >
            static_cast<std::size_t>(
                std::numeric_limits<Eigen::Index>::max()) ||
        coefficient_count >
            static_cast<std::size_t>(
                std::numeric_limits<Eigen::Index>::max()))
      return PICASSO_MULTINOMIAL_INVALID_INPUT;

    for (int index = 0; index < nlambda; ++index) {
      if (!std::isfinite(lambda[index]) || !(lambda[index] >= 0.0))
        return PICASSO_MULTINOMIAL_INVALID_INPUT;
      if (index > 0 && !(lambda[index] < lambda[index - 1]))
        return PICASSO_MULTINOMIAL_INVALID_INPUT;
    }

    Eigen::VectorXi labels(n);
    for (int observation = 0; observation < n; ++observation) {
      const double label = Y_int[observation];
      if (!std::isfinite(label) || label != std::floor(label) ||
          !(label >= 0.0) || !(label < static_cast<double>(K)))
        return PICASSO_MULTINOMIAL_INVALID_INPUT;
      labels[observation] = static_cast<int>(label);
    }

    typedef picasso::detail::MultinomialProblemView MultinomialProblemView;
    const bool borrow_column_major_design =
        !use_python &&
        MultinomialProblemView::design_pointer_is_aligned(X);
    Eigen::MatrixXd owned_design;
    if (use_python) {
      owned_design.resize(n, d);
      for (int observation = 0; observation < n; ++observation) {
        for (int feature = 0; feature < d; ++feature) {
          const std::size_t index =
              static_cast<std::size_t>(observation) *
                  static_cast<std::size_t>(d) +
              static_cast<std::size_t>(feature);
          const double value = X[index];
          if (!std::isfinite(value))
            return PICASSO_MULTINOMIAL_INVALID_INPUT;
          owned_design(observation, feature) = value;
        }
      }
    } else {
      if (!borrow_column_major_design) owned_design.resize(n, d);
      for (int feature = 0; feature < d; ++feature) {
        for (int observation = 0; observation < n; ++observation) {
          const std::size_t index =
              static_cast<std::size_t>(feature) *
                  static_cast<std::size_t>(n) +
              static_cast<std::size_t>(observation);
          const double value = X[index];
          if (!std::isfinite(value))
            return PICASSO_MULTINOMIAL_INVALID_INPUT;
          if (!borrow_column_major_design)
            owned_design(observation, feature) = value;
        }
      }
    }

    std::unique_ptr<MultinomialProblemView> problem;
    if (borrow_column_major_design) {
      problem.reset(new MultinomialProblemView(
          X, n, d, labels, K, static_cast<const void *>(X)));
    } else {
      problem.reset(new MultinomialProblemView(
          owned_design, labels, K, static_cast<const void *>(&owned_design)));
    }
    const double null_negative_log_likelihood =
        path_early_stop
            ? multinomial_c_api_null_negative_log_likelihood(
                  problem->labels(), K, intercept_flag)
            : std::numeric_limits<double>::quiet_NaN();
    double previous_deviance_ratio =
        std::numeric_limits<double>::quiet_NaN();
    picasso::solver::MultinomialActNewtonOptions options;
    options.include_intercept = intercept_flag;
    options.max_outer_iterations = max_ite;
    options.max_inner_sweeps = max_ite;
    options.outer_kkt_tolerance = pprec;
    const double inner_floor =
        100.0 * std::numeric_limits<double>::epsilon();
    options.inner_kkt_tolerance =
        std::min(pprec, std::max(inner_floor, 0.01 * pprec));
    options.use_adaptive_inner_tolerance = true;
    options.use_vectorized_coordinate_kernels = true;
    options.reuse_line_search_probabilities = true;
    options.use_compact_inner_active_set = true;

    if (reg_type == 1) {
      picasso::solver::internal::MultinomialPathViewState path_state;
      for (int lambda_index = 0; lambda_index < nlambda; ++lambda_index) {
        if (picasso::solver::interrupt_requested())
          return PICASSO_MULTINOMIAL_INTERRUPTED;
        current_lambda = lambda_index;
        const std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();
        picasso::solver::MultinomialActNewtonPathResult path_result =
            picasso::solver::internal::solve_multinomial_actnewton_path_view(
                *problem, options, lambda[lambda_index], &path_state);
        picasso::solver::MultinomialActNewtonResult result =
            std::move(path_result.solution);
        const double elapsed =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start)
                .count();
        multinomial_c_api_record_actnewton(lambda_index, result, diagnostics);
        if (diagnostics != nullptr) runt[lambda_index] = elapsed;
        if (!result.converged() ||
            !multinomial_c_api_valid_solution(
                result.beta, result.intercept, d, K)) {
          if (diagnostics != nullptr)
            *diagnostics->failed_lambda = lambda_index;
          return result.converged()
                     ? PICASSO_MULTINOMIAL_NUMERICAL_FAILURE
                     : multinomial_c_api_solver_status(result.status);
        }
        const int nonzero_count =
            multinomial_c_api_nonzero_count(result.beta);
        double smooth_nll = std::numeric_limits<double>::quiet_NaN();
        if (!multinomial_c_api_l1_negative_log_likelihood(
                result, lambda[lambda_index], &smooth_nll)) {
          if (diagnostics != nullptr) {
            *diagnostics->failed_lambda = lambda_index;
            *diagnostics->failed_stage = -1;
          }
          return PICASSO_MULTINOMIAL_NUMERICAL_FAILURE;
        }
        if (diagnostics != nullptr && diagnostics->smooth_nll != nullptr)
          diagnostics->smooth_nll[lambda_index] = smooth_nll;
        multinomial_c_api_write_solution(
            lambda_index, d, K, coefficient_count, result.beta,
            result.intercept, result.total_inner_sweeps, nonzero_count,
            diagnostics == nullptr ? 0.0 : elapsed, beta_out, intcpt_out,
            ite_lamb, size_act, runt, num_fit);
        if (dfmax >= 0 && nonzero_count > dfmax)
          return PICASSO_MULTINOMIAL_DFMAX_REACHED;
        if (path_early_stop && multinomial_c_api_path_should_stop(
                smooth_nll,
                null_negative_log_likelihood, lambda_index + 1,
                &previous_deviance_ratio))
          return PICASSO_MULTINOMIAL_COMPLETED;
      }
      return PICASSO_MULTINOMIAL_COMPLETED;
    }

    const picasso::solver::MultinomialLlaPenalty penalty =
        reg_type == 2
            ? picasso::solver::MultinomialLlaPenalty::kMCP
            : picasso::solver::MultinomialLlaPenalty::kSCAD;
    picasso::solver::internal::MultinomialPathViewState master_state;
    picasso::solver::MultinomialLlaOptions lla_options;
    lla_options.maximum_stages = lla_max_stages;
    int path_status = PICASSO_MULTINOMIAL_COMPLETED;
    for (int lambda_index = 0; lambda_index < nlambda; ++lambda_index) {
      if (picasso::solver::interrupt_requested())
        return PICASSO_MULTINOMIAL_INTERRUPTED;
      current_lambda = lambda_index;
      const std::chrono::steady_clock::time_point start =
          std::chrono::steady_clock::now();
      // PathSolver commits atomically. This state is local to the C call, and
      // every later LLA/diagnostic failure returns immediately, so a second
      // d-by-K rollback copy would have no observable effect.
      picasso::solver::MultinomialActNewtonPathResult master_path_result =
          picasso::solver::internal::solve_multinomial_actnewton_path_view(
              *problem, options, lambda[lambda_index], &master_state);
      picasso::solver::MultinomialLlaResult result =
          picasso::solver::internal::
              solve_multinomial_lla_from_l1_master_view(
                  *problem, options, lla_options, penalty,
                  lambda[lambda_index], gamma,
                  std::move(master_path_result.solution));
      const double elapsed =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - start)
              .count();
      multinomial_c_api_record_lla(lambda_index, result, diagnostics);
      if (diagnostics != nullptr) runt[lambda_index] = elapsed;
      if (!result.has_valid_solution() ||
          !multinomial_c_api_valid_solution(
              result.beta, result.intercept, d, K) ||
          !multinomial_c_api_valid_solution(
              result.l1_master_beta, result.l1_master_intercept, d, K)) {
        if (diagnostics != nullptr) {
          *diagnostics->failed_lambda = lambda_index;
          *diagnostics->failed_stage = result.failed_stage;
        }
        if (result.has_valid_solution())
          return PICASSO_MULTINOMIAL_NUMERICAL_FAILURE;
        return multinomial_c_api_lla_status(result);
      }
      const int nonzero_count =
          multinomial_c_api_nonzero_count(result.beta);
      double smooth_nll = std::numeric_limits<double>::quiet_NaN();
      if (!multinomial_c_api_lla_negative_log_likelihood(
              result, penalty, lambda[lambda_index], gamma,
              &smooth_nll)) {
        if (diagnostics != nullptr) {
          *diagnostics->failed_lambda = lambda_index;
          *diagnostics->failed_stage = result.failed_stage;
        }
        return PICASSO_MULTINOMIAL_NUMERICAL_FAILURE;
      }
      if (diagnostics != nullptr && diagnostics->smooth_nll != nullptr)
        diagnostics->smooth_nll[lambda_index] = smooth_nll;
      multinomial_c_api_write_solution(
          lambda_index, d, K, coefficient_count, result.beta,
          result.intercept, result.total_inner_sweeps, nonzero_count,
          diagnostics == nullptr ? 0.0 : elapsed, beta_out, intcpt_out,
          ite_lamb, size_act, runt, num_fit);
      if (!result.completed())
        path_status = PICASSO_MULTINOMIAL_LLA_STATIONARITY_LIMIT;
      if (dfmax >= 0 && nonzero_count > dfmax)
        return PICASSO_MULTINOMIAL_DFMAX_REACHED;
      if (path_early_stop && multinomial_c_api_path_should_stop(
              smooth_nll,
              null_negative_log_likelihood, lambda_index + 1,
              &previous_deviance_ratio))
        return path_status;
    }
    return path_status;
  } catch (...) {
    if (diagnostics != nullptr && diagnostics->failed_lambda != nullptr)
      *diagnostics->failed_lambda = current_lambda;
    return PICASSO_MULTINOMIAL_EXCEPTION;
  }
}

}  // namespace

extern "C" const char *PicassoMultinomialPathStatusString(int status) {
  switch (status) {
    case PICASSO_MULTINOMIAL_COMPLETED:
      return "completed";
    case PICASSO_MULTINOMIAL_DFMAX_REACHED:
      return "dfmax_reached";
    case PICASSO_MULTINOMIAL_INVALID_INPUT:
      return "invalid_input";
    case PICASSO_MULTINOMIAL_OUTER_ITERATION_LIMIT:
      return "outer_iteration_limit";
    case PICASSO_MULTINOMIAL_INNER_ITERATION_LIMIT:
      return "inner_iteration_limit";
    case PICASSO_MULTINOMIAL_LINE_SEARCH_FAILED:
      return "line_search_failed";
    case PICASSO_MULTINOMIAL_NO_DESCENT_DIRECTION:
      return "no_descent_direction";
    case PICASSO_MULTINOMIAL_NUMERICAL_FAILURE:
      return "numerical_failure";
    case PICASSO_MULTINOMIAL_LLA_MAJORIZATION_FAILED:
      return "lla_majorization_failed";
    case PICASSO_MULTINOMIAL_EXCEPTION:
      return "exception";
    case PICASSO_MULTINOMIAL_LLA_STATIONARITY_LIMIT:
      return "lla_stationarity_limit";
    case PICASSO_MULTINOMIAL_INTERRUPTED:
      return "interrupted";
  }
  return "unknown";
}

extern "C" void SolveMultinomialRegression(
    double *Y_int, double *X, int n, int d, int K, double *lambda,
    int nlambda, double gamma, int max_ite, double pprec, int reg_type,
    bool intercept_flag, int dfmax, double *beta_out, double *intcpt_out,
    int *ite_lamb, int *size_act, double *runt, int *num_fit,
    bool use_python) {
  solve_multinomial_regression_impl(
      Y_int, X, n, d, K, lambda, nlambda, gamma, max_ite, pprec,
      reg_type, intercept_flag, dfmax, beta_out, intcpt_out, ite_lamb,
      size_act, runt, num_fit, use_python,
      picasso::solver::MultinomialLlaOptions().maximum_stages, false,
      nullptr);
}

extern "C" int SolveMultinomialRegressionV2(
    double *Y_int, double *X, int n, int d, int K, double *lambda,
    int nlambda, double gamma, int max_ite, double pprec, int reg_type,
    bool intercept_flag, int dfmax, double *beta_out, double *intcpt_out,
    int *ite_lamb, int *size_act, double *runt, int *num_fit,
    bool use_python, int *failed_lambda, int *failed_stage,
    int *outer_ite, long long *inner_sweeps,
    long long *coordinate_updates, double *objective, double *kkt,
    double *stationarity) {
  MultinomialCapiDiagnostics diagnostics;
  diagnostics.failed_lambda = failed_lambda;
  diagnostics.failed_stage = failed_stage;
  diagnostics.outer_iterations = outer_ite;
  diagnostics.inner_sweeps = inner_sweeps;
  diagnostics.coordinate_updates = coordinate_updates;
  diagnostics.objective = objective;
  diagnostics.kkt = kkt;
  diagnostics.stationarity = stationarity;
  diagnostics.smooth_nll = nullptr;
  return solve_multinomial_regression_impl(
      Y_int, X, n, d, K, lambda, nlambda, gamma, max_ite, pprec,
      reg_type, intercept_flag, dfmax, beta_out, intcpt_out, ite_lamb,
      size_act, runt, num_fit, use_python,
      picasso::solver::MultinomialLlaOptions().maximum_stages,
      false, &diagnostics);
}

extern "C" int SolveMultinomialRegressionV3(
    double *Y_int, double *X, int n, int d, int K, double *lambda,
    int nlambda, double gamma, int max_ite, double pprec, int reg_type,
    bool intercept_flag, int dfmax, double *beta_out, double *intcpt_out,
    int *ite_lamb, int *size_act, double *runt, int *num_fit,
    bool use_python, int lla_max_stages, int *failed_lambda,
    int *failed_stage, int *outer_ite, long long *inner_sweeps,
    long long *coordinate_updates, double *objective, double *kkt,
    double *stationarity) {
  MultinomialCapiDiagnostics diagnostics;
  diagnostics.failed_lambda = failed_lambda;
  diagnostics.failed_stage = failed_stage;
  diagnostics.outer_iterations = outer_ite;
  diagnostics.inner_sweeps = inner_sweeps;
  diagnostics.coordinate_updates = coordinate_updates;
  diagnostics.objective = objective;
  diagnostics.kkt = kkt;
  diagnostics.stationarity = stationarity;
  diagnostics.smooth_nll = nullptr;
  return solve_multinomial_regression_impl(
      Y_int, X, n, d, K, lambda, nlambda, gamma, max_ite, pprec,
      reg_type, intercept_flag, dfmax, beta_out, intcpt_out, ite_lamb,
      size_act, runt, num_fit, use_python, lla_max_stages, false,
      &diagnostics);
}

extern "C" int SolveMultinomialRegressionV4(
    double *Y_int, double *X, int n, int d, int K, double *lambda,
    int nlambda, double gamma, int max_ite, double pprec, int reg_type,
    bool intercept_flag, int dfmax, double *beta_out, double *intcpt_out,
    int *ite_lamb, int *size_act, double *runt, int *num_fit,
    bool use_python, int lla_max_stages, bool path_early_stop,
    int *failed_lambda, int *failed_stage, int *outer_ite,
    long long *inner_sweeps, long long *coordinate_updates,
    double *objective, double *kkt, double *stationarity) {
  MultinomialCapiDiagnostics diagnostics;
  diagnostics.failed_lambda = failed_lambda;
  diagnostics.failed_stage = failed_stage;
  diagnostics.outer_iterations = outer_ite;
  diagnostics.inner_sweeps = inner_sweeps;
  diagnostics.coordinate_updates = coordinate_updates;
  diagnostics.objective = objective;
  diagnostics.kkt = kkt;
  diagnostics.stationarity = stationarity;
  diagnostics.smooth_nll = nullptr;
  return solve_multinomial_regression_impl(
      Y_int, X, n, d, K, lambda, nlambda, gamma, max_ite, pprec,
      reg_type, intercept_flag, dfmax, beta_out, intcpt_out, ite_lamb,
      size_act, runt, num_fit, use_python, lla_max_stages,
      path_early_stop, &diagnostics);
}

extern "C" int SolveMultinomialRegressionV5(
    double *Y_int, double *X, int n, int d, int K, double *lambda,
    int nlambda, double gamma, int max_ite, double pprec, int reg_type,
    bool intercept_flag, int dfmax, double *beta_out, double *intcpt_out,
    int *ite_lamb, int *size_act, double *runt, int *num_fit,
    bool use_python, int lla_max_stages, bool path_early_stop,
    int *failed_lambda, int *failed_stage, int *outer_ite,
    long long *inner_sweeps, long long *coordinate_updates,
    double *objective, double *kkt, double *stationarity,
    double *smooth_nll) {
  MultinomialCapiDiagnostics diagnostics;
  diagnostics.failed_lambda = failed_lambda;
  diagnostics.failed_stage = failed_stage;
  diagnostics.outer_iterations = outer_ite;
  diagnostics.inner_sweeps = inner_sweeps;
  diagnostics.coordinate_updates = coordinate_updates;
  diagnostics.objective = objective;
  diagnostics.kkt = kkt;
  diagnostics.stationarity = stationarity;
  diagnostics.smooth_nll = smooth_nll;
  return solve_multinomial_regression_impl(
      Y_int, X, n, d, K, lambda, nlambda, gamma, max_ite, pprec,
      reg_type, intercept_flag, dfmax, beta_out, intcpt_out, ite_lamb,
      size_act, runt, num_fit, use_python, lla_max_stages,
      path_early_stop, &diagnostics);
}
