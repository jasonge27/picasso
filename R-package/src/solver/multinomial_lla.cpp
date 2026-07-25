#include <picasso/multinomial_lla.hpp>

#include "../internal/multinomial_solver_view.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace picasso {
namespace solver {
namespace {

typedef ::picasso::detail::MultinomialProblemView MultinomialProblemView;

MultinomialProblemView lla_problem_view(
    const MultinomialObjective &objective) {
  return MultinomialProblemView(
      objective.design_matrix(), objective.labels(), objective.class_num(),
      &objective);
}

void validate_lla_parameters(MultinomialLlaPenalty penalty, double lambda,
                             double gamma) {
  if (!(lambda >= 0.0) || !std::isfinite(lambda))
    throw std::invalid_argument(
        "multinomial LLA lambda must be finite and nonnegative");
  if (!std::isfinite(gamma))
    throw std::invalid_argument("multinomial LLA gamma must be finite");
  switch (penalty) {
    case MultinomialLlaPenalty::kMCP:
      if (!(gamma > 1.0))
        throw std::invalid_argument("multinomial MCP gamma must exceed one");
      return;
    case MultinomialLlaPenalty::kSCAD:
      if (!(gamma > 2.0))
        throw std::invalid_argument("multinomial SCAD gamma must exceed two");
      return;
  }
  throw std::invalid_argument("unknown multinomial LLA penalty");
}

double lla_penalty_derivative_unchecked(MultinomialLlaPenalty penalty,
                                        double absolute_value,
                                        double lambda, double gamma) {
  if (lambda == 0.0) return 0.0;
  if (penalty == MultinomialLlaPenalty::kMCP)
    return std::max(0.0, lambda - absolute_value / gamma);
  if (absolute_value <= lambda) return lambda;
  return std::max(
      0.0, lambda - (absolute_value - lambda) / (gamma - 1.0));
}

double lla_penalty_value_unchecked(MultinomialLlaPenalty penalty,
                                   double absolute_value, double lambda,
                                   double gamma) {
  if (lambda == 0.0 || absolute_value == 0.0) return 0.0;
  if (penalty == MultinomialLlaPenalty::kMCP) {
    const double scaled_value = absolute_value / gamma;
    if (scaled_value < lambda)
      return absolute_value * (lambda - 0.5 * scaled_value);
    return 0.5 * gamma * lambda * lambda;
  }
  if (absolute_value <= lambda) return lambda * absolute_value;
  const double offset = absolute_value - lambda;
  const double derivative_drop = offset / (gamma - 1.0);
  if (derivative_drop < lambda)
    return lambda * lambda +
           offset * (lambda - 0.5 * derivative_drop);
  return 0.5 * (gamma + 1.0) * lambda * lambda;
}

Eigen::VectorXd lla_empirical_null_intercept(
    const MultinomialProblemView &objective, bool include_intercept) {
  Eigen::VectorXd intercept = Eigen::VectorXd::Zero(objective.class_num());
  if (!include_intercept) return intercept;
  const Eigen::VectorXi &labels = objective.labels();
  for (Eigen::Index observation = 0; observation < labels.size();
       ++observation)
    intercept[labels[observation]] += 1.0;
  intercept.array() /= static_cast<double>(objective.sample_num());
  for (Eigen::Index klass = 0; klass < intercept.size(); ++klass)
    intercept[klass] = std::log(std::max(intercept[klass], 1e-8));
  intercept.array() -= intercept.mean();
  return intercept;
}

bool lla_make_weights_and_penalty(
    const Eigen::MatrixXd &beta, MultinomialLlaPenalty penalty,
    double lambda, double gamma, Eigen::MatrixXd *weights,
    double *penalty_sum, double *weighted_l1_sum,
    double *tangent_constant) {
  weights->resize(beta.rows(), beta.cols());
  double target_sum = 0.0;
  double weighted_sum = 0.0;
  double constant_sum = 0.0;
  for (Eigen::Index feature = 0; feature < beta.rows(); ++feature) {
    for (Eigen::Index klass = 0; klass < beta.cols(); ++klass) {
      const double absolute_value = std::fabs(beta(feature, klass));
      const double weight = lla_penalty_derivative_unchecked(
          penalty, absolute_value, lambda, gamma);
      const double value = lla_penalty_value_unchecked(
          penalty, absolute_value, lambda, gamma);
      (*weights)(feature, klass) = weight;
      target_sum += value;
      weighted_sum += weight * absolute_value;
      constant_sum += value - weight * absolute_value;
    }
  }
  *penalty_sum = target_sum;
  *weighted_l1_sum = weighted_sum;
  *tangent_constant = constant_sum;
  return weights->allFinite() && std::isfinite(target_sum) &&
         std::isfinite(weighted_sum) && std::isfinite(constant_sum);
}

bool lla_target_objective(const MultinomialProblemView &objective,
                          const Eigen::MatrixXd &beta,
                          const Eigen::VectorXd &intercept,
                          MultinomialLlaPenalty penalty, double lambda,
                          double gamma, double *value) {
  double penalty_sum = 0.0;
  for (Eigen::Index feature = 0; feature < beta.rows(); ++feature) {
    for (Eigen::Index klass = 0; klass < beta.cols(); ++klass) {
      penalty_sum += lla_penalty_value_unchecked(
          penalty, std::fabs(beta(feature, klass)), lambda, gamma);
    }
  }
  if (!std::isfinite(penalty_sum)) return false;
  try {
    *value = objective.negative_log_likelihood(beta, intercept) + penalty_sum;
  } catch (const std::invalid_argument &) {
    return false;
  }
  return std::isfinite(*value);
}

double lla_coefficient_stationarity(double coefficient, double gradient,
                                    double derivative,
                                    double zero_tolerance) {
  if (coefficient > zero_tolerance)
    return std::fabs(gradient + derivative);
  if (coefficient < -zero_tolerance)
    return std::fabs(gradient - derivative);
  return std::max(0.0, std::fabs(gradient) - derivative);
}

bool lla_target_stationarity(
    const MultinomialProblemView &objective, const Eigen::MatrixXd &beta,
    const Eigen::VectorXd &intercept, MultinomialLlaPenalty penalty,
    double lambda, double gamma,
    const MultinomialActNewtonOptions &options, double *residual) {
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  try {
    objective.smooth_gradient(beta, intercept, &beta_gradient,
                              &intercept_gradient);
  } catch (const std::invalid_argument &) {
    return false;
  }
  if (!beta_gradient.allFinite() || !intercept_gradient.allFinite())
    return false;
  double maximum = 0.0;
  for (Eigen::Index feature = 0; feature < beta.rows(); ++feature) {
    for (Eigen::Index klass = 0; klass < beta.cols(); ++klass) {
      const double derivative = lla_penalty_derivative_unchecked(
          penalty, std::fabs(beta(feature, klass)), lambda, gamma);
      maximum = std::max(
          maximum,
          lla_coefficient_stationarity(
              beta(feature, klass), beta_gradient(feature, klass), derivative,
              options.zero_tolerance));
    }
  }
  if (options.include_intercept)
    maximum = std::max(maximum,
                       intercept_gradient.cwiseAbs().maxCoeff());
  *residual = maximum;
  return std::isfinite(maximum);
}

MultinomialLlaStageRecord lla_stage_record(
    int stage, bool is_l1_master,
    const MultinomialActNewtonResult &subproblem) {
  MultinomialLlaStageRecord record;
  record.stage = stage;
  record.is_l1_master = is_l1_master;
  record.subproblem_status = subproblem.status;
  record.outer_iterations = subproblem.outer_iterations;
  record.inner_sweeps = subproblem.total_inner_sweeps;
  record.coordinate_updates = subproblem.total_coordinate_updates;
  record.surrogate_objective = subproblem.final_objective;
  record.subproblem_kkt_residual = subproblem.final_kkt_residual;
  record.target_objective = std::numeric_limits<double>::quiet_NaN();
  record.tangent_constant = std::numeric_limits<double>::quiet_NaN();
  record.majorizer_at_anchor = std::numeric_limits<double>::quiet_NaN();
  record.majorizer_at_solution = std::numeric_limits<double>::quiet_NaN();
  record.target_stationarity = std::numeric_limits<double>::infinity();
  return record;
}

void lla_append_stage(const MultinomialLlaStageRecord &record,
                      MultinomialLlaResult *result) {
  result->stages.push_back(record);
  result->total_outer_iterations += record.outer_iterations;
  result->total_inner_sweeps += record.inner_sweeps;
  result->total_coordinate_updates += record.coordinate_updates;
}

MultinomialLlaResult lla_initial_result(
    const Eigen::MatrixXd &fallback_beta,
    const Eigen::VectorXd &fallback_intercept) {
  MultinomialLlaResult result;
  result.beta = fallback_beta;
  result.intercept = fallback_intercept;
  result.status = MultinomialLlaStatus::kSubproblemFailed;
  result.failed_stage = 0;
  result.completed_stages = 0;
  result.total_outer_iterations = 0;
  result.total_inner_sweeps = 0;
  result.total_coordinate_updates = 0;
  result.final_target_objective =
      std::numeric_limits<double>::quiet_NaN();
  result.final_target_stationarity =
      std::numeric_limits<double>::infinity();
  return result;
}

double lla_roundoff_allowance(double configured_tolerance,
                              double first, double second,
                              double third, double fourth) {
  double scale = 1.0;
  scale = std::max(scale, std::fabs(first));
  scale = std::max(scale, std::fabs(second));
  scale = std::max(scale, std::fabs(third));
  scale = std::max(scale, std::fabs(fourth));
  return (configured_tolerance +
          64.0 * std::numeric_limits<double>::epsilon()) * scale;
}

MultinomialActNewtonOptions lla_proximal_newton_options(
    const MultinomialActNewtonOptions &options,
    const MultinomialLlaOptions &lla_options) {
  MultinomialActNewtonOptions effective = options;
  if (lla_options.stopping_rule !=
      MultinomialLlaStoppingRule::kTargetStationarity)
    return effective;

  const double target_tolerance =
      lla_options.stationarity_tolerance == 0.0
          ? options.outer_kkt_tolerance
          : lla_options.stationarity_tolerance;
  if (target_tolerance > 0.0 && std::isfinite(target_tolerance) &&
      options.outer_kkt_tolerance > 0.0 &&
      std::isfinite(options.outer_kkt_tolerance)) {
    // A weighted-L1 stage only needs to be solved as accurately as the two
    // outer certifications can observe. Requiring the historical 0.01*outer
    // inner tolerance can exhaust max_inner_sweeps on dense, nearly singular
    // full-K surrogates without improving target-stationarity certification.
    effective.inner_kkt_tolerance =
        std::min(options.outer_kkt_tolerance, target_tolerance);
  }
  return effective;
}

void validate_and_normalize_lla_options(
    MultinomialActNewtonOptions *proximal_newton_options,
    MultinomialLlaOptions *lla_options) {
  if (proximal_newton_options == 0 || lla_options == 0)
    throw std::invalid_argument("multinomial LLA options are null");
  *proximal_newton_options = lla_proximal_newton_options(
      *proximal_newton_options, *lla_options);
  if (lla_options->stopping_rule !=
          MultinomialLlaStoppingRule::kTargetStationarity &&
      lla_options->stopping_rule !=
          MultinomialLlaStoppingRule::kFixedStages)
    throw std::invalid_argument("unknown multinomial LLA stopping rule");
  if (lla_options->minimum_stages < 3)
    throw std::invalid_argument(
        "multinomial LLA minimum stages must be at least three");
  if (lla_options->maximum_stages < lla_options->minimum_stages)
    throw std::invalid_argument(
        "multinomial LLA maximum stages must not be smaller than the minimum");
  if (lla_options->stopping_rule ==
          MultinomialLlaStoppingRule::kFixedStages &&
      lla_options->maximum_stages != lla_options->minimum_stages)
    throw std::invalid_argument(
        "fixed-stage multinomial LLA requires equal minimum and maximum stages");
  if (lla_options->stationarity_tolerance == 0.0)
    lla_options->stationarity_tolerance =
        proximal_newton_options->outer_kkt_tolerance;
  if (!(lla_options->stationarity_tolerance > 0.0) ||
      !std::isfinite(lla_options->stationarity_tolerance))
    throw std::invalid_argument(
        "multinomial LLA stationarity tolerance must be finite and positive");
  if (!(lla_options->majorization_tolerance >= 0.0) ||
      !std::isfinite(lla_options->majorization_tolerance))
    throw std::invalid_argument(
        "multinomial LLA majorization tolerance must be finite and nonnegative");
}

MultinomialLlaResult lla_complete_from_master(
    const MultinomialProblemView &objective,
    const MultinomialActNewtonOptions &proximal_newton_options,
    const MultinomialLlaOptions &lla_options,
    MultinomialLlaPenalty penalty, double lambda, double gamma,
    MultinomialActNewtonResult master,
    const Eigen::MatrixXd &fallback_beta,
    const Eigen::VectorXd &fallback_intercept) {
  MultinomialLlaResult result =
      lla_initial_result(fallback_beta, fallback_intercept);
  MultinomialLlaStageRecord master_record =
      lla_stage_record(0, true, master);
  lla_append_stage(master_record, &result);

  if (!master.converged()) {
    lla_target_objective(objective, result.beta, result.intercept, penalty,
                         lambda, gamma, &result.final_target_objective);
    lla_target_stationarity(
        objective, result.beta, result.intercept, penalty, lambda, gamma,
        proximal_newton_options, &result.final_target_stationarity);
    return result;
  }

  double current_target = 0.0;
  double current_stationarity = 0.0;
  if (!lla_target_objective(objective, master.beta, master.intercept,
                            penalty, lambda, gamma, &current_target) ||
      !lla_target_stationarity(
          objective, master.beta, master.intercept, penalty, lambda, gamma,
          proximal_newton_options, &current_stationarity)) {
    result.status = MultinomialLlaStatus::kNumericalFailure;
    return result;
  }
  result.stages.back().target_objective = current_target;
  result.stages.back().target_stationarity = current_stationarity;
  std::vector<unsigned char> active_features = master.active_features;
  if (active_features.size() !=
      static_cast<std::size_t>(objective.feature_num()))
    active_features.assign(
        static_cast<std::size_t>(objective.feature_num()), 0);
  result.l1_master_beta.swap(master.beta);
  result.l1_master_intercept.swap(master.intercept);
  std::vector<MultinomialIterationRecord>().swap(master.history);
  result.beta = result.l1_master_beta;
  result.intercept = result.l1_master_intercept;
  result.completed_stages = 1;
  result.failed_stage = -1;
  result.final_target_objective = current_target;
  result.final_target_stationarity = current_stationarity;

  const bool fixed_stage_compatibility =
      lla_options.stopping_rule == MultinomialLlaStoppingRule::kFixedStages;
  const int stage_limit = fixed_stage_compatibility
                              ? lla_options.minimum_stages
                              : lla_options.maximum_stages;
  for (int stage = 1; stage < stage_limit; ++stage) {
    Eigen::MatrixXd weights;
    double anchor_penalty = 0.0;
    double anchor_weighted_l1 = 0.0;
    double tangent_constant = 0.0;
    if (!lla_make_weights_and_penalty(
            result.beta, penalty, lambda, gamma, &weights, &anchor_penalty,
            &anchor_weighted_l1, &tangent_constant)) {
      result.status = MultinomialLlaStatus::kNumericalFailure;
      result.failed_stage = stage;
      return result;
    }

    double anchor_nll = 0.0;
    try {
      anchor_nll = objective.negative_log_likelihood(
          result.beta, result.intercept);
    } catch (const std::invalid_argument &) {
      result.status = MultinomialLlaStatus::kNumericalFailure;
      result.failed_stage = stage;
      return result;
    }
    const double majorizer_anchor =
        anchor_nll + anchor_weighted_l1 + tangent_constant;
    MultinomialActNewtonResult candidate =
        internal::solve_multinomial_actnewton_weighted_view(
            objective, proximal_newton_options, weights, result.beta,
            result.intercept, active_features);
    MultinomialLlaStageRecord record =
        lla_stage_record(stage, false, candidate);
    record.tangent_constant = tangent_constant;
    record.majorizer_at_anchor = majorizer_anchor;

    if (!candidate.converged()) {
      lla_append_stage(record, &result);
      result.status = MultinomialLlaStatus::kSubproblemFailed;
      result.failed_stage = stage;
      return result;
    }

    double candidate_target = 0.0;
    double candidate_stationarity = 0.0;
    const double majorizer_solution =
        candidate.final_objective + tangent_constant;
    record.majorizer_at_solution = majorizer_solution;
    const bool target_is_finite = lla_target_objective(
        objective, candidate.beta, candidate.intercept, penalty, lambda,
        gamma, &candidate_target);
    const bool stationarity_is_finite = lla_target_stationarity(
        objective, candidate.beta, candidate.intercept, penalty, lambda,
        gamma, proximal_newton_options, &candidate_stationarity);
    record.target_objective = candidate_target;
    record.target_stationarity = candidate_stationarity;
    lla_append_stage(record, &result);

    const double start_majorizer =
        candidate.history.empty()
            ? std::numeric_limits<double>::infinity()
            : candidate.history.front().objective + tangent_constant;
    const double allowance = lla_roundoff_allowance(
        lla_options.majorization_tolerance, current_target,
        majorizer_anchor, majorizer_solution, candidate_target);
    if (!target_is_finite || !stationarity_is_finite ||
        !std::isfinite(anchor_nll) || !std::isfinite(anchor_penalty) ||
        !std::isfinite(majorizer_anchor) ||
        !std::isfinite(majorizer_solution) ||
        !std::isfinite(start_majorizer)) {
      result.status = MultinomialLlaStatus::kNumericalFailure;
      result.failed_stage = stage;
      return result;
    }
    if (std::fabs(majorizer_anchor - current_target) > allowance ||
        start_majorizer > current_target + allowance ||
        majorizer_solution > current_target + allowance ||
        candidate_target > majorizer_solution + allowance ||
        candidate_target > current_target + allowance) {
      result.status = MultinomialLlaStatus::kMajorizationFailed;
      result.failed_stage = stage;
      return result;
    }

    result.beta.swap(candidate.beta);
    result.intercept.swap(candidate.intercept);
    active_features.swap(candidate.active_features);
    current_target = candidate_target;
    current_stationarity = candidate_stationarity;
    result.completed_stages = stage + 1;
    result.failed_stage = -1;
    result.final_target_objective = current_target;
    result.final_target_stationarity = current_stationarity;
    if (!fixed_stage_compatibility &&
        result.completed_stages >= lla_options.minimum_stages &&
        current_stationarity <= lla_options.stationarity_tolerance) {
      result.status = MultinomialLlaStatus::kCompleted;
      return result;
    }
  }

  if (fixed_stage_compatibility) {
    result.status = MultinomialLlaStatus::kCompleted;
    return result;
  }
  result.status = MultinomialLlaStatus::kStationarityLimit;
  // No weighted-L1 solve failed. The last accepted, majorization-checked
  // iterate remains a valid best-effort model without a stationarity
  // certificate.
  result.failed_stage = -1;
  return result;
}

}  // namespace

double multinomial_lla_penalty_value(
    MultinomialLlaPenalty penalty, double absolute_value,
    double lambda, double gamma) {
  validate_lla_parameters(penalty, lambda, gamma);
  if (!(absolute_value >= 0.0) || !std::isfinite(absolute_value))
    throw std::invalid_argument(
        "multinomial LLA absolute coefficient must be finite and nonnegative");
  return lla_penalty_value_unchecked(
      penalty, absolute_value, lambda, gamma);
}

double multinomial_lla_penalty_derivative(
    MultinomialLlaPenalty penalty, double absolute_value,
    double lambda, double gamma) {
  validate_lla_parameters(penalty, lambda, gamma);
  if (!(absolute_value >= 0.0) || !std::isfinite(absolute_value))
    throw std::invalid_argument(
        "multinomial LLA absolute coefficient must be finite and nonnegative");
  return lla_penalty_derivative_unchecked(
      penalty, absolute_value, lambda, gamma);
}

const char *multinomial_lla_status_string(MultinomialLlaStatus status) {
  switch (status) {
    case MultinomialLlaStatus::kCompleted:
      return "completed";
    case MultinomialLlaStatus::kStationarityLimit:
      return "stationarity_limit";
    case MultinomialLlaStatus::kSubproblemFailed:
      return "subproblem_failed";
    case MultinomialLlaStatus::kMajorizationFailed:
      return "majorization_failed";
    case MultinomialLlaStatus::kNumericalFailure:
      return "numerical_failure";
  }
  return "unknown";
}

MultinomialLlaOptions::MultinomialLlaOptions()
    : stopping_rule(MultinomialLlaStoppingRule::kTargetStationarity),
      minimum_stages(3),
      maximum_stages(3),
      stationarity_tolerance(0.0),
      majorization_tolerance(1e-12) {}

MultinomialLlaOptions MultinomialLlaOptions::fixed_stage_compatibility(
    int stage_count) {
  MultinomialLlaOptions options;
  options.stopping_rule = MultinomialLlaStoppingRule::kFixedStages;
  options.minimum_stages = stage_count;
  options.maximum_stages = stage_count;
  return options;
}

MultinomialLlaSolver::MultinomialLlaSolver(
    const MultinomialObjective &objective,
    const MultinomialActNewtonOptions &proximal_newton_options,
    const MultinomialLlaOptions &lla_options)
    : m_objective(objective),
      m_proximal_newton_options(
          lla_proximal_newton_options(proximal_newton_options, lla_options)),
      m_lla_options(lla_options),
      m_l1_master_solver(objective, proximal_newton_options),
      m_proximal_newton_solver(objective, m_proximal_newton_options) {
  validate_and_normalize_lla_options(
      &m_proximal_newton_options, &m_lla_options);
}

MultinomialLlaResult MultinomialLlaSolver::solve(
    MultinomialLlaPenalty penalty, double lambda, double gamma) const {
  validate_lla_parameters(penalty, lambda, gamma);
  const MultinomialProblemView objective = lla_problem_view(m_objective);
  const Eigen::MatrixXd fallback_beta = Eigen::MatrixXd::Zero(
      objective.feature_num(), objective.class_num());
  const Eigen::VectorXd fallback_intercept = lla_empirical_null_intercept(
      objective, m_proximal_newton_options.include_intercept);
  MultinomialActNewtonResult master =
      m_l1_master_solver.solve(lambda);
  return lla_complete_from_master(
      objective, m_proximal_newton_options, m_lla_options, penalty, lambda,
      gamma, std::move(master), fallback_beta, fallback_intercept);
}

MultinomialLlaResult MultinomialLlaSolver::solve(
    MultinomialLlaPenalty penalty, double lambda, double gamma,
    const Eigen::MatrixXd &initial_beta,
    const Eigen::VectorXd &initial_intercept) const {
  validate_lla_parameters(penalty, lambda, gamma);
  if (initial_beta.rows() != m_objective.feature_num() ||
      initial_beta.cols() != m_objective.class_num())
    throw std::invalid_argument("initial multinomial LLA beta has wrong shape");
  if (initial_intercept.size() != m_objective.class_num())
    throw std::invalid_argument(
        "initial multinomial LLA intercept has wrong length");
  if (!initial_beta.allFinite() || !initial_intercept.allFinite())
    throw std::invalid_argument(
        "initial multinomial LLA parameters must be finite");
  MultinomialActNewtonResult master = m_l1_master_solver.solve(
      lambda, initial_beta, initial_intercept);
  const MultinomialProblemView objective = lla_problem_view(m_objective);
  return lla_complete_from_master(
      objective, m_proximal_newton_options, m_lla_options, penalty, lambda,
      gamma, std::move(master), initial_beta, initial_intercept);
}

MultinomialLlaResult MultinomialLlaSolver::solve_from_l1_master(
    MultinomialLlaPenalty penalty, double lambda, double gamma,
    MultinomialActNewtonResult master) const {
  const MultinomialProblemView objective = lla_problem_view(m_objective);
  return internal::solve_multinomial_lla_from_l1_master_view(
      objective, m_proximal_newton_options, m_lla_options, penalty, lambda,
      gamma, std::move(master));
}

namespace internal {

MultinomialLlaResult solve_multinomial_lla_from_l1_master_view(
    const ::picasso::detail::MultinomialProblemView &problem,
    const MultinomialActNewtonOptions &proximal_newton_options,
    const MultinomialLlaOptions &lla_options,
    MultinomialLlaPenalty penalty, double lambda, double gamma,
    MultinomialActNewtonResult master) {
  validate_lla_parameters(penalty, lambda, gamma);
  if (master.beta.rows() != problem.feature_num() ||
      master.beta.cols() != problem.class_num())
    throw std::invalid_argument(
        "multinomial LLA master beta has wrong shape");
  if (master.intercept.size() != problem.class_num())
    throw std::invalid_argument(
        "multinomial LLA master intercept has wrong length");
  if (!master.beta.allFinite() || !master.intercept.allFinite())
    throw std::invalid_argument(
        "multinomial LLA master parameters must be finite");

  MultinomialActNewtonOptions effective_proximal_options =
      proximal_newton_options;
  MultinomialLlaOptions effective_lla_options = lla_options;
  validate_and_normalize_lla_options(
      &effective_proximal_options, &effective_lla_options);
  const Eigen::MatrixXd fallback_beta = master.beta;
  const Eigen::VectorXd fallback_intercept = master.intercept;
  return lla_complete_from_master(
      problem, effective_proximal_options, effective_lla_options, penalty,
      lambda, gamma, std::move(master), fallback_beta, fallback_intercept);
}

}  // namespace internal

}  // namespace solver
}  // namespace picasso
