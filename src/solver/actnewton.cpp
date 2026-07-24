#include <picasso/actnewton.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace picasso {
namespace solver {
namespace {

const double kCoefficientZeroTolerance = 1e-8;
const double kFastCoordinateUpdateTolerance = 1e-4;
// The legacy scalar objectives update residual summaries incrementally and use
// 1e-8 as their coefficient zero/update threshold. A tighter majorization
// allowance misclassifies sub-nanounit solver noise as a hard LLA failure.
const double kMajorizationTolerance = 1e-8;

double path_deviance(double smooth_objective, bool square_root_loss) {
  const double magnitude = std::fabs(smooth_objective);
  return square_root_loss ? 0.5 * magnitude * magnitude : magnitude;
}

struct LlaMetrics {
  double smooth_objective;
  double target_penalty;
  double weighted_penalty;
  double target_objective;
  double surrogate_objective;
  double weighted_l1_kkt;
  double target_stationarity;

  LlaMetrics()
      : smooth_objective(std::numeric_limits<double>::quiet_NaN()),
        target_penalty(std::numeric_limits<double>::quiet_NaN()),
        weighted_penalty(std::numeric_limits<double>::quiet_NaN()),
        target_objective(std::numeric_limits<double>::quiet_NaN()),
        surrogate_objective(std::numeric_limits<double>::quiet_NaN()),
        weighted_l1_kkt(std::numeric_limits<double>::infinity()),
        target_stationarity(std::numeric_limits<double>::infinity()) {}
};

enum class WeightedSubproblemStatus {
  kConverged,
  kIterationLimit,
  kNumericalFailure
};

bool valid_penalty_parameters(RegType penalty, double gamma) {
  if (penalty == L1) return true;
  if (!std::isfinite(gamma)) return false;
  if (penalty == MCP) return gamma > 1.0;
  if (penalty == SCAD) return gamma > 2.0;
  return false;
}

double penalty_derivative(RegType penalty, double absolute_value,
                          double lambda, double gamma) {
  if (penalty == L1) return lambda;
  if (lambda == 0.0) return 0.0;
  if (penalty == MCP)
    return std::max(0.0, lambda - absolute_value / gamma);
  if (absolute_value <= lambda) return lambda;
  return std::max(
      0.0, lambda - (absolute_value - lambda) / (gamma - 1.0));
}

double penalty_value(RegType penalty, double absolute_value,
                     double lambda, double gamma) {
  if (absolute_value == 0.0 || lambda == 0.0) return 0.0;
  if (penalty == L1) return lambda * absolute_value;
  if (penalty == MCP) {
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

double coefficient_stationarity(double coefficient, double smooth_gradient,
                                double derivative) {
  if (coefficient > kCoefficientZeroTolerance)
    return std::fabs(smooth_gradient + derivative);
  if (coefficient < -kCoefficientZeroTolerance)
    return std::fabs(smooth_gradient - derivative);
  return std::max(0.0, std::fabs(smooth_gradient) - derivative);
}

bool finite_model(ObjFunction *objective) {
  const ModelParam &model = objective->get_model_param_ref();
  return model.beta.allFinite() && std::isfinite(model.intercept) &&
         objective->get_model_Xb_ref().allFinite();
}

bool compute_metrics(ObjFunction *objective,
                     const PicassoSolverParams &parameters,
                     RegType target_penalty, double lambda,
                     const std::vector<double> &weighted_l1,
                     LlaMetrics *metrics,
                     std::vector<double> *gradient_magnitudes,
                     bool refresh_objective = true) {
  if (metrics == nullptr || weighted_l1.size() !=
                                static_cast<std::size_t>(objective->get_dim()))
    return false;
  if (!finite_model(objective)) return false;

  if (refresh_objective) {
    objective->update_auxiliary();
    if (!finite_model(objective)) return false;
    objective->update_all_gradients();
  }
  const double smooth_objective = objective->eval();
  if (!std::isfinite(smooth_objective)) return false;

  const ModelParam &model = objective->get_model_param_ref();
  double target_penalty_sum = 0.0;
  double weighted_penalty_sum = 0.0;
  double weighted_kkt = 0.0;
  double target_stationarity = 0.0;
  if (gradient_magnitudes != nullptr)
    gradient_magnitudes->assign(model.d, 0.0);

  for (int feature = 0; feature < model.d; ++feature) {
    const double smooth_gradient = -objective->get_grad(feature);
    const double absolute_gradient = std::fabs(smooth_gradient);
    const double coefficient = model.beta[feature];
    const double absolute_value = std::fabs(coefficient);
    const double target_derivative = penalty_derivative(
        target_penalty, absolute_value, lambda, parameters.gamma);
    const double weight = weighted_l1[feature];
    const double value = penalty_value(
        target_penalty, absolute_value, lambda, parameters.gamma);
    if (!std::isfinite(smooth_gradient) ||
        !std::isfinite(target_derivative) || !std::isfinite(weight) ||
        weight < 0.0 || !std::isfinite(value))
      return false;
    if (gradient_magnitudes != nullptr)
      (*gradient_magnitudes)[feature] = absolute_gradient;
    target_penalty_sum += value;
    weighted_penalty_sum += weight * absolute_value;
    weighted_kkt = std::max(
        weighted_kkt,
        coefficient_stationarity(coefficient, smooth_gradient, weight));
    target_stationarity = std::max(
        target_stationarity,
        coefficient_stationarity(
            coefficient, smooth_gradient, target_derivative));
  }

  if (parameters.include_intercept) {
    const double intercept_gradient = objective->get_intercept_gradient();
    if (!std::isfinite(intercept_gradient)) return false;
    weighted_kkt = std::max(weighted_kkt, std::fabs(intercept_gradient));
    target_stationarity =
        std::max(target_stationarity, std::fabs(intercept_gradient));
  }

  metrics->smooth_objective = smooth_objective;
  metrics->target_penalty = target_penalty_sum;
  metrics->weighted_penalty = weighted_penalty_sum;
  metrics->target_objective = smooth_objective + target_penalty_sum;
  metrics->surrogate_objective = smooth_objective + weighted_penalty_sum;
  metrics->weighted_l1_kkt = weighted_kkt;
  metrics->target_stationarity = target_stationarity;
  return std::isfinite(metrics->target_objective) &&
         std::isfinite(metrics->surrogate_objective) &&
         std::isfinite(weighted_kkt) &&
         std::isfinite(target_stationarity);
}

bool compute_active_weighted_l1_kkt(
    ObjFunction *objective, const PicassoSolverParams &parameters,
    const std::vector<double> &weighted_l1,
    const std::vector<int> &active_set, double *active_kkt) {
  const int dimension = objective->get_dim();
  if (active_kkt == nullptr ||
      weighted_l1.size() != static_cast<std::size_t>(dimension) ||
      active_set.size() != static_cast<std::size_t>(dimension))
    return false;

  objective->update_auxiliary();
  if (!finite_model(objective)) return false;

  double residual = 0.0;
  for (int feature = 0; feature < dimension; ++feature) {
    if (!active_set[feature]) continue;
    objective->update_gradient(feature);
    const double smooth_gradient = -objective->get_grad(feature);
    const double weight = weighted_l1[feature];
    if (!std::isfinite(smooth_gradient) || !std::isfinite(weight) ||
        weight < 0.0)
      return false;
    residual = std::max(
        residual,
        coefficient_stationarity(objective->get_model_coef(feature),
                                 smooth_gradient, weight));
  }

  if (parameters.include_intercept) {
    const double intercept_gradient = objective->get_intercept_gradient();
    if (!std::isfinite(intercept_gradient)) return false;
    residual = std::max(residual, std::fabs(intercept_gradient));
  }
  *active_kkt = residual;
  return std::isfinite(residual);
}

bool make_lla_weights(const ModelParam &model, RegType penalty,
                      double lambda, double gamma,
                      std::vector<double> *weights,
                      double *target_penalty_sum,
                      double *weighted_penalty_sum,
                      double *tangent_constant) {
  if (weights == nullptr || target_penalty_sum == nullptr ||
      weighted_penalty_sum == nullptr || tangent_constant == nullptr)
    return false;
  weights->resize(model.d);
  *target_penalty_sum = 0.0;
  *weighted_penalty_sum = 0.0;
  *tangent_constant = 0.0;
  for (int feature = 0; feature < model.d; ++feature) {
    const double absolute_value = std::fabs(model.beta[feature]);
    const double weight =
        penalty_derivative(penalty, absolute_value, lambda, gamma);
    const double value =
        penalty_value(penalty, absolute_value, lambda, gamma);
    if (!std::isfinite(weight) || weight < 0.0 || !std::isfinite(value))
      return false;
    (*weights)[feature] = weight;
    *target_penalty_sum += value;
    *weighted_penalty_sum += weight * absolute_value;
    *tangent_constant += value - weight * absolute_value;
  }
  return std::isfinite(*target_penalty_sum) &&
         std::isfinite(*weighted_penalty_sum) &&
         std::isfinite(*tangent_constant);
}

double majorization_allowance(double first, double second,
                              double third, double fourth) {
  double scale = 1.0;
  scale = std::max(scale, std::fabs(first));
  scale = std::max(scale, std::fabs(second));
  scale = std::max(scale, std::fabs(third));
  scale = std::max(scale, std::fabs(fourth));
  return (kMajorizationTolerance +
          64.0 * std::numeric_limits<double>::epsilon()) * scale;
}

WeightedSubproblemStatus solve_weighted_l1_subproblem(
    ObjFunction *objective, const PicassoSolverParams &parameters,
    RegType target_penalty, double lambda,
    const std::vector<double> &weighted_l1,
    std::vector<int> *active_set, int *iteration_count,
    LlaMetrics *final_metrics, std::vector<double> *final_gradients,
    bool refresh_initial_metrics) {
  const int dimension = objective->get_dim();
  if (active_set == nullptr || iteration_count == nullptr ||
      final_metrics == nullptr || final_gradients == nullptr ||
      active_set->size() != static_cast<std::size_t>(dimension) ||
      weighted_l1.size() != static_cast<std::size_t>(dimension) ||
      parameters.max_iter <= 0)
    return WeightedSubproblemStatus::kNumericalFailure;

  const double deviance_threshold =
      std::fabs(objective->get_deviance()) * std::sqrt(parameters.prec);
  // This is an inexact-Newton forcing threshold for the moving quadratic
  // surrogate, not the public convergence tolerance. Every outer candidate
  // still has to satisfy the full weighted-L1 KKT residual <= prec below.
  if (!std::isfinite(deviance_threshold))
    return WeightedSubproblemStatus::kNumericalFailure;

  RegL1 regularizer;
  PoissonObjective *fast_poisson =
      dynamic_cast<PoissonObjective *>(objective);
  const bool defer_linear_predictor =
      fast_poisson != nullptr &&
      parameters.prec >= kFastCoordinateUpdateTolerance &&
      objective->get_sample_num() > dimension;
  std::vector<int> active_indices;
  active_indices.reserve(dimension);
  const bool supports_active_kkt_gate =
      dynamic_cast<GLMObjective *>(objective) != nullptr ||
      dynamic_cast<SqrtMSEObjective *>(objective) != nullptr;

  // Avoid perturbing an anchor that already solves the new weighted-L1
  // subproblem. This matters near an LLA fixed point: another approximate
  // Newton sweep can otherwise introduce a tiny objective increase even
  // though the current point already has the requested KKT certificate.
  LlaMetrics initial_metrics;
  // Stage zero or a changed model must refresh auxiliary state and gradients.
  // Consecutive LLA stages share the exact same anchor, so they can reuse the
  // just-certified state and only recompute the O(d) penalty/KKT terms.
  if (!compute_metrics(objective, parameters, target_penalty, lambda,
                       weighted_l1, &initial_metrics, final_gradients,
                       refresh_initial_metrics))
    return WeightedSubproblemStatus::kNumericalFailure;
  *final_metrics = initial_metrics;
  if (initial_metrics.weighted_l1_kkt <= parameters.prec)
    return WeightedSubproblemStatus::kConverged;

  for (int outer = 0; outer < parameters.max_iter; ++outer) {
    active_indices.clear();
    for (int feature = 0; feature < dimension; ++feature) {
      if (!(*active_set)[feature]) continue;
      regularizer.set_param(weighted_l1[feature], 0.0);
      const double updated =
          defer_linear_predictor
              ? fast_poisson->coordinate_descent_deferred(
                    &regularizer, feature)
              : objective->coordinate_descent(&regularizer, feature);
      if (!std::isfinite(updated))
        return WeightedSubproblemStatus::kNumericalFailure;
      if (std::fabs(updated) > 0.0) active_indices.push_back(feature);
    }
    if (!(defer_linear_predictor
              ? fast_poisson->coordinate_state_all_finite()
              : finite_model(objective)))
      return WeightedSubproblemStatus::kNumericalFailure;

    bool inner_converged = false;
    for (int inner = 0; inner < parameters.max_iter; ++inner) {
      bool small_local_change = true;
      for (std::size_t index = 0; index < active_indices.size(); ++index) {
        const int feature = active_indices[index];
        const double old_coefficient = objective->get_model_coef(feature);
        regularizer.set_param(weighted_l1[feature], 0.0);
        const double updated =
            defer_linear_predictor
                ? fast_poisson->coordinate_descent_deferred(
                      &regularizer, feature)
                : objective->coordinate_descent(&regularizer, feature);
        if (!std::isfinite(updated))
          return WeightedSubproblemStatus::kNumericalFailure;
        if (!objective->can_skip_local_change(
                old_coefficient, feature, deviance_threshold)) {
          const double local_change =
              objective->get_local_change(old_coefficient, feature);
          if (!std::isfinite(local_change))
            return WeightedSubproblemStatus::kNumericalFailure;
          if (local_change > deviance_threshold) small_local_change = false;
        }
      }

      if (parameters.include_intercept) {
        const double old_intercept = objective->get_model_coef(-1);
        objective->intercept_update();
        const double local_change =
            objective->get_local_change(old_intercept, -1);
        if (!std::isfinite(local_change))
          return WeightedSubproblemStatus::kNumericalFailure;
        if (local_change > deviance_threshold) small_local_change = false;
      }

      if (!(defer_linear_predictor
                ? fast_poisson->coordinate_state_all_finite()
                : finite_model(objective)))
        return WeightedSubproblemStatus::kNumericalFailure;

      ++(*iteration_count);
      if (small_local_change) {
        inner_converged = true;
        break;
      }
    }
    if (!inner_converged)
      return WeightedSubproblemStatus::kIterationLimit;
    if (defer_linear_predictor &&
        !fast_poisson->rebuild_linear_predictor(active_indices))
      return WeightedSubproblemStatus::kNumericalFailure;

    LlaMetrics metrics;
    if (supports_active_kkt_gate) {
      double active_kkt = 0.0;
      if (!compute_active_weighted_l1_kkt(
              objective, parameters, weighted_l1, *active_set,
              &active_kkt))
        return WeightedSubproblemStatus::kNumericalFailure;
      if (active_kkt > parameters.prec) continue;
      objective->update_all_gradients();
      if (!compute_metrics(objective, parameters, target_penalty, lambda,
                           weighted_l1, &metrics, final_gradients, false))
        return WeightedSubproblemStatus::kNumericalFailure;
    } else if (!compute_metrics(
                   objective, parameters, target_penalty, lambda,
                   weighted_l1, &metrics, final_gradients)) {
      return WeightedSubproblemStatus::kNumericalFailure;
    }

    bool added_active_coordinate = false;
    for (int feature = 0; feature < dimension; ++feature) {
      if (!(*active_set)[feature] &&
          (*final_gradients)[feature] > weighted_l1[feature]) {
        (*active_set)[feature] = 1;
        added_active_coordinate = true;
      }
    }

    *final_metrics = metrics;
    if (!added_active_coordinate &&
        metrics.weighted_l1_kkt <= parameters.prec)
      return WeightedSubproblemStatus::kConverged;
  }

  // Keep a failed point's diagnostics tied to the final candidate even when
  // every full scan was correctly deferred by the active-set gate.
  LlaMetrics terminal_metrics;
  if (!compute_metrics(objective, parameters, target_penalty, lambda,
                       weighted_l1, &terminal_metrics, final_gradients))
    return WeightedSubproblemStatus::kNumericalFailure;
  *final_metrics = terminal_metrics;
  return WeightedSubproblemStatus::kIterationLimit;
}

ActNewtonLlaStatus lla_failure_status(WeightedSubproblemStatus status) {
  return status == WeightedSubproblemStatus::kNumericalFailure
             ? ActNewtonLlaStatus::kNumericalFailure
             : ActNewtonLlaStatus::kSubproblemFailed;
}

}  // namespace

ActNewtonSolver::ActNewtonSolver(ObjFunction *obj, PicassoSolverParams param)
    : m_param(param),
      m_obj(obj),
      lla_path_status(ActNewtonLlaStatus::kNotRun),
      failed_lambda(-1),
      failed_stage(-1) {
  itercnt_path.clear();
  runtime_path.clear();
  solution_path.clear();
}

struct ActNewtonSolver::CommitSink {
  CommitSink(double *beta_out, double *intercept_out, int *iterations_out,
             int *active_size_out, double *runtime_out,
             double *smooth_objective_out, int *committed_count_out,
             int *last_nonzero_count_out)
      : beta(beta_out),
        intercept(intercept_out),
        iterations(iterations_out),
        active_size(active_size_out),
        runtime(runtime_out),
        smooth_objective(smooth_objective_out),
        committed_count(committed_count_out),
        last_nonzero_count(last_nonzero_count_out) {
    if (committed_count != nullptr) *committed_count = 0;
    if (last_nonzero_count != nullptr) *last_nonzero_count = 0;
  }

  void commit(int path_index, const ModelParam &model, int dimension,
              int iteration_count, double runtime_value,
              double smooth_objective_value) {
    int nonzero_count = 0;
    for (int feature = 0; feature < dimension; ++feature) {
      const double coefficient = model.beta[feature];
      if (beta != nullptr)
        beta[static_cast<std::size_t>(path_index) * dimension + feature] =
            coefficient;
      if (std::fabs(coefficient) > kCoefficientZeroTolerance)
        ++nonzero_count;
    }
    if (intercept != nullptr) intercept[path_index] = model.intercept;
    if (iterations != nullptr) iterations[path_index] = iteration_count;
    if (active_size != nullptr) active_size[path_index] = nonzero_count;
    if (runtime != nullptr) runtime[path_index] = runtime_value;
    if (smooth_objective != nullptr)
      smooth_objective[path_index] = smooth_objective_value;
    if (last_nonzero_count != nullptr)
      *last_nonzero_count = nonzero_count;
    // Publish the prefix length last. All outputs for this model are complete
    // before an exception at a later lambda can expose the committed prefix.
    if (committed_count != nullptr) *committed_count = path_index + 1;
  }

  double *beta;
  double *intercept;
  int *iterations;
  int *active_size;
  double *runtime;
  double *smooth_objective;
  int *committed_count;
  int *last_nonzero_count;
};

void ActNewtonSolver::solve() { solve_impl(false); }

void ActNewtonSolver::solve_preinitialized() {
  solve_impl(true);
}

void ActNewtonSolver::solve_impl(bool objective_state_preinitialized) {
  (void)solve_impl(objective_state_preinitialized, nullptr);
}

int ActNewtonSolver::solve_to_buffers(
    double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, double *smooth_objective, int *committed_count,
    int *last_nonzero_count) {
  CommitSink sink(beta, intcpt, ite_lamb, size_act, runt, smooth_objective,
                  committed_count, last_nonzero_count);
  return solve_impl(false, &sink);
}

int ActNewtonSolver::solve_preinitialized_to_buffers(
    double *beta, double *intcpt, int *ite_lamb, int *size_act,
    double *runt, double *smooth_objective, int *committed_count,
    int *last_nonzero_count) {
  CommitSink sink(beta, intcpt, ite_lamb, size_act, runt, smooth_objective,
                  committed_count, last_nonzero_count);
  return solve_impl(true, &sink);
}

int ActNewtonSolver::solve_impl(bool objective_state_preinitialized,
                                CommitSink *sink) {
  const std::vector<double> &lambdas = m_param.get_lambda_path();
  const std::size_t path_size = lambdas.size();
  const double missing = std::numeric_limits<double>::quiet_NaN();

  itercnt_path.assign(path_size, 0);
  runtime_path.assign(path_size, 0.0);
  solution_path.clear();
  if (sink == nullptr) solution_path.reserve(path_size);
  lla_status_path.assign(path_size, ActNewtonLlaStatus::kNotRun);
  lla_stages_path.assign(path_size, 0);
  objective_path.assign(path_size, missing);
  smooth_objective_path.assign(path_size, missing);
  kkt_path.assign(path_size, missing);
  stationarity_path.assign(path_size, missing);
  lla_path_status = ActNewtonLlaStatus::kNotRun;
  failed_lambda = -1;
  failed_stage = -1;
  int committed_count = 0;

  if (m_obj == nullptr) {
    if (!lla_status_path.empty()) {
      lla_status_path[0] = ActNewtonLlaStatus::kNumericalFailure;
      failed_lambda = 0;
    }
    lla_path_status = ActNewtonLlaStatus::kNumericalFailure;
    return committed_count;
  }

  const int dimension = m_obj->get_dim();
  const bool nonconvex = m_param.reg_type == MCP || m_param.reg_type == SCAD;
  const bool square_root_loss =
      dynamic_cast<SqrtMSEObjective *>(m_obj) != nullptr;
  if (dimension <= 0 || !(m_param.prec > 0.0) ||
      !std::isfinite(m_param.prec) || m_param.max_iter <= 0 ||
      !valid_penalty_parameters(m_param.reg_type, m_param.gamma) ||
      (nonconvex &&
       (m_param.num_relaxation_round < 3 ||
        m_param.num_relaxation_round >
            static_cast<unsigned>(std::numeric_limits<int>::max())))) {
    if (!lla_status_path.empty()) {
      lla_status_path[0] = ActNewtonLlaStatus::kNumericalFailure;
      failed_lambda = 0;
    }
    lla_path_status = ActNewtonLlaStatus::kNumericalFailure;
    return committed_count;
  }

  GLMObjective *glm_objective = dynamic_cast<GLMObjective *>(m_obj);
  if (glm_objective != nullptr) {
    glm_objective->set_fused_coordinate_updates(
        m_param.reg_type == L1 ||
        m_param.prec >= kFastCoordinateUpdateTolerance);
    const bool fast_poisson_reductions =
        dynamic_cast<PoissonObjective *>(m_obj) != nullptr &&
        m_param.prec >= kFastCoordinateUpdateTolerance;
    glm_objective->set_fast_residual_dot(fast_poisson_reductions);
    glm_objective->set_fast_weighted_sq_sum(fast_poisson_reductions);
  }

  std::vector<int> master_active_set(dimension, 0);
  std::vector<double> master_gradients(dimension, 0.0);
  if (!objective_state_preinitialized) m_obj->update_auxiliary();
  if (!finite_model(m_obj)) {
    if (!lla_status_path.empty()) {
      lla_status_path[0] = ActNewtonLlaStatus::kNumericalFailure;
      failed_lambda = 0;
    }
    lla_path_status = ActNewtonLlaStatus::kNumericalFailure;
    return committed_count;
  }
  if (!objective_state_preinitialized) m_obj->update_all_gradients();
  for (int feature = 0; feature < dimension; ++feature) {
    const double gradient = m_obj->get_grad(feature);
    if (!std::isfinite(gradient)) {
      if (!lla_status_path.empty()) {
        lla_status_path[0] = ActNewtonLlaStatus::kNumericalFailure;
        failed_lambda = 0;
      }
      lla_path_status = ActNewtonLlaStatus::kNumericalFailure;
      return committed_count;
    }
    master_gradients[feature] = std::fabs(gradient);
  }

  ModelParam model_master = m_obj->get_model_param();
  Eigen::ArrayXd xb_master = m_obj->get_model_Xb();
  bool master_state_current = true;
  std::vector<double> deviance_path;
  lla_path_status = ActNewtonLlaStatus::kCompleted;

  for (std::size_t lambda_index = 0; lambda_index < path_size;
       ++lambda_index) {
    if (interrupt_requested()) {
      lla_path_status = ActNewtonLlaStatus::kInterrupted;
      break;
    }
    const double lambda = lambdas[lambda_index];
    if (!(lambda >= 0.0) || !std::isfinite(lambda)) {
      lla_status_path[lambda_index] =
          ActNewtonLlaStatus::kNumericalFailure;
      lla_path_status = ActNewtonLlaStatus::kNumericalFailure;
      failed_lambda = static_cast<int>(lambda_index);
      failed_stage = -1;
      break;
    }

    const ModelParam lambda_start_model = model_master;
    const Eigen::ArrayXd lambda_start_xb = xb_master;
    if (!master_state_current) {
      m_obj->set_model_param(model_master);
      m_obj->set_model_Xb(xb_master);
    }
    std::vector<int> active_set = master_active_set;

    const double strong_threshold =
        lambda_index > 0
            ? 2.0 * lambda - lambdas[lambda_index - 1]
            : 2.0 * lambda;
    for (int feature = 0; feature < dimension; ++feature) {
      if (master_gradients[feature] > strong_threshold)
        active_set[feature] = 1;
    }

    std::vector<double> stage_weights(dimension, lambda);
    LlaMetrics current_metrics;
    std::vector<double> current_gradients;
    WeightedSubproblemStatus subproblem = solve_weighted_l1_subproblem(
        m_obj, m_param, m_param.reg_type, lambda, stage_weights,
        &active_set, &itercnt_path[lambda_index], &current_metrics,
        &current_gradients, !master_state_current);
    if (subproblem != WeightedSubproblemStatus::kConverged) {
      lla_status_path[lambda_index] = lla_failure_status(subproblem);
      objective_path[lambda_index] = current_metrics.target_objective;
      smooth_objective_path[lambda_index] =
          current_metrics.smooth_objective;
      kkt_path[lambda_index] = current_metrics.weighted_l1_kkt;
      stationarity_path[lambda_index] =
          current_metrics.target_stationarity;
      lla_path_status = lla_status_path[lambda_index];
      failed_lambda = static_cast<int>(lambda_index);
      failed_stage = 0;
      m_obj->set_model_param(lambda_start_model);
      m_obj->set_model_Xb(lambda_start_xb);
      m_obj->update_auxiliary();
      break;
    }

    int completed_stages = 1;
    const ModelParam candidate_master = m_obj->get_model_param();
    const Eigen::ArrayXd candidate_master_xb = m_obj->get_model_Xb();
    const std::vector<int> candidate_master_active_set = active_set;
    const std::vector<double> candidate_master_gradients = current_gradients;

    bool hard_failure = false;
    if (nonconvex) {
      const int maximum_stages =
          static_cast<int>(m_param.num_relaxation_round);
      std::vector<double> candidate_gradients(dimension);
      for (int stage = 1; stage < maximum_stages; ++stage) {
        const ModelParam anchor_model = m_obj->get_model_param();
        const Eigen::ArrayXd anchor_xb = m_obj->get_model_Xb();
        const std::vector<int> anchor_active_set = active_set;
        const LlaMetrics anchor_metrics = current_metrics;

        double anchor_penalty = 0.0;
        double anchor_weighted_penalty = 0.0;
        double tangent_constant = 0.0;
        if (!make_lla_weights(
                anchor_model, m_param.reg_type, lambda, m_param.gamma,
                &stage_weights, &anchor_penalty, &anchor_weighted_penalty,
                &tangent_constant)) {
          lla_status_path[lambda_index] =
              ActNewtonLlaStatus::kNumericalFailure;
          failed_stage = stage;
          hard_failure = true;
          break;
        }

        LlaMetrics candidate_metrics;
        subproblem = solve_weighted_l1_subproblem(
            m_obj, m_param, m_param.reg_type, lambda, stage_weights,
            &active_set, &itercnt_path[lambda_index], &candidate_metrics,
            &candidate_gradients, false);
        if (subproblem != WeightedSubproblemStatus::kConverged) {
          lla_status_path[lambda_index] = lla_failure_status(subproblem);
          objective_path[lambda_index] =
              candidate_metrics.target_objective;
          smooth_objective_path[lambda_index] =
              candidate_metrics.smooth_objective;
          kkt_path[lambda_index] = candidate_metrics.weighted_l1_kkt;
          stationarity_path[lambda_index] =
              candidate_metrics.target_stationarity;
          failed_stage = stage;
          hard_failure = true;
          m_obj->set_model_param(anchor_model);
          m_obj->set_model_Xb(anchor_xb);
          active_set = anchor_active_set;
          m_obj->update_auxiliary();
          break;
        }

        const double majorizer_at_anchor =
            anchor_metrics.smooth_objective + anchor_weighted_penalty +
            tangent_constant;
        const double majorizer_at_candidate =
            candidate_metrics.surrogate_objective + tangent_constant;
        const double allowance = majorization_allowance(
            anchor_metrics.target_objective, majorizer_at_anchor,
            majorizer_at_candidate, candidate_metrics.target_objective);
        if (!std::isfinite(anchor_penalty) ||
            !std::isfinite(majorizer_at_anchor) ||
            !std::isfinite(majorizer_at_candidate) ||
            std::fabs(anchor_penalty - anchor_metrics.target_penalty) >
                allowance ||
            std::fabs(majorizer_at_anchor -
                      anchor_metrics.target_objective) > allowance ||
            majorizer_at_candidate >
                anchor_metrics.target_objective + allowance ||
            candidate_metrics.target_objective >
                majorizer_at_candidate + allowance ||
            candidate_metrics.target_objective >
                anchor_metrics.target_objective + allowance) {
          lla_status_path[lambda_index] =
              ActNewtonLlaStatus::kMajorizationFailed;
          objective_path[lambda_index] =
              candidate_metrics.target_objective;
          smooth_objective_path[lambda_index] =
              candidate_metrics.smooth_objective;
          kkt_path[lambda_index] = candidate_metrics.weighted_l1_kkt;
          stationarity_path[lambda_index] =
              candidate_metrics.target_stationarity;
          failed_stage = stage;
          hard_failure = true;
          m_obj->set_model_param(anchor_model);
          m_obj->set_model_Xb(anchor_xb);
          active_set = anchor_active_set;
          m_obj->update_auxiliary();
          break;
        }

        current_metrics = candidate_metrics;
        current_gradients.swap(candidate_gradients);
        completed_stages = stage + 1;
        if (completed_stages >= 3 &&
            current_metrics.target_stationarity <= m_param.prec)
          break;
      }
    }

    lla_stages_path[lambda_index] = completed_stages;
    if (hard_failure) {
      lla_path_status = lla_status_path[lambda_index];
      failed_lambda = static_cast<int>(lambda_index);
      m_obj->set_model_param(lambda_start_model);
      m_obj->set_model_Xb(lambda_start_xb);
      m_obj->update_auxiliary();
      break;
    }

    if (nonconvex &&
        current_metrics.target_stationarity > m_param.prec) {
      lla_status_path[lambda_index] =
          ActNewtonLlaStatus::kStationarityLimit;
      if (lla_path_status == ActNewtonLlaStatus::kCompleted)
        lla_path_status = ActNewtonLlaStatus::kStationarityLimit;
    } else {
      lla_status_path[lambda_index] = ActNewtonLlaStatus::kCompleted;
    }
    objective_path[lambda_index] = current_metrics.target_objective;
    smooth_objective_path[lambda_index] = current_metrics.smooth_objective;
    kkt_path[lambda_index] = current_metrics.weighted_l1_kkt;
    stationarity_path[lambda_index] =
        current_metrics.target_stationarity;

    const ModelParam &committed_model = m_obj->get_model_param_ref();
    if (sink == nullptr) solution_path.push_back(committed_model);
    if (sink != nullptr)
      sink->commit(committed_count, committed_model, dimension,
                   itercnt_path[lambda_index], runtime_path[lambda_index],
                   current_metrics.smooth_objective);
    ++committed_count;
    model_master = candidate_master;
    xb_master = candidate_master_xb;
    master_active_set = candidate_master_active_set;
    master_gradients = candidate_master_gradients;
    // A convex path commits the model whose auxiliary state and full gradient
    // were just certified. Nonconvex LLA commits a later stage while the next
    // lambda deliberately warm-starts from its L1 master instead.
    master_state_current = !nonconvex;

    const double current_deviance = path_deviance(
        current_metrics.smooth_objective, square_root_loss);
    deviance_path.push_back(current_deviance);
    const int number_fit = committed_count;
    if (number_fit >= m_param.min_lambda_count) {
      int nonzero_count = 0;
      for (int feature = 0; feature < dimension; ++feature) {
        if (std::fabs(m_obj->get_model_coef(feature)) >
            kCoefficientZeroTolerance)
          ++nonzero_count;
      }
      if (m_param.dfmax >= 0 && nonzero_count > m_param.dfmax) break;

      if (nonzero_count > 0) {
        const double null_deviance = path_deviance(
            m_obj->get_deviance(), square_root_loss);
        if (null_deviance > 0.0) {
          const double deviance_ratio =
              1.0 - current_deviance / null_deviance;
          if (deviance_ratio > m_param.dev_ratio_max) break;

          const int previous_index =
              number_fit - 1 - m_param.min_lambda_count;
          if (previous_index >= 0) {
            const double previous_deviance =
                deviance_path[previous_index];
            const double change =
                std::fabs(previous_deviance - current_deviance);
            if (current_deviance > 0.0 &&
                change / current_deviance < m_param.dev_change_min)
              break;
          }
        }
      }
    }
  }
  return committed_count;
}

}  // namespace solver
}  // namespace picasso
