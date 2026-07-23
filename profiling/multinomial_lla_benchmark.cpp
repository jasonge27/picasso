#include <picasso/multinomial_lla.hpp>
#include <picasso/multinomial_objective.hpp>

// Streaming benchmark for the stationarity-certified multinomial LLA driver.
// The benchmark intentionally runs one production/reference path per
// process so wall time and ru_maxrss can be compared with an external ABBA
// harness.  Example standalone build from the repository root:
//   c++ -O3 -DNDEBUG -std=c++11 -Iinclude \
//     -isystem R-package/src/include/eigen3 \
//     profiling/multinomial_lla_benchmark.cpp \
//     src/objective/multinomial_objective.cpp \
//     src/solver/multinomial_actnewton.cpp \
//     src/solver/multinomial_lla.cpp -o multinomial_lla_benchmark

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

namespace {

const int kPathPointCount = 18;
const int kMinimumLlaStageCount = 3;
const int kMaximumLlaStageCount = 25;
const double kGamma = 3.0;
const double kZeroTolerance = 1e-12;
const double kOuterKktTolerance = 2e-7;
const double kStationarityTolerance = 2e-7;

enum class PenaltyKind { kMcp, kScad };
enum class BenchmarkMode { kProduction, kReference };

struct CaseConfiguration {
  int sample_num;
  int feature_num;
  int class_num;
  int signal_feature_num;
  unsigned int seed;
  double within_block_correlation;
  double signal_scale;
  double final_lambda_ratio;
  bool dense_control;
};

struct DataSet {
  Eigen::MatrixXd x;
  Eigen::VectorXi labels;
  double digest;
};

struct TargetOracle {
  double negative_log_likelihood;
  double target_objective;
  double stationarity_residual;
  double probability_digest;
  bool finite;
};

struct StageTotals {
  long long outer_iterations;
  long long inner_sweeps;
  long long coordinate_visits;

  StageTotals()
      : outer_iterations(0), inner_sweeps(0), coordinate_visits(0) {}
};

// Narrow adapter result.  Only fit_lla_point depends on the production LLA
// header; all data generation, streaming, oracle, and JSON code below is
// deliberately independent of its concrete class and field names.
struct PointFit {
  Eigen::MatrixXd beta;
  Eigen::VectorXd intercept;
  Eigen::MatrixXd l1_master_beta;
  Eigen::VectorXd l1_master_intercept;
  bool converged;
  std::string status;
  int completed_stages;
  std::array<StageTotals, kMaximumLlaStageCount> stages;
  double reported_target_objective;
  double reported_target_stationarity;
  double maximum_majorization_violation;
  double maximum_target_objective_increase;

  PointFit()
      : converged(false),
        status("not_started"),
        completed_stages(0),
        reported_target_objective(
            std::numeric_limits<double>::quiet_NaN()),
        reported_target_stationarity(
            std::numeric_limits<double>::infinity()),
        maximum_majorization_violation(
            std::numeric_limits<double>::infinity()),
        maximum_target_objective_increase(
            std::numeric_limits<double>::infinity()) {}
};

struct PathSummary {
  int requested_points;
  int completed_points;
  int failed_point;
  std::string terminal_status;
  std::array<StageTotals, kMaximumLlaStageCount> stages;
  int maximum_completed_stages;
  long long total_outer_iterations;
  long long total_inner_sweeps;
  long long total_coordinate_visits;
  double wall_seconds;
  double oracle_seconds;
  double lambda_digest;
  double target_objective_digest;
  double target_stationarity_digest;
  double probability_digest;
  double final_probability_digest;
  double final_target_objective;
  double final_target_stationarity;
  double maximum_target_stationarity;
  double maximum_reported_target_stationarity;
  double maximum_target_objective_report_error;
  double maximum_target_objective_scale;
  double maximum_majorization_violation;
  double maximum_target_objective_increase;
  int final_nonzero_features;
  int final_nonzero_coefficients;
  bool all_converged;
  bool all_finite;

  explicit PathSummary(int point_count)
      : requested_points(point_count),
        completed_points(0),
        failed_point(-1),
        terminal_status("not_started"),
        maximum_completed_stages(0),
        total_outer_iterations(0),
        total_inner_sweeps(0),
        total_coordinate_visits(0),
        wall_seconds(0.0),
        oracle_seconds(0.0),
        lambda_digest(0.0),
        target_objective_digest(0.0),
        target_stationarity_digest(0.0),
        probability_digest(0.0),
        final_probability_digest(
            std::numeric_limits<double>::quiet_NaN()),
        final_target_objective(std::numeric_limits<double>::quiet_NaN()),
        final_target_stationarity(std::numeric_limits<double>::infinity()),
        maximum_target_stationarity(0.0),
        maximum_reported_target_stationarity(0.0),
        maximum_target_objective_report_error(0.0),
        maximum_target_objective_scale(1.0),
        maximum_majorization_violation(0.0),
        maximum_target_objective_increase(0.0),
        final_nonzero_features(0),
        final_nonzero_coefficients(0),
        all_converged(true),
        all_finite(true) {}
};

void usage(const char *program) {
  std::cerr
      << "Usage: " << program
      << " --case sparse-k3|wide-k4|high-k12|dense-control"
         " --penalty mcp|scad --mode production|reference"
         " [--scan-interval positive-integer]"
         " [--inner-kkt-tolerance positive-number]"
         " [--hessian-damping positive-number]\n"
      << "Run production and reference in separate processes. Production"
         " enables the exact feature active set and direction cache;"
         " reference disables only the active set. Both run the same adaptive"
         " warm LLA path with target-stationarity certification.\n";
}

PenaltyKind parse_penalty(const std::string &value) {
  if (value == "mcp") return PenaltyKind::kMcp;
  if (value == "scad") return PenaltyKind::kScad;
  throw std::invalid_argument("--penalty must be mcp or scad");
}

BenchmarkMode parse_mode(const std::string &value) {
  if (value == "production") return BenchmarkMode::kProduction;
  if (value == "reference") return BenchmarkMode::kReference;
  throw std::invalid_argument("--mode must be production or reference");
}

const char *penalty_name(PenaltyKind penalty) {
  return penalty == PenaltyKind::kMcp ? "mcp" : "scad";
}

const char *mode_name(BenchmarkMode mode) {
  return mode == BenchmarkMode::kProduction ? "production" : "reference";
}

CaseConfiguration configuration_for(const std::string &case_name) {
  CaseConfiguration configuration;
  if (case_name == "sparse-k3") {
    configuration.sample_num = 1200;
    configuration.feature_num = 280;
    configuration.class_num = 3;
    configuration.signal_feature_num = 9;
    configuration.seed = 9101u;
    configuration.within_block_correlation = 0.15;
    configuration.signal_scale = 0.54;
    configuration.final_lambda_ratio = 0.22;
    configuration.dense_control = false;
  } else if (case_name == "wide-k4") {
    configuration.sample_num = 360;
    configuration.feature_num = 1100;
    configuration.class_num = 4;
    configuration.signal_feature_num = 11;
    configuration.seed = 9103u;
    configuration.within_block_correlation = 0.25;
    configuration.signal_scale = 0.48;
    configuration.final_lambda_ratio = 0.28;
    configuration.dense_control = false;
  } else if (case_name == "high-k12") {
    configuration.sample_num = 640;
    configuration.feature_num = 240;
    configuration.class_num = 12;
    configuration.signal_feature_num = 13;
    configuration.seed = 9109u;
    configuration.within_block_correlation = 0.10;
    configuration.signal_scale = 0.30;
    configuration.final_lambda_ratio = 0.38;
    configuration.dense_control = false;
  } else if (case_name == "dense-control") {
    configuration.sample_num = 400;
    configuration.feature_num = 96;
    configuration.class_num = 4;
    configuration.signal_feature_num = 82;
    configuration.seed = 9113u;
    configuration.within_block_correlation = 0.10;
    configuration.signal_scale = 0.14;
    configuration.final_lambda_ratio = 0.05;
    configuration.dense_control = true;
  } else {
    throw std::invalid_argument("unknown LLA benchmark case: " + case_name);
  }
  return configuration;
}

DataSet make_data(const CaseConfiguration &configuration) {
  std::mt19937 generator(configuration.seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  DataSet data;
  data.x.resize(configuration.sample_num, configuration.feature_num);
  const int block_size = 8;
  const double common_scale =
      std::sqrt(configuration.within_block_correlation);
  const double independent_scale =
      std::sqrt(1.0 - configuration.within_block_correlation);
  for (int observation = 0; observation < configuration.sample_num;
       ++observation) {
    double common = normal(generator);
    for (int feature = 0; feature < configuration.feature_num; ++feature) {
      if (feature % block_size == 0) common = normal(generator);
      data.x(observation, feature) =
          common_scale * common + independent_scale * normal(generator);
    }
  }
  for (int feature = 0; feature < configuration.feature_num; ++feature) {
    data.x.col(feature).array() -= data.x.col(feature).mean();
    const double scale =
        std::sqrt(data.x.col(feature).squaredNorm() /
                  static_cast<double>(configuration.sample_num));
    if (!(scale > 0.0) || !std::isfinite(scale))
      throw std::runtime_error("generated a degenerate design column");
    data.x.col(feature) /= scale;
  }

  Eigen::MatrixXd true_beta = Eigen::MatrixXd::Zero(
      configuration.feature_num, configuration.class_num);
  for (int signal = 0; signal < configuration.signal_feature_num; ++signal) {
    const int feature = configuration.dense_control
                            ? signal
                            : (19 + 41 * signal) % configuration.feature_num;
    for (int klass = 0; klass < configuration.class_num; ++klass)
      true_beta(feature, klass) = normal(generator);
    true_beta.row(feature).array() -= true_beta.row(feature).mean();
    const double row_rms = std::sqrt(
        true_beta.row(feature).squaredNorm() /
        static_cast<double>(configuration.class_num));
    if (!(row_rms > 0.0) || !std::isfinite(row_rms))
      throw std::runtime_error("generated a degenerate signal row");
    true_beta.row(feature) *= configuration.signal_scale / row_rms;
  }

  Eigen::VectorXd true_intercept(configuration.class_num);
  for (int klass = 0; klass < configuration.class_num; ++klass) {
    true_intercept[klass] =
        0.06 * (static_cast<double>(klass) -
                0.5 * static_cast<double>(configuration.class_num - 1));
  }
  const Eigen::MatrixXd logits =
      data.x * true_beta +
      true_intercept.transpose().replicate(configuration.sample_num, 1);
  data.labels.resize(configuration.sample_num);
  Eigen::VectorXi class_counts =
      Eigen::VectorXi::Zero(configuration.class_num);
  Eigen::VectorXd class_weights(configuration.class_num);
  for (int observation = 0; observation < configuration.sample_num;
       ++observation) {
    const double maximum = logits.row(observation).maxCoeff();
    double weight_sum = 0.0;
    for (int klass = 0; klass < configuration.class_num; ++klass) {
      class_weights[klass] =
          std::exp(logits(observation, klass) - maximum);
      weight_sum += class_weights[klass];
    }
    const double draw = uniform(generator) * weight_sum;
    double cumulative = 0.0;
    int sampled_class = configuration.class_num - 1;
    for (int klass = 0; klass < configuration.class_num; ++klass) {
      cumulative += class_weights[klass];
      if (draw <= cumulative) {
        sampled_class = klass;
        break;
      }
    }
    data.labels[observation] = sampled_class;
    ++class_counts[sampled_class];
  }
  if (class_counts.minCoeff() <= 0)
    throw std::runtime_error("generated data contain an empty class");

  double design_digest = 0.0;
  for (int observation = 0; observation < configuration.sample_num;
       ++observation) {
    for (int feature = 0; feature < configuration.feature_num; ++feature) {
      const int weight = 1 + (observation * 11 + feature * 17) % 31;
      design_digest += static_cast<double>(weight) *
                       data.x(observation, feature);
    }
    design_digest +=
        static_cast<double>((observation % 13) + 1) *
        static_cast<double>(data.labels[observation] + 1);
  }
  data.digest =
      design_digest /
      static_cast<double>(configuration.sample_num *
                          configuration.feature_num);
  return data;
}

Eigen::VectorXd empirical_null_intercept(const Eigen::VectorXi &labels,
                                         int class_num) {
  Eigen::VectorXd counts = Eigen::VectorXd::Zero(class_num);
  for (Eigen::Index index = 0; index < labels.size(); ++index)
    counts[labels[index]] += 1.0;
  Eigen::VectorXd intercept(class_num);
  for (int klass = 0; klass < class_num; ++klass) {
    if (!(counts[klass] > 0.0))
      throw std::runtime_error("null intercept encountered an empty class");
    intercept[klass] =
        std::log(counts[klass] / static_cast<double>(labels.size()));
  }
  intercept.array() -= intercept.mean();
  return intercept;
}

std::vector<double> make_lambda_ratios(double final_ratio) {
  if (!(final_ratio > 0.0) || !(final_ratio < 1.0))
    throw std::invalid_argument("final lambda ratio must lie in (0,1)");
  const double first_ratio = 1.02;
  std::vector<double> ratios(kPathPointCount);
  for (int point = 0; point < kPathPointCount; ++point) {
    const double fraction =
        static_cast<double>(point) /
        static_cast<double>(kPathPointCount - 1);
    ratios[static_cast<std::size_t>(point)] =
        std::exp((1.0 - fraction) * std::log(first_ratio) +
                 fraction * std::log(final_ratio));
  }
  return ratios;
}

double penalty_value(PenaltyKind penalty, double coefficient,
                     double lambda, double gamma) {
  const double value = std::fabs(coefficient);
  if (penalty == PenaltyKind::kMcp) {
    if (value <= gamma * lambda)
      return lambda * value - value * value / (2.0 * gamma);
    return 0.5 * gamma * lambda * lambda;
  }
  if (value <= lambda) return lambda * value;
  if (value <= gamma * lambda) {
    return (-value * value + 2.0 * gamma * lambda * value -
            lambda * lambda) /
           (2.0 * (gamma - 1.0));
  }
  return 0.5 * (gamma + 1.0) * lambda * lambda;
}

double penalty_derivative(PenaltyKind penalty, double coefficient,
                          double lambda, double gamma) {
  const double value = std::fabs(coefficient);
  if (penalty == PenaltyKind::kMcp)
    return std::max(0.0, lambda - value / gamma);
  if (value <= lambda) return lambda;
  if (value < gamma * lambda)
    return (gamma * lambda - value) / (gamma - 1.0);
  return 0.0;
}

double coefficient_stationarity(double coefficient, double gradient,
                                double weight) {
  if (coefficient > kZeroTolerance)
    return std::fabs(gradient + weight);
  if (coefficient < -kZeroTolerance)
    return std::fabs(gradient - weight);
  return std::max(0.0, std::fabs(gradient) - weight);
}

TargetOracle evaluate_target(const picasso::MultinomialObjective &objective,
                             const Eigen::MatrixXd &beta,
                             const Eigen::VectorXd &intercept,
                             double lambda, PenaltyKind penalty,
                             double gamma) {
  TargetOracle oracle;
  Eigen::MatrixXd probabilities;
  oracle.negative_log_likelihood =
      objective.negative_log_likelihood(beta, intercept, &probabilities);
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  objective.smooth_gradient_from_probabilities(
      probabilities, &beta_gradient, &intercept_gradient);

  double penalty_sum = 0.0;
  oracle.stationarity_residual = 0.0;
  for (Eigen::Index feature = 0; feature < beta.rows(); ++feature) {
    for (Eigen::Index klass = 0; klass < beta.cols(); ++klass) {
      penalty_sum += penalty_value(
          penalty, beta(feature, klass), lambda, gamma);
      const double weight = penalty_derivative(
          penalty, beta(feature, klass), lambda, gamma);
      oracle.stationarity_residual = std::max(
          oracle.stationarity_residual,
          coefficient_stationarity(beta(feature, klass),
                                   beta_gradient(feature, klass), weight));
    }
  }
  oracle.stationarity_residual = std::max(
      oracle.stationarity_residual,
      intercept_gradient.cwiseAbs().maxCoeff());
  oracle.target_objective = oracle.negative_log_likelihood + penalty_sum;

  double probability_digest = 0.0;
  for (Eigen::Index observation = 0;
       observation < probabilities.rows(); ++observation) {
    for (Eigen::Index klass = 0; klass < probabilities.cols(); ++klass) {
      const int weight =
          1 + (static_cast<int>(observation) * 17 +
               static_cast<int>(klass) * 13) % 29;
      probability_digest +=
          static_cast<double>(weight) * probabilities(observation, klass);
    }
  }
  oracle.probability_digest =
      probability_digest / static_cast<double>(probabilities.size());
  oracle.finite =
      beta.allFinite() && intercept.allFinite() && probabilities.allFinite() &&
      beta_gradient.allFinite() && intercept_gradient.allFinite() &&
      std::isfinite(oracle.negative_log_likelihood) &&
      std::isfinite(oracle.target_objective) &&
      std::isfinite(oracle.stationarity_residual) &&
      std::isfinite(oracle.probability_digest);
  return oracle;
}

int count_nonzero_features(const Eigen::MatrixXd &beta, double tolerance) {
  int count = 0;
  for (Eigen::Index feature = 0; feature < beta.rows(); ++feature) {
    if (beta.row(feature).cwiseAbs().maxCoeff() > tolerance) ++count;
  }
  return count;
}

int count_nonzero_coefficients(const Eigen::MatrixXd &beta,
                               double tolerance) {
  int count = 0;
  for (Eigen::Index feature = 0; feature < beta.rows(); ++feature) {
    for (Eigen::Index klass = 0; klass < beta.cols(); ++klass) {
      if (std::fabs(beta(feature, klass)) > tolerance) ++count;
    }
  }
  return count;
}

picasso::solver::MultinomialLlaPenalty solver_penalty(
    PenaltyKind penalty) {
  return penalty == PenaltyKind::kMcp
             ? picasso::solver::MultinomialLlaPenalty::kMCP
             : picasso::solver::MultinomialLlaPenalty::kSCAD;
}

PointFit fit_lla_point(
    const picasso::solver::MultinomialLlaSolver &solver,
    PenaltyKind penalty, double lambda, double gamma,
    const Eigen::MatrixXd &initial_beta,
    const Eigen::VectorXd &initial_intercept) {
  picasso::solver::MultinomialLlaResult result = solver.solve(
      solver_penalty(penalty), lambda, gamma,
      initial_beta, initial_intercept);

  PointFit fit;
  fit.converged = result.completed();
  fit.status = picasso::solver::multinomial_lla_status_string(result.status);
  fit.completed_stages = result.completed_stages;
  fit.reported_target_objective = result.final_target_objective;
  fit.reported_target_stationarity = result.final_target_stationarity;
  fit.maximum_majorization_violation = 0.0;
  fit.maximum_target_objective_increase = 0.0;

  std::array<bool, kMaximumLlaStageCount> stage_seen;
  stage_seen.fill(false);
  int summed_outer_iterations = 0;
  int summed_inner_sweeps = 0;
  long long summed_coordinate_visits = 0;
  bool stage_structure_valid = true;
  for (std::size_t index = 0; index < result.stages.size(); ++index) {
    const picasso::solver::MultinomialLlaStageRecord &record =
        result.stages[index];
    if (record.stage < 0 || record.stage >= kMaximumLlaStageCount)
      throw std::runtime_error("LLA result returned an invalid stage index");
    if (record.stage != static_cast<int>(index))
      throw std::runtime_error("LLA stage records are not ordered");
    const std::size_t stage = static_cast<std::size_t>(record.stage);
    if (stage_seen[stage])
      throw std::runtime_error("LLA result returned a duplicate stage index");
    stage_seen[stage] = true;
    fit.stages[stage].outer_iterations = record.outer_iterations;
    fit.stages[stage].inner_sweeps = record.inner_sweeps;
    fit.stages[stage].coordinate_visits = record.coordinate_updates;
    summed_outer_iterations += record.outer_iterations;
    summed_inner_sweeps += record.inner_sweeps;
    summed_coordinate_visits += record.coordinate_updates;
    stage_structure_valid =
        stage_structure_valid &&
        record.subproblem_status ==
            picasso::solver::MultinomialSolverStatus::kConverged &&
        record.is_l1_master == (record.stage == 0);

    if (record.stage > 0) {
      const picasso::solver::MultinomialLlaStageRecord &previous =
          result.stages[index - 1];
      // The LLA majorizer must touch the previous target at its anchor,
      // dominate the target at the new solution, and not increase from its
      // anchor.  Track all three violations with one deterministic scalar.
      fit.maximum_majorization_violation = std::max(
          fit.maximum_majorization_violation,
          std::fabs(record.majorizer_at_anchor -
                    previous.target_objective));
      fit.maximum_majorization_violation = std::max(
          fit.maximum_majorization_violation,
          std::max(0.0, record.target_objective -
                            record.majorizer_at_solution));
      fit.maximum_majorization_violation = std::max(
          fit.maximum_majorization_violation,
          std::max(0.0, record.majorizer_at_solution -
                            record.majorizer_at_anchor));
      fit.maximum_target_objective_increase = std::max(
          fit.maximum_target_objective_increase,
          std::max(0.0, record.target_objective -
                            previous.target_objective));
    }
  }
  if (fit.converged) {
    bool all_stages_seen =
        result.completed_stages >= kMinimumLlaStageCount &&
        result.completed_stages <= kMaximumLlaStageCount &&
        result.stages.size() ==
            static_cast<std::size_t>(result.completed_stages) &&
        result.total_outer_iterations == summed_outer_iterations &&
        result.total_inner_sweeps == summed_inner_sweeps &&
        result.total_coordinate_updates == summed_coordinate_visits &&
        stage_structure_valid &&
        result.final_target_stationarity <= kStationarityTolerance;
    for (int stage = 0; stage < result.completed_stages; ++stage)
      all_stages_seen =
          all_stages_seen && stage_seen[static_cast<std::size_t>(stage)];
    if (!all_stages_seen) {
      fit.converged = false;
      fit.status = "invalid_completed_stage_count";
    }
  }

  fit.beta = std::move(result.beta);
  fit.intercept = std::move(result.intercept);
  fit.l1_master_beta = std::move(result.l1_master_beta);
  fit.l1_master_intercept = std::move(result.l1_master_intercept);
  return fit;
}

PathSummary run_streaming_path(
    const picasso::MultinomialObjective &objective,
    PenaltyKind penalty, BenchmarkMode mode,
    const std::vector<double> &lambda_ratios, double lambda_max,
    const Eigen::VectorXd &initial_intercept,
    int exact_kkt_scan_interval, double inner_kkt_tolerance,
    double hessian_damping) {
  const int feature_num = objective.feature_num();
  const int class_num = objective.class_num();
  Eigen::MatrixXd l1_master_beta =
      Eigen::MatrixXd::Zero(feature_num, class_num);
  Eigen::VectorXd l1_master_intercept = initial_intercept;
  Eigen::MatrixXd final_beta = l1_master_beta;
  Eigen::VectorXd final_intercept = initial_intercept;
  PathSummary summary(static_cast<int>(lambda_ratios.size()));

  picasso::solver::MultinomialActNewtonOptions proximal_newton_options;
  proximal_newton_options.max_outer_iterations = 100;
  proximal_newton_options.max_inner_sweeps = 4000;
  proximal_newton_options.outer_kkt_tolerance = kOuterKktTolerance;
  proximal_newton_options.inner_kkt_tolerance = inner_kkt_tolerance;
  proximal_newton_options.hessian_damping = hessian_damping;
  proximal_newton_options.exact_kkt_scan_interval =
      exact_kkt_scan_interval;
  proximal_newton_options.use_active_set =
      mode == BenchmarkMode::kProduction;
  proximal_newton_options.use_probability_dot_direction_cache = true;
  picasso::solver::MultinomialLlaOptions lla_options;
  lla_options.minimum_stages = kMinimumLlaStageCount;
  lla_options.maximum_stages = kMaximumLlaStageCount;
  lla_options.stationarity_tolerance = kStationarityTolerance;
  lla_options.majorization_tolerance = 2e-9;
  const picasso::solver::MultinomialLlaSolver solver(
      objective, proximal_newton_options, lla_options);

  for (std::size_t point = 0; point < lambda_ratios.size(); ++point) {
    const double lambda = lambda_max * lambda_ratios[point];
    const double point_weight = static_cast<double>(point + 1);
    summary.lambda_digest += point_weight * lambda;

    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    PointFit fit = fit_lla_point(
        solver, penalty, lambda, kGamma,
        l1_master_beta, l1_master_intercept);
    if (fit.converged) {
      // Both state assignments are deliberately inside the timed operation.
      // A failed point never commits either master or final state.
      l1_master_beta = std::move(fit.l1_master_beta);
      l1_master_intercept = std::move(fit.l1_master_intercept);
      final_beta = std::move(fit.beta);
      final_intercept = std::move(fit.intercept);
    }
    summary.wall_seconds +=
        std::chrono::duration_cast<std::chrono::duration<double> >(
            std::chrono::steady_clock::now() - start)
            .count();

    summary.terminal_status = fit.status;
    summary.maximum_completed_stages = std::max(
        summary.maximum_completed_stages, fit.completed_stages);
    for (int stage = 0; stage < kMaximumLlaStageCount; ++stage) {
      summary.stages[static_cast<std::size_t>(stage)].outer_iterations +=
          fit.stages[static_cast<std::size_t>(stage)].outer_iterations;
      summary.stages[static_cast<std::size_t>(stage)].inner_sweeps +=
          fit.stages[static_cast<std::size_t>(stage)].inner_sweeps;
      summary.stages[static_cast<std::size_t>(stage)].coordinate_visits +=
          fit.stages[static_cast<std::size_t>(stage)].coordinate_visits;
    }
    summary.maximum_majorization_violation = std::max(
        summary.maximum_majorization_violation,
        std::max(0.0, fit.maximum_majorization_violation));
    summary.maximum_target_objective_increase = std::max(
        summary.maximum_target_objective_increase,
        std::max(0.0, fit.maximum_target_objective_increase));
    summary.maximum_reported_target_stationarity = std::max(
        summary.maximum_reported_target_stationarity,
        fit.reported_target_stationarity);
    summary.all_finite =
        summary.all_finite &&
        std::isfinite(fit.reported_target_objective) &&
        std::isfinite(fit.reported_target_stationarity) &&
        std::isfinite(fit.maximum_majorization_violation) &&
        std::isfinite(fit.maximum_target_objective_increase);
    if (!fit.converged) {
      summary.all_converged = false;
      summary.failed_point = static_cast<int>(point);
      break;
    }
    ++summary.completed_points;

    const std::chrono::steady_clock::time_point oracle_start =
        std::chrono::steady_clock::now();
    const TargetOracle oracle = evaluate_target(
        objective, final_beta, final_intercept, lambda, penalty, kGamma);
    summary.oracle_seconds +=
        std::chrono::duration_cast<std::chrono::duration<double> >(
            std::chrono::steady_clock::now() - oracle_start)
            .count();
    summary.all_finite = summary.all_finite && oracle.finite;
    summary.maximum_target_objective_report_error = std::max(
        summary.maximum_target_objective_report_error,
        std::fabs(fit.reported_target_objective -
                  oracle.target_objective));
    summary.maximum_target_stationarity = std::max(
        summary.maximum_target_stationarity,
        oracle.stationarity_residual);
    summary.maximum_target_objective_scale = std::max(
        summary.maximum_target_objective_scale,
        std::max(std::fabs(fit.reported_target_objective),
                 std::fabs(oracle.target_objective)));
    summary.target_objective_digest +=
        point_weight * oracle.target_objective;
    summary.target_stationarity_digest +=
        point_weight * oracle.stationarity_residual;
    summary.probability_digest +=
        point_weight * oracle.probability_digest;
    summary.final_probability_digest = oracle.probability_digest;
    summary.final_target_objective = oracle.target_objective;
    summary.final_target_stationarity = oracle.stationarity_residual;
  }

  for (int stage = 0; stage < kMaximumLlaStageCount; ++stage) {
    summary.total_outer_iterations +=
        summary.stages[static_cast<std::size_t>(stage)].outer_iterations;
    summary.total_inner_sweeps +=
        summary.stages[static_cast<std::size_t>(stage)].inner_sweeps;
    summary.total_coordinate_visits +=
        summary.stages[static_cast<std::size_t>(stage)].coordinate_visits;
  }
  summary.final_nonzero_features =
      count_nonzero_features(final_beta, 1e-10);
  summary.final_nonzero_coefficients =
      count_nonzero_coefficients(final_beta, 1e-10);
  if (summary.completed_points == summary.requested_points) {
    summary.failed_point = -1;
    summary.terminal_status = "converged";
  }
  return summary;
}

long long peak_rss_raw() {
#if defined(_WIN32)
  return 0;
#else
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) return -1;
  return static_cast<long long>(usage.ru_maxrss);
#endif
}

long long peak_rss_bytes(long long raw_value) {
  if (raw_value < 0) return raw_value;
#if defined(__APPLE__) || defined(_WIN32)
  return raw_value;
#else
  return raw_value * 1024LL;
#endif
}

bool validation_passed(const PathSummary &summary) {
  const double tolerance =
      1e-10 + 2e-9 * summary.maximum_target_objective_scale;
  const bool objective_valid =
      summary.maximum_target_objective_report_error <= tolerance;
  const bool majorization_valid =
      summary.maximum_majorization_violation <= tolerance;
  const bool target_descent_valid =
      summary.maximum_target_objective_increase <= tolerance;
  const bool stationarity_valid =
      std::isfinite(summary.maximum_target_stationarity) &&
      std::isfinite(summary.maximum_reported_target_stationarity) &&
      summary.maximum_target_stationarity <= kStationarityTolerance &&
      summary.maximum_reported_target_stationarity <=
          kStationarityTolerance;
  return summary.all_converged && summary.all_finite && objective_valid &&
         majorization_valid && target_descent_valid && stationarity_valid;
}

std::string json_number(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

void print_json(const std::string &case_name,
                const CaseConfiguration &configuration,
                PenaltyKind penalty, BenchmarkMode mode,
                double data_digest,
                const std::vector<double> &lambda_ratios,
                double lambda_max, int exact_kkt_scan_interval,
                double inner_kkt_tolerance, double hessian_damping,
                const PathSummary &summary) {
  const double objective_tolerance =
      1e-10 + 2e-9 * summary.maximum_target_objective_scale;
  const double mm_tolerance = objective_tolerance;
  const bool objective_valid =
      summary.maximum_target_objective_report_error <= objective_tolerance;
  const bool majorization_valid =
      summary.maximum_majorization_violation <= mm_tolerance;
  const bool target_descent_valid =
      summary.maximum_target_objective_increase <= mm_tolerance;
  const bool stationarity_valid =
      std::isfinite(summary.maximum_target_stationarity) &&
      std::isfinite(summary.maximum_reported_target_stationarity) &&
      summary.maximum_target_stationarity <= kStationarityTolerance &&
      summary.maximum_reported_target_stationarity <=
          kStationarityTolerance;
  const bool passed = validation_passed(summary);
  const long long rss_raw = peak_rss_raw();

  std::cout
      << std::setprecision(17)
      << "{\n"
      << "  \"schema\": \"multinomial-lla-benchmark-v2\",\n"
      << "  \"case\": \"" << case_name << "\",\n"
      << "  \"penalty\": \"" << penalty_name(penalty) << "\",\n"
      << "  \"mode\": \"" << mode_name(mode) << "\",\n"
      << "  \"gamma\": " << json_number(kGamma) << ",\n"
      << "  \"n\": " << configuration.sample_num << ",\n"
      << "  \"d\": " << configuration.feature_num << ",\n"
      << "  \"classes\": " << configuration.class_num << ",\n"
      << "  \"signal_features\": "
      << configuration.signal_feature_num << ",\n"
      << "  \"dense_control\": "
      << (configuration.dense_control ? "true" : "false") << ",\n"
      << "  \"path_points_requested\": " << summary.requested_points
      << ",\n"
      << "  \"path_points_completed\": " << summary.completed_points
      << ",\n"
      << "  \"failed_point_zero_based\": " << summary.failed_point
      << ",\n"
      << "  \"terminal_status\": \"" << summary.terminal_status
      << "\",\n"
      << "  \"lambda_max\": " << json_number(lambda_max) << ",\n"
      << "  \"first_lambda_ratio\": "
      << json_number(lambda_ratios.front())
      << ",\n"
      << "  \"final_lambda_ratio\": "
      << json_number(lambda_ratios.back())
      << ",\n"
      << "  \"configuration\": {\n"
      << "    \"active_set\": "
      << (mode == BenchmarkMode::kProduction ? "true" : "false")
      << ",\n"
      << "    \"direction_cache\": true,\n"
      << "    \"exact_kkt_scan_interval\": "
      << exact_kkt_scan_interval << ",\n"
      << "    \"requested_inner_kkt_tolerance\": "
      << json_number(inner_kkt_tolerance) << ",\n"
      << "    \"adaptive_weighted_inner_kkt_tolerance\": "
      << json_number(std::min(kOuterKktTolerance,
                              kStationarityTolerance))
      << ",\n"
      << "    \"hessian_damping\": "
      << json_number(hessian_damping) << ",\n"
      << "    \"warm_l1_master_path\": true,\n"
      << "    \"warm_weighted_stages\": true,\n"
      << "    \"minimum_lla_stages_per_lambda\": "
      << kMinimumLlaStageCount << ",\n"
      << "    \"maximum_lla_stages_per_lambda\": "
      << kMaximumLlaStageCount << ",\n"
      << "    \"maximum_observed_lla_stages_per_lambda\": "
      << summary.maximum_completed_stages << ",\n"
      << "    \"target_stationarity_tolerance\": "
      << json_number(kStationarityTolerance)
      << "\n  },\n"
      << "  \"digests\": {\n"
      << "    \"data\": " << json_number(data_digest) << ",\n"
      << "    \"lambda\": " << json_number(summary.lambda_digest)
      << ",\n"
      << "    \"target_objective\": "
      << json_number(summary.target_objective_digest) << ",\n"
      << "    \"target_stationarity\": "
      << json_number(summary.target_stationarity_digest) << ",\n"
      << "    \"probability_path\": "
      << json_number(summary.probability_digest) << ",\n"
      << "    \"probability_final\": "
      << json_number(summary.final_probability_digest)
      << "\n  },\n"
      << "  \"stage_totals\": [\n";
  for (int stage = 0; stage < summary.maximum_completed_stages; ++stage) {
    const StageTotals &totals = summary.stages[static_cast<std::size_t>(stage)];
    std::cout
        << "    {\"stage\": " << stage
        << ", \"kind\": \"" << (stage == 0 ? "l1_master" : "weighted_l1")
        << "\", \"outer_iterations\": " << totals.outer_iterations
        << ", \"inner_sweeps\": " << totals.inner_sweeps
        << ", \"coordinate_visits\": " << totals.coordinate_visits
        << "}";
    if (stage + 1 != summary.maximum_completed_stages) std::cout << ",";
    std::cout << "\n";
  }
  std::cout
      << "  ],\n"
      << "  \"totals\": {\n"
      << "    \"outer_iterations\": " << summary.total_outer_iterations
      << ",\n"
      << "    \"inner_sweeps\": " << summary.total_inner_sweeps
      << ",\n"
      << "    \"coordinate_visits\": "
      << summary.total_coordinate_visits << ",\n"
      << "    \"wall_seconds_solve_and_state_commit\": "
      << json_number(summary.wall_seconds) << ",\n"
      << "    \"oracle_seconds_outside_timer\": "
      << json_number(summary.oracle_seconds) << "\n"
      << "  },\n"
      << "  \"final\": {\n"
      << "    \"target_objective\": "
      << json_number(summary.final_target_objective)
      << ",\n"
      << "    \"target_stationarity\": "
      << json_number(summary.final_target_stationarity) << ",\n"
      << "    \"nonzero_features\": "
      << summary.final_nonzero_features << ",\n"
      << "    \"nonzero_coefficients\": "
      << summary.final_nonzero_coefficients << "\n"
      << "  },\n"
      << "  \"validation\": {\n"
      << "    \"all_converged\": "
      << (summary.all_converged ? "true" : "false") << ",\n"
      << "    \"all_finite\": "
      << (summary.all_finite ? "true" : "false") << ",\n"
      << "    \"maximum_target_stationarity\": "
      << json_number(summary.maximum_target_stationarity) << ",\n"
      << "    \"maximum_reported_target_stationarity\": "
      << json_number(summary.maximum_reported_target_stationarity) << ",\n"
      << "    \"maximum_target_objective_report_error\": "
      << json_number(summary.maximum_target_objective_report_error)
      << ",\n"
      << "    \"maximum_majorization_violation\": "
      << json_number(summary.maximum_majorization_violation) << ",\n"
      << "    \"maximum_target_objective_increase\": "
      << json_number(summary.maximum_target_objective_increase) << ",\n"
      << "    \"objective_valid\": "
      << (objective_valid ? "true" : "false") << ",\n"
      << "    \"majorization_valid\": "
      << (majorization_valid ? "true" : "false") << ",\n"
      << "    \"target_descent_valid\": "
      << (target_descent_valid ? "true" : "false") << ",\n"
      << "    \"stationarity_within_tolerance\": "
      << (stationarity_valid ? "true" : "false") << "\n"
      << "  },\n"
      << "  \"ru_maxrss_raw\": " << rss_raw << ",\n"
      << "  \"ru_maxrss_bytes\": " << peak_rss_bytes(rss_raw)
      << ",\n"
      << "  \"passed\": " << (passed ? "true" : "false") << "\n"
      << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
  std::string case_name;
  std::string penalty_value_string;
  std::string mode_value;
  int exact_kkt_scan_interval = 4;
  double inner_kkt_tolerance = 2e-9;
  double hessian_damping = 1e-10;
  try {
    for (int argument = 1; argument < argc; ++argument) {
      const std::string key(argv[argument]);
      if (key == "--help" || key == "-h") {
        usage(argv[0]);
        return 0;
      }
      if (key == "--case" && argument + 1 < argc) {
        case_name = argv[++argument];
      } else if (key == "--penalty" && argument + 1 < argc) {
        penalty_value_string = argv[++argument];
      } else if (key == "--mode" && argument + 1 < argc) {
        mode_value = argv[++argument];
      } else if (key == "--scan-interval" && argument + 1 < argc) {
        exact_kkt_scan_interval = std::stoi(argv[++argument]);
      } else if (key == "--inner-kkt-tolerance" &&
                 argument + 1 < argc) {
        inner_kkt_tolerance = std::stod(argv[++argument]);
      } else if (key == "--hessian-damping" && argument + 1 < argc) {
        hessian_damping = std::stod(argv[++argument]);
      } else {
        usage(argv[0]);
        throw std::invalid_argument("unknown or incomplete argument: " + key);
      }
    }
    if (case_name.empty() || penalty_value_string.empty() ||
        mode_value.empty()) {
      usage(argv[0]);
      throw std::invalid_argument(
          "--case, --penalty, and --mode are all required");
    }
    if (exact_kkt_scan_interval <= 0)
      throw std::invalid_argument("--scan-interval must be positive");
    if (!(inner_kkt_tolerance > 0.0) ||
        !std::isfinite(inner_kkt_tolerance))
      throw std::invalid_argument(
          "--inner-kkt-tolerance must be finite and positive");
    if (!(hessian_damping > 0.0) || !std::isfinite(hessian_damping))
      throw std::invalid_argument(
          "--hessian-damping must be finite and positive");
    const PenaltyKind penalty = parse_penalty(penalty_value_string);
    const BenchmarkMode mode = parse_mode(mode_value);
    const CaseConfiguration configuration = configuration_for(case_name);
    DataSet data = make_data(configuration);
    const double data_digest = data.digest;
    const Eigen::VectorXd null_intercept =
        empirical_null_intercept(data.labels, configuration.class_num);
    picasso::MultinomialObjective objective(
        std::move(data.x), std::move(data.labels), configuration.class_num);

    const Eigen::MatrixXd zero_beta = Eigen::MatrixXd::Zero(
        configuration.feature_num, configuration.class_num);
    Eigen::MatrixXd null_gradient;
    Eigen::VectorXd ignored_intercept_gradient;
    objective.smooth_gradient(zero_beta, null_intercept,
                              &null_gradient,
                              &ignored_intercept_gradient);
    const double lambda_max = null_gradient.cwiseAbs().maxCoeff();
    if (!(lambda_max > 0.0) || !std::isfinite(lambda_max))
      throw std::runtime_error("generated a nonpositive lambda_max");
    const std::vector<double> lambda_ratios =
        make_lambda_ratios(configuration.final_lambda_ratio);
    const PathSummary summary = run_streaming_path(
        objective, penalty, mode, lambda_ratios, lambda_max,
        null_intercept, exact_kkt_scan_interval, inner_kkt_tolerance,
        hessian_damping);
    print_json(case_name, configuration, penalty, mode, data_digest,
               lambda_ratios, lambda_max, exact_kkt_scan_interval,
               inner_kkt_tolerance, hessian_damping, summary);
    return validation_passed(summary) ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "multinomial LLA benchmark error: " << error.what()
              << "\n";
    return 1;
  }
}
