#include <picasso/multinomial_actnewton.hpp>
#include <picasso/multinomial_objective.hpp>

// Streaming active-set/path-ActNewton benchmark. Build from the repository root:
//   c++ -O3 -DNDEBUG -std=c++11 -Wall -Wextra -Wpedantic -Werror \
//     -DEIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS \
//     -Iinclude -isystem R-package/src/include/eigen3 \
//     profiling/multinomial_active_set_benchmark.cpp \
//     src/objective/multinomial_objective.cpp \
//     src/solver/multinomial_actnewton.cpp \
//     -o /tmp/multinomial_active_set_benchmark
//
// Run active-set A/B measurements in separate processes, for example:
//   /tmp/multinomial_active_set_benchmark --case sparse-k3 --active on
//   /tmp/multinomial_active_set_benchmark --case sparse-k3 --active off

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

namespace {

const int kPathPointCount = 24;

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
};

struct OracleSummary {
  double objective;
  double kkt_residual;
  double probability_digest;
  bool finite;
};

struct PathSummary {
  std::array<int, 6> status_counts;
  int outer_iterations;
  int inner_sweeps;
  long long coordinate_visits;
  int full_subproblem_kkt_scans;
  int reactivated_features;
  int subproblem_reactivated_features;
  int outer_reactivated_features;
  long long initial_strong_features;
  long long strong_rule_activated_features;
  long long full_kkt_path_reactivations;
  int final_active_features;
  int peak_active_features;
  int final_nonzero_features;
  int final_nonzero_coefficients;
  picasso::solver::MultinomialSolverStatus final_status;
  double final_reported_objective;
  double final_oracle_objective;
  double final_reported_kkt;
  double final_independent_kkt;
  double lambda_digest;
  double status_digest;
  double reported_objective_digest;
  double oracle_objective_digest;
  double reported_kkt_digest;
  double independent_kkt_digest;
  double probability_digest;
  double maximum_independent_kkt;
  double maximum_kkt_report_error;
  double maximum_objective_report_error;
  double maximum_objective_scale;
  double wall_seconds;
  double oracle_seconds;
  bool all_converged;
  bool all_finite;

  PathSummary()
      : outer_iterations(0),
        inner_sweeps(0),
        coordinate_visits(0),
        full_subproblem_kkt_scans(0),
        reactivated_features(0),
        subproblem_reactivated_features(0),
        outer_reactivated_features(0),
        initial_strong_features(0),
        strong_rule_activated_features(0),
        full_kkt_path_reactivations(0),
        final_active_features(0),
        peak_active_features(0),
        final_nonzero_features(0),
        final_nonzero_coefficients(0),
        final_status(
            picasso::solver::MultinomialSolverStatus::kNumericalFailure),
        final_reported_objective(std::numeric_limits<double>::quiet_NaN()),
        final_oracle_objective(std::numeric_limits<double>::quiet_NaN()),
        final_reported_kkt(std::numeric_limits<double>::infinity()),
        final_independent_kkt(std::numeric_limits<double>::infinity()),
        lambda_digest(0.0),
        status_digest(0.0),
        reported_objective_digest(0.0),
        oracle_objective_digest(0.0),
        reported_kkt_digest(0.0),
        independent_kkt_digest(0.0),
        probability_digest(0.0),
        maximum_independent_kkt(0.0),
        maximum_kkt_report_error(0.0),
        maximum_objective_report_error(0.0),
        maximum_objective_scale(1.0),
        wall_seconds(0.0),
        oracle_seconds(0.0),
        all_converged(true),
        all_finite(true) {
    status_counts.fill(0);
  }
};

void usage(const char *program) {
  std::cerr
      << "Usage: " << program
      << " --case sparse-k3|wide-k4|high-k12|dense-control"
         " --active on|off\n"
      << "Run active=on and active=off as separate processes so wall time and"
         " ru_maxrss are independent.\n";
}

bool parse_active_switch(const std::string &value) {
  if (value == "on") return true;
  if (value == "off") return false;
  throw std::invalid_argument("--active must be on or off");
}

CaseConfiguration configuration_for(const std::string &case_name) {
  CaseConfiguration configuration;
  if (case_name == "sparse-k3") {
    configuration.sample_num = 1950;
    configuration.feature_num = 320;
    configuration.class_num = 3;
    configuration.signal_feature_num = 9;
    configuration.seed = 8101u;
    configuration.within_block_correlation = 0.12;
    configuration.signal_scale = 0.52;
    configuration.final_lambda_ratio = 0.30;
    configuration.dense_control = false;
  } else if (case_name == "wide-k4") {
    configuration.sample_num = 480;
    configuration.feature_num = 1400;
    configuration.class_num = 4;
    configuration.signal_feature_num = 11;
    configuration.seed = 8107u;
    configuration.within_block_correlation = 0.10;
    configuration.signal_scale = 0.50;
    configuration.final_lambda_ratio = 0.32;
    configuration.dense_control = false;
  } else if (case_name == "high-k12") {
    configuration.sample_num = 720;
    configuration.feature_num = 360;
    configuration.class_num = 12;
    configuration.signal_feature_num = 13;
    configuration.seed = 8111u;
    configuration.within_block_correlation = 0.08;
    configuration.signal_scale = 0.30;
    configuration.final_lambda_ratio = 0.42;
    configuration.dense_control = false;
  } else if (case_name == "dense-control") {
    configuration.sample_num = 400;
    configuration.feature_num = 96;
    configuration.class_num = 4;
    configuration.signal_feature_num = 82;
    configuration.seed = 8117u;
    configuration.within_block_correlation = 0.10;
    configuration.signal_scale = 0.13;
    configuration.final_lambda_ratio = 0.015;
    configuration.dense_control = true;
  } else {
    throw std::invalid_argument("unknown active-set benchmark case: " +
                                case_name);
  }
  return configuration;
}

DataSet make_data(const CaseConfiguration &configuration) {
  std::mt19937 generator(configuration.seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  DataSet data;
  data.x.resize(configuration.sample_num, configuration.feature_num);
  const int correlation_block_size = 8;
  const double common_scale =
      std::sqrt(configuration.within_block_correlation);
  const double independent_scale =
      std::sqrt(1.0 - configuration.within_block_correlation);
  for (int observation = 0; observation < configuration.sample_num;
       ++observation) {
    double common = normal(generator);
    for (int feature = 0; feature < configuration.feature_num; ++feature) {
      if (feature % correlation_block_size == 0)
        common = normal(generator);
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
    const int feature =
        configuration.dense_control
            ? signal
            : (17 + 37 * signal) % configuration.feature_num;
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
        0.08 * (static_cast<double>(klass) -
                0.5 * static_cast<double>(configuration.class_num - 1));
  }

  const Eigen::MatrixXd logits =
      data.x * true_beta +
      true_intercept.transpose().replicate(configuration.sample_num, 1);
  data.labels.resize(configuration.sample_num);
  Eigen::VectorXi class_counts =
      Eigen::VectorXi::Zero(configuration.class_num);
  Eigen::VectorXd weights(configuration.class_num);
  for (int observation = 0; observation < configuration.sample_num;
       ++observation) {
    const double maximum = logits.row(observation).maxCoeff();
    double weight_sum = 0.0;
    for (int klass = 0; klass < configuration.class_num; ++klass) {
      weights[klass] = std::exp(logits(observation, klass) - maximum);
      weight_sum += weights[klass];
    }
    const double draw = uniform(generator) * weight_sum;
    double cumulative = 0.0;
    int sampled_class = configuration.class_num - 1;
    for (int klass = 0; klass < configuration.class_num; ++klass) {
      cumulative += weights[klass];
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
    throw std::invalid_argument("final lambda ratio must lie in (0, 1)");
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

double coefficient_kkt(double coefficient, double gradient, double lambda,
                       double zero_tolerance) {
  if (coefficient > zero_tolerance)
    return std::fabs(gradient + lambda);
  if (coefficient < -zero_tolerance)
    return std::fabs(gradient - lambda);
  return std::max(0.0, std::fabs(gradient) - lambda);
}

OracleSummary streaming_oracle(
    const picasso::MultinomialObjective &objective,
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    double lambda,
    const picasso::solver::MultinomialActNewtonOptions &options) {
  const Eigen::MatrixXd &x = objective.design_matrix();
  const Eigen::VectorXi &labels = objective.labels();
  const int n = objective.sample_num();
  const int d = objective.feature_num();
  const int class_num = objective.class_num();
  const double inverse_n = 1.0 / static_cast<double>(n);

  Eigen::MatrixXd beta_gradient = Eigen::MatrixXd::Zero(d, class_num);
  Eigen::VectorXd intercept_gradient = Eigen::VectorXd::Zero(class_num);
  Eigen::VectorXd logits(class_num);
  Eigen::VectorXd exponentials(class_num);
  double negative_log_likelihood = 0.0;
  double probability_digest = 0.0;

  for (int observation = 0; observation < n; ++observation) {
    for (int klass = 0; klass < class_num; ++klass) {
      double value = intercept[klass];
      for (int feature = 0; feature < d; ++feature)
        value += x(observation, feature) * beta(feature, klass);
      logits[klass] = value;
    }
    const double maximum = logits.maxCoeff();
    double exponential_sum = 0.0;
    for (int klass = 0; klass < class_num; ++klass) {
      exponentials[klass] = std::exp(logits[klass] - maximum);
      exponential_sum += exponentials[klass];
    }
    if (!std::isfinite(maximum) || !std::isfinite(exponential_sum) ||
        !(exponential_sum > 0.0)) {
      OracleSummary failure;
      failure.objective = std::numeric_limits<double>::quiet_NaN();
      failure.kkt_residual = std::numeric_limits<double>::infinity();
      failure.probability_digest =
          std::numeric_limits<double>::quiet_NaN();
      failure.finite = false;
      return failure;
    }
    negative_log_likelihood +=
        std::log(exponential_sum) +
        maximum - logits[labels[observation]];

    for (int klass = 0; klass < class_num; ++klass) {
      const double probability = exponentials[klass] / exponential_sum;
      const double residual =
          probability - (labels[observation] == klass ? 1.0 : 0.0);
      intercept_gradient[klass] += inverse_n * residual;
      for (int feature = 0; feature < d; ++feature) {
        beta_gradient(feature, klass) +=
            inverse_n * x(observation, feature) * residual;
      }
      const int digest_integer =
          1 + (observation * 17 + klass * 13) % 29;
      probability_digest +=
          static_cast<double>(digest_integer) * probability;
    }
  }

  OracleSummary summary;
  summary.objective =
      inverse_n * negative_log_likelihood +
      lambda * beta.cwiseAbs().sum();
  summary.kkt_residual = 0.0;
  for (int feature = 0; feature < d; ++feature) {
    for (int klass = 0; klass < class_num; ++klass) {
      summary.kkt_residual = std::max(
          summary.kkt_residual,
          coefficient_kkt(beta(feature, klass),
                          beta_gradient(feature, klass), lambda,
                          options.zero_tolerance));
    }
  }
  if (options.include_intercept) {
    summary.kkt_residual = std::max(
        summary.kkt_residual,
        intercept_gradient.cwiseAbs().maxCoeff());
  }
  summary.probability_digest =
      probability_digest / static_cast<double>(n * class_num);
  summary.finite =
      std::isfinite(summary.objective) &&
      std::isfinite(summary.kkt_residual) &&
      std::isfinite(summary.probability_digest);
  return summary;
}

int status_index(picasso::solver::MultinomialSolverStatus status) {
  switch (status) {
    case picasso::solver::MultinomialSolverStatus::kConverged:
      return 0;
    case picasso::solver::MultinomialSolverStatus::kOuterIterationLimit:
      return 1;
    case picasso::solver::MultinomialSolverStatus::kInnerIterationLimit:
      return 2;
    case picasso::solver::MultinomialSolverStatus::kLineSearchFailed:
      return 3;
    case picasso::solver::MultinomialSolverStatus::kNoDescentDirection:
      return 4;
    case picasso::solver::MultinomialSolverStatus::kNumericalFailure:
      return 5;
  }
  return 5;
}

int count_nonzero_features(const Eigen::MatrixXd &beta,
                           double tolerance) {
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

PathSummary run_streaming_path(
    const picasso::MultinomialObjective &objective,
    const picasso::solver::MultinomialActNewtonSolver &solver,
    const picasso::solver::MultinomialActNewtonOptions &options,
    const std::vector<double> &lambda_ratios, double lambda_max,
    const Eigen::VectorXd &initial_intercept) {
  const int d = objective.feature_num();
  const int class_num = objective.class_num();
  Eigen::MatrixXd warm_beta = Eigen::MatrixXd::Zero(d, class_num);
  Eigen::VectorXd warm_intercept = initial_intercept;
  picasso::solver::MultinomialActNewtonPathSolver path_solver(objective,
                                                               options);
  picasso::solver::MultinomialActNewtonPathState path_state;
  PathSummary summary;

  for (std::size_t point = 0; point < lambda_ratios.size(); ++point) {
    const double lambda = lambda_max * lambda_ratios[point];
    const double point_weight = static_cast<double>(point + 1);
    picasso::solver::MultinomialSolverStatus point_status =
        picasso::solver::MultinomialSolverStatus::kNumericalFailure;
    double reported_objective = std::numeric_limits<double>::quiet_NaN();
    double reported_kkt = std::numeric_limits<double>::infinity();
    int point_outer_iterations = 0;
    int point_inner_sweeps = 0;
    long long point_coordinate_visits = 0;
    int point_full_scans = 0;
    int point_reactivations = 0;
    int point_subproblem_reactivations = 0;
    int point_outer_reactivations = 0;
    int point_final_active = 0;
    int point_peak_active = 0;
    int point_initial_strong = 0;
    int point_strong_rule_activated = 0;
    int point_full_kkt_path_reactivations = 0;

    {
      const std::chrono::steady_clock::time_point start =
          std::chrono::steady_clock::now();
      picasso::solver::MultinomialActNewtonResult result;
      if (options.use_active_set) {
        picasso::solver::MultinomialActNewtonPathResult path_result =
            path_solver.solve(lambda, &path_state);
        point_initial_strong = path_result.initial_strong_features;
        point_strong_rule_activated =
            path_result.strong_rule_activated_features;
        point_full_kkt_path_reactivations =
            path_result.full_kkt_reactivated_features;
        result = std::move(path_result.solution);
      } else {
        result = solver.solve(lambda, warm_beta, warm_intercept);
      }
      // Persisting the next warm start is part of the timed path operation.
      warm_beta = result.beta;
      warm_intercept = result.intercept;
      const std::chrono::steady_clock::time_point finish =
          std::chrono::steady_clock::now();
      summary.wall_seconds +=
          std::chrono::duration_cast<std::chrono::duration<double> >(
              finish - start)
              .count();

      point_status = result.status;
      reported_objective = result.final_objective;
      reported_kkt = result.final_kkt_residual;
      point_outer_iterations = result.outer_iterations;
      point_inner_sweeps = result.total_inner_sweeps;
      point_coordinate_visits = result.total_coordinate_updates;
      point_full_scans = result.total_full_subproblem_kkt_scans;
      point_reactivations = result.total_reactivated_features;
      point_subproblem_reactivations =
          result.total_subproblem_reactivated_features;
      point_outer_reactivations = result.total_outer_reactivated_features;
      point_final_active = result.final_active_features;
      point_peak_active = result.final_active_features;
      for (std::size_t record = 0; record < result.history.size(); ++record) {
        point_peak_active =
            std::max(point_peak_active,
                     result.history[record].active_features);
      }
      // result (including its probability-free accepted history) is destroyed
      // here; no per-point object is retained across the lambda path.
    }

    const std::chrono::steady_clock::time_point oracle_start =
        std::chrono::steady_clock::now();
    const OracleSummary oracle = streaming_oracle(
        objective, warm_beta, warm_intercept, lambda, options);
    summary.oracle_seconds +=
        std::chrono::duration_cast<std::chrono::duration<double> >(
            std::chrono::steady_clock::now() - oracle_start)
            .count();

    const int point_status_index = status_index(point_status);
    ++summary.status_counts[static_cast<std::size_t>(point_status_index)];
    summary.outer_iterations += point_outer_iterations;
    summary.inner_sweeps += point_inner_sweeps;
    summary.coordinate_visits += point_coordinate_visits;
    summary.full_subproblem_kkt_scans += point_full_scans;
    summary.reactivated_features += point_reactivations;
    summary.subproblem_reactivated_features +=
        point_subproblem_reactivations;
    summary.outer_reactivated_features += point_outer_reactivations;
    summary.initial_strong_features += point_initial_strong;
    summary.strong_rule_activated_features +=
        point_strong_rule_activated;
    summary.full_kkt_path_reactivations +=
        point_full_kkt_path_reactivations;
    summary.final_active_features = point_final_active;
    summary.peak_active_features =
        std::max(summary.peak_active_features, point_peak_active);
    summary.final_status = point_status;
    summary.final_reported_objective = reported_objective;
    summary.final_oracle_objective = oracle.objective;
    summary.final_reported_kkt = reported_kkt;
    summary.final_independent_kkt = oracle.kkt_residual;

    summary.lambda_digest += point_weight * lambda;
    summary.status_digest +=
        point_weight * static_cast<double>(point_status_index + 1);
    summary.reported_objective_digest +=
        point_weight * reported_objective;
    summary.oracle_objective_digest += point_weight * oracle.objective;
    summary.reported_kkt_digest += point_weight * reported_kkt;
    summary.independent_kkt_digest +=
        point_weight * oracle.kkt_residual;
    summary.probability_digest +=
        point_weight * oracle.probability_digest;
    summary.maximum_independent_kkt =
        std::max(summary.maximum_independent_kkt, oracle.kkt_residual);
    summary.maximum_kkt_report_error =
        std::max(summary.maximum_kkt_report_error,
                 std::fabs(reported_kkt - oracle.kkt_residual));
    summary.maximum_objective_report_error =
        std::max(summary.maximum_objective_report_error,
                 std::fabs(reported_objective - oracle.objective));
    summary.maximum_objective_scale =
        std::max(summary.maximum_objective_scale,
                 std::max(std::fabs(reported_objective),
                          std::fabs(oracle.objective)));
    summary.all_converged =
        summary.all_converged &&
        point_status ==
            picasso::solver::MultinomialSolverStatus::kConverged;
    summary.all_finite =
        summary.all_finite && oracle.finite &&
        std::isfinite(reported_objective) && std::isfinite(reported_kkt);
  }

  summary.final_nonzero_features =
      count_nonzero_features(warm_beta, 1e-10);
  summary.final_nonzero_coefficients =
      count_nonzero_coefficients(warm_beta, 1e-10);
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

void print_json(const std::string &case_name, bool active_set,
                const CaseConfiguration &configuration,
                const std::vector<double> &lambda_ratios,
                double lambda_max,
                const picasso::solver::MultinomialActNewtonOptions &options,
                const PathSummary &summary) {
  const double active_fraction =
      static_cast<double>(summary.final_active_features) /
      static_cast<double>(configuration.feature_num);
  const double nonzero_feature_fraction =
      static_cast<double>(summary.final_nonzero_features) /
      static_cast<double>(configuration.feature_num);
  const long long intercept_coordinate_visits =
      static_cast<long long>(summary.inner_sweeps) *
      static_cast<long long>(configuration.class_num);
  const long long beta_coordinate_visits =
      summary.coordinate_visits - intercept_coordinate_visits;
  const double objective_tolerance =
      1e-10 + 2e-9 * summary.maximum_objective_scale;
  const bool kkt_valid =
      summary.maximum_independent_kkt <=
          1.10 * options.outer_kkt_tolerance &&
      summary.maximum_kkt_report_error <= 5e-10;
  const bool objective_valid =
      summary.maximum_objective_report_error <= objective_tolerance;
  const bool coordinate_split_valid = beta_coordinate_visits >= 0;
  const bool profile_gate_applicable =
      configuration.dense_control || active_set;
  const bool profile_gate_passed =
      configuration.dense_control
          ? nonzero_feature_fraction >= 0.60
          : (!active_set || active_fraction < 0.25);
  const bool passed =
      summary.all_converged && summary.all_finite && kkt_valid &&
      objective_valid && coordinate_split_valid && profile_gate_passed;
  const long long rss_raw = peak_rss_raw();

  std::cout
      << std::setprecision(17)
      << "{\n"
      << "  \"schema\": \"multinomial-active-set-benchmark-v2\",\n"
      << "  \"case\": \"" << case_name << "\",\n"
      << "  \"active\": \"" << (active_set ? "on" : "off")
      << "\",\n"
      << "  \"n\": " << configuration.sample_num << ",\n"
      << "  \"d\": " << configuration.feature_num << ",\n"
      << "  \"classes\": " << configuration.class_num << ",\n"
      << "  \"path_points\": " << lambda_ratios.size() << ",\n"
      << "  \"lambda_max\": " << lambda_max << ",\n"
      << "  \"first_lambda_ratio\": " << lambda_ratios.front()
      << ",\n"
      << "  \"final_lambda_ratio\": " << lambda_ratios.back()
      << ",\n"
      << "  \"status_counts\": {\n"
      << "    \"converged\": " << summary.status_counts[0] << ",\n"
      << "    \"outer_iteration_limit\": " << summary.status_counts[1]
      << ",\n"
      << "    \"inner_iteration_limit\": " << summary.status_counts[2]
      << ",\n"
      << "    \"line_search_failed\": " << summary.status_counts[3]
      << ",\n"
      << "    \"no_descent_direction\": " << summary.status_counts[4]
      << ",\n"
      << "    \"numerical_failure\": " << summary.status_counts[5]
      << "\n  },\n"
      << "  \"digests\": {\n"
      << "    \"lambda\": " << summary.lambda_digest << ",\n"
      << "    \"status\": " << summary.status_digest << ",\n"
      << "    \"reported_objective\": "
      << summary.reported_objective_digest << ",\n"
      << "    \"oracle_objective\": "
      << summary.oracle_objective_digest << ",\n"
      << "    \"reported_kkt\": " << summary.reported_kkt_digest
      << ",\n"
      << "    \"independent_kkt\": " << summary.independent_kkt_digest
      << ",\n"
      << "    \"probability\": " << summary.probability_digest
      << "\n  },\n"
      << "  \"totals\": {\n"
      << "    \"outer_iterations\": " << summary.outer_iterations
      << ",\n"
      << "    \"inner_sweeps\": " << summary.inner_sweeps << ",\n"
      << "    \"coordinate_visits_total_from_api\": "
      << summary.coordinate_visits << ",\n"
      << "    \"beta_coordinate_visits_derived\": "
      << beta_coordinate_visits << ",\n"
      << "    \"intercept_coordinate_visits_derived\": "
      << intercept_coordinate_visits << ",\n"
      << "    \"coordinate_split_method\": "
         "\"total-minus-inner-sweeps-times-k\",\n"
      << "    \"full_subproblem_kkt_scans\": "
      << summary.full_subproblem_kkt_scans << ",\n"
      << "    \"reactivated_features\": "
      << summary.reactivated_features << ",\n"
      << "    \"subproblem_reactivated_features\": "
      << summary.subproblem_reactivated_features << ",\n"
      << "    \"outer_reactivated_features\": "
      << summary.outer_reactivated_features << ",\n"
      << "    \"initial_strong_features_sum\": "
      << summary.initial_strong_features << ",\n"
      << "    \"strong_rule_activated_features\": "
      << summary.strong_rule_activated_features << ",\n"
      << "    \"full_kkt_path_reactivations\": "
      << summary.full_kkt_path_reactivations << ",\n"
      << "    \"wall_seconds_solve_and_state_copy\": "
      << summary.wall_seconds << ",\n"
      << "    \"oracle_seconds_outside_timer\": "
      << summary.oracle_seconds << "\n"
      << "  },\n"
      << "  \"active_features\": {\n"
      << "    \"final\": " << summary.final_active_features << ",\n"
      << "    \"peak\": " << summary.peak_active_features << ",\n"
      << "    \"final_fraction\": " << active_fraction << "\n"
      << "  },\n"
      << "  \"final_support\": {\n"
      << "    \"nonzero_features\": " << summary.final_nonzero_features
      << ",\n"
      << "    \"nonzero_coefficients\": "
      << summary.final_nonzero_coefficients << ",\n"
      << "    \"nonzero_feature_fraction\": "
      << nonzero_feature_fraction << "\n"
      << "  },\n"
      << "  \"final_point\": {\n"
      << "    \"status\": \""
      << picasso::solver::multinomial_solver_status_string(
             summary.final_status)
      << "\",\n"
      << "    \"reported_objective\": "
      << summary.final_reported_objective << ",\n"
      << "    \"oracle_objective\": " << summary.final_oracle_objective
      << ",\n"
      << "    \"reported_kkt\": " << summary.final_reported_kkt
      << ",\n"
      << "    \"independent_kkt\": " << summary.final_independent_kkt
      << "\n  },\n"
      << "  \"validation\": {\n"
      << "    \"all_converged\": "
      << (summary.all_converged ? "true" : "false") << ",\n"
      << "    \"all_finite\": "
      << (summary.all_finite ? "true" : "false") << ",\n"
      << "    \"maximum_independent_kkt\": "
      << summary.maximum_independent_kkt << ",\n"
      << "    \"maximum_kkt_report_error\": "
      << summary.maximum_kkt_report_error << ",\n"
      << "    \"maximum_objective_report_error\": "
      << summary.maximum_objective_report_error << ",\n"
      << "    \"kkt_valid\": " << (kkt_valid ? "true" : "false")
      << ",\n"
      << "    \"objective_valid\": "
      << (objective_valid ? "true" : "false") << ",\n"
      << "    \"coordinate_split_valid\": "
      << (coordinate_split_valid ? "true" : "false") << ",\n"
      << "    \"profile_gate_applicable\": "
      << (profile_gate_applicable ? "true" : "false") << ",\n"
      << "    \"profile_gate_passed\": "
      << (profile_gate_passed ? "true" : "false") << "\n"
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
  std::string active_value;
  try {
    for (int argument = 1; argument < argc; ++argument) {
      const std::string key(argv[argument]);
      if (key == "--help" || key == "-h") {
        usage(argv[0]);
        return 0;
      }
      if (key == "--case" && argument + 1 < argc) {
        case_name = argv[++argument];
      } else if (key == "--active" && argument + 1 < argc) {
        active_value = argv[++argument];
      } else {
        usage(argv[0]);
        throw std::invalid_argument("unknown or incomplete argument: " + key);
      }
    }
    if (case_name.empty() || active_value.empty()) {
      usage(argv[0]);
      throw std::invalid_argument("both --case and --active are required");
    }

    const bool active_set = parse_active_switch(active_value);
    const CaseConfiguration configuration = configuration_for(case_name);
    DataSet data = make_data(configuration);
    const Eigen::VectorXd null_intercept =
        empirical_null_intercept(data.labels, configuration.class_num);
    picasso::MultinomialObjective objective(
        std::move(data.x), std::move(data.labels), configuration.class_num);

    const Eigen::MatrixXd zero_beta = Eigen::MatrixXd::Zero(
        configuration.feature_num, configuration.class_num);
    Eigen::MatrixXd lambda_gradient;
    Eigen::VectorXd ignored_intercept_gradient;
    objective.smooth_gradient(zero_beta, null_intercept, &lambda_gradient,
                              &ignored_intercept_gradient);
    const double lambda_max = lambda_gradient.cwiseAbs().maxCoeff();
    if (!(lambda_max > 0.0) || !std::isfinite(lambda_max))
      throw std::runtime_error("computed lambda_max is not positive finite");
    const std::vector<double> lambda_ratios =
        make_lambda_ratios(configuration.final_lambda_ratio);

    picasso::solver::MultinomialActNewtonOptions options;
    options.max_outer_iterations = 100;
    options.max_inner_sweeps = 4000;
    options.outer_kkt_tolerance = 2e-7;
    options.inner_kkt_tolerance = 2e-9;
    options.use_probability_dot_direction_cache = true;
    options.use_active_set = active_set;
    options.canonicalize_feature_l1_gauge = true;
    options.use_compact_inner_active_set = active_set;
    picasso::solver::MultinomialActNewtonSolver solver(objective, options);

    const PathSummary summary = run_streaming_path(
        objective, solver, options, lambda_ratios, lambda_max,
        null_intercept);
    print_json(case_name, active_set, configuration, lambda_ratios,
               lambda_max, options, summary);

    const double active_fraction =
        static_cast<double>(summary.final_active_features) /
        static_cast<double>(configuration.feature_num);
    const double nonzero_fraction =
        static_cast<double>(summary.final_nonzero_features) /
        static_cast<double>(configuration.feature_num);
    const bool profile_gate_passed =
        configuration.dense_control
            ? nonzero_fraction >= 0.60
            : (!active_set || active_fraction < 0.25);
    const double objective_tolerance =
        1e-10 + 2e-9 * summary.maximum_objective_scale;
    const bool passed =
        summary.all_converged && summary.all_finite &&
        summary.maximum_independent_kkt <=
            1.10 * options.outer_kkt_tolerance &&
        summary.maximum_kkt_report_error <= 5e-10 &&
        summary.maximum_objective_report_error <= objective_tolerance &&
        profile_gate_passed;
    return passed ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "active-set benchmark error: " << error.what() << "\n";
    return 1;
  }
}
