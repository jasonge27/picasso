#include <picasso/multinomial_actnewton.hpp>
#include <picasso/multinomial_objective.hpp>

// Standalone Phase-7 experiment; it does not require a path API in the solver.
// Build from the repository root:
//   c++ -O3 -DNDEBUG -std=c++11 -Iinclude \
//     -IR-package/src/include/eigen3 \
//     profiling/multinomial_path_benchmark.cpp \
//     src/objective/multinomial_objective.cpp \
//     src/solver/multinomial_actnewton.cpp -o /tmp/multinomial_path_benchmark
// Correctness comparison and independent-process timing examples:
//   /tmp/multinomial_path_benchmark --case k3 --mode compare
//   /tmp/multinomial_path_benchmark --case k3 --mode cold-null
//   /tmp/multinomial_path_benchmark --case k3 --mode warm

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

namespace {

const double kObjectiveAbsoluteTolerance = 1e-9;
const double kObjectiveRelativeTolerance = 2e-8;
const double kProbabilityTolerance = 5e-6;

struct CaseConfiguration {
  int sample_num;
  int feature_num;
  int class_num;
  unsigned int seed;
};

struct DataSet {
  Eigen::MatrixXd x;
  Eigen::VectorXi labels;
};

struct FitMeasurement {
  picasso::solver::MultinomialActNewtonResult result;
  Eigen::MatrixXd probabilities;
  double independent_kkt;
  double seconds;
  int nonzero_coefficients;
};

struct Totals {
  int outer_iterations;
  int inner_sweeps;
  long long coordinate_updates;
  double seconds;

  Totals()
      : outer_iterations(0),
        inner_sweeps(0),
        coordinate_updates(0),
        seconds(0.0) {}

  void add(const FitMeasurement &measurement) {
    outer_iterations += measurement.result.outer_iterations;
    inner_sweeps += measurement.result.total_inner_sweeps;
    coordinate_updates += measurement.result.total_coordinate_updates;
    seconds += measurement.seconds;
  }
};

enum class BenchmarkMode {
  kColdZero,
  kColdNull,
  kWarm,
  kCompare
};

struct PathRun {
  BenchmarkMode mode;
  std::vector<FitMeasurement> fits;
  Totals totals;
  double objective_digest;
  double probability_digest;
  double kkt_digest;
  bool all_converged;

  PathRun()
      : mode(BenchmarkMode::kColdZero),
        objective_digest(0.0),
        probability_digest(0.0),
        kkt_digest(0.0),
        all_converged(true) {}
};

struct PointComparison {
  double lambda_ratio;
  double lambda;
  FitMeasurement cold_zero;
  FitMeasurement cold_null;
  FitMeasurement warm;
  double maximum_objective_difference;
  double maximum_probability_difference;
  double maximum_kkt_report_error;
  bool equivalent;
};

void usage(const char *program) {
  std::cerr << "Usage: " << program
            << " --case k3|k4|k8 "
               "--mode cold-zero|cold-null|warm|compare\n"
            << "Single modes run exactly one complete path per process; "
               "compare validates all three paths point by point.\n";
}

BenchmarkMode parse_mode(const std::string &mode) {
  if (mode == "cold-zero") return BenchmarkMode::kColdZero;
  if (mode == "cold-null") return BenchmarkMode::kColdNull;
  if (mode == "warm") return BenchmarkMode::kWarm;
  if (mode == "compare") return BenchmarkMode::kCompare;
  throw std::invalid_argument("unknown benchmark mode: " + mode);
}

const char *mode_name(BenchmarkMode mode) {
  switch (mode) {
    case BenchmarkMode::kColdZero:
      return "cold-zero";
    case BenchmarkMode::kColdNull:
      return "cold-null";
    case BenchmarkMode::kWarm:
      return "warm";
    case BenchmarkMode::kCompare:
      return "compare";
  }
  return "unknown";
}

CaseConfiguration configuration_for(const std::string &case_name) {
  CaseConfiguration configuration;
  if (case_name == "k3") {
    configuration.sample_num = 320;
    configuration.feature_num = 64;
    configuration.class_num = 3;
    configuration.seed = 7001u;
  } else if (case_name == "k4") {
    configuration.sample_num = 360;
    configuration.feature_num = 80;
    configuration.class_num = 4;
    configuration.seed = 7003u;
  } else if (case_name == "k8") {
    configuration.sample_num = 480;
    configuration.feature_num = 96;
    configuration.class_num = 8;
    configuration.seed = 7007u;
  } else {
    throw std::invalid_argument("unknown path benchmark case: " + case_name);
  }
  return configuration;
}

DataSet make_data(const CaseConfiguration &configuration) {
  std::mt19937 generator(configuration.seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  DataSet data;
  data.x.resize(configuration.sample_num, configuration.feature_num);
  for (int i = 0; i < configuration.sample_num; ++i) {
    for (int j = 0; j < configuration.feature_num; ++j)
      data.x(i, j) = normal(generator);
  }
  for (int j = 0; j < configuration.feature_num; ++j) {
    data.x.col(j).array() -= data.x.col(j).mean();
    const double scale =
        std::sqrt(data.x.col(j).squaredNorm() /
                  static_cast<double>(configuration.sample_num));
    data.x.col(j) /= scale;
  }

  Eigen::MatrixXd true_beta = Eigen::MatrixXd::Zero(
      configuration.feature_num, configuration.class_num);
  for (int klass = 0; klass < configuration.class_num; ++klass) {
    for (int active = 0; active < 6; ++active) {
      const int feature =
          (klass * 13 + active * 19) % configuration.feature_num;
      const double sign = ((klass + active) % 2 == 0) ? 1.0 : -1.0;
      true_beta(feature, klass) += sign * (0.75 - 0.08 * active);
    }
  }
  Eigen::VectorXd true_intercept(configuration.class_num);
  for (int klass = 0; klass < configuration.class_num; ++klass) {
    true_intercept[klass] =
        0.10 * (static_cast<double>(klass) -
                0.5 * static_cast<double>(configuration.class_num - 1));
  }

  const Eigen::MatrixXd logits =
      data.x * true_beta +
      true_intercept.transpose().replicate(configuration.sample_num, 1);
  data.labels.resize(configuration.sample_num);
  for (int i = 0; i < configuration.sample_num; ++i) {
    const double maximum = logits.row(i).maxCoeff();
    Eigen::VectorXd weights(configuration.class_num);
    double weight_sum = 0.0;
    for (int klass = 0; klass < configuration.class_num; ++klass) {
      weights[klass] = std::exp(logits(i, klass) - maximum);
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
    data.labels[i] = sampled_class;
  }
  return data;
}

Eigen::VectorXd null_intercept(const Eigen::VectorXi &labels,
                               int num_classes) {
  Eigen::VectorXd counts = Eigen::VectorXd::Zero(num_classes);
  for (Eigen::Index i = 0; i < labels.size(); ++i) counts[labels[i]] += 1.0;
  Eigen::VectorXd intercept(num_classes);
  for (int klass = 0; klass < num_classes; ++klass) {
    if (!(counts[klass] > 0.0))
      throw std::runtime_error("path benchmark generated an empty class");
    intercept[klass] =
        std::log(counts[klass] / static_cast<double>(labels.size()));
  }
  intercept.array() -= intercept.mean();
  return intercept;
}

double coefficient_kkt(double coefficient, double gradient, double lambda,
                       double zero_tolerance) {
  if (coefficient > zero_tolerance)
    return std::fabs(gradient + lambda);
  if (coefficient < -zero_tolerance)
    return std::fabs(gradient - lambda);
  return std::max(0.0, std::fabs(gradient) - lambda);
}

double independent_kkt(
    const picasso::MultinomialObjective &objective,
    const Eigen::MatrixXd &beta, const Eigen::VectorXd &intercept,
    double lambda,
    const picasso::solver::MultinomialActNewtonOptions &options) {
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  objective.smooth_gradient(beta, intercept, &beta_gradient,
                            &intercept_gradient);
  double residual = 0.0;
  for (Eigen::Index j = 0; j < beta.rows(); ++j) {
    for (Eigen::Index klass = 0; klass < beta.cols(); ++klass) {
      residual = std::max(
          residual,
          coefficient_kkt(beta(j, klass), beta_gradient(j, klass), lambda,
                          options.zero_tolerance));
    }
  }
  if (options.include_intercept)
    residual = std::max(residual,
                        intercept_gradient.cwiseAbs().maxCoeff());
  return residual;
}

FitMeasurement measure_fit(
    const picasso::MultinomialObjective &objective,
    const picasso::solver::MultinomialActNewtonSolver &solver,
    const picasso::solver::MultinomialActNewtonOptions &options,
    double lambda, const Eigen::MatrixXd &initial_beta,
    const Eigen::VectorXd &initial_intercept) {
  const std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  picasso::solver::MultinomialActNewtonResult result =
      solver.solve(lambda, initial_beta, initial_intercept);
  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double> >(
          std::chrono::steady_clock::now() - start)
          .count();

  FitMeasurement measurement;
  measurement.result = std::move(result);
  objective.negative_log_likelihood(measurement.result.beta,
                                    measurement.result.intercept,
                                    &measurement.probabilities);
  measurement.independent_kkt = independent_kkt(
      objective, measurement.result.beta, measurement.result.intercept,
      lambda, options);
  measurement.seconds = seconds;
  measurement.nonzero_coefficients = 0;
  for (Eigen::Index j = 0; j < measurement.result.beta.rows(); ++j) {
    for (Eigen::Index klass = 0;
         klass < measurement.result.beta.cols(); ++klass) {
      if (std::fabs(measurement.result.beta(j, klass)) > 1e-10)
        ++measurement.nonzero_coefficients;
    }
  }
  return measurement;
}

PathRun run_path(
    BenchmarkMode mode,
    const picasso::MultinomialObjective &objective,
    const picasso::solver::MultinomialActNewtonSolver &solver,
    const picasso::solver::MultinomialActNewtonOptions &options,
    const std::vector<double> &lambdas, const Eigen::MatrixXd &zero_beta,
    const Eigen::VectorXd &zero_intercept,
    const Eigen::VectorXd &empirical_null_intercept) {
  if (mode == BenchmarkMode::kCompare)
    throw std::invalid_argument("compare is not a single path mode");

  PathRun path;
  path.mode = mode;
  Eigen::MatrixXd warm_beta = zero_beta;
  Eigen::VectorXd warm_intercept = empirical_null_intercept;
  for (std::size_t index = 0; index < lambdas.size(); ++index) {
    const Eigen::MatrixXd *initial_beta = &zero_beta;
    const Eigen::VectorXd *initial_intercept = &zero_intercept;
    if (mode == BenchmarkMode::kColdNull) {
      initial_intercept = &empirical_null_intercept;
    } else if (mode == BenchmarkMode::kWarm) {
      initial_beta = &warm_beta;
      initial_intercept = &warm_intercept;
    }
    FitMeasurement measurement = measure_fit(
        objective, solver, options, lambdas[index], *initial_beta,
        *initial_intercept);
    if (mode == BenchmarkMode::kWarm && measurement.result.converged()) {
      warm_beta = measurement.result.beta;
      warm_intercept = measurement.result.intercept;
    }

    const double point_weight = static_cast<double>(index + 1);
    path.objective_digest +=
        point_weight * measurement.result.final_objective;
    path.kkt_digest += point_weight * measurement.independent_kkt;
    double point_probability_digest = 0.0;
    for (Eigen::Index i = 0; i < measurement.probabilities.rows(); ++i) {
      for (Eigen::Index klass = 0;
           klass < measurement.probabilities.cols(); ++klass) {
        const double entry_weight =
            1.0 + static_cast<double>((i * 7 + klass * 11) % 23);
        point_probability_digest +=
            entry_weight * measurement.probabilities(i, klass);
      }
    }
    path.probability_digest +=
        point_weight * point_probability_digest /
        static_cast<double>(measurement.probabilities.size());
    path.all_converged =
        path.all_converged && measurement.result.converged();
    path.totals.add(measurement);
    path.fits.push_back(std::move(measurement));
  }
  return path;
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

double max_pairwise_objective_difference(const FitMeasurement &first,
                                         const FitMeasurement &second,
                                         const FitMeasurement &third) {
  return std::max(
      std::fabs(first.result.final_objective -
                second.result.final_objective),
      std::max(std::fabs(first.result.final_objective -
                         third.result.final_objective),
               std::fabs(second.result.final_objective -
                         third.result.final_objective)));
}

double max_pairwise_probability_difference(const FitMeasurement &first,
                                           const FitMeasurement &second,
                                           const FitMeasurement &third) {
  return std::max(
      (first.probabilities - second.probabilities).cwiseAbs().maxCoeff(),
      std::max((first.probabilities - third.probabilities)
                       .cwiseAbs()
                       .maxCoeff(),
               (second.probabilities - third.probabilities)
                   .cwiseAbs()
                   .maxCoeff()));
}

double reduction_fraction(long long warm, long long cold) {
  if (cold == 0) return warm == 0 ? 0.0 : -1.0;
  return 1.0 - static_cast<double>(warm) / static_cast<double>(cold);
}

void print_fit(const char *name, const FitMeasurement &measurement) {
  std::cout << "      \"" << name << "\": {"
            << "\"status\": \""
            << picasso::solver::multinomial_solver_status_string(
                   measurement.result.status)
            << "\", \"objective\": "
            << measurement.result.final_objective
            << ", \"reported_kkt\": "
            << measurement.result.final_kkt_residual
            << ", \"independent_kkt\": " << measurement.independent_kkt
            << ", \"nonzero\": " << measurement.nonzero_coefficients
            << ", \"outer\": " << measurement.result.outer_iterations
            << ", \"inner_sweeps\": "
            << measurement.result.total_inner_sweeps
            << ", \"coordinate_updates\": "
            << measurement.result.total_coordinate_updates
            << ", \"wall_seconds\": " << measurement.seconds << "}";
}

void print_totals(const char *name, const Totals &totals) {
  std::cout << "    \"" << name << "\": {"
            << "\"outer\": " << totals.outer_iterations
            << ", \"inner_sweeps\": " << totals.inner_sweeps
            << ", \"coordinate_updates\": " << totals.coordinate_updates
            << ", \"wall_seconds\": " << totals.seconds << "}";
}

bool print_single_path(
    const std::string &case_name,
    const CaseConfiguration &configuration, double lambda_max,
    const std::vector<double> &lambda_ratios,
    const std::vector<double> &lambdas, const PathRun &path,
    const picasso::solver::MultinomialActNewtonOptions &options) {
  bool kkt_valid = true;
  for (std::size_t index = 0; index < path.fits.size(); ++index) {
    const FitMeasurement &fit = path.fits[index];
    kkt_valid =
        kkt_valid &&
        fit.independent_kkt <= 1.05 * options.outer_kkt_tolerance &&
        std::fabs(fit.independent_kkt -
                  fit.result.final_kkt_residual) <= 1e-12;
  }
  const bool passed = path.all_converged && kkt_valid;
  const long long rss_raw = peak_rss_raw();
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"case\": \"" << case_name << "\",\n"
            << "  \"mode\": \"" << mode_name(path.mode) << "\",\n"
            << "  \"n\": " << configuration.sample_num << ",\n"
            << "  \"d\": " << configuration.feature_num << ",\n"
            << "  \"classes\": " << configuration.class_num << ",\n"
            << "  \"lambda_max\": " << lambda_max << ",\n"
            << "  \"points\": [\n";
  for (std::size_t index = 0; index < path.fits.size(); ++index) {
    const FitMeasurement &fit = path.fits[index];
    std::cout << "    {\"lambda_ratio\": " << lambda_ratios[index]
              << ", \"lambda\": " << lambdas[index]
              << ", \"status\": \""
              << picasso::solver::multinomial_solver_status_string(
                     fit.result.status)
              << "\", \"objective\": " << fit.result.final_objective
              << ", \"reported_kkt\": "
              << fit.result.final_kkt_residual
              << ", \"independent_kkt\": " << fit.independent_kkt
              << ", \"nonzero\": " << fit.nonzero_coefficients << "}";
    if (index + 1 != path.fits.size()) std::cout << ",";
    std::cout << "\n";
  }
  std::cout << "  ],\n"
            << "  \"totals\": {\"outer\": "
            << path.totals.outer_iterations
            << ", \"inner_sweeps\": " << path.totals.inner_sweeps
            << ", \"coordinate_updates\": "
            << path.totals.coordinate_updates
            << ", \"wall_seconds\": " << path.totals.seconds << "},\n"
            << "  \"digests\": {\"objective\": "
            << path.objective_digest
            << ", \"probability\": " << path.probability_digest
            << ", \"independent_kkt\": " << path.kkt_digest << "},\n"
            << "  \"ru_maxrss_raw\": " << rss_raw << ",\n"
            << "  \"ru_maxrss_bytes\": " << peak_rss_bytes(rss_raw)
            << ",\n"
            << "  \"all_converged\": "
            << (path.all_converged ? "true" : "false") << ",\n"
            << "  \"kkt_valid\": " << (kkt_valid ? "true" : "false")
            << ",\n"
            << "  \"passed\": " << (passed ? "true" : "false")
            << "\n}\n";
  return passed;
}

}  // namespace

int main(int argc, char **argv) {
  std::string case_name;
  std::string mode_value;
  try {
    for (int argument = 1; argument < argc; ++argument) {
      const std::string key(argv[argument]);
      if (key == "--help" || key == "-h") {
        usage(argv[0]);
        return 0;
      }
      if (key == "--case" && argument + 1 < argc) {
        case_name = argv[++argument];
      } else if (key == "--mode" && argument + 1 < argc) {
        mode_value = argv[++argument];
      } else {
        usage(argv[0]);
        throw std::invalid_argument("unknown or incomplete argument: " + key);
      }
    }
    if (case_name.empty() || mode_value.empty()) {
      usage(argv[0]);
      throw std::invalid_argument("both --case and --mode are required");
    }
    const BenchmarkMode mode = parse_mode(mode_value);

    const CaseConfiguration configuration = configuration_for(case_name);
    DataSet data = make_data(configuration);
    const Eigen::VectorXd empirical_null_intercept =
        null_intercept(data.labels, configuration.class_num);
    picasso::MultinomialObjective objective(
        std::move(data.x), std::move(data.labels), configuration.class_num);
    const Eigen::MatrixXd zero_beta = Eigen::MatrixXd::Zero(
        configuration.feature_num, configuration.class_num);
    const Eigen::VectorXd zero_intercept =
        Eigen::VectorXd::Zero(configuration.class_num);

    Eigen::MatrixXd lambda_gradient;
    Eigen::VectorXd ignored_intercept_gradient;
    objective.smooth_gradient(zero_beta, empirical_null_intercept,
                              &lambda_gradient,
                              &ignored_intercept_gradient);
    const double lambda_max = lambda_gradient.cwiseAbs().maxCoeff();
    const double ratio_values[] =
        {1.02, 0.94, 0.78, 0.60, 0.43, 0.29, 0.18, 0.10, 0.04};
    const std::vector<double> lambda_ratios(
        ratio_values,
        ratio_values + sizeof(ratio_values) / sizeof(ratio_values[0]));
    std::vector<double> lambdas(lambda_ratios.size());
    for (std::size_t index = 0; index < lambda_ratios.size(); ++index)
      lambdas[index] = lambda_max * lambda_ratios[index];

    picasso::solver::MultinomialActNewtonOptions options;
    options.max_outer_iterations = 100;
    options.max_inner_sweeps = 4000;
    options.outer_kkt_tolerance = 1e-7;
    options.inner_kkt_tolerance = 1e-9;
    // Isolate Phase-7 warm-start gains from the later active-set default.
    options.use_active_set = false;
    picasso::solver::MultinomialActNewtonSolver solver(objective, options);

    if (mode != BenchmarkMode::kCompare) {
      const PathRun path = run_path(
          mode, objective, solver, options, lambdas, zero_beta,
          zero_intercept, empirical_null_intercept);
      const bool passed = print_single_path(
          case_name, configuration, lambda_max, lambda_ratios, lambdas, path,
          options);
      return passed ? 0 : 2;
    }

    const PathRun cold_zero_path = run_path(
        BenchmarkMode::kColdZero, objective, solver, options, lambdas,
        zero_beta, zero_intercept, empirical_null_intercept);
    const PathRun cold_null_path = run_path(
        BenchmarkMode::kColdNull, objective, solver, options, lambdas,
        zero_beta, zero_intercept, empirical_null_intercept);
    const PathRun warm_path = run_path(
        BenchmarkMode::kWarm, objective, solver, options, lambdas, zero_beta,
        zero_intercept, empirical_null_intercept);
    const Totals &cold_zero_totals = cold_zero_path.totals;
    const Totals &cold_null_totals = cold_null_path.totals;
    const Totals &warm_totals = warm_path.totals;
    std::vector<PointComparison> points;
    bool every_point_equivalent = true;
    bool all_converged = cold_zero_path.all_converged &&
                         cold_null_path.all_converged &&
                         warm_path.all_converged;

    for (std::size_t index = 0; index < lambda_ratios.size(); ++index) {
      PointComparison point;
      point.lambda_ratio = lambda_ratios[index];
      point.lambda = lambdas[index];
      point.cold_zero = cold_zero_path.fits[index];
      point.cold_null = cold_null_path.fits[index];
      point.warm = warm_path.fits[index];
      point.maximum_objective_difference = max_pairwise_objective_difference(
          point.cold_zero, point.cold_null, point.warm);
      point.maximum_probability_difference =
          max_pairwise_probability_difference(
              point.cold_zero, point.cold_null, point.warm);
      point.maximum_kkt_report_error = std::max(
          std::fabs(point.cold_zero.result.final_kkt_residual -
                    point.cold_zero.independent_kkt),
          std::max(std::fabs(point.cold_null.result.final_kkt_residual -
                             point.cold_null.independent_kkt),
                   std::fabs(point.warm.result.final_kkt_residual -
                             point.warm.independent_kkt)));
      const double objective_scale = std::max(
          1.0, std::max(std::fabs(point.cold_zero.result.final_objective),
                        std::max(std::fabs(
                                     point.cold_null.result.final_objective),
                                 std::fabs(point.warm.result.final_objective))));
      const bool point_converged =
          point.cold_zero.result.converged() &&
          point.cold_null.result.converged() && point.warm.result.converged();
      const bool kkt_valid =
          point.cold_zero.independent_kkt <=
              1.05 * options.outer_kkt_tolerance &&
          point.cold_null.independent_kkt <=
              1.05 * options.outer_kkt_tolerance &&
          point.warm.independent_kkt <=
              1.05 * options.outer_kkt_tolerance &&
          point.maximum_kkt_report_error <= 1e-12;
      point.equivalent =
          point_converged && kkt_valid &&
          point.maximum_objective_difference <=
              kObjectiveAbsoluteTolerance +
                  kObjectiveRelativeTolerance * objective_scale &&
          point.maximum_probability_difference <= kProbabilityTolerance;
      every_point_equivalent = every_point_equivalent && point.equivalent;
      points.push_back(std::move(point));
    }

    const bool sparse_path =
        points.front().warm.nonzero_coefficients == 0 &&
        points.back().warm.nonzero_coefficients >
            points.front().warm.nonzero_coefficients;
    const bool warm_counts_lower =
        warm_totals.outer_iterations < cold_null_totals.outer_iterations &&
        warm_totals.inner_sweeps < cold_null_totals.inner_sweeps &&
        warm_totals.coordinate_updates <
            cold_null_totals.coordinate_updates;
    const bool passed = all_converged && every_point_equivalent &&
                        sparse_path && warm_counts_lower;
    const long long rss_raw = peak_rss_raw();

    std::cout << std::setprecision(17)
              << "{\n"
              << "  \"case\": \"" << case_name << "\",\n"
              << "  \"mode\": \"compare\",\n"
              << "  \"n\": " << configuration.sample_num << ",\n"
              << "  \"d\": " << configuration.feature_num << ",\n"
              << "  \"classes\": " << configuration.class_num << ",\n"
              << "  \"lambda_max\": " << lambda_max << ",\n"
              << "  \"thresholds\": {\"outer_kkt\": "
              << options.outer_kkt_tolerance
              << ", \"objective_abs\": "
              << kObjectiveAbsoluteTolerance
              << ", \"objective_rel\": "
              << kObjectiveRelativeTolerance
              << ", \"probability_max_abs\": "
              << kProbabilityTolerance << "},\n"
              << "  \"points\": [\n";
    for (std::size_t index = 0; index < points.size(); ++index) {
      const PointComparison &point = points[index];
      std::cout << "    {\"lambda_ratio\": " << point.lambda_ratio
                << ", \"lambda\": " << point.lambda << ",\n";
      print_fit("cold_zero", point.cold_zero);
      std::cout << ",\n";
      print_fit("cold_null", point.cold_null);
      std::cout << ",\n";
      print_fit("warm", point.warm);
      std::cout << ",\n"
                << "      \"max_objective_difference\": "
                << point.maximum_objective_difference << ",\n"
                << "      \"max_probability_difference\": "
                << point.maximum_probability_difference << ",\n"
                << "      \"max_kkt_report_error\": "
                << point.maximum_kkt_report_error << ",\n"
                << "      \"equivalent\": "
                << (point.equivalent ? "true" : "false") << "\n"
                << "    }";
      if (index + 1 != points.size()) std::cout << ",";
      std::cout << "\n";
    }
    std::cout << "  ],\n"
              << "  \"totals\": {\n";
    print_totals("cold_zero", cold_zero_totals);
    std::cout << ",\n";
    print_totals("cold_null", cold_null_totals);
    std::cout << ",\n";
    print_totals("warm", warm_totals);
    std::cout << "\n  },\n"
              << "  \"digests\": {\n"
              << "    \"cold_zero\": {\"objective\": "
              << cold_zero_path.objective_digest
              << ", \"probability\": "
              << cold_zero_path.probability_digest
              << ", \"independent_kkt\": "
              << cold_zero_path.kkt_digest << "},\n"
              << "    \"cold_null\": {\"objective\": "
              << cold_null_path.objective_digest
              << ", \"probability\": "
              << cold_null_path.probability_digest
              << ", \"independent_kkt\": "
              << cold_null_path.kkt_digest << "},\n"
              << "    \"warm\": {\"objective\": "
              << warm_path.objective_digest
              << ", \"probability\": " << warm_path.probability_digest
              << ", \"independent_kkt\": " << warm_path.kkt_digest
              << "}\n"
              << "  },\n"
              << "  \"warm_reduction_vs_cold_null\": {"
              << "\"outer\": "
              << reduction_fraction(warm_totals.outer_iterations,
                                    cold_null_totals.outer_iterations)
              << ", \"inner_sweeps\": "
              << reduction_fraction(warm_totals.inner_sweeps,
                                    cold_null_totals.inner_sweeps)
              << ", \"coordinate_updates\": "
              << reduction_fraction(warm_totals.coordinate_updates,
                                    cold_null_totals.coordinate_updates)
              << ", \"wall_seconds\": "
              << reduction_fraction(
                     static_cast<long long>(warm_totals.seconds * 1e12),
                     static_cast<long long>(cold_null_totals.seconds * 1e12))
              << "},\n"
              << "  \"ru_maxrss_raw\": " << rss_raw << ",\n"
              << "  \"ru_maxrss_bytes\": "
              << peak_rss_bytes(rss_raw) << ",\n"
              << "  \"gates\": {"
              << "\"all_converged\": "
              << (all_converged ? "true" : "false")
              << ", \"every_point_equivalent\": "
              << (every_point_equivalent ? "true" : "false")
              << ", \"sparse_path\": "
              << (sparse_path ? "true" : "false")
              << ", \"warm_counts_lower_than_cold_null\": "
              << (warm_counts_lower ? "true" : "false") << "},\n"
              << "  \"passed\": " << (passed ? "true" : "false")
              << "\n}\n";
    return passed ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "path benchmark error: " << error.what() << "\n";
    return 1;
  }
}
