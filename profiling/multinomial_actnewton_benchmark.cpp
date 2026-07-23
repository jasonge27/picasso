#include <picasso/multinomial_actnewton.hpp>
#include <picasso/multinomial_objective.hpp>

// Example (from the repository root):
//   c++ -O3 -DNDEBUG -std=c++11 -Iinclude \
//     -IR-package/src/include/eigen3 \
//     profiling/multinomial_actnewton_benchmark.cpp \
//     src/objective/multinomial_objective.cpp \
//     src/solver/multinomial_actnewton.cpp -o /tmp/multinomial_benchmark
//   /tmp/multinomial_benchmark --case wide --cache on --gauge on

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

namespace {

struct CaseConfiguration {
  int sample_num;
  int feature_num;
  int class_num;
  double lambda;
  unsigned int seed;
};

struct DataSet {
  Eigen::MatrixXd x;
  Eigen::VectorXi labels;
};

void usage(const char *program) {
  std::cerr << "Usage: " << program
            << " --case small|wide|many-classes --cache on|off "
               "[--gauge on|off]\n"
            << "Run cache modes in separate processes so ru_maxrss is not "
               "contaminated by the other mode. Feature gauge defaults on.\n";
}

CaseConfiguration configuration_for(const std::string &case_name) {
  CaseConfiguration configuration;
  if (case_name == "small") {
    configuration.sample_num = 240;
    configuration.feature_num = 24;
    configuration.class_num = 3;
    configuration.lambda = 0.018;
    configuration.seed = 1729u;
  } else if (case_name == "wide") {
    configuration.sample_num = 320;
    configuration.feature_num = 140;
    configuration.class_num = 5;
    configuration.lambda = 0.020;
    configuration.seed = 2718u;
  } else if (case_name == "many-classes") {
    configuration.sample_num = 320;
    configuration.feature_num = 48;
    configuration.class_num = 10;
    configuration.lambda = 0.020;
    configuration.seed = 31415u;
  } else {
    throw std::invalid_argument("unknown benchmark case: " + case_name);
  }
  return configuration;
}

bool parse_on_off(const std::string &value, const std::string &option) {
  if (value == "on") return true;
  if (value == "off") return false;
  throw std::invalid_argument(option + " must be 'on' or 'off'");
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
    const double mean = data.x.col(j).mean();
    data.x.col(j).array() -= mean;
    const double scale =
        std::sqrt(data.x.col(j).squaredNorm() /
                  static_cast<double>(configuration.sample_num));
    data.x.col(j) /= scale;
  }

  Eigen::MatrixXd true_beta = Eigen::MatrixXd::Zero(
      configuration.feature_num, configuration.class_num);
  for (int klass = 0; klass < configuration.class_num; ++klass) {
    for (int offset = 0; offset < 5; ++offset) {
      const int feature =
          (klass * 11 + offset * 17) % configuration.feature_num;
      const double sign = ((klass + offset) % 2 == 0) ? 1.0 : -1.0;
      true_beta(feature, klass) += sign * (0.55 - 0.07 * offset);
    }
  }
  Eigen::VectorXd true_intercept(configuration.class_num);
  for (int klass = 0; klass < configuration.class_num; ++klass)
    true_intercept[klass] =
        0.12 * (static_cast<double>(klass) -
                0.5 * static_cast<double>(configuration.class_num - 1));

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

}  // namespace

int main(int argc, char **argv) {
  std::string case_name;
  bool cache_enabled = false;
  bool cache_was_set = false;
  bool gauge_enabled = true;
  try {
    for (int argument = 1; argument < argc; ++argument) {
      const std::string key(argv[argument]);
      if (key == "--help" || key == "-h") {
        usage(argv[0]);
        return 0;
      }
      if (key == "--case" && argument + 1 < argc) {
        case_name = argv[++argument];
      } else if (key == "--cache" && argument + 1 < argc) {
        cache_enabled = parse_on_off(argv[++argument], "--cache");
        cache_was_set = true;
      } else if (key == "--gauge" && argument + 1 < argc) {
        gauge_enabled = parse_on_off(argv[++argument], "--gauge");
      } else {
        usage(argv[0]);
        throw std::invalid_argument("unknown or incomplete argument: " + key);
      }
    }
    if (case_name.empty() || !cache_was_set) {
      usage(argv[0]);
      throw std::invalid_argument("both --case and --cache are required");
    }

    const CaseConfiguration configuration = configuration_for(case_name);
    DataSet data = make_data(configuration);
    picasso::MultinomialObjective objective(
        std::move(data.x), std::move(data.labels), configuration.class_num);
    picasso::solver::MultinomialActNewtonOptions options;
    options.max_outer_iterations = 80;
    options.max_inner_sweeps = 4000;
    options.outer_kkt_tolerance = 1e-6;
    options.inner_kkt_tolerance = 1e-8;
    options.use_probability_dot_direction_cache = cache_enabled;
    // This benchmark isolates cache/gauge changes from the later active-set
    // phase so archived Phase-5/6 measurements remain reproducible.
    options.use_active_set = false;
    options.canonicalize_feature_l1_gauge = gauge_enabled;
    picasso::solver::MultinomialActNewtonSolver solver(objective, options);

    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    const picasso::solver::MultinomialActNewtonResult result =
        solver.solve(configuration.lambda);
    const std::chrono::steady_clock::time_point finish =
        std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration_cast<std::chrono::duration<double> >(
            finish - start)
            .count();
    const long long rss_raw = peak_rss_raw();

    std::cout << std::setprecision(17)
              << "{\n"
              << "  \"case\": \"" << case_name << "\",\n"
              << "  \"cache\": "
              << (cache_enabled ? "true" : "false") << ",\n"
              << "  \"feature_gauge\": "
              << (gauge_enabled ? "true" : "false") << ",\n"
              << "  \"n\": " << configuration.sample_num << ",\n"
              << "  \"d\": " << configuration.feature_num << ",\n"
              << "  \"classes\": " << configuration.class_num << ",\n"
              << "  \"lambda\": " << configuration.lambda << ",\n"
              << "  \"status\": \""
              << picasso::solver::multinomial_solver_status_string(
                     result.status)
              << "\",\n"
              << "  \"wall_seconds\": " << wall_seconds << ",\n"
              << "  \"ru_maxrss_raw\": " << rss_raw << ",\n"
              << "  \"ru_maxrss_bytes\": " << peak_rss_bytes(rss_raw)
              << ",\n"
              << "  \"cache_vector_bytes\": "
              << (cache_enabled
                      ? static_cast<long long>(configuration.sample_num) *
                            static_cast<long long>(sizeof(double))
                      : 0LL)
              << ",\n"
              << "  \"objective\": " << result.final_objective << ",\n"
              << "  \"kkt_residual\": " << result.final_kkt_residual
              << ",\n"
              << "  \"outer_iterations\": " << result.outer_iterations
              << ",\n"
              << "  \"inner_sweeps\": " << result.total_inner_sweeps
              << ",\n"
              << "  \"coordinate_updates\": "
              << result.total_coordinate_updates << "\n"
              << "}\n";
    return result.converged() ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "benchmark error: " << error.what() << "\n";
    return 1;
  }
}
