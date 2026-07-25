#include <picasso/c_api.hpp>
#include <picasso/multinomial_actnewton.hpp>
#include <picasso/multinomial_lla.hpp>
#include <picasso/multinomial_objective.hpp>

#include "../src/internal/multinomial_problem_view.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// The legacy sibling is intentionally retained as an internal regression
// oracle rather than exposed in the public header.
extern "C" void SolveMultinomialRegressionLegacy(
    double *Y_int, double *X, int n, int d, int num_classes,
    double *lambda, int nlambda, double gamma, int max_ite, double pprec,
    int reg_type, bool intercept, int dfmax, double *beta, double *intcpt,
    int *ite_lamb, int *size_act, double *runt, int *num_fit,
    bool usePython);

namespace {

typedef picasso::solver::MultinomialActNewtonResult PnResult;
typedef picasso::solver::MultinomialSolverStatus PnStatus;

const double kOutputSentinel = 91.125;
const int kIntegerSentinel = 9125;
const double kCoefficientZeroTolerance = 1e-8;
const int kMinimumLlaStageCount = 3;
const int kMaximumLlaStageCount = 25;
const double kLlaZeroTolerance = 1e-12;

bool require(bool condition, const std::string &message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

struct Fixture {
  Eigen::MatrixXd x;
  Eigen::VectorXi labels;
  int num_classes;
};

Fixture make_signal_fixture() {
  Fixture fixture;
  const int n = 96;
  const int d = 4;
  fixture.num_classes = 3;
  fixture.x.resize(n, d);
  fixture.labels.resize(n);

  Eigen::MatrixXd true_beta(d, fixture.num_classes);
  true_beta << 0.95, -0.55, -0.20,
              -0.35, 0.75, -0.30,
               0.20, -0.45, 0.35,
              -0.55, 0.15, 0.40;
  Eigen::VectorXd true_intercept(fixture.num_classes);
  true_intercept << 0.25, -0.05, -0.20;

  for (int i = 0; i < n; ++i) {
    fixture.x(i, 0) = std::sin(0.087 * static_cast<double>(i + 1));
    fixture.x(i, 1) = std::cos(0.131 * static_cast<double>(i + 2));
    fixture.x(i, 2) =
        static_cast<double>((i * 7) % 19 - 9) / 9.0;
    fixture.x(i, 3) =
        std::sin(0.037 * static_cast<double>((i + 3) * (i + 1)));
  }

  const Eigen::MatrixXd logits =
      fixture.x * true_beta +
      true_intercept.transpose().replicate(n, 1);
  for (int i = 0; i < n; ++i) {
    const double row_maximum = logits.row(i).maxCoeff();
    Eigen::VectorXd probabilities(fixture.num_classes);
    for (int klass = 0; klass < fixture.num_classes; ++klass)
      probabilities[klass] = std::exp(logits(i, klass) - row_maximum);
    probabilities.array() /= probabilities.sum();
    const double draw =
        std::fmod(0.6180339887498949 * static_cast<double>(i + 1), 1.0);
    double cumulative = 0.0;
    fixture.labels[i] = fixture.num_classes - 1;
    for (int klass = 0; klass < fixture.num_classes; ++klass) {
      cumulative += probabilities[klass];
      if (draw <= cumulative) {
        fixture.labels[i] = klass;
        break;
      }
    }
  }
  return fixture;
}

std::vector<double> double_labels(const Fixture &fixture) {
  std::vector<double> labels(
      static_cast<std::size_t>(fixture.labels.size()));
  for (Eigen::Index i = 0; i < fixture.labels.size(); ++i)
    labels[static_cast<std::size_t>(i)] =
        static_cast<double>(fixture.labels[i]);
  return labels;
}

std::vector<double> packed_design(const Fixture &fixture, bool row_major) {
  const int n = static_cast<int>(fixture.x.rows());
  const int d = static_cast<int>(fixture.x.cols());
  std::vector<double> packed(static_cast<std::size_t>(n) * d);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      const std::size_t index =
          row_major ? static_cast<std::size_t>(i) * d + j
                    : static_cast<std::size_t>(j) * n + i;
      packed[index] = fixture.x(i, j);
    }
  }
  return packed;
}

struct ApiResult {
  std::vector<double> beta;
  std::vector<double> intercept;
  std::vector<int> iterations;
  std::vector<int> size_active;
  std::vector<double> runtime;
  int num_fit;

  ApiResult(int d, int num_classes, int nlambda, double double_fill,
            int integer_fill)
      : beta(static_cast<std::size_t>(d) * num_classes * nlambda,
             double_fill),
        intercept(static_cast<std::size_t>(num_classes) * nlambda,
                  double_fill),
        iterations(static_cast<std::size_t>(nlambda), integer_fill),
        size_active(static_cast<std::size_t>(nlambda), integer_fill),
        runtime(static_cast<std::size_t>(nlambda), double_fill),
        num_fit(integer_fill) {}
};

bool all_zero(const std::vector<double> &values);
bool all_zero(const std::vector<int> &values);

picasso::solver::MultinomialActNewtonOptions make_options(
    int max_iterations, double precision, bool include_intercept) {
  picasso::solver::MultinomialActNewtonOptions options;
  options.max_outer_iterations = max_iterations;
  options.max_inner_sweeps = max_iterations;
  options.outer_kkt_tolerance = precision;
  const double roundoff_floor =
      100.0 * std::numeric_limits<double>::epsilon();
  options.inner_kkt_tolerance =
      std::min(precision, std::max(roundoff_floor, 0.01 * precision));
  options.use_adaptive_inner_tolerance = true;
  options.use_vectorized_coordinate_kernels = true;
  options.reuse_line_search_probabilities = true;
  options.use_compact_inner_active_set = true;
  options.include_intercept = include_intercept;
  return options;
}

Eigen::MatrixXd lla_weights(const Eigen::MatrixXd &anchor, int reg_type,
                            double lambda, double gamma) {
  Eigen::MatrixXd weights(anchor.rows(), anchor.cols());
  for (Eigen::Index feature = 0; feature < anchor.rows(); ++feature) {
    for (Eigen::Index klass = 0; klass < anchor.cols(); ++klass) {
      const double absolute_value = std::fabs(anchor(feature, klass));
      double derivative = 0.0;
      if (reg_type == 2) {
        derivative = std::max(0.0, lambda - absolute_value / gamma);
      } else if (absolute_value <= lambda) {
        derivative = lambda;
      } else if (absolute_value <= gamma * lambda) {
        derivative =
            (gamma * lambda - absolute_value) / (gamma - 1.0);
      }
      weights(feature, klass) = derivative;
    }
  }
  return weights;
}

double lla_target_stationarity(
    const picasso::MultinomialObjective &objective,
    const PnResult &solution, int reg_type, double lambda, double gamma,
    bool include_intercept) {
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  objective.smooth_gradient(solution.beta, solution.intercept,
                            &beta_gradient, &intercept_gradient);
  double maximum = 0.0;
  for (Eigen::Index feature = 0; feature < solution.beta.rows(); ++feature) {
    for (Eigen::Index klass = 0; klass < solution.beta.cols(); ++klass) {
      const double coefficient = solution.beta(feature, klass);
      const double absolute_value = std::fabs(coefficient);
      double derivative = 0.0;
      if (reg_type == 2) {
        derivative = std::max(0.0, lambda - absolute_value / gamma);
      } else if (absolute_value <= lambda) {
        derivative = lambda;
      } else if (absolute_value <= gamma * lambda) {
        derivative =
            (gamma * lambda - absolute_value) / (gamma - 1.0);
      }
      double residual = 0.0;
      if (coefficient > kLlaZeroTolerance) {
        residual = std::fabs(beta_gradient(feature, klass) + derivative);
      } else if (coefficient < -kLlaZeroTolerance) {
        residual = std::fabs(beta_gradient(feature, klass) - derivative);
      } else {
        residual = std::max(
            0.0, std::fabs(beta_gradient(feature, klass)) - derivative);
      }
      maximum = std::max(maximum, residual);
    }
  }
  if (include_intercept)
    maximum = std::max(maximum,
                       intercept_gradient.cwiseAbs().maxCoeff());
  return maximum;
}

void commit_solution(const PnResult &solution, int lambda_index,
                     int total_inner_sweeps, ApiResult *path) {
  const int d = static_cast<int>(solution.beta.rows());
  const int num_classes = static_cast<int>(solution.beta.cols());
  const std::size_t beta_base =
      static_cast<std::size_t>(lambda_index) * num_classes * d;
  const std::size_t intercept_base =
      static_cast<std::size_t>(lambda_index) * num_classes;
  int nonzero = 0;
  for (int klass = 0; klass < num_classes; ++klass) {
    for (int feature = 0; feature < d; ++feature) {
      const double coefficient = solution.beta(feature, klass);
      path->beta[beta_base + static_cast<std::size_t>(klass) * d +
                 feature] = coefficient;
      if (std::fabs(coefficient) > kCoefficientZeroTolerance) ++nonzero;
    }
    path->intercept[intercept_base + klass] = solution.intercept[klass];
  }
  path->iterations[static_cast<std::size_t>(lambda_index)] =
      total_inner_sweeps;
  path->size_active[static_cast<std::size_t>(lambda_index)] = nonzero;
  path->runtime[static_cast<std::size_t>(lambda_index)] = 0.0;
  path->num_fit = lambda_index + 1;
}

struct ManualPath {
  ApiResult output;
  std::vector<PnStatus> attempted_statuses;

  ManualPath(int d, int num_classes, int nlambda)
      : output(d, num_classes, nlambda, 0.0, 0) {}
};

ManualPath manual_path(const Fixture &fixture,
                       const std::vector<double> &lambdas, double gamma,
                       int max_iterations, double precision, int reg_type,
                       bool include_intercept, int dfmax) {
  const int d = static_cast<int>(fixture.x.cols());
  const int num_classes = fixture.num_classes;
  ManualPath expected(d, num_classes, static_cast<int>(lambdas.size()));
  picasso::MultinomialObjective objective(fixture.x, fixture.labels,
                                          num_classes);
  const picasso::solver::MultinomialActNewtonOptions options =
      make_options(max_iterations, precision, include_intercept);
  picasso::solver::MultinomialActNewtonPathSolver master_solver(
      objective, options);
  picasso::solver::MultinomialActNewtonPathState master_state;
  picasso::solver::MultinomialLlaSolver lla_solver(objective, options);
  for (std::size_t lambda_index = 0; lambda_index < lambdas.size();
       ++lambda_index) {
    const double lambda = lambdas[lambda_index];
    picasso::solver::MultinomialActNewtonPathState candidate_master_state =
        master_state;
    picasso::solver::MultinomialActNewtonPathResult master_path =
        master_solver.solve(lambda, &candidate_master_state);
    PnResult master = std::move(master_path.solution);
    expected.attempted_statuses.push_back(master.status);
    if (!master.converged()) break;

    PnResult final_solution = master;
    int total_inner_sweeps = master.total_inner_sweeps;

    if (reg_type != 1) {
      const picasso::solver::MultinomialLlaPenalty penalty =
          reg_type == 2
              ? picasso::solver::MultinomialLlaPenalty::kMCP
              : picasso::solver::MultinomialLlaPenalty::kSCAD;
      picasso::solver::MultinomialLlaResult lla =
          lla_solver.solve_from_l1_master(
              penalty, lambda, gamma, std::move(master));
      for (std::size_t stage = 1; stage < lla.stages.size(); ++stage)
        expected.attempted_statuses.push_back(
            lla.stages[stage].subproblem_status);
      if (!lla.has_valid_solution()) break;
      final_solution.beta = std::move(lla.beta);
      final_solution.intercept = std::move(lla.intercept);
      final_solution.status = PnStatus::kConverged;
      total_inner_sweeps = lla.total_inner_sweeps;
    }

    commit_solution(final_solution, static_cast<int>(lambda_index),
                    total_inner_sweeps, &expected.output);
    master_state = candidate_master_state;
    const int nonzero =
        expected.output.size_active[lambda_index];
    if (dfmax >= 0 && nonzero > dfmax) break;
  }
  return expected;
}

ApiResult public_path(const Fixture &fixture,
                      const std::vector<double> &input_lambdas, double gamma,
                      int max_iterations, double precision, int reg_type,
                      bool include_intercept, int dfmax, bool row_major) {
  std::vector<double> labels = double_labels(fixture);
  std::vector<double> design = packed_design(fixture, row_major);
  std::vector<double> lambdas = input_lambdas;
  const int n = static_cast<int>(fixture.x.rows());
  const int d = static_cast<int>(fixture.x.cols());
  const int nlambda = static_cast<int>(lambdas.size());
  ApiResult result(d, fixture.num_classes, nlambda, kOutputSentinel,
                   kIntegerSentinel);
  SolveMultinomialRegression(
      labels.data(), design.data(), n, d, fixture.num_classes,
      lambdas.data(), nlambda, gamma, max_iterations, precision, reg_type,
      include_intercept, dfmax, result.beta.data(), result.intercept.data(),
      result.iterations.data(), result.size_active.data(),
      result.runtime.data(), &result.num_fit, row_major);
  return result;
}

ApiResult public_path_from_raw_design(
    const Fixture &fixture, std::vector<double> *labels, double *design,
    std::vector<double> *lambdas, double gamma, int max_iterations,
    double precision, int reg_type, bool include_intercept,
    bool row_major) {
  const int n = static_cast<int>(fixture.x.rows());
  const int d = static_cast<int>(fixture.x.cols());
  const int nlambda = static_cast<int>(lambdas->size());
  ApiResult result(d, fixture.num_classes, nlambda, kOutputSentinel,
                   kIntegerSentinel);
  SolveMultinomialRegression(
      labels->data(), design, n, d, fixture.num_classes, lambdas->data(),
      nlambda, gamma, max_iterations, precision, reg_type,
      include_intercept, -1, result.beta.data(), result.intercept.data(),
      result.iterations.data(), result.size_active.data(),
      result.runtime.data(), &result.num_fit, row_major);
  return result;
}

struct V2Result {
  ApiResult output;
  int status;
  int failed_lambda;
  int failed_stage;
  std::vector<int> outer_iterations;
  std::vector<long long> inner_sweeps;
  std::vector<long long> coordinate_updates;
  std::vector<double> objective;
  std::vector<double> kkt;
  std::vector<double> stationarity;

  V2Result(int d, int num_classes, int nlambda)
      : output(d, num_classes, nlambda, kOutputSentinel,
               kIntegerSentinel),
        status(kIntegerSentinel),
        failed_lambda(kIntegerSentinel),
        failed_stage(kIntegerSentinel),
        outer_iterations(static_cast<std::size_t>(nlambda),
                         kIntegerSentinel),
        inner_sweeps(static_cast<std::size_t>(nlambda),
                     static_cast<long long>(kIntegerSentinel)),
        coordinate_updates(static_cast<std::size_t>(nlambda),
                           static_cast<long long>(kIntegerSentinel)),
        objective(static_cast<std::size_t>(nlambda), kOutputSentinel),
        kkt(static_cast<std::size_t>(nlambda), kOutputSentinel),
        stationarity(static_cast<std::size_t>(nlambda), kOutputSentinel) {}
};

V2Result public_path_v2(const Fixture &fixture,
                        const std::vector<double> &input_lambdas,
                        double gamma, int max_iterations, double precision,
                        int reg_type, bool include_intercept, int dfmax,
                        bool row_major) {
  std::vector<double> labels = double_labels(fixture);
  std::vector<double> design = packed_design(fixture, row_major);
  std::vector<double> lambdas = input_lambdas;
  const int n = static_cast<int>(fixture.x.rows());
  const int d = static_cast<int>(fixture.x.cols());
  const int nlambda = static_cast<int>(lambdas.size());
  V2Result result(d, fixture.num_classes, nlambda);
  result.status = SolveMultinomialRegressionV2(
      labels.data(), design.data(), n, d, fixture.num_classes,
      lambdas.data(), nlambda, gamma, max_iterations, precision, reg_type,
      include_intercept, dfmax, result.output.beta.data(),
      result.output.intercept.data(), result.output.iterations.data(),
      result.output.size_active.data(), result.output.runtime.data(),
      &result.output.num_fit, row_major, &result.failed_lambda,
      &result.failed_stage, result.outer_iterations.data(),
      result.inner_sweeps.data(), result.coordinate_updates.data(),
      result.objective.data(), result.kkt.data(),
      result.stationarity.data());
  return result;
}

V2Result public_path_v3(const Fixture &fixture,
                        const std::vector<double> &input_lambdas,
                        double gamma, int max_iterations, double precision,
                        int reg_type, bool include_intercept, int dfmax,
                        bool row_major, int lla_max_stages) {
  std::vector<double> labels = double_labels(fixture);
  std::vector<double> design = packed_design(fixture, row_major);
  std::vector<double> lambdas = input_lambdas;
  const int n = static_cast<int>(fixture.x.rows());
  const int d = static_cast<int>(fixture.x.cols());
  const int nlambda = static_cast<int>(lambdas.size());
  V2Result result(d, fixture.num_classes, nlambda);
  result.status = SolveMultinomialRegressionV3(
      labels.data(), design.data(), n, d, fixture.num_classes,
      lambdas.data(), nlambda, gamma, max_iterations, precision, reg_type,
      include_intercept, dfmax, result.output.beta.data(),
      result.output.intercept.data(), result.output.iterations.data(),
      result.output.size_active.data(), result.output.runtime.data(),
      &result.output.num_fit, row_major, lla_max_stages,
      &result.failed_lambda, &result.failed_stage,
      result.outer_iterations.data(), result.inner_sweeps.data(),
      result.coordinate_updates.data(), result.objective.data(),
      result.kkt.data(), result.stationarity.data());
  return result;
}

V2Result public_path_v4(const Fixture &fixture,
                        const std::vector<double> &input_lambdas,
                        double gamma, int max_iterations, double precision,
                        int reg_type, bool include_intercept, int dfmax,
                        bool row_major, int lla_max_stages,
                        bool path_early_stop) {
  std::vector<double> labels = double_labels(fixture);
  std::vector<double> design = packed_design(fixture, row_major);
  std::vector<double> lambdas = input_lambdas;
  const int n = static_cast<int>(fixture.x.rows());
  const int d = static_cast<int>(fixture.x.cols());
  const int nlambda = static_cast<int>(lambdas.size());
  V2Result result(d, fixture.num_classes, nlambda);
  result.status = SolveMultinomialRegressionV4(
      labels.data(), design.data(), n, d, fixture.num_classes,
      lambdas.data(), nlambda, gamma, max_iterations, precision, reg_type,
      include_intercept, dfmax, result.output.beta.data(),
      result.output.intercept.data(), result.output.iterations.data(),
      result.output.size_active.data(), result.output.runtime.data(),
      &result.output.num_fit, row_major, lla_max_stages, path_early_stop,
      &result.failed_lambda, &result.failed_stage,
      result.outer_iterations.data(), result.inner_sweeps.data(),
      result.coordinate_updates.data(), result.objective.data(),
      result.kkt.data(), result.stationarity.data());
  return result;
}

struct V5Result {
  V2Result diagnostics;
  std::vector<double> smooth_nll;

  V5Result(int d, int num_classes, int nlambda)
      : diagnostics(d, num_classes, nlambda),
        smooth_nll(static_cast<std::size_t>(nlambda), kOutputSentinel) {}
};

V5Result public_path_v5(const Fixture &fixture,
                        const std::vector<double> &input_lambdas,
                        double gamma, double precision, int reg_type,
                        bool row_major, bool path_early_stop = false,
                        int max_iterations = 5000, int dfmax = -1,
                        int lla_max_stages = 25) {
  std::vector<double> labels = double_labels(fixture);
  std::vector<double> design = packed_design(fixture, row_major);
  std::vector<double> lambdas = input_lambdas;
  const int n = static_cast<int>(fixture.x.rows());
  const int d = static_cast<int>(fixture.x.cols());
  const int nlambda = static_cast<int>(lambdas.size());
  V5Result result(d, fixture.num_classes, nlambda);
  V2Result &output = result.diagnostics;
  output.status = SolveMultinomialRegressionV5(
      labels.data(), design.data(), n, d, fixture.num_classes,
      lambdas.data(), nlambda, gamma, max_iterations, precision, reg_type,
      true, dfmax,
      output.output.beta.data(), output.output.intercept.data(),
      output.output.iterations.data(), output.output.size_active.data(),
      output.output.runtime.data(), &output.output.num_fit, row_major,
      lla_max_stages, path_early_stop, &output.failed_lambda,
      &output.failed_stage,
      output.outer_iterations.data(), output.inner_sweeps.data(),
      output.coordinate_updates.data(), output.objective.data(),
      output.kkt.data(), output.stationarity.data(), result.smooth_nll.data());
  return result;
}

double maximum_difference(const std::vector<double> &left,
                          const std::vector<double> &right) {
  if (left.size() != right.size())
    return std::numeric_limits<double>::infinity();
  double maximum = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (!std::isfinite(left[index]) || !std::isfinite(right[index]))
      return std::numeric_limits<double>::infinity();
    maximum = std::max(maximum, std::fabs(left[index] - right[index]));
  }
  return maximum;
}

bool compare_path(const ApiResult &actual, const ApiResult &expected,
                  const std::string &case_name) {
  bool ok = true;
  ok &= require(actual.num_fit == expected.num_fit,
                case_name + " num_fit must match the manual path");
  ok &= require(actual.iterations == expected.iterations,
                case_name + " ite_lamb must equal total PN inner sweeps");
  ok &= require(actual.size_active == expected.size_active,
                case_name + " size_act must count |beta| > 1e-8");
  const double beta_difference =
      maximum_difference(actual.beta, expected.beta);
  const double intercept_difference =
      maximum_difference(actual.intercept, expected.intercept);
  ok &= require(beta_difference <= 2e-13,
                case_name + " beta layout/value mismatch (max diff " +
                    std::to_string(beta_difference) + ")");
  ok &= require(intercept_difference <= 2e-13,
                case_name + " intercept layout/value mismatch (max diff " +
                    std::to_string(intercept_difference) + ")");
  ok &= require(actual.runtime.size() == expected.runtime.size(),
                case_name + " runtime output length mismatch");
  if (actual.runtime.size() == expected.runtime.size()) {
    for (std::size_t index = 0; index < actual.runtime.size(); ++index) {
      ok &= require(actual.runtime[index] == 0.0,
                    case_name + " runt must be zero at every path slot");
    }
  }
  return ok;
}

bool compare_layouts(const ApiResult &row_major,
                     const ApiResult &column_major,
                     const std::string &case_name) {
  bool ok = true;
  ok &= require(row_major.num_fit == column_major.num_fit,
                case_name + " layouts must have equal num_fit");
  ok &= require(row_major.iterations == column_major.iterations &&
                    row_major.size_active == column_major.size_active &&
                    row_major.runtime == column_major.runtime,
                case_name + " layouts must have equal diagnostics");
  ok &= require(maximum_difference(row_major.beta, column_major.beta) == 0.0 &&
                    maximum_difference(row_major.intercept,
                                       column_major.intercept) == 0.0,
                case_name + " row/column-major calls must be bitwise equal");
  return ok;
}

bool test_three_penalties_and_both_layouts() {
  const Fixture fixture = make_signal_fixture();
  const std::vector<double> lambdas = {0.08, 0.035, 0.018};
  const int max_iterations = 5000;
  const double precision = 5e-7;
  const double gammas[] = {3.0, 3.0, 3.7};
  const char *names[] = {"L1", "MCP", "SCAD"};
  bool ok = true;

  for (int reg_type = 1; reg_type <= 3; ++reg_type) {
    const ManualPath expected =
        manual_path(fixture, lambdas, gammas[reg_type - 1],
                    max_iterations, precision, reg_type, true, -1);
    ok &= require(expected.output.num_fit ==
                      static_cast<int>(lambdas.size()),
                  std::string(names[reg_type - 1]) +
                      " manual oracle fixture must finish the full path");
    const ApiResult column_major = public_path(
        fixture, lambdas, gammas[reg_type - 1], max_iterations, precision,
        reg_type, true, -1, false);
    const ApiResult row_major = public_path(
        fixture, lambdas, gammas[reg_type - 1], max_iterations, precision,
        reg_type, true, -1, true);
    const std::string prefix = names[reg_type - 1];
    ok &= compare_path(column_major, expected.output,
                       prefix + " column-major");
    ok &= compare_path(row_major, expected.output, prefix + " row-major");
    ok &= compare_layouts(row_major, column_major, prefix);
  }
  return ok;
}

bool test_intercept_false_and_layout() {
  const Fixture fixture = make_signal_fixture();
  const std::vector<double> lambdas = {0.06, 0.025};
  const int max_iterations = 5000;
  const double precision = 5e-7;
  const ManualPath expected = manual_path(
      fixture, lambdas, 3.0, max_iterations, precision, 1, false, -1);
  const ApiResult actual = public_path(
      fixture, lambdas, 3.0, max_iterations, precision, 1, false, -1, true);
  bool ok = compare_path(actual, expected.output,
                         "L1 intercept=false row-major");
  ok &= require(actual.num_fit == static_cast<int>(lambdas.size()),
                "intercept=false fixture must finish the full path");
  for (std::size_t index = 0; index < actual.intercept.size(); ++index)
    ok &= require(actual.intercept[index] == 0.0,
                  "intercept=false must write exact zero intercepts");
  return ok;
}

bool test_borrowed_column_major_boundaries() {
  const Fixture fixture = make_signal_fixture();
  const int n = static_cast<int>(fixture.x.rows());
  const int d = static_cast<int>(fixture.x.cols());
  const std::size_t design_size = static_cast<std::size_t>(n) * d;
  const std::vector<double> lambda_template = {0.08, 0.035};
  const double gammas[] = {3.0, 3.0, 3.7};
  bool ok = true;

  // Exercise both production precision profiles and intercept choices for
  // every penalty.  Python-style row-major remains owning; ordinary aligned
  // column-major storage is borrowed, and the paths must remain bitwise equal.
  const double precisions[] = {1e-4, 1e-7};
  for (int reg_type = 1; reg_type <= 3; ++reg_type) {
    for (int precision_index = 0; precision_index < 2;
         ++precision_index) {
      for (int intercept_index = 0; intercept_index < 2;
           ++intercept_index) {
        const bool include_intercept = intercept_index != 0;
        const ApiResult column_major = public_path(
            fixture, lambda_template, gammas[reg_type - 1], 5000,
            precisions[precision_index], reg_type, include_intercept, -1,
            false);
        const ApiResult row_major = public_path(
            fixture, lambda_template, gammas[reg_type - 1], 5000,
            precisions[precision_index], reg_type, include_intercept, -1,
            true);
        ok &= compare_layouts(
            row_major, column_major,
            "precision/intercept borrowed parity reg=" +
                std::to_string(reg_type) + " precision=" +
                std::to_string(precisions[precision_index]) +
                " intercept=" + std::to_string(intercept_index));
      }
    }
  }

  // Eigen owns this aligned column-major buffer.  The synchronous borrowed
  // call must not alter a single input byte.
  Eigen::MatrixXd aligned_design = fixture.x;
  std::vector<double> labels = double_labels(fixture);
  std::vector<double> lambdas = lambda_template;
  std::vector<unsigned char> aligned_snapshot(
      design_size * sizeof(double));
  std::memcpy(aligned_snapshot.data(), aligned_design.data(),
              aligned_snapshot.size());
  const ApiResult aligned_result = public_path_from_raw_design(
      fixture, &labels, aligned_design.data(), &lambdas, 3.0, 5000, 1e-7,
      1, true, false);
  ok &= require(std::memcmp(aligned_snapshot.data(), aligned_design.data(),
                            aligned_snapshot.size()) == 0,
                "borrowed aligned column-major X must remain byte-identical");
  ok &= compare_layouts(
      public_path(fixture, lambda_template, 3.0, 5000, 1e-7, 1, true,
                  -1, true),
      aligned_result, "aligned borrowed input");

  // data()+1 deliberately exercises the unaligned C-input fallback.  It must
  // remain safe, immutable, and numerically identical to the borrowed path.
  const std::vector<double> packed = packed_design(fixture, false);
  Eigen::VectorXd misaligned_storage =
      Eigen::VectorXd::Constant(static_cast<Eigen::Index>(design_size + 2),
                                -917.25);
  std::copy(packed.begin(), packed.end(), misaligned_storage.data() + 1);
  std::vector<unsigned char> misaligned_snapshot(
      static_cast<std::size_t>(misaligned_storage.size()) * sizeof(double));
  std::memcpy(misaligned_snapshot.data(), misaligned_storage.data(),
              misaligned_snapshot.size());
  const std::size_t required_alignment =
      picasso::detail::MultinomialProblemView::design_alignment_bytes();
  ok &= require(
      required_alignment <= sizeof(double) ||
          !picasso::detail::MultinomialProblemView::
              design_pointer_is_aligned(misaligned_storage.data() + 1),
      "Eigen-aligned data()+1 must select the owning safety fallback when "
      "the configured map needs more than double alignment");
  labels = double_labels(fixture);
  lambdas = lambda_template;
  const ApiResult misaligned_result = public_path_from_raw_design(
      fixture, &labels, misaligned_storage.data() + 1, &lambdas, 3.0,
      5000, 1e-7, 1, true, false);
  ok &= require(std::memcmp(misaligned_snapshot.data(),
                            misaligned_storage.data(),
                            misaligned_snapshot.size()) == 0,
                "misaligned column-major X must remain byte-identical");
  ok &= compare_layouts(aligned_result, misaligned_result,
                        "misaligned owning fallback");

  // Invalid aligned column-major buffers are rejected before any solver state
  // is committed, preserving the C API's all-zero transactional outputs.
  const double invalid_values[] = {
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity()};
  for (int invalid_index = 0; invalid_index < 2; ++invalid_index) {
    Eigen::MatrixXd invalid_design = fixture.x;
    invalid_design.data()[3] = invalid_values[invalid_index];
    labels = double_labels(fixture);
    lambdas = lambda_template;
    const ApiResult rejected = public_path_from_raw_design(
        fixture, &labels, invalid_design.data(), &lambdas, 3.0, 5000,
        1e-7, 1, true, false);
    ok &= require(
        rejected.num_fit == 0 && all_zero(rejected.beta) &&
            all_zero(rejected.intercept) && all_zero(rejected.iterations) &&
            all_zero(rejected.size_active) && all_zero(rejected.runtime),
        std::string(invalid_index == 0 ? "NaN" : "infinite") +
            " borrowed X rejection must be transactional");
  }
  return ok;
}

Eigen::VectorXd empirical_null_intercept(const Fixture &fixture) {
  Eigen::VectorXd intercept = Eigen::VectorXd::Zero(fixture.num_classes);
  for (Eigen::Index i = 0; i < fixture.labels.size(); ++i)
    intercept[fixture.labels[i]] += 1.0;
  intercept.array() /= static_cast<double>(fixture.labels.size());
  for (Eigen::Index klass = 0; klass < intercept.size(); ++klass)
    intercept[klass] = std::log(std::max(intercept[klass], 1e-8));
  intercept.array() -= intercept.mean();
  return intercept;
}

double null_gradient_maximum(const Fixture &fixture) {
  picasso::MultinomialObjective objective(fixture.x, fixture.labels,
                                          fixture.num_classes);
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  objective.smooth_gradient(
      Eigen::MatrixXd::Zero(fixture.x.cols(), fixture.num_classes),
      empirical_null_intercept(fixture), &beta_gradient,
      &intercept_gradient);
  return beta_gradient.cwiseAbs().maxCoeff();
}

bool suffix_is_zero(const ApiResult &result, int d, int num_classes,
                    int first_zero_lambda, const std::string &case_name) {
  bool ok = true;
  const int nlambda = static_cast<int>(result.iterations.size());
  for (int lambda_index = first_zero_lambda; lambda_index < nlambda;
       ++lambda_index) {
    const std::size_t beta_base =
        static_cast<std::size_t>(lambda_index) * d * num_classes;
    const std::size_t intercept_base =
        static_cast<std::size_t>(lambda_index) * num_classes;
    for (int index = 0; index < d * num_classes; ++index)
      ok &= require(result.beta[beta_base + index] == 0.0,
                    case_name + " beta suffix must stay zero");
    for (int klass = 0; klass < num_classes; ++klass)
      ok &= require(result.intercept[intercept_base + klass] == 0.0,
                    case_name + " intercept suffix must stay zero");
    ok &= require(result.iterations[lambda_index] == 0 &&
                      result.size_active[lambda_index] == 0 &&
                      result.runtime[lambda_index] == 0.0,
                  case_name + " diagnostic suffix must stay zero");
  }
  return ok;
}

double fitted_negative_log_likelihood(const Fixture &fixture,
                                      const ApiResult &path,
                                      int lambda_index) {
  const int d = static_cast<int>(fixture.x.cols());
  const int num_classes = fixture.num_classes;
  const std::size_t beta_base =
      static_cast<std::size_t>(lambda_index) *
      static_cast<std::size_t>(d) *
      static_cast<std::size_t>(num_classes);
  const std::size_t intercept_base =
      static_cast<std::size_t>(lambda_index) *
      static_cast<std::size_t>(num_classes);
  Eigen::MatrixXd beta(d, num_classes);
  Eigen::VectorXd intercept(num_classes);
  for (int klass = 0; klass < num_classes; ++klass) {
    for (int feature = 0; feature < d; ++feature) {
      beta(feature, klass) =
          path.beta[beta_base + static_cast<std::size_t>(klass) *
                                    static_cast<std::size_t>(d) +
                    static_cast<std::size_t>(feature)];
    }
    intercept[klass] =
        path.intercept[intercept_base + static_cast<std::size_t>(klass)];
  }
  const picasso::MultinomialObjective objective(
      fixture.x, fixture.labels, fixture.num_classes);
  return objective.negative_log_likelihood(beta, intercept);
}

bool test_v5_smooth_nll_path() {
  const Fixture fixture = make_signal_fixture();
  const double lambda_max = null_gradient_maximum(fixture);
  const std::vector<double> lambdas{
      lambda_max, 0.72 * lambda_max, 0.5 * lambda_max};
  bool ok = true;
  for (int reg_type = 1; reg_type <= 3; ++reg_type) {
    for (int layout = 0; layout < 2; ++layout) {
      const V5Result result = public_path_v5(
          fixture, lambdas, 3.0, 5e-7, reg_type, layout != 0);
      const V2Result &diagnostics = result.diagnostics;
      ok &= require(
          diagnostics.status == PICASSO_MULTINOMIAL_COMPLETED &&
              diagnostics.output.num_fit ==
                  static_cast<int>(lambdas.size()),
          "V5 smooth-NLL path did not complete");
      for (int index = 0; index < diagnostics.output.num_fit; ++index) {
        const double expected = fitted_negative_log_likelihood(
            fixture, diagnostics.output, index);
        const double smooth_nll = result.smooth_nll[index];
        const double derived_penalty =
            diagnostics.objective[index] - smooth_nll;
        ok &= require(
            std::isfinite(smooth_nll) && smooth_nll >= 0.0 &&
                std::fabs(smooth_nll - expected) <= 2e-12,
            "V5 smooth NLL does not match returned coefficients");
        ok &= require(
            std::isfinite(diagnostics.objective[index]) &&
                std::isfinite(derived_penalty) &&
                diagnostics.objective[index] >= -2e-12 &&
                derived_penalty >= -2e-12,
            "V5 committed objective/penalty decomposition is invalid");
      }
    }
  }

  std::vector<double> invalid_lambdas{lambda_max, lambda_max};
  const V5Result invalid = public_path_v5(
      fixture, invalid_lambdas, 3.0, 5e-7, 1, true);
  ok &= require(
      invalid.diagnostics.status == PICASSO_MULTINOMIAL_INVALID_INPUT &&
          invalid.diagnostics.output.num_fit == 0 &&
          std::isnan(invalid.smooth_nll[0]) &&
          std::isnan(invalid.smooth_nll[1]),
      "V5 invalid path did not leave smooth-NLL diagnostics transactional");

  // The public API cannot independently corrupt a solver-returned objective
  // while leaving its solution valid.  Exercise the same commit boundary with
  // a deterministic second-lambda solver failure: every committed smooth NLL
  // must be finite, while the failed point and suffix remain NaN and expose no
  // model.
  const std::vector<double> failing_lambdas{10.0, 0.006, 0.003};
  const V5Result failed = public_path_v5(
      fixture, failing_lambdas, 3.0, 1e-6, 1, true, false, 1);
  ok &= require(
      failed.diagnostics.status != PICASSO_MULTINOMIAL_COMPLETED &&
          failed.diagnostics.status != PICASSO_MULTINOMIAL_DFMAX_REACHED &&
          failed.diagnostics.status != PICASSO_MULTINOMIAL_INVALID_INPUT &&
          failed.diagnostics.output.num_fit == 1 &&
          failed.diagnostics.failed_lambda == 1 &&
          failed.diagnostics.failed_stage == -1 &&
          std::isfinite(failed.smooth_nll[0]) &&
          failed.smooth_nll[0] >= 0.0 &&
          std::isnan(failed.smooth_nll[1]) &&
          std::isnan(failed.smooth_nll[2]),
      "V5 solver failure did not preserve an atomic smooth-NLL prefix");
  const int d = static_cast<int>(fixture.x.cols());
  for (int lambda_index = failed.diagnostics.output.num_fit;
       lambda_index < static_cast<int>(failing_lambdas.size());
       ++lambda_index) {
    const std::size_t beta_base =
        static_cast<std::size_t>(lambda_index) * d * fixture.num_classes;
    const std::size_t intercept_base =
        static_cast<std::size_t>(lambda_index) * fixture.num_classes;
    for (int coefficient = 0; coefficient < d * fixture.num_classes;
         ++coefficient)
      ok &= require(
          failed.diagnostics.output.beta[beta_base + coefficient] == 0.0,
          "V5 failed beta point/suffix must remain uncommitted");
    for (int klass = 0; klass < fixture.num_classes; ++klass)
      ok &= require(
          failed.diagnostics.output.intercept[intercept_base + klass] ==
              0.0,
          "V5 failed intercept point/suffix must remain uncommitted");
    ok &= require(
        failed.diagnostics.output.iterations[lambda_index] == 0 &&
            failed.diagnostics.output.size_active[lambda_index] == 0,
        "V5 failed iteration/model-size slots must remain uncommitted");
  }

  const std::vector<double> dfmax_lambdas{
      2.0 * lambda_max, 0.8 * lambda_max, 0.4 * lambda_max,
      0.2 * lambda_max};
  const V5Result dfmax = public_path_v5(
      fixture, dfmax_lambdas, 3.0, 5e-7, 1, false, false, 5000, 0);
  const int dfmax_fit = dfmax.diagnostics.output.num_fit;
  ok &= require(
      dfmax.diagnostics.status == PICASSO_MULTINOMIAL_DFMAX_REACHED &&
          dfmax_fit > 1 &&
          dfmax_fit < static_cast<int>(dfmax_lambdas.size()) &&
          dfmax.diagnostics.output.size_active[dfmax_fit - 1] > 0,
      "V5 dfmax path did not retain exactly the crossing prefix");
  for (int index = 0; index < static_cast<int>(dfmax_lambdas.size());
       ++index) {
    ok &= require(
        index < dfmax_fit
            ? (std::isfinite(dfmax.smooth_nll[index]) &&
               dfmax.smooth_nll[index] >= 0.0)
            : std::isnan(dfmax.smooth_nll[index]),
        "V5 dfmax smooth-NLL path violated prefix transactionality");
  }
  ok &= suffix_is_zero(
      dfmax.diagnostics.output, d, fixture.num_classes, dfmax_fit,
      "V5 dfmax smooth-NLL prefix");
  return ok;
}

bool test_glmnet_style_saturated_path_early_stop() {
  const Fixture fixture = make_signal_fixture();
  const int requested = 12;
  const double first_lambda = 0.45 * null_gradient_maximum(fixture);
  std::vector<double> lambdas(static_cast<std::size_t>(requested));
  for (int index = 0; index < requested; ++index) {
    lambdas[static_cast<std::size_t>(index)] =
        first_lambda * (1.0 - 1e-7 * static_cast<double>(index));
  }

  picasso::MultinomialObjective objective(
      fixture.x, fixture.labels, fixture.num_classes);
  const double null_nll = objective.negative_log_likelihood(
      Eigen::MatrixXd::Zero(fixture.x.cols(), fixture.num_classes),
      empirical_null_intercept(fixture));
  bool ok = true;
  const int penalties[] = {1, 2};
  for (int penalty_index = 0; penalty_index < 2; ++penalty_index) {
    const int reg_type = penalties[penalty_index];
    const V2Result v3 = public_path_v3(
        fixture, lambdas, 3.0, 5000, 5e-7, reg_type, true, -1, true, 3);
    const V2Result v4_disabled = public_path_v4(
        fixture, lambdas, 3.0, 5000, 5e-7, reg_type, true, -1, true, 3,
        false);
    const V2Result result = public_path_v4(
        fixture, lambdas, 3.0, 5000, 5e-7, reg_type, true, -1, true, 3,
        true);
    const std::string prefix =
        reg_type == 1 ? "L1 saturated path" : "MCP saturated path";
    ok &= require(v3.output.num_fit == requested &&
                      v4_disabled.output.num_fit == requested,
                  prefix + " V3 and V4(false) must retain the full path");
    ok &= require(v3.status == v4_disabled.status &&
                      v3.failed_lambda == -1 && v3.failed_stage == -1 &&
                      v4_disabled.failed_lambda == -1 &&
                      v4_disabled.failed_stage == -1,
                  prefix + " disabled early stopping must preserve status");
    ok &= require(v3.output.iterations == v4_disabled.output.iterations &&
                      v3.output.size_active ==
                          v4_disabled.output.size_active &&
                      maximum_difference(v3.output.beta,
                                         v4_disabled.output.beta) == 0.0 &&
                      maximum_difference(v3.output.intercept,
                                         v4_disabled.output.intercept) == 0.0,
                  prefix + " V4(false) must be solution-identical to V3");
    ok &= require(
        reg_type == 1
            ? result.status == PICASSO_MULTINOMIAL_COMPLETED
            : (result.status == PICASSO_MULTINOMIAL_COMPLETED ||
               result.status == PICASSO_MULTINOMIAL_LLA_STATIONARITY_LIMIT),
        prefix + " must terminate with a usable path status");
    ok &= require(result.failed_lambda == -1 && result.failed_stage == -1,
                  prefix + " early stopping must not report a failure");
    ok &= require(result.output.num_fit >= 5 &&
                      result.output.num_fit < requested,
                  prefix + " must skip the saturated lambda tail");
    if (result.output.num_fit > 0) {
      ok &= require(
          result.output.size_active[static_cast<std::size_t>(
              result.output.num_fit - 1)] > 0,
          prefix + " must not use the zero-model shortcut");
      const int final_index = result.output.num_fit - 1;
      const double final_nll = fitted_negative_log_likelihood(
          fixture, result.output, final_index);
      const double deviance_ratio = 1.0 - final_nll / null_nll;
      bool stopping_rule_holds = deviance_ratio > 0.999;
      const int previous_index = final_index - 1;
      if (previous_index >= 0) {
        const double previous_nll = fitted_negative_log_likelihood(
            fixture, result.output, previous_index);
        const double previous_deviance_ratio =
            1.0 - previous_nll / null_nll;
        stopping_rule_holds =
            stopping_rule_holds ||
            deviance_ratio - previous_deviance_ratio < 1e-5;
      }
      ok &= require(stopping_rule_holds,
                    prefix + " must stop only after a documented "
                             "deviance rule is met");
    }
    ok &= suffix_is_zero(result.output,
                         static_cast<int>(fixture.x.cols()),
                         fixture.num_classes, result.output.num_fit,
                         prefix);
    for (int index = result.output.num_fit; index < requested; ++index) {
      const std::size_t slot = static_cast<std::size_t>(index);
      ok &= require(result.outer_iterations[slot] == 0 &&
                        result.inner_sweeps[slot] == 0 &&
                        result.coordinate_updates[slot] == 0 &&
                        std::isnan(result.objective[slot]) &&
                        std::isnan(result.kkt[slot]) &&
                        std::isnan(result.stationarity[slot]),
                    prefix + " must leave V4 diagnostic suffix unused");
    }
  }
  return ok;
}

bool test_dfmax_includes_crossing_lambda() {
  const Fixture fixture = make_signal_fixture();
  const double gradient_maximum = null_gradient_maximum(fixture);
  const std::vector<double> lambdas = {
      2.0 * gradient_maximum, 0.8 * gradient_maximum,
      0.4 * gradient_maximum, 0.2 * gradient_maximum};
  const int max_iterations = 5000;
  const double precision = 5e-7;
  const ManualPath expected = manual_path(
      fixture, lambdas, 3.0, max_iterations, precision, 1, true, 0);
  const ApiResult actual = public_path(
      fixture, lambdas, 3.0, max_iterations, precision, 1, true, 0, false);
  bool ok = true;
  ok &= require(expected.output.num_fit > 1 &&
                    expected.output.num_fit <
                        static_cast<int>(lambdas.size()),
                "dfmax fixture must cross zero df before the final lambda");
  if (expected.output.num_fit > 0) {
    ok &= require(expected.output.size_active[
                      static_cast<std::size_t>(expected.output.num_fit - 1)] >
                      0,
                  "the dfmax-violating crossing lambda must be included");
  }
  ok &= compare_path(actual, expected.output, "dfmax crossing");
  ok &= suffix_is_zero(actual, static_cast<int>(fixture.x.cols()),
                       fixture.num_classes, actual.num_fit,
                       "dfmax crossing");
  return ok;
}

bool test_iteration_limit_is_atomic_fail_stop() {
  const Fixture fixture = make_signal_fixture();
  const std::vector<double> lambdas = {10.0, 0.006, 0.003};
  const int max_iterations = 1;
  const double precision = 1e-6;
  const ManualPath expected = manual_path(
      fixture, lambdas, 3.0, max_iterations, precision, 1, true, -1);
  const ApiResult actual = public_path(
      fixture, lambdas, 3.0, max_iterations, precision, 1, true, -1, true);
  bool ok = true;
  ok &= require(expected.output.num_fit == 1 &&
                    expected.attempted_statuses.size() == 2 &&
                    expected.attempted_statuses[0] == PnStatus::kConverged &&
                    expected.attempted_statuses[1] != PnStatus::kConverged,
                "iteration-cap fixture must converge once, then fail");
  ok &= compare_path(actual, expected.output,
                     "max_ite atomic fail-stop");
  ok &= suffix_is_zero(actual, static_cast<int>(fixture.x.cols()),
                       fixture.num_classes, actual.num_fit,
                       "max_ite atomic fail-stop");
  return ok;
}

bool test_v2_status_and_diagnostics() {
  const Fixture fixture = make_signal_fixture();
  const std::vector<double> lambdas = {0.08, 0.035, 0.018};
  const int max_iterations = 5000;
  const double precision = 5e-7;
  bool ok = true;

  for (int reg_type = 1; reg_type <= 3; ++reg_type) {
    const double gamma = reg_type == 3 ? 3.7 : 3.0;
    const ApiResult v1 = public_path(
        fixture, lambdas, gamma, max_iterations, precision, reg_type, true,
        -1, true);
    const V2Result v2 = public_path_v2(
        fixture, lambdas, gamma, max_iterations, precision, reg_type, true,
        -1, true);
    const std::string prefix = "V2 reg_type=" + std::to_string(reg_type);
    ok &= require(v2.failed_lambda == -1 && v2.failed_stage == -1,
                  prefix + " must not report a failed point");
    ok &= require(v2.output.num_fit == static_cast<int>(lambdas.size()),
                  prefix + " must finish the full path");
    ok &= require(v2.output.num_fit == v1.num_fit &&
                      v2.output.iterations == v1.iterations &&
                      v2.output.size_active == v1.size_active &&
                      maximum_difference(v2.output.beta, v1.beta) == 0.0 &&
                      maximum_difference(v2.output.intercept,
                                         v1.intercept) == 0.0,
                  prefix + " must preserve the V1 solution/layout");
    bool saw_work = false;
    bool saw_positive_runtime = false;
    bool has_uncertified_point = false;
    for (std::size_t index = 0; index < lambdas.size(); ++index) {
      ok &= require(v2.outer_iterations[index] >= 0 &&
                        v2.inner_sweeps[index] ==
                            v2.output.iterations[index] &&
                        v2.coordinate_updates[index] >= 0,
                    prefix + " integer diagnostics must be consistent");
      ok &= require(std::isfinite(v2.objective[index]) &&
                        std::isfinite(v2.kkt[index]) &&
                        std::isfinite(v2.stationarity[index]) &&
                        std::isfinite(v2.output.runtime[index]) &&
                        v2.output.runtime[index] >= 0.0,
                    prefix + " floating diagnostics must be finite");
      saw_work = saw_work || v2.coordinate_updates[index] > 0;
      saw_positive_runtime =
          saw_positive_runtime || v2.output.runtime[index] > 0.0;
      if (reg_type != 1) {
        has_uncertified_point = has_uncertified_point ||
                                v2.stationarity[index] > precision;
      }
    }
    const int expected_status =
        has_uncertified_point
            ? PICASSO_MULTINOMIAL_LLA_STATIONARITY_LIMIT
            : PICASSO_MULTINOMIAL_COMPLETED;
    ok &= require(v2.status == expected_status,
                  prefix + " status must distinguish a useful capped path "
                           "from a stationarity-certified path");
    ok &= require(saw_work && saw_positive_runtime,
                  prefix + " must expose work and actual runtime");
  }

  const V2Result default_v2 = public_path_v2(
      fixture, lambdas, 3.7, max_iterations, precision, 3, true, -1, true);
  const V2Result default_v3 = public_path_v3(
      fixture, lambdas, 3.7, max_iterations, precision, 3, true, -1, true,
      kMinimumLlaStageCount);
  ok &= require(default_v2.status == default_v3.status &&
                    default_v2.output.num_fit == default_v3.output.num_fit &&
                    default_v2.output.iterations ==
                        default_v3.output.iterations &&
                    maximum_difference(default_v2.output.beta,
                                       default_v3.output.beta) == 0.0 &&
                    maximum_difference(default_v2.output.intercept,
                                       default_v3.output.intercept) == 0.0 &&
                    default_v2.stationarity == default_v3.stationarity,
                "V3 max_stages=3 must preserve the V2 default path exactly");

  const V2Result strict_v3 = public_path_v3(
      fixture, lambdas, 3.7, max_iterations, precision, 3, true, -1, true,
      kMaximumLlaStageCount);
  ok &= require(strict_v3.status == PICASSO_MULTINOMIAL_COMPLETED &&
                    strict_v3.output.num_fit ==
                        static_cast<int>(lambdas.size()) &&
                    strict_v3.failed_lambda == -1 &&
                    strict_v3.failed_stage == -1,
                "raising the V3 LLA budget must permit strict stationarity");
  for (std::size_t index = 0; index < lambdas.size(); ++index) {
    ok &= require(strict_v3.stationarity[index] <= precision,
                  "higher V3 LLA budget must certify every SCAD path point");
  }

  const V2Result invalid_v3 = public_path_v3(
      fixture, std::vector<double>(1, 0.05), 3.7, max_iterations,
      precision, 3, true, -1, true, 2);
  ok &= require(invalid_v3.status == PICASSO_MULTINOMIAL_INVALID_INPUT &&
                    invalid_v3.output.num_fit == 0 &&
                    invalid_v3.failed_lambda == -1 &&
                    invalid_v3.failed_stage == -1 &&
                    all_zero(invalid_v3.output.beta) &&
                    all_zero(invalid_v3.output.intercept),
                "V3 must reject an LLA stage budget below three "
                "transactionally");

  const V2Result unordered_lambda = public_path_v2(
      fixture, std::vector<double>{0.05, 0.05}, 3.0, max_iterations,
      precision, 1, true, -1, true);
  ok &= require(unordered_lambda.status ==
                        PICASSO_MULTINOMIAL_INVALID_INPUT &&
                    unordered_lambda.output.num_fit == 0 &&
                    all_zero(unordered_lambda.output.beta) &&
                    all_zero(unordered_lambda.output.intercept),
                "V2 must reject a nondecreasing lambda path transactionally");

  const V2Result negative_lambda = public_path_v2(
      fixture, std::vector<double>(1, -0.05), 3.0, max_iterations,
      precision, 1, true, -1, true);
  ok &= require(negative_lambda.status ==
                        PICASSO_MULTINOMIAL_INVALID_INPUT &&
                    negative_lambda.output.num_fit == 0 &&
                    all_zero(negative_lambda.output.beta) &&
                    all_zero(negative_lambda.output.intercept),
                "V2 must reject a negative lambda transactionally");

  const double gradient_maximum = null_gradient_maximum(fixture);
  const std::vector<double> dfmax_lambdas = {
      2.0 * gradient_maximum, 0.8 * gradient_maximum,
      0.4 * gradient_maximum, 0.2 * gradient_maximum};
  const V2Result dfmax = public_path_v2(
      fixture, dfmax_lambdas, 3.0, max_iterations, precision, 1, true, 0,
      false);
  ok &= require(dfmax.status == PICASSO_MULTINOMIAL_DFMAX_REACHED &&
                    dfmax.output.num_fit > 1 &&
                    dfmax.output.num_fit <
                        static_cast<int>(dfmax_lambdas.size()) &&
                    dfmax.failed_lambda == -1 && dfmax.failed_stage == -1,
                "V2 dfmax must be an explicit nonfailure termination");
  ok &= require(std::string(PicassoMultinomialPathStatusString(dfmax.status)) ==
                    "dfmax_reached",
                "V2 status string must describe dfmax");
  ok &= require(
      PICASSO_MULTINOMIAL_LLA_STATIONARITY_LIMIT == 10 &&
          std::string(PicassoMultinomialPathStatusString(
              PICASSO_MULTINOMIAL_LLA_STATIONARITY_LIMIT)) ==
              "lla_stationarity_limit",
      "V2 must expose a stable, descriptive LLA stationarity-limit status");

  const std::vector<double> failing_lambdas = {10.0, 0.006, 0.003};
  const V2Result failed = public_path_v2(
      fixture, failing_lambdas, 3.0, 1, 1e-6, 1, true, -1, true);
  ok &= require(failed.status != PICASSO_MULTINOMIAL_COMPLETED &&
                    failed.status != PICASSO_MULTINOMIAL_DFMAX_REACHED &&
                    failed.status != PICASSO_MULTINOMIAL_INVALID_INPUT &&
                    failed.output.num_fit == 1 &&
                    failed.failed_lambda == 1,
                "V2 iteration failure must identify the failed lambda");
  ok &= require(failed.outer_iterations[1] >= 0 &&
                    failed.inner_sweeps[1] >= 0 &&
                    failed.coordinate_updates[1] >= 0 &&
                    std::isfinite(failed.objective[1]) &&
                    std::isfinite(failed.kkt[1]),
                "V2 failed point must retain solver diagnostics");
  const int failed_d = static_cast<int>(fixture.x.cols());
  for (int lambda_index = failed.output.num_fit;
       lambda_index < static_cast<int>(failing_lambdas.size());
       ++lambda_index) {
    const std::size_t beta_base =
        static_cast<std::size_t>(lambda_index) * failed_d *
        fixture.num_classes;
    const std::size_t intercept_base =
        static_cast<std::size_t>(lambda_index) * fixture.num_classes;
    for (int index = 0; index < failed_d * fixture.num_classes; ++index)
      ok &= require(failed.output.beta[beta_base + index] == 0.0,
                    "V2 failed beta point/suffix must stay zero");
    for (int klass = 0; klass < fixture.num_classes; ++klass)
      ok &= require(failed.output.intercept[intercept_base + klass] == 0.0,
                    "V2 failed intercept point/suffix must stay zero");
    ok &= require(failed.output.iterations[lambda_index] == 0 &&
                      failed.output.size_active[lambda_index] == 0,
                  "V2 failed solution counters must stay zero");
  }

  std::vector<double> labels = double_labels(fixture);
  std::vector<double> design = packed_design(fixture, true);
  std::vector<double> invalid_lambdas(1, 0.05);
  V2Result invalid(static_cast<int>(fixture.x.cols()), fixture.num_classes, 1);
  invalid.status = SolveMultinomialRegressionV2(
      labels.data(), design.data(), static_cast<int>(fixture.x.rows()),
      static_cast<int>(fixture.x.cols()), fixture.num_classes,
      invalid_lambdas.data(), 1, 3.0, 100, 1e-6, 1, true, -2,
      invalid.output.beta.data(), invalid.output.intercept.data(),
      invalid.output.iterations.data(), invalid.output.size_active.data(),
      invalid.output.runtime.data(), &invalid.output.num_fit, true,
      &invalid.failed_lambda, &invalid.failed_stage,
      invalid.outer_iterations.data(), invalid.inner_sweeps.data(),
      invalid.coordinate_updates.data(), invalid.objective.data(),
      invalid.kkt.data(), invalid.stationarity.data());
  ok &= require(invalid.status == PICASSO_MULTINOMIAL_INVALID_INPUT &&
                    invalid.output.num_fit == 0 &&
                    invalid.failed_lambda == -1 &&
                    invalid.failed_stage == -1 &&
                    all_zero(invalid.output.beta) &&
                    all_zero(invalid.output.intercept) &&
                    all_zero(invalid.output.iterations) &&
                    all_zero(invalid.output.size_active) &&
                    all_zero(invalid.output.runtime),
                "V2 invalid input must be explicit and transactional");
  return ok;
}

struct MutableCall {
  int n;
  int d;
  int num_classes;
  int nlambda;
  double gamma;
  int max_iterations;
  double precision;
  int reg_type;
  bool include_intercept;
  int dfmax;
  bool row_major;

  std::vector<double> labels;
  std::vector<double> design;
  std::vector<double> lambdas;
  ApiResult output;

  double *labels_pointer;
  double *design_pointer;
  double *lambda_pointer;
  double *beta_pointer;
  double *intercept_pointer;
  int *iterations_pointer;
  int *size_active_pointer;
  double *runtime_pointer;
  int *num_fit_pointer;

  explicit MutableCall(const Fixture &fixture)
      : n(static_cast<int>(fixture.x.rows())),
        d(static_cast<int>(fixture.x.cols())),
        num_classes(fixture.num_classes),
        nlambda(2),
        gamma(3.0),
        max_iterations(100),
        precision(1e-6),
        reg_type(1),
        include_intercept(true),
        dfmax(-1),
        row_major(true),
        labels(double_labels(fixture)),
        design(packed_design(fixture, true)),
        lambdas(),
        output(d, num_classes, nlambda, kOutputSentinel,
               kIntegerSentinel),
        labels_pointer(0),
        design_pointer(0),
        lambda_pointer(0),
        beta_pointer(0),
        intercept_pointer(0),
        iterations_pointer(0),
        size_active_pointer(0),
        runtime_pointer(0),
        num_fit_pointer(0) {
    lambdas.push_back(0.08);
    lambdas.push_back(0.03);
    labels_pointer = labels.data();
    design_pointer = design.data();
    lambda_pointer = lambdas.data();
    beta_pointer = output.beta.data();
    intercept_pointer = output.intercept.data();
    iterations_pointer = output.iterations.data();
    size_active_pointer = output.size_active.data();
    runtime_pointer = output.runtime.data();
    num_fit_pointer = &output.num_fit;
  }

 private:
  MutableCall(const MutableCall &);
  MutableCall &operator=(const MutableCall &);
};

bool all_zero(const std::vector<double> &values) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (values[index] != 0.0) return false;
  }
  return true;
}

bool all_zero(const std::vector<int> &values) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (values[index] != 0) return false;
  }
  return true;
}

typedef std::function<void(MutableCall *)> CallMutation;

bool expect_rejected(const Fixture &fixture, const std::string &case_name,
                     const CallMutation &mutate) {
  MutableCall call(fixture);
  mutate(&call);
  bool threw = false;
  try {
    SolveMultinomialRegression(
        call.labels_pointer, call.design_pointer, call.n, call.d,
        call.num_classes, call.lambda_pointer, call.nlambda, call.gamma,
        call.max_iterations, call.precision, call.reg_type,
        call.include_intercept, call.dfmax, call.beta_pointer,
        call.intercept_pointer, call.iterations_pointer,
        call.size_active_pointer, call.runtime_pointer,
        call.num_fit_pointer, call.row_major);
  } catch (...) {
    threw = true;
  }

  bool ok = true;
  ok &= require(!threw, case_name + " must not throw across the C ABI");
  if (call.num_fit_pointer != 0)
    ok &= require(call.output.num_fit == 0,
                  case_name + " must set num_fit=0");
  if (call.beta_pointer != 0)
    ok &= require(all_zero(call.output.beta),
                  case_name + " must zero writable beta");
  if (call.intercept_pointer != 0)
    ok &= require(all_zero(call.output.intercept),
                  case_name + " must zero writable intercepts");
  if (call.iterations_pointer != 0)
    ok &= require(all_zero(call.output.iterations),
                  case_name + " must zero writable ite_lamb");
  if (call.size_active_pointer != 0)
    ok &= require(all_zero(call.output.size_active),
                  case_name + " must zero writable size_act");
  if (call.runtime_pointer != 0)
    ok &= require(all_zero(call.output.runtime),
                  case_name + " must zero writable runt");
  return ok;
}

bool test_invalid_inputs_are_transactional() {
  const Fixture fixture = make_signal_fixture();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  bool ok = true;

  ok &= expect_rejected(fixture, "fractional label",
                        [](MutableCall *call) { call->labels[0] = 1.5; });
  ok &= expect_rejected(fixture, "negative label",
                        [](MutableCall *call) { call->labels[0] = -1.0; });
  ok &= expect_rejected(fixture, "out-of-range label",
                        [](MutableCall *call) {
                          call->labels[0] =
                              static_cast<double>(call->num_classes);
                        });
  ok &= expect_rejected(fixture, "NaN label",
                        [nan](MutableCall *call) { call->labels[0] = nan; });
  ok &= expect_rejected(fixture, "NaN design",
                        [nan](MutableCall *call) { call->design[3] = nan; });
  ok &= expect_rejected(fixture, "infinite design",
                        [infinity](MutableCall *call) {
                          call->design[4] = infinity;
                        });
  ok &= expect_rejected(fixture, "negative lambda",
                        [](MutableCall *call) { call->lambdas[1] = -0.01; });
  ok &= expect_rejected(fixture, "NaN lambda",
                        [nan](MutableCall *call) { call->lambdas[0] = nan; });
  ok &= expect_rejected(fixture, "infinite lambda",
                        [infinity](MutableCall *call) {
                          call->lambdas[0] = infinity;
                        });
  ok &= expect_rejected(fixture, "MCP gamma boundary",
                        [](MutableCall *call) {
                          call->reg_type = 2;
                          call->gamma = 1.0;
                        });
  ok &= expect_rejected(fixture, "SCAD gamma boundary",
                        [](MutableCall *call) {
                          call->reg_type = 3;
                          call->gamma = 2.0;
                        });
  ok &= expect_rejected(fixture, "NaN gamma",
                        [nan](MutableCall *call) {
                          call->reg_type = 2;
                          call->gamma = nan;
                        });
  ok &= expect_rejected(fixture, "infinite gamma",
                        [infinity](MutableCall *call) {
                          call->reg_type = 3;
                          call->gamma = infinity;
                        });
  ok &= expect_rejected(fixture, "reg_type below range",
                        [](MutableCall *call) { call->reg_type = 0; });
  ok &= expect_rejected(fixture, "reg_type above range",
                        [](MutableCall *call) { call->reg_type = 4; });
  ok &= expect_rejected(fixture, "zero precision",
                        [](MutableCall *call) { call->precision = 0.0; });
  ok &= expect_rejected(fixture, "negative precision",
                        [](MutableCall *call) { call->precision = -1e-6; });
  ok &= expect_rejected(fixture, "NaN precision",
                        [nan](MutableCall *call) { call->precision = nan; });
  ok &= expect_rejected(fixture, "infinite precision",
                        [infinity](MutableCall *call) {
                          call->precision = infinity;
                        });
  ok &= expect_rejected(fixture, "zero max_ite",
                        [](MutableCall *call) {
                          call->max_iterations = 0;
                        });
  ok &= expect_rejected(fixture, "negative max_ite",
                        [](MutableCall *call) {
                          call->max_iterations = -1;
                        });
  ok &= expect_rejected(fixture, "dfmax below disabled sentinel",
                        [](MutableCall *call) { call->dfmax = -2; });
  ok &= expect_rejected(fixture, "zero n",
                        [](MutableCall *call) { call->n = 0; });

  ok &= expect_rejected(fixture, "null Y",
                        [](MutableCall *call) {
                          call->labels_pointer = 0;
                        });
  ok &= expect_rejected(fixture, "null X",
                        [](MutableCall *call) {
                          call->design_pointer = 0;
                        });
  ok &= expect_rejected(fixture, "null lambda",
                        [](MutableCall *call) {
                          call->lambda_pointer = 0;
                        });
  ok &= expect_rejected(fixture, "null beta output",
                        [](MutableCall *call) { call->beta_pointer = 0; });
  ok &= expect_rejected(fixture, "null intercept output",
                        [](MutableCall *call) {
                          call->intercept_pointer = 0;
                        });
  ok &= expect_rejected(fixture, "null iteration output",
                        [](MutableCall *call) {
                          call->iterations_pointer = 0;
                        });
  ok &= expect_rejected(fixture, "null size output",
                        [](MutableCall *call) {
                          call->size_active_pointer = 0;
                        });
  ok &= expect_rejected(fixture, "null runtime output",
                        [](MutableCall *call) {
                          call->runtime_pointer = 0;
                        });
  ok &= expect_rejected(fixture, "null num_fit output",
                        [](MutableCall *call) {
                          call->num_fit_pointer = 0;
                        });
  return ok;
}

bool test_oversized_dimensions_return_safely() {
  const Fixture fixture = make_signal_fixture();
  MutableCall call(fixture);
  call.d = std::numeric_limits<int>::max();
  call.num_classes = 2;
  bool threw = false;
  try {
    SolveMultinomialRegression(
        call.labels_pointer, call.design_pointer, call.n, call.d,
        call.num_classes, call.lambda_pointer, call.nlambda, call.gamma,
        call.max_iterations, call.precision, call.reg_type,
        call.include_intercept, call.dfmax, call.beta_pointer,
        call.intercept_pointer, call.iterations_pointer,
        call.size_active_pointer, call.runtime_pointer,
        call.num_fit_pointer, call.row_major);
  } catch (...) {
    threw = true;
  }

  bool ok = true;
  ok &= require(!threw,
                "oversized d*K must return without allocation or throw");
  ok &= require(call.output.num_fit == 0,
                "oversized d*K must set num_fit=0");
  ok &= require(all_zero(call.output.iterations) &&
                    all_zero(call.output.size_active) &&
                    all_zero(call.output.runtime),
                "oversized d*K must zero path diagnostics");
  const std::size_t safe_intercept_count =
      static_cast<std::size_t>(call.num_classes) * call.nlambda;
  for (std::size_t index = 0; index < safe_intercept_count; ++index)
    ok &= require(call.output.intercept[index] == 0.0,
                  "oversized d*K must zero the safely sized intercept output");
  return ok;
}

bool test_legacy_sibling_smoke() {
  const Fixture fixture = make_signal_fixture();
  std::vector<double> labels = double_labels(fixture);
  std::vector<double> design = packed_design(fixture, true);
  std::vector<double> lambdas(1, 10.0);
  const int n = static_cast<int>(fixture.x.rows());
  const int d = static_cast<int>(fixture.x.cols());
  ApiResult result(d, fixture.num_classes, 1, kOutputSentinel,
                   kIntegerSentinel);
  bool threw = false;
  try {
    SolveMultinomialRegressionLegacy(
        labels.data(), design.data(), n, d, fixture.num_classes,
        lambdas.data(), 1, 3.0, 100, 1e-6, 1, true, -1,
        result.beta.data(), result.intercept.data(), result.iterations.data(),
        result.size_active.data(), result.runtime.data(), &result.num_fit,
        true);
  } catch (...) {
    threw = true;
  }
  bool ok = true;
  ok &= require(!threw, "legacy sibling smoke call must not throw");
  ok &= require(result.num_fit == 1,
                "legacy sibling smoke call must fit one lambda");
  for (std::size_t index = 0; index < result.beta.size(); ++index)
    ok &= require(std::isfinite(result.beta[index]),
                  "legacy sibling beta must be finite");
  for (std::size_t index = 0; index < result.intercept.size(); ++index)
    ok &= require(std::isfinite(result.intercept[index]),
                  "legacy sibling intercept must be finite");
  ok &= require(result.iterations[0] > 0 && result.size_active[0] >= 0 &&
                    result.runtime[0] == 0.0,
                "legacy sibling diagnostics must be writable and finite");
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_three_penalties_and_both_layouts();
  ok &= test_intercept_false_and_layout();
  ok &= test_borrowed_column_major_boundaries();
  ok &= test_v5_smooth_nll_path();
  ok &= test_glmnet_style_saturated_path_early_stop();
  ok &= test_dfmax_includes_crossing_lambda();
  ok &= test_iteration_limit_is_atomic_fail_stop();
  ok &= test_v2_status_and_diagnostics();
  ok &= test_invalid_inputs_are_transactional();
  ok &= test_oversized_dimensions_return_safely();
  ok &= test_legacy_sibling_smoke();
  if (!ok) return 1;
  std::cout << "multinomial_c_api_test: PASS\n";
  return 0;
}
