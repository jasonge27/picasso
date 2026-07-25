#include <picasso/c_api.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

enum class Family { kBinomial, kPoisson, kSqrtLasso };

const int kN = 180;
const int kD = 12;
const int kNlambda = 4;
const double kPrecision = 1e-5;

bool require(bool condition, const std::string &message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

struct Fixture {
  std::vector<double> x;
  std::vector<double> binomial;
  std::vector<double> poisson;
  std::vector<double> sqrt_lasso;
  std::vector<double> offset;
};

Fixture make_fixture() {
  Fixture result;
  result.x.resize(static_cast<std::size_t>(kN) * kD);
  result.binomial.resize(kN);
  result.poisson.resize(kN);
  result.sqrt_lasso.resize(kN);
  result.offset.resize(kN);
  std::vector<double> means(kD, 0.0);
  std::vector<double> scales(kD, 0.0);
  for (int i = 0; i < kN; ++i) {
    for (int j = 0; j < kD; ++j) {
      const std::size_t index = static_cast<std::size_t>(i) * kD + j;
      result.x[index] =
          std::sin(0.07 * (i + 1.0) * (j + 1.0)) +
          0.25 * std::cos(0.13 * (i + j + 2.0));
      means[j] += result.x[index];
    }
  }
  for (int j = 0; j < kD; ++j) means[j] /= kN;
  for (int i = 0; i < kN; ++i) {
    for (int j = 0; j < kD; ++j) {
      const std::size_t index = static_cast<std::size_t>(i) * kD + j;
      result.x[index] -= means[j];
      scales[j] += result.x[index] * result.x[index];
    }
  }
  for (int j = 0; j < kD; ++j)
    scales[j] = std::sqrt(scales[j] / (kN - 1.0));

  const double beta[] = {1.0, -0.8, 0.65, -0.5};
  for (int i = 0; i < kN; ++i) {
    double signal = 0.0;
    for (int j = 0; j < kD; ++j) {
      const std::size_t index = static_cast<std::size_t>(i) * kD + j;
      result.x[index] /= scales[j];
      if (j < 4) signal += beta[j] * result.x[index];
    }
    const double draw =
        std::fmod(0.6180339887498949 * static_cast<double>(i + 1), 1.0);
    result.binomial[i] =
        draw < 1.0 / (1.0 + std::exp(-(-0.2 + signal))) ? 1.0 : 0.0;
    result.poisson[i] =
        std::floor(std::exp(0.15 + 0.28 * signal) + draw);
    result.sqrt_lasso[i] =
        0.4 + signal + 0.35 * std::sin(0.31 * (i + 1.0));
    result.offset[i] = 0.12 * std::sin(0.19 * (i + 1.0));
  }
  return result;
}

const std::vector<double> &response(const Fixture &fixture, Family family) {
  if (family == Family::kBinomial) return fixture.binomial;
  if (family == Family::kPoisson) return fixture.poisson;
  return fixture.sqrt_lasso;
}

std::vector<double> lambda_path(const Fixture &fixture, Family family,
                                bool failure_path = false) {
  const std::vector<double> &y = response(fixture, family);
  double y_mean = 0.0;
  for (int i = 0; i < kN; ++i) y_mean += y[i];
  y_mean /= kN;
  double loss_scale = 1.0;
  if (family == Family::kSqrtLasso) {
    double squared = 0.0;
    for (int i = 0; i < kN; ++i)
      squared += (y[i] - y_mean) * (y[i] - y_mean);
    loss_scale = std::sqrt(squared / kN);
  }
  double lambda_max = 0.0;
  for (int j = 0; j < kD; ++j) {
    double gradient = 0.0;
    for (int i = 0; i < kN; ++i) {
      gradient += fixture.x[static_cast<std::size_t>(i) * kD + j] *
                  (y[i] - y_mean);
    }
    lambda_max = std::max(
        lambda_max, std::fabs(gradient) / (kN * loss_scale));
  }
  if (failure_path)
    return std::vector<double>{lambda_max, 0.2 * lambda_max};
  const double ratios[] = {1.0, 0.72, 0.52, 0.37};
  std::vector<double> result(kNlambda);
  for (int index = 0; index < kNlambda; ++index)
    result[index] = lambda_max * ratios[index];
  return result;
}

struct Outputs {
  std::vector<double> beta;
  std::vector<double> intercept;
  std::vector<int> iterations;
  std::vector<int> active_size;
  std::vector<double> runtime;
  int number_fit;
  int failed_lambda;
  int failed_stage;
  std::vector<int> stages;
  std::vector<double> objective;
  std::vector<double> kkt;
  std::vector<double> stationarity;
  std::vector<double> smooth_objective;

  explicit Outputs(int nlambda)
      : beta(static_cast<std::size_t>(nlambda) * kD, 91.0),
        intercept(nlambda, 91.0),
        iterations(nlambda, 91),
        active_size(nlambda, 91),
        runtime(nlambda, 91.0),
        number_fit(91),
        failed_lambda(91),
        failed_stage(91),
        stages(nlambda, 91),
        objective(nlambda, 91.0),
        kkt(nlambda, 91.0),
        stationarity(nlambda, 91.0),
        smooth_objective(nlambda, 91.0) {}
};

int call_v2(const Fixture &fixture, Family family, std::vector<double> *lambda,
            int reg_type, int maximum_iterations, double precision,
            int maximum_stages, int dfmax, double gamma, Outputs *output,
            bool include_intercept = true, bool pass_offset = true) {
  std::vector<double> y = response(fixture, family);
  const int nlambda = static_cast<int>(lambda->size());
  if (family == Family::kBinomial) {
    return SolveLogisticRegressionV2(
        y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
        lambda->data(), nlambda, gamma, maximum_iterations, precision,
        reg_type, include_intercept, dfmax,
        pass_offset ? const_cast<double *>(fixture.offset.data()) : nullptr,
        output->beta.data(),
        output->intercept.data(), output->iterations.data(),
        output->active_size.data(), output->runtime.data(),
        &output->number_fit, true, maximum_stages, &output->failed_lambda,
        &output->failed_stage, output->stages.data(), output->objective.data(),
        output->kkt.data(), output->stationarity.data());
  }
  if (family == Family::kPoisson) {
    return SolvePoissonRegressionV2(
        y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
        lambda->data(), nlambda, gamma, maximum_iterations, precision,
        reg_type, include_intercept, dfmax,
        pass_offset ? const_cast<double *>(fixture.offset.data()) : nullptr,
        output->beta.data(),
        output->intercept.data(), output->iterations.data(),
        output->active_size.data(), output->runtime.data(),
        &output->number_fit, true, maximum_stages, &output->failed_lambda,
        &output->failed_stage, output->stages.data(), output->objective.data(),
        output->kkt.data(), output->stationarity.data());
  }
  return SolveSqrtLinearRegressionV2(
      y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
      lambda->data(), nlambda, gamma, maximum_iterations, precision, reg_type,
      include_intercept, dfmax, output->beta.data(), output->intercept.data(),
      output->iterations.data(), output->active_size.data(),
      output->runtime.data(), &output->number_fit, true, maximum_stages,
      &output->failed_lambda, &output->failed_stage, output->stages.data(),
      output->objective.data(), output->kkt.data(),
      output->stationarity.data());
}

int call_v3(const Fixture &fixture, Family family, std::vector<double> *lambda,
            int reg_type, Outputs *output, bool include_intercept = true,
            int maximum_iterations = 1000,
            double precision = kPrecision, int maximum_stages = 25,
            int dfmax = -1) {
  std::vector<double> y = response(fixture, family);
  const int nlambda = static_cast<int>(lambda->size());
  if (family == Family::kBinomial) {
    return SolveLogisticRegressionV3(
        y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
        lambda->data(), nlambda, 3.5, maximum_iterations, precision,
        reg_type, include_intercept, dfmax,
        const_cast<double *>(fixture.offset.data()), output->beta.data(),
        output->intercept.data(), output->iterations.data(),
        output->active_size.data(), output->runtime.data(),
        &output->number_fit, true, maximum_stages, &output->failed_lambda,
        &output->failed_stage, output->stages.data(), output->objective.data(),
        output->kkt.data(), output->stationarity.data(),
        output->smooth_objective.data());
  }
  if (family == Family::kPoisson) {
    return SolvePoissonRegressionV3(
        y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
        lambda->data(), nlambda, 3.5, maximum_iterations, precision,
        reg_type, include_intercept, dfmax,
        const_cast<double *>(fixture.offset.data()), output->beta.data(),
        output->intercept.data(), output->iterations.data(),
        output->active_size.data(), output->runtime.data(),
        &output->number_fit, true, maximum_stages, &output->failed_lambda,
        &output->failed_stage, output->stages.data(), output->objective.data(),
        output->kkt.data(), output->stationarity.data(),
        output->smooth_objective.data());
  }
  return SolveSqrtLinearRegressionV3(
      y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
      lambda->data(), nlambda, 3.5, maximum_iterations, precision, reg_type,
      include_intercept, dfmax,
      output->beta.data(), output->intercept.data(),
      output->iterations.data(), output->active_size.data(),
      output->runtime.data(), &output->number_fit, true, maximum_stages,
      &output->failed_lambda, &output->failed_stage, output->stages.data(),
      output->objective.data(), output->kkt.data(),
      output->stationarity.data(), output->smooth_objective.data());
}

double explicit_smooth_objective(const Fixture &fixture, Family family,
                                 const Outputs &output, int lambda_index) {
  const std::vector<double> &y = response(fixture, family);
  long double total = 0.0L;
  for (int row = 0; row < kN; ++row) {
    long double eta = output.intercept[lambda_index];
    for (int feature = 0; feature < kD; ++feature) {
      eta += fixture.x[static_cast<std::size_t>(row) * kD + feature] *
             output.beta[static_cast<std::size_t>(lambda_index) * kD +
                         feature];
    }
    if (family != Family::kSqrtLasso) eta += fixture.offset[row];
    if (family == Family::kBinomial) {
      total += y[row] > 0.5 ? std::log1p(std::exp(-eta))
                            : std::log1p(std::exp(eta));
    } else if (family == Family::kPoisson) {
      total += std::exp(eta) - y[row] * eta;
    } else {
      const long double residual = y[row] - eta;
      total += residual * residual;
    }
  }
  const long double mean = total / kN;
  return family == Family::kSqrtLasso
             ? std::sqrt(static_cast<double>(mean))
             : static_cast<double>(mean);
}

Outputs call_legacy(const Fixture &fixture, Family family,
                    std::vector<double> *lambda, int reg_type) {
  std::vector<double> y = response(fixture, family);
  Outputs output(static_cast<int>(lambda->size()));
  if (family == Family::kBinomial) {
    SolveLogisticRegression(
        y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
        lambda->data(), static_cast<int>(lambda->size()), 3.5, 1000,
        kPrecision, reg_type, true, -1,
        const_cast<double *>(fixture.offset.data()), output.beta.data(),
        output.intercept.data(), output.iterations.data(),
        output.active_size.data(), output.runtime.data(), &output.number_fit,
        true);
  } else if (family == Family::kPoisson) {
    SolvePoissonRegression(
        y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
        lambda->data(), static_cast<int>(lambda->size()), 3.5, 1000,
        kPrecision, reg_type, true, -1,
        const_cast<double *>(fixture.offset.data()), output.beta.data(),
        output.intercept.data(), output.iterations.data(),
        output.active_size.data(), output.runtime.data(), &output.number_fit,
        true);
  } else {
    SolveSqrtLinearRegression(
        y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
        lambda->data(), static_cast<int>(lambda->size()), 3.5, 1000,
        kPrecision, reg_type, true, -1, output.beta.data(),
        output.intercept.data(), output.iterations.data(),
        output.active_size.data(), output.runtime.data(), &output.number_fit,
        true);
  }
  return output;
}

bool same_prefix(const Outputs &left, const Outputs &right) {
  if (left.number_fit != right.number_fit) return false;
  const std::size_t beta_count =
      static_cast<std::size_t>(left.number_fit) * kD;
  return std::equal(left.beta.begin(), left.beta.begin() + beta_count,
                    right.beta.begin()) &&
         std::equal(left.intercept.begin(),
                    left.intercept.begin() + left.number_fit,
                    right.intercept.begin());
}

int call_v3_with_null_outputs(const Fixture &fixture, Family family,
                              std::vector<double> *lambda) {
  std::vector<double> y = response(fixture, family);
  const int nlambda = static_cast<int>(lambda->size());
  if (family == Family::kBinomial) {
    return SolveLogisticRegressionV3(
        y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
        lambda->data(), nlambda, 3.5, 1000, kPrecision, 1, true, -1,
        const_cast<double *>(fixture.offset.data()), nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, true, 3, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr);
  }
  if (family == Family::kPoisson) {
    return SolvePoissonRegressionV3(
        y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
        lambda->data(), nlambda, 3.5, 1000, kPrecision, 1, true, -1,
        const_cast<double *>(fixture.offset.data()), nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, true, 3, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr);
  }
  return SolveSqrtLinearRegressionV3(
      y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
      lambda->data(), nlambda, 3.5, 1000, kPrecision, 1, true, -1,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, true, 3,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

std::string family_name(Family family) {
  if (family == Family::kBinomial) return "binomial";
  if (family == Family::kPoisson) return "poisson";
  return "sqrt-lasso";
}

bool test_zero_curvature_path() {
  const int n = 8;
  const int d = 2;
  const int nlambda = 2;
  const double precision = 1e-7;
  std::vector<double> design(static_cast<std::size_t>(n) * d, 0.0);
  std::vector<double> offset(n);
  for (int row = 0; row < n; ++row)
    offset[row] = 0.1 * std::sin(0.7 * (row + 1.0));
  const std::vector<double> binomial = {0.0, 1.0, 0.0, 1.0,
                                        1.0, 0.0, 1.0, 0.0};
  const std::vector<double> poisson = {0.0, 1.0, 2.0, 0.0,
                                       3.0, 1.0, 0.0, 2.0};
  const std::vector<double> sqrt_lasso = {
      -1.0, 0.5, 2.0, -0.25, 1.5, -0.75, 0.25, 1.0};
  const Family families[] = {
      Family::kBinomial, Family::kPoisson, Family::kSqrtLasso};
  bool ok = true;

  for (int family_index = 0; family_index < 3; ++family_index) {
    const Family family = families[family_index];
    for (int reg_type = 1; reg_type <= 3; ++reg_type) {
      for (int intercept_value = 0; intercept_value <= 1;
           ++intercept_value) {
        for (int python_layout = 0; python_layout <= 1; ++python_layout) {
          std::vector<double> response_values =
              family == Family::kBinomial
                  ? binomial
                  : (family == Family::kPoisson ? poisson : sqrt_lasso);
          double lambda[nlambda] = {1.0, 0.0};
          double beta[nlambda * d] = {91.0, 91.0, 91.0, 91.0};
          double intercept[nlambda] = {91.0, 91.0};
          int iterations[nlambda] = {91, 91};
          int active_size[nlambda] = {91, 91};
          double runtime[nlambda] = {91.0, 91.0};
          int number_fit = 91;
          int failed_lambda = 91;
          int failed_stage = 91;
          int stages[nlambda] = {91, 91};
          double objective[nlambda] = {91.0, 91.0};
          double kkt[nlambda] = {91.0, 91.0};
          double stationarity[nlambda] = {91.0, 91.0};
          double smooth[nlambda] = {91.0, 91.0};
          int status = PICASSO_LLA_INVALID_INPUT;
          const bool include_intercept = intercept_value != 0;
          const bool use_python = python_layout != 0;

          if (family == Family::kBinomial) {
            status = SolveLogisticRegressionV3(
                response_values.data(), design.data(), n, d, lambda,
                nlambda, 3.5, 1000, precision, reg_type,
                include_intercept, -1, offset.data(), beta, intercept,
                iterations, active_size, runtime, &number_fit, use_python, 5,
                &failed_lambda, &failed_stage, stages, objective, kkt,
                stationarity, smooth);
          } else if (family == Family::kPoisson) {
            status = SolvePoissonRegressionV3(
                response_values.data(), design.data(), n, d, lambda,
                nlambda, 3.5, 1000, precision, reg_type,
                include_intercept, -1, offset.data(), beta, intercept,
                iterations, active_size, runtime, &number_fit, use_python, 5,
                &failed_lambda, &failed_stage, stages, objective, kkt,
                stationarity, smooth);
          } else {
            status = SolveSqrtLinearRegressionV3(
                response_values.data(), design.data(), n, d, lambda,
                nlambda, 3.5, 1000, precision, reg_type,
                include_intercept, -1, beta, intercept, iterations,
                active_size, runtime, &number_fit, use_python, 5,
                &failed_lambda, &failed_stage, stages, objective, kkt,
                stationarity, smooth);
          }

          std::ostringstream context;
          context << family_name(family) << " penalty=" << reg_type
                  << " intercept=" << include_intercept
                  << " python-layout=" << use_python;
          const std::string label = context.str();
          ok &= require((status == PICASSO_LLA_COMPLETED ||
                         status == PICASSO_LLA_STATIONARITY_LIMIT) &&
                            number_fit == nlambda && failed_lambda == -1 &&
                            failed_stage == -1,
                        label + " did not complete its zero-curvature path");
          for (int lambda_index = 0; lambda_index < nlambda;
               ++lambda_index) {
            ok &= require(beta[lambda_index * d] == 0.0 &&
                              beta[lambda_index * d + 1] == 0.0 &&
                              active_size[lambda_index] == 0 &&
                              std::isfinite(intercept[lambda_index]) &&
                              std::isfinite(objective[lambda_index]) &&
                              std::isfinite(kkt[lambda_index]) &&
                              std::isfinite(stationarity[lambda_index]) &&
                              std::isfinite(smooth[lambda_index]) &&
                              kkt[lambda_index] <= precision + 1e-12,
                          label + " produced an invalid flat-coordinate fit");
          }
        }
      }
    }
  }
  return ok;
}

bool test_success_and_legacy_compatibility() {
  bool ok = true;
  const Fixture fixture = make_fixture();
  const Family families[] = {
      Family::kBinomial, Family::kPoisson, Family::kSqrtLasso};
  for (int family_index = 0; family_index < 3; ++family_index) {
    const Family family = families[family_index];
    const std::string label = family_name(family);
    for (int reg_type = 1; reg_type <= 3; ++reg_type) {
      std::vector<double> lambdas = lambda_path(fixture, family);
      Outputs capped(kNlambda);
      const int status = call_v2(
          fixture, family, &lambdas, reg_type, 1000, kPrecision, 3, -1, 3.5,
          &capped);
      ok &= require(status == PICASSO_LLA_COMPLETED ||
                        status == PICASSO_LLA_STATIONARITY_LIMIT,
                    label + " V2 returned a hard status");
      ok &= require(capped.number_fit == kNlambda &&
                        capped.failed_lambda == -1 &&
                        capped.failed_stage == -1,
                    label + " V2 did not commit the complete path");
      for (int index = 0; index < kNlambda; ++index) {
        ok &= require(capped.stages[index] == (reg_type == 1 ? 1 : 3) &&
                          std::isfinite(capped.objective[index]) &&
                          std::isfinite(capped.kkt[index]) &&
                          std::isfinite(capped.stationarity[index]) &&
                          capped.kkt[index] <= kPrecision + 1e-12,
                      label + " V2 diagnostics are not certified");
      }

      Outputs legacy = call_legacy(fixture, family, &lambdas, reg_type);
      ok &= require(same_prefix(capped, legacy),
                    label + " legacy ABI does not match V2 cap=3");

      if (reg_type != 1) {
        Outputs adaptive(kNlambda);
        const int adaptive_status = call_v2(
            fixture, family, &lambdas, reg_type, 1000, kPrecision, 25, -1,
            3.5, &adaptive);
        ok &= require(adaptive_status == PICASSO_LLA_COMPLETED &&
                          adaptive.number_fit == kNlambda,
                      label + " raised cap did not complete");
        for (int index = 0; index < kNlambda; ++index) {
          ok &= require(adaptive.stages[index] >= 3 &&
                            adaptive.stages[index] <= 25 &&
                            adaptive.stationarity[index] <=
                                kPrecision + 1e-12 &&
                            adaptive.objective[index] <=
                                capped.objective[index] + 1e-7,
                        label + " raised cap lost its adaptive certificate");
        }
      }
    }
  }
  return ok;
}

bool test_native_smooth_objective_paths() {
  bool ok = true;
  const Fixture fixture = make_fixture();
  const Family families[] = {
      Family::kBinomial, Family::kPoisson, Family::kSqrtLasso};
  for (int family_index = 0; family_index < 3; ++family_index) {
    const Family family = families[family_index];
    for (int reg_type = 1; reg_type <= 3; ++reg_type) {
      std::vector<double> lambdas = lambda_path(fixture, family);
      Outputs output(kNlambda);
      const int status = call_v3(
          fixture, family, &lambdas, reg_type, &output);
      ok &= require(status == PICASSO_LLA_COMPLETED &&
                        output.number_fit == kNlambda,
                    family_name(family) + " V3 did not complete");
      for (int index = 0; index < output.number_fit; ++index) {
        const double expected = explicit_smooth_objective(
            fixture, family, output, index);
        const double difference =
            std::fabs(output.smooth_objective[index] - expected);
        const double tolerance = family == Family::kBinomial
                                     ? 2e-12
                                     : 5e-9;
        std::ostringstream difference_text;
        difference_text << std::setprecision(17) << difference;
        ok &= require(
            difference <= tolerance,
            family_name(family) + " V3 smooth objective mismatch (" +
                difference_text.str() + ")");
        ok &= require(output.objective[index] + 1e-13 >=
                          output.smooth_objective[index],
                      family_name(family) +
                          " target objective is below smooth objective");
      }
    }
  }

  std::vector<double> lambdas =
      lambda_path(fixture, Family::kSqrtLasso);
  Outputs naive(kNlambda);
  SolveLinearRegressionNaiveUpdateV2(
      const_cast<double *>(fixture.sqrt_lasso.data()),
      const_cast<double *>(fixture.x.data()), kN, kD, lambdas.data(),
      kNlambda, 3.5, 1000, kPrecision, 1, true, -1, naive.beta.data(),
      naive.intercept.data(), naive.iterations.data(),
      naive.active_size.data(), naive.runtime.data(), &naive.number_fit,
      true, naive.smooth_objective.data());
  Outputs covariance(kNlambda);
  SolveLinearRegressionCovUpdateV2(
      const_cast<double *>(fixture.sqrt_lasso.data()),
      const_cast<double *>(fixture.x.data()), kN, kD, lambdas.data(),
      kNlambda, 3.5, 1000, kPrecision, 1, true, -1,
      covariance.beta.data(), covariance.intercept.data(),
      covariance.iterations.data(), covariance.active_size.data(),
      covariance.runtime.data(), &covariance.number_fit, true,
      covariance.smooth_objective.data());
  ok &= require(naive.number_fit == kNlambda &&
                    covariance.number_fit == kNlambda,
                "Gaussian V2 did not complete");
  for (int index = 0; index < kNlambda; ++index) {
    const double naive_rmse = explicit_smooth_objective(
        fixture, Family::kSqrtLasso, naive, index);
    const double covariance_rmse = explicit_smooth_objective(
        fixture, Family::kSqrtLasso, covariance, index);
    ok &= require(std::fabs(naive.smooth_objective[index] -
                            naive_rmse * naive_rmse) <= 2e-12,
                  "Gaussian naive V2 MSE mismatch");
    ok &= require(std::fabs(covariance.smooth_objective[index] -
                            covariance_rmse * covariance_rmse) <= 2e-12,
                  "Gaussian covariance V2 MSE mismatch");
  }
  return ok;
}

bool test_intercept_dfmax_and_optional_outputs() {
  bool ok = true;
  const Fixture fixture = make_fixture();
  const Family families[] = {
      Family::kBinomial, Family::kPoisson, Family::kSqrtLasso};

  for (int family_index = 0; family_index < 3; ++family_index) {
    const Family family = families[family_index];
    std::vector<double> lambdas = lambda_path(fixture, family);
    Outputs no_intercept(kNlambda);
    const int no_intercept_status = call_v2(
        fixture, family, &lambdas, 1, 1000, kPrecision, 3, -1, 3.5,
        &no_intercept, false, true);
    bool intercept_is_zero = true;
    for (int index = 0; index < no_intercept.number_fit; ++index) {
      if (no_intercept.intercept[index] != 0.0)
        intercept_is_zero = false;
    }
    ok &= require(no_intercept_status == PICASSO_LLA_COMPLETED &&
                      no_intercept.number_fit == kNlambda &&
                      intercept_is_zero,
                  family_name(family) +
                      " no-intercept path did not complete correctly");

    ok &= require(call_v3_with_null_outputs(fixture, family, &lambdas) ==
                      PICASSO_LLA_COMPLETED,
                  family_name(family) +
                      " V3 rejected null optional outputs");
  }

  std::vector<double> logistic_y = fixture.binomial;
  std::vector<double> logistic_lambdas =
      lambda_path(fixture, Family::kBinomial);
  SolveLogisticRegression(
      logistic_y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
      logistic_lambdas.data(), kNlambda, 3.5, 1000, kPrecision, 1, true, -1,
      const_cast<double *>(fixture.offset.data()), nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, true);
  ok &= require(SolveLogisticRegressionV2(
                    logistic_y.data(),
                    const_cast<double *>(fixture.x.data()), kN, kD,
                    logistic_lambdas.data(), kNlambda, 3.5, 1000,
                    kPrecision, 1, true, -1,
                    const_cast<double *>(fixture.offset.data()), nullptr,
                    nullptr, nullptr, nullptr, nullptr, nullptr, true, 3,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) ==
                    PICASSO_LLA_COMPLETED,
                "V1/V2 logistic APIs did not tolerate null outputs");

  const double lambda_max = logistic_lambdas.front();
  const double ratios[] = {0.50, 0.42, 0.34, 0.27,
                           0.21, 0.16, 0.12, 0.08};
  std::vector<double> dfmax_lambdas(8);
  for (int index = 0; index < 8; ++index)
    dfmax_lambdas[index] = lambda_max * ratios[index];
  Outputs dfmax_output(8);
  const int dfmax_status = SolveLogisticRegressionV3(
      logistic_y.data(), const_cast<double *>(fixture.x.data()), kN, kD,
      dfmax_lambdas.data(), 8, 3.5, 1000, kPrecision, 1, true, 0,
      const_cast<double *>(fixture.offset.data()), dfmax_output.beta.data(),
      dfmax_output.intercept.data(), dfmax_output.iterations.data(), nullptr,
      dfmax_output.runtime.data(), &dfmax_output.number_fit, true, 3,
      &dfmax_output.failed_lambda, &dfmax_output.failed_stage,
      dfmax_output.stages.data(), dfmax_output.objective.data(),
      dfmax_output.kkt.data(), dfmax_output.stationarity.data(),
      dfmax_output.smooth_objective.data());
  int final_nonzero = 0;
  if (dfmax_output.number_fit > 0) {
    const int final_index = dfmax_output.number_fit - 1;
    for (int feature = 0; feature < kD; ++feature) {
      if (std::fabs(dfmax_output.beta[
              static_cast<std::size_t>(final_index) * kD + feature]) > 1e-8)
        ++final_nonzero;
    }
  }
  ok &= require(dfmax_status == PICASSO_LLA_DFMAX_REACHED &&
                    dfmax_output.number_fit == 5 && final_nonzero > 0,
                "dfmax status must use independent sparsity when size_act "
                "is null");
  return ok;
}

bool all_zero(const std::vector<double> &values) {
  for (std::size_t index = 0; index < values.size(); ++index)
    if (values[index] != 0.0) return false;
  return true;
}

bool test_input_validation_and_transactionality() {
  bool ok = true;
  const Fixture fixture = make_fixture();
  std::vector<double> one_lambda(1, 0.2);
  struct InvalidCase {
    double gamma;
    int max_iterations;
    double precision;
    int reg_type;
    int dfmax;
    int maximum_stages;
  };
  const InvalidCase cases[] = {
      {3.5, 1000, kPrecision, 2, -1, 2},
      {3.5, 1000, kPrecision, 0, -1, 3},
      {1.0, 1000, kPrecision, 2, -1, 3},
      {2.0, 1000, kPrecision, 3, -1, 3},
      {3.5, 0, kPrecision, 2, -1, 3},
      {3.5, 1000, std::numeric_limits<double>::quiet_NaN(), 2, -1, 3},
      {3.5, 1000, kPrecision, 2, -2, 3}};
  for (std::size_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
       ++index) {
    Outputs output(1);
    const int status = call_v2(
        fixture, Family::kBinomial, &one_lambda, cases[index].reg_type,
        cases[index].max_iterations, cases[index].precision,
        cases[index].maximum_stages, cases[index].dfmax,
        cases[index].gamma, &output);
    ok &= require(status == PICASSO_LLA_INVALID_INPUT &&
                      output.number_fit == 0 &&
                      output.failed_lambda == -1 &&
                      output.failed_stage == -1 &&
                      all_zero(output.beta) &&
                      std::isnan(output.objective[0]) &&
                      std::isnan(output.kkt[0]) &&
                      std::isnan(output.stationarity[0]),
                  "invalid scalar V2 input was not rejected transactionally");
  }

  std::vector<double> unordered{0.2, 0.2};
  Outputs unordered_output(2);
  ok &= require(
      call_v2(fixture, Family::kBinomial, &unordered, 2, 1000,
              kPrecision, 3, -1, 3.5, &unordered_output) ==
          PICASSO_LLA_INVALID_INPUT,
      "nondecreasing lambda path was accepted");
  std::vector<double> negative(1, -0.1);
  Outputs negative_output(1);
  ok &= require(
      call_v2(fixture, Family::kBinomial, &negative, 2, 1000,
              kPrecision, 3, -1, 3.5, &negative_output) ==
          PICASSO_LLA_INVALID_INPUT,
      "negative lambda was accepted");

  Outputs gaussian_unordered(2);
  SolveLinearRegressionNaiveUpdate(
      const_cast<double *>(fixture.sqrt_lasso.data()),
      const_cast<double *>(fixture.x.data()), kN, kD, unordered.data(), 2,
      3.5, 1000, kPrecision, 1, true, -1, gaussian_unordered.beta.data(),
      gaussian_unordered.intercept.data(),
      gaussian_unordered.iterations.data(),
      gaussian_unordered.active_size.data(),
      gaussian_unordered.runtime.data(), &gaussian_unordered.number_fit,
      true);
  ok &= require(gaussian_unordered.number_fit == 0 &&
                    all_zero(gaussian_unordered.beta) &&
                    all_zero(gaussian_unordered.intercept),
                "Gaussian naive C API accepted a nondecreasing lambda path");

  Outputs gaussian_negative(1);
  SolveLinearRegressionCovUpdate(
      const_cast<double *>(fixture.sqrt_lasso.data()),
      const_cast<double *>(fixture.x.data()), kN, kD, negative.data(), 1,
      3.5, 1000, kPrecision, 1, true, -1, gaussian_negative.beta.data(),
      gaussian_negative.intercept.data(), gaussian_negative.iterations.data(),
      gaussian_negative.active_size.data(), gaussian_negative.runtime.data(),
      &gaussian_negative.number_fit, true);
  ok &= require(gaussian_negative.number_fit == 0 &&
                    all_zero(gaussian_negative.beta) &&
                    all_zero(gaussian_negative.intercept),
                "Gaussian covariance C API accepted a negative lambda");

  // Reject flattened dimensions before touching the coefficient buffer.  The
  // output is deliberately one element long so this also protects the legacy
  // wrappers, which delegate to V2, from integer-overflow-sized zero fills.
  double tiny_response = 0.0;
  double tiny_design = 0.0;
  double overflow_lambdas[2] = {0.2, 0.1};
  double beta_sentinel = 73.0;
  double overflow_intercept[2] = {0.0, 0.0};
  int overflow_iterations[2] = {0, 0};
  int overflow_active[2] = {0, 0};
  double overflow_runtime[2] = {0.0, 0.0};
  int overflow_num_fit = 91;
  double tiny_offset = 0.0;
  int overflow_failed_lambda = 91;
  int overflow_failed_stage = 91;
  int overflow_stages[2] = {91, 91};
  double overflow_objective[2] = {91.0, 91.0};
  double overflow_kkt[2] = {91.0, 91.0};
  double overflow_stationarity[2] = {91.0, 91.0};
  const int overflow_status = SolveLogisticRegressionV2(
      &tiny_response, &tiny_design, 1, std::numeric_limits<int>::max(),
      overflow_lambdas, 2, 3.5, 1000, kPrecision, 2, true, -1,
      &tiny_offset, &beta_sentinel, overflow_intercept,
      overflow_iterations, overflow_active, overflow_runtime,
      &overflow_num_fit, false, 3, &overflow_failed_lambda,
      &overflow_failed_stage, overflow_stages, overflow_objective,
      overflow_kkt, overflow_stationarity);
  ok &= require(overflow_status == PICASSO_LLA_INVALID_INPUT &&
                    overflow_num_fit == 0 && beta_sentinel == 73.0 &&
                    overflow_failed_lambda == -1 &&
                    overflow_failed_stage == -1,
                "overflow-sized scalar path touched its output buffer");

  std::vector<double> failure_lambdas =
      lambda_path(fixture, Family::kBinomial, true);
  failure_lambdas.push_back(0.15 * failure_lambdas.front());
  failure_lambdas.push_back(0.10 * failure_lambdas.front());
  Outputs failed(4);
  const int failed_status = call_v3(
      fixture, Family::kBinomial, &failure_lambdas, 2, &failed, true, 1,
      1e-7, 3, -1);
  bool uncommitted_models_are_zero = true;
  for (int lambda_index = 1; lambda_index < 4; ++lambda_index) {
    for (int feature = 0; feature < kD; ++feature) {
      if (failed.beta[static_cast<std::size_t>(lambda_index) * kD +
                      feature] != 0.0)
        uncommitted_models_are_zero = false;
    }
    if (failed.intercept[lambda_index] != 0.0 ||
        failed.iterations[lambda_index] != 0 ||
        failed.active_size[lambda_index] != 0 ||
        failed.runtime[lambda_index] != 0.0)
      uncommitted_models_are_zero = false;
  }
  ok &= require(failed_status == PICASSO_LLA_SUBPROBLEM_FAILED &&
                    failed.number_fit == 1 && failed.failed_lambda == 1 &&
                    failed.failed_stage == 0 && failed.stages[0] == 3 &&
                    failed.stages[1] == 0 &&
                    uncommitted_models_are_zero &&
                    std::isnan(failed.objective[2]) &&
                    std::isnan(failed.objective[3]) &&
                    std::isnan(failed.kkt[2]) &&
                    std::isnan(failed.kkt[3]) &&
                    std::isnan(failed.stationarity[2]) &&
                    std::isnan(failed.stationarity[3]) &&
                    std::isnan(failed.smooth_objective[2]) &&
                    std::isnan(failed.smooth_objective[3]),
                "hard failure exposed an uncommitted scalar model");
  return ok;
}

bool test_status_strings() {
  bool ok = true;
  ok &= require(std::string(PicassoLlaPathStatusString(
                    PICASSO_LLA_COMPLETED)) == "completed",
                "completed status string mismatch");
  ok &= require(std::string(PicassoLlaPathStatusString(
                    PICASSO_LLA_STATIONARITY_LIMIT)) ==
                    "lla_stationarity_limit",
                "stationarity-limit status string mismatch");
  ok &= require(std::string(PicassoLlaPathStatusString(
                    PICASSO_LLA_INTERRUPTED)) == "interrupted",
                "interrupted status string mismatch");
  ok &= require(std::string(PicassoLlaPathStatusString(91)) == "unknown",
                "unknown status string mismatch");
  return ok;
}

int g_interrupt_polls = 0;
int g_interrupt_after = 0;

int interrupt_after_polls() {
  ++g_interrupt_polls;
  return g_interrupt_polls > g_interrupt_after ? 1 : 0;
}

bool test_cooperative_interrupt() {
  bool ok = true;
  const Fixture fixture = make_fixture();
  std::vector<double> lambda = lambda_path(fixture, Family::kBinomial);

  // Interrupt after two lambda-boundary polls: the committed two-lambda
  // prefix must remain usable and the failing suffix untouched.
  g_interrupt_polls = 0;
  g_interrupt_after = 2;
  PicassoSetInterruptCallback(interrupt_after_polls);
  Outputs interrupted(kNlambda);
  const int status =
      call_v3(fixture, Family::kBinomial, &lambda, 1, &interrupted);
  PicassoSetInterruptCallback(nullptr);
  ok &= require(status == PICASSO_LLA_INTERRUPTED,
                "interrupt did not report PICASSO_LLA_INTERRUPTED");
  ok &= require(interrupted.number_fit == 2,
                "interrupt did not keep the committed two-lambda prefix");
  // These entry points zero-initialize outputs before solving, so the
  // uncommitted suffix must still hold that initialization, not a model.
  ok &= require(interrupted.intercept[2] == 0.0 &&
                    interrupted.intercept[3] == 0.0,
                "interrupt committed a model past the reported prefix");

  // A cleared callback must restore uninterrupted full-path behavior.
  Outputs completed(kNlambda);
  const int full_status =
      call_v3(fixture, Family::kBinomial, &lambda, 1, &completed);
  ok &= require(full_status == PICASSO_LLA_COMPLETED &&
                    completed.number_fit == kNlambda,
                "cleared interrupt callback still truncated the path");
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_status_strings();
  ok &= test_zero_curvature_path();
  ok &= test_success_and_legacy_compatibility();
  ok &= test_native_smooth_objective_paths();
  ok &= test_intercept_dfmax_and_optional_outputs();
  ok &= test_input_validation_and_transactionality();
  ok &= test_cooperative_interrupt();
  if (!ok) return 1;
  std::cout << "Scalar adaptive-LLA C API tests passed\n";
  return 0;
}
