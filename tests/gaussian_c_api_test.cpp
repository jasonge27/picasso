#include <picasso/actgd.hpp>
#include <picasso/c_api.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace allocation_failure {
bool fail_next = false;
bool count_allocations = false;
std::size_t allocation_count = 0;
long long allocations_before_failure = -1;
bool injected_failure = false;

void begin_counting() {
  allocation_count = 0;
  count_allocations = true;
}

std::size_t end_counting() {
  count_allocations = false;
  return allocation_count;
}

void fail_after(std::size_t successful_allocations) {
  allocations_before_failure =
      static_cast<long long>(successful_allocations);
  injected_failure = false;
}

bool disarm() {
  allocations_before_failure = -1;
  const bool injected = injected_failure;
  injected_failure = false;
  return injected;
}
}

// This test is linked directly with the scalar implementation so allocation
// failure can be injected portably, including on platforms whose shared-library
// symbol binding does not interpose operator new from the test executable.
void *operator new(std::size_t size) {
  if (allocation_failure::count_allocations)
    ++allocation_failure::allocation_count;
  if (allocation_failure::fail_next) {
    allocation_failure::fail_next = false;
    throw std::bad_alloc();
  }
  if (allocation_failure::allocations_before_failure >= 0) {
    if (allocation_failure::allocations_before_failure == 0) {
      allocation_failure::allocations_before_failure = -1;
      allocation_failure::injected_failure = true;
      throw std::bad_alloc();
    }
    --allocation_failure::allocations_before_failure;
  }
  void *memory = std::malloc(size == 0 ? 1 : size);
  if (memory == 0) throw std::bad_alloc();
  return memory;
}

void *operator new[](std::size_t size) {
  return ::operator new(size);
}

void operator delete(void *memory) noexcept {
  std::free(memory);
}

void operator delete[](void *memory) noexcept {
  std::free(memory);
}

namespace {

const int kN = 12;
const int kD = 3;
const int kNlambda = 2;
const double kDoubleSentinel = 91.125;
const int kIntegerSentinel = 9125;

enum class Backend { kNaive, kCovariance };
enum class ApiVersion { kV1, kV2 };

bool require(bool condition, const std::string &message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

struct Fixture {
  std::vector<double> response;
  std::vector<double> design;
  std::vector<double> lambda;

  Fixture() : response(kN), design(kN * kD), lambda() {
    for (int row = 0; row < kN; ++row) {
      const double x0 = static_cast<double>(row - 5) / 5.0;
      const double x1 = std::sin(0.43 * static_cast<double>(row + 1));
      const double x2 = static_cast<double>((row * 7) % 11 - 5) / 5.0;
      design[static_cast<std::size_t>(row) * kD] = x0;
      design[static_cast<std::size_t>(row) * kD + 1] = x1;
      design[static_cast<std::size_t>(row) * kD + 2] = x2;
      response[row] = 0.35 + 1.2 * x0 - 0.7 * x1 + 0.15 * x2;
    }
    lambda.push_back(0.35);
    lambda.push_back(0.08);
  }
};

struct Settings {
  double gamma;
  int max_iterations;
  double precision;
  int reg_type;
  int dfmax;

  Settings()
      : gamma(3.5),
        max_iterations(1000),
        precision(1e-8),
        reg_type(1),
        dfmax(-1) {}
};

struct Outputs {
  std::vector<double> beta;
  std::vector<double> intercept;
  std::vector<int> iterations;
  std::vector<int> active_size;
  std::vector<double> runtime;
  std::vector<double> smooth_objective;
  int num_fit;

  Outputs(int dimension = kD, int path_length = kNlambda)
      : beta(dimension * path_length, kDoubleSentinel),
        intercept(path_length, kDoubleSentinel),
        iterations(path_length, kIntegerSentinel),
        active_size(path_length, kIntegerSentinel),
        runtime(path_length, kDoubleSentinel),
        smooth_objective(path_length, kDoubleSentinel),
        num_fit(kIntegerSentinel) {}
};

void call_gaussian_raw(
    Backend backend, ApiVersion version, double *response, double *design,
    int sample_count, int dimension, double *lambda, int path_length,
    const Settings &settings, bool include_intercept, double *beta,
    double *intercept, int *iterations, int *active_size, double *runtime,
    int *num_fit, double *smooth_objective, bool use_python = true) {
  if (version == ApiVersion::kV2) {
    if (backend == Backend::kNaive) {
      SolveLinearRegressionNaiveUpdateV2(
          response, design, sample_count, dimension, lambda, path_length,
          settings.gamma, settings.max_iterations, settings.precision,
          settings.reg_type, include_intercept, settings.dfmax, beta,
          intercept, iterations, active_size, runtime, num_fit, use_python,
          smooth_objective);
    } else {
      SolveLinearRegressionCovUpdateV2(
          response, design, sample_count, dimension, lambda, path_length,
          settings.gamma, settings.max_iterations, settings.precision,
          settings.reg_type, include_intercept, settings.dfmax, beta,
          intercept, iterations, active_size, runtime, num_fit, use_python,
          smooth_objective);
    }
    return;
  }

  if (backend == Backend::kNaive) {
    SolveLinearRegressionNaiveUpdate(
        response, design, sample_count, dimension, lambda, path_length,
        settings.gamma, settings.max_iterations, settings.precision,
        settings.reg_type, include_intercept, settings.dfmax, beta, intercept,
        iterations, active_size, runtime, num_fit, use_python);
  } else {
    SolveLinearRegressionCovUpdate(
        response, design, sample_count, dimension, lambda, path_length,
        settings.gamma, settings.max_iterations, settings.precision,
        settings.reg_type, include_intercept, settings.dfmax, beta, intercept,
        iterations, active_size, runtime, num_fit, use_python);
  }
}

void call_gaussian(Backend backend, ApiVersion version, Fixture *fixture,
                   const Settings &settings, Outputs *output,
                   double *response = 0, double *design = 0,
                   double *lambda = 0, bool include_intercept = true) {
  double *response_pointer = response == 0 ? fixture->response.data() : response;
  double *design_pointer = design == 0 ? fixture->design.data() : design;
  double *lambda_pointer = lambda == 0 ? fixture->lambda.data() : lambda;
  call_gaussian_raw(
      backend, version, response_pointer, design_pointer, kN, kD,
      lambda_pointer, kNlambda, settings, include_intercept,
      output->beta.data(), output->intercept.data(), output->iterations.data(),
      output->active_size.data(), output->runtime.data(), &output->num_fit,
      output->smooth_objective.data());
}

picasso::solver::PicassoSolverParams make_test_params(
    double *lambda, int path_length, const Settings &settings,
    bool include_intercept) {
  picasso::solver::PicassoSolverParams params;
  params.set_lambdas(lambda, path_length);
  params.gamma = settings.gamma;
  if (settings.reg_type == 1)
    params.reg_type = picasso::solver::L1;
  else if (settings.reg_type == 2)
    params.reg_type = picasso::solver::MCP;
  else
    params.reg_type = picasso::solver::SCAD;
  params.include_intercept = include_intercept;
  params.prec = settings.precision;
  params.max_iter = settings.max_iterations;
  params.dfmax = settings.dfmax;
  return params;
}

template <typename ObjectiveType>
int solve_gaussian_retained(
    ApiVersion version, double *response, double *design, int sample_count,
    int dimension, double *lambda, int path_length, const Settings &settings,
    bool include_intercept, Outputs *output) {
  ObjectiveType objective(design, response, sample_count, dimension,
                          include_intercept, true);
  const picasso::solver::PicassoSolverParams params = make_test_params(
      lambda, path_length, settings, include_intercept);
  picasso::solver::ActGDSolver solver(&objective, params);
  solver.solve();

  const int actual_fit = solver.get_num_lambdas_fit();
  const std::vector<int> &iteration_path = solver.get_itercnt_path();
  const std::vector<double> &runtime_path = solver.get_runtime_path();
  const std::vector<double> &smooth_path =
      solver.get_smooth_objective_path();
  for (int path_index = 0; path_index < actual_fit; ++path_index) {
    const picasso::ModelParam &model = solver.get_model_param(path_index);
    int nonzero_count = 0;
    for (int feature = 0; feature < dimension; ++feature) {
      const double coefficient = model.beta[feature];
      output->beta[static_cast<std::size_t>(path_index) * dimension +
                   feature] = coefficient;
      if (std::fabs(coefficient) > 1e-8) ++nonzero_count;
    }
    output->intercept[path_index] = model.intercept;
    output->iterations[path_index] = iteration_path[path_index];
    output->active_size[path_index] = nonzero_count;
    output->runtime[path_index] = runtime_path[path_index];
    if (version == ApiVersion::kV2)
      output->smooth_objective[path_index] = smooth_path[path_index];
  }
  return actual_fit;
}

void reset_failed_outputs(Outputs *output, ApiVersion version) {
  std::fill(output->beta.begin(), output->beta.end(), 0.0);
  std::fill(output->intercept.begin(), output->intercept.end(), 0.0);
  std::fill(output->iterations.begin(), output->iterations.end(), 0);
  std::fill(output->active_size.begin(), output->active_size.end(), 0);
  std::fill(output->runtime.begin(), output->runtime.end(), 0.0);
  if (version == ApiVersion::kV2) {
    const double missing = std::numeric_limits<double>::quiet_NaN();
    std::fill(output->smooth_objective.begin(),
              output->smooth_objective.end(), missing);
  }
  output->num_fit = 0;
}

void call_gaussian_retained_raw(
    Backend backend, ApiVersion version, double *response, double *design,
    int sample_count, int dimension, double *lambda, int path_length,
    const Settings &settings, bool include_intercept, Outputs *output) {
  output->num_fit = 0;
  if (version == ApiVersion::kV2) {
    const double missing = std::numeric_limits<double>::quiet_NaN();
    std::fill(output->smooth_objective.begin(),
              output->smooth_objective.end(), missing);
  }
  try {
    if (backend == Backend::kNaive) {
      output->num_fit = solve_gaussian_retained<
          picasso::GaussianNaiveUpdateObjective>(
          version, response, design, sample_count, dimension, lambda,
          path_length, settings, include_intercept, output);
    } else {
      output->num_fit = solve_gaussian_retained<
          picasso::GaussianCovUpdateObjective>(
          version, response, design, sample_count, dimension, lambda,
          path_length, settings, include_intercept, output);
    }
  } catch (...) {
    reset_failed_outputs(output, version);
  }
}

void call_gaussian_retained(Backend backend, ApiVersion version,
                            Fixture *fixture, const Settings &settings,
                            Outputs *output,
                            bool include_intercept = true) {
  call_gaussian_retained_raw(
      backend, version, fixture->response.data(), fixture->design.data(), kN,
      kD, fixture->lambda.data(), kNlambda, settings, include_intercept,
      output);
}

bool all_zero(const std::vector<double> &values) {
  for (std::size_t index = 0; index < values.size(); ++index)
    if (values[index] != 0.0) return false;
  return true;
}

bool all_zero(const std::vector<int> &values) {
  for (std::size_t index = 0; index < values.size(); ++index)
    if (values[index] != 0) return false;
  return true;
}

bool transactional_failure(const Outputs &output, ApiVersion version) {
  bool ok = output.num_fit == 0 && all_zero(output.beta) &&
            all_zero(output.intercept) && all_zero(output.iterations) &&
            all_zero(output.active_size) && all_zero(output.runtime);
  if (version == ApiVersion::kV2) {
    for (std::size_t index = 0; index < output.smooth_objective.size(); ++index)
      ok = ok && std::isnan(output.smooth_objective[index]);
  }
  return ok;
}

std::string api_name(Backend backend, ApiVersion version) {
  return std::string(backend == Backend::kNaive ? "naive" : "covariance") +
         (version == ApiVersion::kV1 ? " V1" : " V2");
}

bool test_strict_scalar_validation() {
  const Backend backends[] = {Backend::kNaive, Backend::kCovariance};
  const ApiVersion versions[] = {ApiVersion::kV1, ApiVersion::kV2};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  bool ok = true;

  for (int backend_index = 0; backend_index < 2; ++backend_index) {
    for (int version_index = 0; version_index < 2; ++version_index) {
      const Backend backend = backends[backend_index];
      const ApiVersion version = versions[version_index];
      const std::string name = api_name(backend, version);
      std::vector<Settings> invalid;

      Settings zero_iterations;
      zero_iterations.max_iterations = 0;
      invalid.push_back(zero_iterations);
      Settings negative_iterations;
      negative_iterations.max_iterations = -1;
      invalid.push_back(negative_iterations);
      Settings zero_precision;
      zero_precision.precision = 0.0;
      invalid.push_back(zero_precision);
      Settings negative_precision;
      negative_precision.precision = -1e-8;
      invalid.push_back(negative_precision);
      Settings nan_precision;
      nan_precision.precision = nan;
      invalid.push_back(nan_precision);
      Settings infinite_precision;
      infinite_precision.precision = infinity;
      invalid.push_back(infinite_precision);
      Settings unknown_penalty;
      unknown_penalty.reg_type = 4;
      invalid.push_back(unknown_penalty);
      Settings zero_penalty;
      zero_penalty.reg_type = 0;
      invalid.push_back(zero_penalty);
      Settings invalid_dfmax;
      invalid_dfmax.dfmax = -2;
      invalid.push_back(invalid_dfmax);
      Settings mcp_boundary;
      mcp_boundary.reg_type = 2;
      mcp_boundary.gamma = 1.0;
      invalid.push_back(mcp_boundary);
      Settings scad_boundary;
      scad_boundary.reg_type = 3;
      scad_boundary.gamma = 2.0;
      invalid.push_back(scad_boundary);
      Settings nan_gamma;
      nan_gamma.reg_type = 2;
      nan_gamma.gamma = nan;
      invalid.push_back(nan_gamma);
      Settings infinite_gamma;
      infinite_gamma.reg_type = 3;
      infinite_gamma.gamma = infinity;
      invalid.push_back(infinite_gamma);

      for (std::size_t case_index = 0; case_index < invalid.size();
           ++case_index) {
        Fixture fixture;
        Outputs output;
        bool threw = false;
        try {
          call_gaussian(backend, version, &fixture, invalid[case_index],
                        &output);
        } catch (...) {
          threw = true;
        }
        ok &= require(!threw, name + " invalid input crossed the C ABI");
        ok &= require(transactional_failure(output, version),
                      name + " invalid input was not transactional");
      }
    }
  }
  return ok;
}

bool test_nonfinite_gaussian_data_is_transactional() {
  const Backend backends[] = {Backend::kNaive, Backend::kCovariance};
  const ApiVersion versions[] = {ApiVersion::kV1, ApiVersion::kV2};
  const double values[] = {
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity()};
  bool ok = true;

  for (int backend_index = 0; backend_index < 2; ++backend_index) {
    for (int version_index = 0; version_index < 2; ++version_index) {
      for (int python_layout = 0; python_layout < 2; ++python_layout) {
        for (int intercept_case = 0; intercept_case < 2; ++intercept_case) {
          for (int response_case = 0; response_case < 2; ++response_case) {
            for (int value_index = 0; value_index < 3; ++value_index) {
              Fixture fixture;
              if (response_case)
                fixture.response[4] = values[value_index];
              else
                fixture.design[5] = values[value_index];
              Outputs output;
              bool threw = false;
              try {
                call_gaussian_raw(
                    backends[backend_index], versions[version_index],
                    fixture.response.data(), fixture.design.data(), kN, kD,
                    fixture.lambda.data(), kNlambda, Settings(),
                    intercept_case != 0, output.beta.data(),
                    output.intercept.data(), output.iterations.data(),
                    output.active_size.data(), output.runtime.data(),
                    &output.num_fit, output.smooth_objective.data(),
                    python_layout != 0);
              } catch (...) {
                threw = true;
              }
              const std::string name =
                  api_name(backends[backend_index], versions[version_index]) +
                  (python_layout ? " row-major" : " column-major") +
                  (intercept_case ? " intercept" : " no-intercept") +
                  (response_case ? " response" : " design");
              ok &= require(
                  !threw, name + " nonfinite data crossed the C ABI");
              ok &= require(
                  transactional_failure(output, versions[version_index]),
                  name + " nonfinite data was not rejected transactionally");
            }
          }
        }
      }
    }
  }
  return ok;
}

bool test_allocation_exceptions_are_isolated() {
  const Backend backends[] = {Backend::kNaive, Backend::kCovariance};
  const ApiVersion versions[] = {ApiVersion::kV1, ApiVersion::kV2};
  bool ok = true;
  for (int backend_index = 0; backend_index < 2; ++backend_index) {
    for (int version_index = 0; version_index < 2; ++version_index) {
      Fixture fixture;
      Settings settings;
      Outputs output;
      bool threw = false;
      allocation_failure::fail_next = true;
      try {
        call_gaussian(backends[backend_index], versions[version_index],
                      &fixture, settings, &output);
      } catch (...) {
        threw = true;
        allocation_failure::fail_next = false;
      }
      const std::string name =
          api_name(backends[backend_index], versions[version_index]);
      ok &= require(!allocation_failure::fail_next,
                    name + " did not reach the injected allocation");
      ok &= require(!threw, name + " leaked an exception across the C ABI");
      ok &= require(transactional_failure(output, versions[version_index]),
                    name + " exception output was not transactional");
    }
  }
  return ok;
}

bool test_each_native_allocation_failure_is_transactional() {
  const Backend backends[] = {Backend::kNaive, Backend::kCovariance};
  const ApiVersion versions[] = {ApiVersion::kV1, ApiVersion::kV2};
  bool ok = true;
  for (int backend_index = 0; backend_index < 2; ++backend_index) {
    for (int version_index = 0; version_index < 2; ++version_index) {
      std::size_t allocation_counts[2] = {0, 0};
      for (int retained_index = 0; retained_index < 2; ++retained_index) {
        const bool retained = retained_index == 0;
        Fixture fixture;
        Settings settings;

        // Warm each path before counting so one-time runtime initialization
        // cannot make the counted sequence differ from injected runs.
        Outputs warmup;
        if (retained)
          call_gaussian_retained(backends[backend_index],
                                 versions[version_index], &fixture, settings,
                                 &warmup);
        else
          call_gaussian(backends[backend_index], versions[version_index],
                        &fixture, settings, &warmup);

        Outputs baseline;
        allocation_failure::begin_counting();
        if (retained)
          call_gaussian_retained(backends[backend_index],
                                 versions[version_index], &fixture, settings,
                                 &baseline);
        else
          call_gaussian(backends[backend_index], versions[version_index],
                        &fixture, settings, &baseline);
        const std::size_t allocation_count =
            allocation_failure::end_counting();
        allocation_counts[retained_index] = allocation_count;
        const std::string name =
            api_name(backends[backend_index], versions[version_index]) +
            (retained ? " retained" : " sink");
        ok &= require(baseline.num_fit == kNlambda && allocation_count > 1,
                      name + " allocation sweep baseline did not complete");

        // Sweep every successful-call allocation. The retained case includes
        // ModelParam copies after regularizer construction; ASan/LSan checks
        // their exceptional cleanup. The sink case verifies the C ABI reset.
        for (std::size_t failure_index = 0;
             failure_index < allocation_count; ++failure_index) {
          Outputs output;
          bool threw = false;
          allocation_failure::fail_after(failure_index);
          try {
            if (retained)
              call_gaussian_retained(
                  backends[backend_index], versions[version_index], &fixture,
                  settings, &output);
            else
              call_gaussian(backends[backend_index], versions[version_index],
                            &fixture, settings, &output);
          } catch (...) {
            threw = true;
          }
          const bool injected = allocation_failure::disarm();
          ok &= require(injected,
                        name + " did not reach allocation index " +
                            std::to_string(failure_index));
          ok &= require(!threw, name + " leaked an injected exception");
          ok &= require(
              transactional_failure(output, versions[version_index]),
              name + " allocation failure was not transactional");
        }
      }
      ok &= require(
          allocation_counts[1] < allocation_counts[0],
          api_name(backends[backend_index], versions[version_index]) +
              " sink did not eliminate retained-path allocations");
    }
  }
  return ok;
}

bool test_v2_initializes_smooth_path_for_invalid_dimensions() {
  Fixture fixture;
  Settings settings;
  bool ok = true;
  const Backend backends[] = {Backend::kNaive, Backend::kCovariance};
  for (int backend_index = 0; backend_index < 2; ++backend_index) {
    Outputs output;
    bool threw = false;
    try {
      if (backends[backend_index] == Backend::kNaive) {
        SolveLinearRegressionNaiveUpdateV2(
            fixture.response.data(), fixture.design.data(), kN, 0,
            fixture.lambda.data(), kNlambda, settings.gamma,
            settings.max_iterations, settings.precision, settings.reg_type,
            true, settings.dfmax, output.beta.data(), output.intercept.data(),
            output.iterations.data(), output.active_size.data(),
            output.runtime.data(), &output.num_fit, true,
            output.smooth_objective.data());
      } else {
        SolveLinearRegressionCovUpdateV2(
            fixture.response.data(), fixture.design.data(), kN, 0,
            fixture.lambda.data(), kNlambda, settings.gamma,
            settings.max_iterations, settings.precision, settings.reg_type,
            true, settings.dfmax, output.beta.data(), output.intercept.data(),
            output.iterations.data(), output.active_size.data(),
            output.runtime.data(), &output.num_fit, true,
            output.smooth_objective.data());
      }
    } catch (...) {
      threw = true;
    }
    const std::string name =
        api_name(backends[backend_index], ApiVersion::kV2);
    ok &= require(!threw, name + " invalid dimension crossed the C ABI");
    ok &= require(output.num_fit == 0,
                  name + " invalid dimension did not clear num_fit");
    for (int index = 0; index < kNlambda; ++index)
      ok &= require(std::isnan(output.smooth_objective[index]),
                    name + " did not initialize its smooth path");
  }
  return ok;
}

bool test_l1_ignores_gamma_consistently() {
  const Backend backends[] = {Backend::kNaive, Backend::kCovariance};
  const ApiVersion versions[] = {ApiVersion::kV1, ApiVersion::kV2};
  bool ok = true;
  for (int backend_index = 0; backend_index < 2; ++backend_index) {
    for (int version_index = 0; version_index < 2; ++version_index) {
      Fixture fixture;
      Settings settings;
      settings.gamma = std::numeric_limits<double>::quiet_NaN();
      Outputs output;
      call_gaussian(backends[backend_index], versions[version_index],
                    &fixture, settings, &output);
      ok &= require(output.num_fit == kNlambda,
                    api_name(backends[backend_index], versions[version_index]) +
                        " treated unused L1 gamma as an error");
    }
  }
  return ok;
}

bool nearly_equal(double left, double right) {
  const double scale = 1.0 + std::fabs(left) + std::fabs(right);
  return std::fabs(left - right) <= 1e-13 * scale;
}

bool equal_or_both_nan(double left, double right) {
  return (std::isnan(left) && std::isnan(right)) ||
         nearly_equal(left, right);
}

bool outputs_match(const Outputs &retained, const Outputs &sink,
                   const std::string &name) {
  bool ok = true;
  ok &= require(retained.num_fit == sink.num_fit,
                name + " fitted path lengths differ");
  for (std::size_t index = 0; index < retained.beta.size(); ++index)
    ok &= require(equal_or_both_nan(retained.beta[index], sink.beta[index]),
                  name + " coefficient differs at index " +
                      std::to_string(index));
  for (std::size_t index = 0; index < retained.intercept.size(); ++index) {
    ok &= require(equal_or_both_nan(retained.intercept[index],
                                    sink.intercept[index]),
                  name + " intercept differs at index " +
                      std::to_string(index));
    ok &= require(retained.iterations[index] == sink.iterations[index],
                  name + " iteration count differs at index " +
                      std::to_string(index));
    ok &= require(retained.active_size[index] == sink.active_size[index],
                  name + " active size differs at index " +
                      std::to_string(index));
    ok &= require(equal_or_both_nan(retained.runtime[index],
                                    sink.runtime[index]),
                  name + " runtime diagnostic differs at index " +
                      std::to_string(index));
    ok &= require(equal_or_both_nan(retained.smooth_objective[index],
                                    sink.smooth_objective[index]),
                  name + " smooth objective differs at index " +
                      std::to_string(index));
  }
  return ok;
}

std::string configuration_name(Backend backend, ApiVersion version,
                               int reg_type, bool include_intercept) {
  const char *penalty = reg_type == 1 ? "L1" : (reg_type == 2 ? "MCP" : "SCAD");
  return api_name(backend, version) + " " + penalty +
         (include_intercept ? " intercept" : " no-intercept");
}

bool test_retained_and_sink_paths_match() {
  const Backend backends[] = {Backend::kNaive, Backend::kCovariance};
  const ApiVersion versions[] = {ApiVersion::kV1, ApiVersion::kV2};
  bool ok = true;
  for (int backend_index = 0; backend_index < 2; ++backend_index) {
    for (int version_index = 0; version_index < 2; ++version_index) {
      for (int reg_type = 1; reg_type <= 3; ++reg_type) {
        for (int intercept_index = 0; intercept_index < 2;
             ++intercept_index) {
          Fixture fixture;
          Settings settings;
          settings.reg_type = reg_type;
          const bool include_intercept = intercept_index != 0;
          Outputs retained;
          Outputs sink;
          call_gaussian_retained(backends[backend_index],
                                 versions[version_index], &fixture, settings,
                                 &retained, include_intercept);
          call_gaussian(backends[backend_index], versions[version_index],
                        &fixture, settings, &sink, 0, 0, 0,
                        include_intercept);
          ok &= outputs_match(
              retained, sink,
              configuration_name(backends[backend_index],
                                 versions[version_index], reg_type,
                                 include_intercept));
        }
      }
    }
  }
  return ok;
}

template <typename ObjectiveType>
bool sink_does_not_retain_for_objective(Fixture *fixture,
                                        const Settings &settings,
                                        const std::string &name) {
  ObjectiveType objective(fixture->design.data(), fixture->response.data(),
                          kN, kD, true, true);
  const picasso::solver::PicassoSolverParams params = make_test_params(
      fixture->lambda.data(), kNlambda, settings, true);
  picasso::solver::ActGDSolver solver(&objective, params);
  Outputs output;
  const int committed = solver.solve_to_buffers(
      output.beta.data(), output.intercept.data(), output.iterations.data(),
      output.active_size.data(), output.runtime.data(),
      output.smooth_objective.data());
  bool ok = true;
  ok &= require(committed == kNlambda,
                name + " sink committed an unexpected path length");
  ok &= require(solver.get_num_lambdas_fit() == 0,
                name + " sink retained coefficient-path models");
  ok &= require(solver.get_smooth_objective_path().size() == kNlambda,
                name + " sink lost smooth-objective diagnostics");
  return ok;
}

bool test_sink_does_not_retain_solution_path() {
  Fixture fixture;
  Settings settings;
  bool ok = sink_does_not_retain_for_objective<
      picasso::GaussianNaiveUpdateObjective>(&fixture, settings, "naive");
  ok &= sink_does_not_retain_for_objective<
      picasso::GaussianCovUpdateObjective>(&fixture, settings, "covariance");
  return ok;
}

bool test_dfmax_crossing_fit_matches_retained_path() {
  const int path_length = 7;
  std::vector<double> lambda;
  lambda.push_back(1.0);
  lambda.push_back(0.5);
  lambda.push_back(0.2);
  lambda.push_back(0.05);
  lambda.push_back(0.001);
  lambda.push_back(0.0005);
  lambda.push_back(0.0001);
  const Backend backends[] = {Backend::kNaive, Backend::kCovariance};
  const ApiVersion versions[] = {ApiVersion::kV1, ApiVersion::kV2};
  bool ok = true;
  for (int backend_index = 0; backend_index < 2; ++backend_index) {
    for (int version_index = 0; version_index < 2; ++version_index) {
      Fixture fixture;
      Settings settings;
      settings.dfmax = 0;
      Outputs retained(kD, path_length);
      Outputs sink(kD, path_length);
      call_gaussian_retained_raw(
          backends[backend_index], versions[version_index],
          fixture.response.data(), fixture.design.data(), kN, kD,
          lambda.data(), path_length, settings, true, &retained);
      call_gaussian_raw(
          backends[backend_index], versions[version_index],
          fixture.response.data(), fixture.design.data(), kN, kD,
          lambda.data(), path_length, settings, true, sink.beta.data(),
          sink.intercept.data(), sink.iterations.data(),
          sink.active_size.data(), sink.runtime.data(), &sink.num_fit,
          sink.smooth_objective.data());
      const std::string name =
          api_name(backends[backend_index], versions[version_index]) +
          " dfmax crossing";
      ok &= require(retained.num_fit == 5,
                    name + " did not retain the fifth crossing fit");
      ok &= outputs_match(retained, sink, name);
    }
  }
  return ok;
}

bool test_null_optional_outputs() {
  const Backend backends[] = {Backend::kNaive, Backend::kCovariance};
  const ApiVersion versions[] = {ApiVersion::kV1, ApiVersion::kV2};
  bool ok = true;
  for (int backend_index = 0; backend_index < 2; ++backend_index) {
    for (int version_index = 0; version_index < 2; ++version_index) {
      for (int reg_type = 1; reg_type <= 3; ++reg_type) {
        for (int intercept_index = 0; intercept_index < 2;
             ++intercept_index) {
          Fixture fixture;
          Settings settings;
          settings.reg_type = reg_type;
          int num_fit = kIntegerSentinel;
          call_gaussian_raw(
              backends[backend_index], versions[version_index],
              fixture.response.data(), fixture.design.data(), kN, kD,
              fixture.lambda.data(), kNlambda, settings,
              intercept_index != 0, nullptr, nullptr, nullptr, nullptr,
              nullptr, &num_fit, nullptr);
          const std::string name = configuration_name(
              backends[backend_index], versions[version_index], reg_type,
              intercept_index != 0);
          ok &= require(num_fit == kNlambda,
                        name + " null outputs changed fitted path length");
          call_gaussian_raw(
              backends[backend_index], versions[version_index],
              fixture.response.data(), fixture.design.data(), kN, kD,
              fixture.lambda.data(), kNlambda, settings,
              intercept_index != 0, nullptr, nullptr, nullptr, nullptr,
              nullptr, nullptr, nullptr);
        }
      }
    }
  }
  return ok;
}

bool test_v1_matches_v2() {
  const Backend backends[] = {Backend::kNaive, Backend::kCovariance};
  bool ok = true;
  for (int backend_index = 0; backend_index < 2; ++backend_index) {
    for (int reg_type = 1; reg_type <= 3; ++reg_type) {
      Fixture fixture;
      Settings settings;
      settings.reg_type = reg_type;
      Outputs legacy;
      Outputs versioned;
      call_gaussian(backends[backend_index], ApiVersion::kV1, &fixture,
                    settings, &legacy);
      call_gaussian(backends[backend_index], ApiVersion::kV2, &fixture,
                    settings, &versioned);
      const std::string name =
          api_name(backends[backend_index], ApiVersion::kV1);
      ok &= require(legacy.num_fit == kNlambda &&
                        versioned.num_fit == legacy.num_fit,
                    name + " and V2 fitted different path lengths");
      for (std::size_t index = 0; index < legacy.beta.size(); ++index)
        ok &= require(nearly_equal(legacy.beta[index], versioned.beta[index]),
                      name + " coefficient differs from V2");
      for (int index = 0; index < kNlambda; ++index) {
        ok &= require(nearly_equal(legacy.intercept[index],
                                   versioned.intercept[index]) &&
                          legacy.iterations[index] ==
                              versioned.iterations[index] &&
                          legacy.active_size[index] ==
                              versioned.active_size[index] &&
                          nearly_equal(legacy.runtime[index],
                                       versioned.runtime[index]),
                      name + " path diagnostics differ from V2");
        ok &= require(std::isfinite(versioned.smooth_objective[index]) &&
                          versioned.smooth_objective[index] >= 0.0,
                      name + " V2 smooth objective is invalid");
      }
    }
  }
  return ok;
}

struct ActNewtonFixture {
  static const int kPathLength = 4;

  std::vector<double> response;
  std::vector<double> offset;
  std::vector<double> design;
  std::vector<double> lambda;

  ActNewtonFixture()
      : response(kN), offset(kN), design(kN * kD), lambda() {
    for (int row = 0; row < kN; ++row) {
      const double x0 = static_cast<double>(row - 5) / 5.0;
      const double x1 = std::sin(0.43 * static_cast<double>(row + 1));
      const double x2 = static_cast<double>((row * 7) % 11 - 5) / 5.0;
      design[static_cast<std::size_t>(row) * kD] = x0;
      design[static_cast<std::size_t>(row) * kD + 1] = x1;
      design[static_cast<std::size_t>(row) * kD + 2] = x2;
      offset[row] = 0.08 * std::sin(0.31 * static_cast<double>(row + 1));
      const double signal = -0.15 + offset[row] + 1.1 * x0 - 0.8 * x1;
      response[row] = signal > 0.0 ? 1.0 : 0.0;
    }
    lambda.push_back(0.40);
    lambda.push_back(0.22);
    lambda.push_back(0.11);
    lambda.push_back(0.05);
  }
};

struct ActNewtonOutputs {
  std::vector<double> beta;
  std::vector<double> intercept;
  std::vector<int> iterations;
  std::vector<int> active_size;
  std::vector<double> runtime;
  std::vector<int> stages;
  std::vector<double> objective;
  std::vector<double> kkt;
  std::vector<double> stationarity;
  std::vector<double> smooth_objective;
  int num_fit;
  int failed_lambda;
  int failed_stage;
  int status;

  ActNewtonOutputs()
      : beta(ActNewtonFixture::kPathLength * kD, kDoubleSentinel),
        intercept(ActNewtonFixture::kPathLength, kDoubleSentinel),
        iterations(ActNewtonFixture::kPathLength, kIntegerSentinel),
        active_size(ActNewtonFixture::kPathLength, kIntegerSentinel),
        runtime(ActNewtonFixture::kPathLength, kDoubleSentinel),
        stages(ActNewtonFixture::kPathLength, kIntegerSentinel),
        objective(ActNewtonFixture::kPathLength, kDoubleSentinel),
        kkt(ActNewtonFixture::kPathLength, kDoubleSentinel),
        stationarity(ActNewtonFixture::kPathLength, kDoubleSentinel),
        smooth_objective(ActNewtonFixture::kPathLength, kDoubleSentinel),
        num_fit(kIntegerSentinel),
        failed_lambda(kIntegerSentinel),
        failed_stage(kIntegerSentinel),
        status(kIntegerSentinel) {}
};

void call_actnewton(ActNewtonFixture *fixture, ActNewtonOutputs *output) {
  output->status = SolveLogisticRegressionV3(
      fixture->response.data(), fixture->design.data(), kN, kD,
      fixture->lambda.data(), ActNewtonFixture::kPathLength, 3.5, 1000,
      1e-6, 1, true, -1, fixture->offset.data(), output->beta.data(),
      output->intercept.data(), output->iterations.data(),
      output->active_size.data(), output->runtime.data(), &output->num_fit,
      true, 3, &output->failed_lambda, &output->failed_stage,
      output->stages.data(), output->objective.data(), output->kkt.data(),
      output->stationarity.data(), output->smooth_objective.data());
}

bool actnewton_prefix_matches(const ActNewtonOutputs &baseline,
                              const ActNewtonOutputs &candidate) {
  for (int path_index = 0; path_index < candidate.num_fit; ++path_index) {
    for (int feature = 0; feature < kD; ++feature) {
      const std::size_t index =
          static_cast<std::size_t>(path_index) * kD + feature;
      if (baseline.beta[index] != candidate.beta[index]) return false;
    }
    if (baseline.intercept[path_index] != candidate.intercept[path_index] ||
        baseline.iterations[path_index] != candidate.iterations[path_index] ||
        baseline.active_size[path_index] != candidate.active_size[path_index] ||
        baseline.runtime[path_index] != candidate.runtime[path_index] ||
        baseline.stages[path_index] != candidate.stages[path_index] ||
        baseline.objective[path_index] != candidate.objective[path_index] ||
        baseline.kkt[path_index] != candidate.kkt[path_index] ||
        baseline.stationarity[path_index] !=
            candidate.stationarity[path_index] ||
        baseline.smooth_objective[path_index] !=
            candidate.smooth_objective[path_index])
      return false;
  }
  return true;
}

bool actnewton_uncommitted_model_tail_is_zero(
    const ActNewtonOutputs &output) {
  for (int path_index = output.num_fit;
       path_index < ActNewtonFixture::kPathLength; ++path_index) {
    for (int feature = 0; feature < kD; ++feature) {
      if (output.beta[static_cast<std::size_t>(path_index) * kD + feature] !=
          0.0)
        return false;
    }
    if (output.intercept[path_index] != 0.0 ||
        output.iterations[path_index] != 0 ||
        output.active_size[path_index] != 0 ||
        output.runtime[path_index] != 0.0)
      return false;
  }
  return true;
}

bool actnewton_unattempted_diagnostic_tail_is_nan(
    const ActNewtonOutputs &output) {
  const int first_unattempted = output.failed_lambda >= 0
                                    ? output.failed_lambda + 1
                                    : output.num_fit;
  for (int path_index = first_unattempted;
       path_index < ActNewtonFixture::kPathLength; ++path_index) {
    if (output.stages[path_index] != 0 ||
        !std::isnan(output.objective[path_index]) ||
        !std::isnan(output.kkt[path_index]) ||
        !std::isnan(output.stationarity[path_index]) ||
        !std::isnan(output.smooth_objective[path_index]))
      return false;
  }
  return true;
}

bool test_actnewton_allocation_failure_preserves_committed_prefix() {
  bool ok = true;
  ActNewtonFixture fixture;

  ActNewtonOutputs warmup;
  call_actnewton(&fixture, &warmup);
  ok &= require(warmup.status == PICASSO_LLA_COMPLETED &&
                    warmup.num_fit == ActNewtonFixture::kPathLength,
                "ActNewton allocation warmup did not complete");

  ActNewtonOutputs baseline;
  allocation_failure::begin_counting();
  call_actnewton(&fixture, &baseline);
  const std::size_t allocation_count = allocation_failure::end_counting();
  ok &= require(baseline.status == PICASSO_LLA_COMPLETED &&
                    baseline.num_fit == ActNewtonFixture::kPathLength &&
                    allocation_count > 1,
                "ActNewton allocation baseline did not complete");

  int failures_after_commit = 0;
  int failures_after_exactly_one_commit = 0;
  for (std::size_t failure_index = 0; failure_index < allocation_count;
       ++failure_index) {
    ActNewtonOutputs output;
    bool threw = false;
    allocation_failure::fail_after(failure_index);
    try {
      call_actnewton(&fixture, &output);
    } catch (...) {
      threw = true;
    }
    const bool injected = allocation_failure::disarm();
    ok &= require(injected,
                  "ActNewton did not reach allocation index " +
                      std::to_string(failure_index));
    ok &= require(!threw,
                  "ActNewton leaked an injected exception across the C ABI");
    ok &= require(output.status == PICASSO_LLA_EXCEPTION,
                  "ActNewton allocation failure returned the wrong status");

    if (output.num_fit == 0) {
      ok &= require(output.failed_lambda == 0,
                    "ActNewton pre-commit exception identified the wrong "
                    "lambda");
      ok &= require(actnewton_uncommitted_model_tail_is_zero(output),
                    "ActNewton pre-commit exception left model sentinels");
      bool diagnostics_are_unattempted = true;
      for (int path_index = 0;
           path_index < ActNewtonFixture::kPathLength; ++path_index) {
        if (output.stages[path_index] != 0 ||
            !std::isnan(output.objective[path_index]) ||
            !std::isnan(output.kkt[path_index]) ||
            !std::isnan(output.stationarity[path_index]) ||
            !std::isnan(output.smooth_objective[path_index]))
          diagnostics_are_unattempted = false;
      }
      ok &= require(diagnostics_are_unattempted,
                    "ActNewton pre-commit exception populated diagnostics");
    } else if (output.num_fit > 0 &&
        output.num_fit < ActNewtonFixture::kPathLength) {
      ++failures_after_commit;
      if (output.num_fit == 1) ++failures_after_exactly_one_commit;
      ok &= require(output.failed_lambda == output.num_fit,
                    "ActNewton exception did not identify the next lambda");
      ok &= require(actnewton_prefix_matches(baseline, output),
                    "ActNewton exception changed its committed prefix");
      ok &= require(actnewton_uncommitted_model_tail_is_zero(output),
                    "ActNewton exception exposed an uncommitted model");
      ok &= require(actnewton_unattempted_diagnostic_tail_is_nan(output),
                    "ActNewton exception populated unattempted diagnostics");
    } else if (output.num_fit == ActNewtonFixture::kPathLength) {
      ok &= require(output.failed_lambda == -1 &&
                        actnewton_prefix_matches(baseline, output),
                    "ActNewton post-path exception changed a committed fit");
    }
  }
  ok &= require(failures_after_commit > 0 &&
                    failures_after_exactly_one_commit > 0,
                "allocation injection never failed after the first "
                "ActNewton model commit");
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_strict_scalar_validation();
  ok &= test_nonfinite_gaussian_data_is_transactional();
  ok &= test_allocation_exceptions_are_isolated();
  ok &= test_each_native_allocation_failure_is_transactional();
  ok &= test_v2_initializes_smooth_path_for_invalid_dimensions();
  ok &= test_l1_ignores_gamma_consistently();
  ok &= test_retained_and_sink_paths_match();
  ok &= test_sink_does_not_retain_solution_path();
  ok &= test_dfmax_crossing_fit_matches_retained_path();
  ok &= test_null_optional_outputs();
  ok &= test_v1_matches_v2();
  ok &= test_actnewton_allocation_failure_preserves_committed_prefix();
  if (!ok) return 1;
  std::cout << "gaussian_c_api_test: PASS\n";
  return 0;
}
