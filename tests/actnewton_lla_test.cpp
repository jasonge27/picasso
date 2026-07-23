#include <picasso/actnewton.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

enum class Family { kBinomial, kPoisson, kSqrtLasso };

const int kSampleCount = 180;
const int kFeatureCount = 12;
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
  Fixture fixture;
  fixture.x.resize(static_cast<std::size_t>(kSampleCount) * kFeatureCount);
  fixture.binomial.resize(kSampleCount);
  fixture.poisson.resize(kSampleCount);
  fixture.sqrt_lasso.resize(kSampleCount);
  fixture.offset.resize(kSampleCount);

  std::vector<double> mean(kFeatureCount, 0.0);
  std::vector<double> scale(kFeatureCount, 0.0);
  for (int sample = 0; sample < kSampleCount; ++sample) {
    for (int feature = 0; feature < kFeatureCount; ++feature) {
      const double value =
          std::sin(0.07 * (sample + 1.0) * (feature + 1.0)) +
          0.25 * std::cos(0.13 * (sample + feature + 2.0));
      fixture.x[static_cast<std::size_t>(sample) * kFeatureCount + feature] =
          value;
      mean[feature] += value;
    }
  }
  for (int feature = 0; feature < kFeatureCount; ++feature)
    mean[feature] /= kSampleCount;
  for (int sample = 0; sample < kSampleCount; ++sample) {
    for (int feature = 0; feature < kFeatureCount; ++feature) {
      const std::size_t index =
          static_cast<std::size_t>(sample) * kFeatureCount + feature;
      const double centered = fixture.x[index] - mean[feature];
      fixture.x[index] = centered;
      scale[feature] += centered * centered;
    }
  }
  for (int feature = 0; feature < kFeatureCount; ++feature)
    scale[feature] = std::sqrt(scale[feature] / (kSampleCount - 1.0));

  const double beta[] = {1.0, -0.8, 0.65, -0.5};
  for (int sample = 0; sample < kSampleCount; ++sample) {
    double signal = 0.0;
    for (int feature = 0; feature < kFeatureCount; ++feature) {
      const std::size_t index =
          static_cast<std::size_t>(sample) * kFeatureCount + feature;
      fixture.x[index] /= scale[feature];
      if (feature < 4) signal += beta[feature] * fixture.x[index];
    }
    const double draw = std::fmod(
        0.6180339887498949 * static_cast<double>(sample + 1), 1.0);
    const double probability = 1.0 / (1.0 + std::exp(-(-0.2 + signal)));
    fixture.binomial[sample] = draw < probability ? 1.0 : 0.0;
    fixture.poisson[sample] =
        std::floor(std::exp(0.15 + 0.28 * signal) + draw);
    fixture.sqrt_lasso[sample] =
        0.4 + signal + 0.35 * std::sin(0.31 * (sample + 1.0));
    fixture.offset[sample] =
        0.12 * std::sin(0.19 * (sample + 1.0));
  }
  return fixture;
}

const std::vector<double> &response(const Fixture &fixture, Family family) {
  if (family == Family::kBinomial) return fixture.binomial;
  if (family == Family::kPoisson) return fixture.poisson;
  return fixture.sqrt_lasso;
}

std::unique_ptr<picasso::ObjFunction> make_objective(
    const Fixture &fixture, Family family, bool include_intercept = true,
    bool use_offset = false) {
  const std::vector<double> &y = response(fixture, family);
  if (family == Family::kBinomial) {
    std::unique_ptr<picasso::LogisticObjective> objective(
        new picasso::LogisticObjective(
            fixture.x.data(), y.data(), kSampleCount, kFeatureCount,
            include_intercept, true));
    if (use_offset)
      objective->set_offset(fixture.offset.data(), kSampleCount);
    return std::unique_ptr<picasso::ObjFunction>(objective.release());
  }
  if (family == Family::kPoisson) {
    std::unique_ptr<picasso::PoissonObjective> objective(
        new picasso::PoissonObjective(
            fixture.x.data(), y.data(), kSampleCount, kFeatureCount,
            include_intercept, true));
    if (use_offset)
      objective->set_offset(fixture.offset.data(), kSampleCount);
    return std::unique_ptr<picasso::ObjFunction>(objective.release());
  }
  return std::unique_ptr<picasso::ObjFunction>(new picasso::SqrtMSEObjective(
      fixture.x.data(), y.data(), kSampleCount, kFeatureCount,
      include_intercept, true));
}

std::vector<double> make_lambda_path(const Fixture &fixture, Family family,
                                     bool short_failure_path = false,
                                     bool include_intercept = true,
                                     bool use_offset = false) {
  std::unique_ptr<picasso::ObjFunction> objective =
      make_objective(fixture, family, include_intercept, use_offset);
  double lambda_max = 0.0;
  for (int feature = 0; feature < kFeatureCount; ++feature)
    lambda_max = std::max(lambda_max,
                          std::fabs(objective->get_grad(feature)));
  if (short_failure_path)
    return std::vector<double>{lambda_max, 0.2 * lambda_max};
  const double ratios[] = {1.0, 0.72, 0.52, 0.37};
  std::vector<double> lambdas(4);
  for (int index = 0; index < 4; ++index)
    lambdas[index] = lambda_max * ratios[index];
  return lambdas;
}

struct RunResult {
  int number_fit;
  int failed_lambda;
  int failed_stage;
  picasso::solver::ActNewtonLlaStatus path_status;
  std::vector<picasso::solver::ActNewtonLlaStatus> statuses;
  std::vector<int> stages;
  std::vector<int> iterations;
  std::vector<double> objective;
  std::vector<double> smooth_objective;
  std::vector<double> kkt;
  std::vector<double> stationarity;
  std::vector<double> runtime;
  std::vector<picasso::ModelParam> models;
};

RunResult run_solver(const Fixture &fixture, Family family,
                     picasso::solver::RegType penalty,
                     const std::vector<double> &lambdas, int maximum_stages,
                     int maximum_iterations = 1000,
                     double precision = kPrecision,
                     bool include_intercept = true,
                     bool use_offset = false) {
  std::unique_ptr<picasso::ObjFunction> objective =
      make_objective(fixture, family, include_intercept, use_offset);
  picasso::solver::PicassoSolverParams parameters;
  parameters.set_lambdas(lambdas.data(), static_cast<int>(lambdas.size()));
  parameters.reg_type = penalty;
  parameters.gamma = 3.5;
  parameters.include_intercept = include_intercept;
  parameters.prec = precision;
  parameters.max_iter = maximum_iterations;
  parameters.num_relaxation_round = maximum_stages;
  parameters.dfmax = -1;

  picasso::solver::ActNewtonSolver solver(objective.get(), parameters);
  solver.solve();

  RunResult result;
  result.number_fit = solver.get_num_lambdas_fit();
  result.failed_lambda = solver.get_failed_lambda();
  result.failed_stage = solver.get_failed_stage();
  result.path_status = solver.get_lla_path_status();
  result.statuses = solver.get_lla_status_path();
  result.stages = solver.get_lla_stages_path();
  result.iterations = solver.get_itercnt_path();
  result.objective = solver.get_objective_path();
  result.smooth_objective = solver.get_smooth_objective_path();
  result.kkt = solver.get_kkt_path();
  result.stationarity = solver.get_stationarity_path();
  result.runtime = solver.get_runtime_path();
  for (int index = 0; index < result.number_fit; ++index)
    result.models.push_back(solver.get_model_param(index));
  return result;
}

struct SinkRunResult {
  RunResult path;
  std::vector<double> raw_beta;
  std::vector<double> raw_intercept;
  std::vector<int> active_size;
  std::vector<int> raw_iterations;
  std::vector<double> raw_runtime;
  std::vector<double> raw_smooth_objective;
  int returned_count;
  int published_count;
  int last_nonzero_count;
  int retained_count;
};

SinkRunResult run_solver_to_buffers(
    const Fixture &fixture, Family family,
    picasso::solver::RegType penalty,
    const std::vector<double> &lambdas, int maximum_stages,
    int maximum_iterations = 1000, double precision = kPrecision,
    bool include_intercept = true, bool use_offset = false) {
  std::unique_ptr<picasso::ObjFunction> objective =
      make_objective(fixture, family, include_intercept, use_offset);
  picasso::solver::PicassoSolverParams parameters;
  parameters.set_lambdas(lambdas.data(), static_cast<int>(lambdas.size()));
  parameters.reg_type = penalty;
  parameters.gamma = 3.5;
  parameters.include_intercept = include_intercept;
  parameters.prec = precision;
  parameters.max_iter = maximum_iterations;
  parameters.num_relaxation_round = maximum_stages;
  parameters.dfmax = -1;

  const int path_size = static_cast<int>(lambdas.size());
  std::vector<double> beta(
      static_cast<std::size_t>(path_size) * kFeatureCount, 91.0);
  std::vector<double> intercept(path_size, 91.0);
  std::vector<int> iterations(path_size, 91);
  std::vector<int> active_size(path_size, 91);
  std::vector<double> runtime(path_size, 91.0);
  std::vector<double> smooth_objective(path_size, 91.0);
  int published_count = 91;
  int last_nonzero_count = 91;

  picasso::solver::ActNewtonSolver solver(objective.get(), parameters);
  const int returned_count = solver.solve_to_buffers(
      beta.data(), intercept.data(), iterations.data(), active_size.data(),
      runtime.data(), smooth_objective.data(), &published_count,
      &last_nonzero_count);

  SinkRunResult result;
  result.returned_count = returned_count;
  result.published_count = published_count;
  result.last_nonzero_count = last_nonzero_count;
  result.retained_count = solver.get_num_lambdas_fit();
  result.path.number_fit = returned_count;
  result.path.failed_lambda = solver.get_failed_lambda();
  result.path.failed_stage = solver.get_failed_stage();
  result.path.path_status = solver.get_lla_path_status();
  result.path.statuses = solver.get_lla_status_path();
  result.path.stages = solver.get_lla_stages_path();
  result.path.iterations = solver.get_itercnt_path();
  result.path.objective = solver.get_objective_path();
  result.path.smooth_objective = solver.get_smooth_objective_path();
  result.path.kkt = solver.get_kkt_path();
  result.path.stationarity = solver.get_stationarity_path();
  result.path.runtime = solver.get_runtime_path();
  result.raw_beta = beta;
  result.raw_intercept = intercept;
  result.active_size = active_size;
  result.raw_iterations = iterations;
  result.raw_runtime = runtime;
  result.raw_smooth_objective = smooth_objective;
  for (int index = 0; index < returned_count; ++index) {
    picasso::ModelParam model(kFeatureCount);
    model.intercept = intercept[index];
    for (int feature = 0; feature < kFeatureCount; ++feature) {
      model.beta[feature] =
          beta[static_cast<std::size_t>(index) * kFeatureCount + feature];
    }
    result.path.models.push_back(model);
  }
  return result;
}

bool finite_committed_diagnostics(const RunResult &result) {
  for (int index = 0; index < result.number_fit; ++index) {
    if (!std::isfinite(result.objective[index]) ||
        !std::isfinite(result.kkt[index]) ||
        !std::isfinite(result.stationarity[index]) ||
        !result.models[index].beta.allFinite() ||
        !std::isfinite(result.models[index].intercept))
      return false;
  }
  return true;
}

std::string family_name(Family family) {
  if (family == Family::kBinomial) return "binomial";
  if (family == Family::kPoisson) return "poisson";
  return "sqrt-lasso";
}

bool same_double(double left, double right) {
  return (std::isnan(left) && std::isnan(right)) || left == right;
}

bool same_double_path(const std::vector<double> &left,
                      const std::vector<double> &right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (!same_double(left[index], right[index])) return false;
  }
  return true;
}

bool same_run_result(const RunResult &retained, const RunResult &sink) {
  if (retained.number_fit != sink.number_fit ||
      retained.failed_lambda != sink.failed_lambda ||
      retained.failed_stage != sink.failed_stage ||
      retained.path_status != sink.path_status ||
      retained.statuses != sink.statuses ||
      retained.stages != sink.stages ||
      retained.iterations != sink.iterations ||
      !same_double_path(retained.objective, sink.objective) ||
      !same_double_path(retained.smooth_objective,
                        sink.smooth_objective) ||
      !same_double_path(retained.kkt, sink.kkt) ||
      !same_double_path(retained.stationarity, sink.stationarity) ||
      !same_double_path(retained.runtime, sink.runtime) ||
      retained.models.size() != sink.models.size())
    return false;
  for (std::size_t index = 0; index < retained.models.size(); ++index) {
    if (retained.models[index].intercept != sink.models[index].intercept ||
        retained.models[index].beta.size() != sink.models[index].beta.size() ||
        !(retained.models[index].beta == sink.models[index].beta).all())
      return false;
  }
  return true;
}

bool test_retained_and_direct_sink_are_identical() {
  bool ok = true;
  const Fixture fixture = make_fixture();
  const Family families[] = {
      Family::kBinomial, Family::kPoisson, Family::kSqrtLasso};
  const picasso::solver::RegType penalties[] = {
      picasso::solver::L1, picasso::solver::MCP, picasso::solver::SCAD};

  for (int family_index = 0; family_index < 3; ++family_index) {
    const Family family = families[family_index];
    for (int penalty_index = 0; penalty_index < 3; ++penalty_index) {
      const picasso::solver::RegType penalty = penalties[penalty_index];
      for (int intercept_index = 0; intercept_index < 2;
           ++intercept_index) {
        const bool include_intercept = intercept_index != 0;
        const bool use_offset = family != Family::kSqrtLasso;
        const int maximum_stages = penalty == picasso::solver::L1 ? 3 : 25;
        const std::vector<double> lambdas = make_lambda_path(
            fixture, family, false, include_intercept, use_offset);
        const RunResult retained = run_solver(
            fixture, family, penalty, lambdas, maximum_stages, 1000,
            kPrecision, include_intercept, use_offset);
        const SinkRunResult sink = run_solver_to_buffers(
            fixture, family, penalty, lambdas, maximum_stages, 1000,
            kPrecision, include_intercept, use_offset);
        const std::string label = family_name(family) + " penalty=" +
            std::to_string(static_cast<int>(penalty)) +
            (include_intercept ? " intercept" : " no-intercept");

        ok &= require(
            retained.number_fit == static_cast<int>(lambdas.size()) &&
                sink.returned_count == retained.number_fit &&
                sink.published_count == retained.number_fit &&
                sink.retained_count == 0,
            label + " direct sink did not publish exactly the fitted path");
        ok &= require(same_run_result(retained, sink.path),
                      label + " direct sink changed retained results");

        bool raw_path_matches = true;
        for (int index = 0; index < retained.number_fit; ++index) {
          if (sink.raw_iterations[index] != retained.iterations[index] ||
              sink.raw_runtime[index] != retained.runtime[index] ||
              sink.raw_smooth_objective[index] !=
                  retained.smooth_objective[index])
            raw_path_matches = false;
        }
        ok &= require(raw_path_matches,
                      label + " direct sink omitted a raw output field");

        int expected_last_nonzero = 0;
        if (!retained.models.empty()) {
          const picasso::ModelParam &last = retained.models.back();
          for (int feature = 0; feature < kFeatureCount; ++feature) {
            if (std::fabs(last.beta[feature]) > 1e-8)
              ++expected_last_nonzero;
          }
        }
        bool active_sizes_match = true;
        for (int index = 0; index < retained.number_fit; ++index) {
          int expected = 0;
          for (int feature = 0; feature < kFeatureCount; ++feature) {
            if (std::fabs(retained.models[index].beta[feature]) > 1e-8)
              ++expected;
          }
          if (sink.active_size[index] != expected)
            active_sizes_match = false;
        }
        ok &= require(active_sizes_match &&
                          sink.last_nonzero_count == expected_last_nonzero,
                      label + " direct sink reported the wrong sparsity");
      }
    }
  }

  const std::vector<double> failure_path =
      make_lambda_path(fixture, Family::kBinomial, true);
  const RunResult retained_failure = run_solver(
      fixture, Family::kBinomial, picasso::solver::MCP, failure_path, 3, 1,
      1e-7);
  const SinkRunResult sink_failure = run_solver_to_buffers(
      fixture, Family::kBinomial, picasso::solver::MCP, failure_path, 3, 1,
      1e-7);
  ok &= require(
      sink_failure.returned_count == 1 &&
          sink_failure.published_count == 1 &&
          sink_failure.retained_count == 0 &&
          same_run_result(retained_failure, sink_failure.path) &&
          sink_failure.raw_iterations[0] ==
              retained_failure.iterations[0] &&
          sink_failure.raw_runtime[0] == retained_failure.runtime[0] &&
          sink_failure.raw_smooth_objective[0] ==
              retained_failure.smooth_objective[0] &&
          sink_failure.raw_iterations[1] == 91 &&
          sink_failure.raw_runtime[1] == 91.0 &&
          sink_failure.raw_smooth_objective[1] == 91.0 &&
          sink_failure.active_size[1] == 91 &&
          sink_failure.raw_intercept[1] == 91.0,
      "direct sink must preserve the retained prefix on a later failure");
  bool failure_beta_tail_is_sentinel = true;
  for (int feature = 0; feature < kFeatureCount; ++feature) {
    if (sink_failure.raw_beta[kFeatureCount + feature] != 91.0)
      failure_beta_tail_is_sentinel = false;
  }
  ok &= require(failure_beta_tail_is_sentinel,
                "direct sink wrote an uncommitted coefficient tail");
  return ok;
}

bool test_adaptive_lla_for_all_scalar_families() {
  bool ok = true;
  const Fixture fixture = make_fixture();
  const Family families[] = {
      Family::kBinomial, Family::kPoisson, Family::kSqrtLasso};
  const picasso::solver::RegType penalties[] = {
      picasso::solver::MCP, picasso::solver::SCAD};

  for (int family_index = 0; family_index < 3; ++family_index) {
    const Family family = families[family_index];
    const std::string label = family_name(family);
    const std::vector<double> lambdas = make_lambda_path(fixture, family);

    const RunResult l1 = run_solver(
        fixture, family, picasso::solver::L1, lambdas, 3);
    ok &= require(l1.number_fit == static_cast<int>(lambdas.size()),
                  label + " L1 must fit the complete deterministic path");
    ok &= require(l1.path_status ==
                      picasso::solver::ActNewtonLlaStatus::kCompleted,
                  label + " L1 path must be certified");
    ok &= require(finite_committed_diagnostics(l1),
                  label + " L1 diagnostics and models must be finite");
    for (std::size_t index = 0; index < lambdas.size(); ++index) {
      ok &= require(l1.statuses[index] ==
                        picasso::solver::ActNewtonLlaStatus::kCompleted &&
                        l1.stages[index] == 1 &&
                        l1.kkt[index] <= kPrecision + 1e-12 &&
                        l1.stationarity[index] <= kPrecision + 1e-12,
                    label + " L1 must use one KKT-certified stage");
    }

    for (int penalty_index = 0; penalty_index < 2; ++penalty_index) {
      const picasso::solver::RegType penalty = penalties[penalty_index];
      const std::string penalty_label =
          penalty == picasso::solver::MCP ? "MCP" : "SCAD";
      const RunResult capped = run_solver(
          fixture, family, penalty, lambdas, 3);
      const RunResult adaptive = run_solver(
          fixture, family, penalty, lambdas, 25);

      ok &= require(capped.number_fit == static_cast<int>(lambdas.size()) &&
                        adaptive.number_fit ==
                            static_cast<int>(lambdas.size()),
                    label + " " + penalty_label +
                        " cap-3 and adaptive paths must both be usable");
      ok &= require(capped.path_status ==
                        picasso::solver::ActNewtonLlaStatus::
                            kStationarityLimit,
                    label + " " + penalty_label +
                        " fixture must expose the default stage limit");
      ok &= require(adaptive.path_status ==
                        picasso::solver::ActNewtonLlaStatus::kCompleted,
                    label + " " + penalty_label +
                        " raised cap must reach target stationarity");
      ok &= require(finite_committed_diagnostics(capped) &&
                        finite_committed_diagnostics(adaptive),
                    label + " " + penalty_label +
                        " diagnostics and models must remain finite");

      for (std::size_t index = 0; index < lambdas.size(); ++index) {
        const double allowance =
            1e-10 * (1.0 + std::fabs(capped.objective[index]));
        ok &= require(capped.stages[index] == 3 &&
                          capped.kkt[index] <= kPrecision + 1e-12,
                      label + " " + penalty_label +
                          " default must complete three valid stages");
        ok &= require(adaptive.statuses[index] ==
                          picasso::solver::ActNewtonLlaStatus::kCompleted &&
                          adaptive.stages[index] >= 3 &&
                          adaptive.stages[index] <= 25 &&
                          adaptive.kkt[index] <= kPrecision + 1e-12 &&
                          adaptive.stationarity[index] <=
                              kPrecision + 1e-12,
                      label + " " + penalty_label +
                          " adaptive result must carry both certifications");
        ok &= require(adaptive.objective[index] <=
                          capped.objective[index] + allowance,
                      label + " " + penalty_label +
                          " additional LLA stages must not raise the target "
                          "objective");
      }
    }
  }
  return ok;
}

bool test_invalid_budget_and_transactional_failure() {
  bool ok = true;
  const Fixture fixture = make_fixture();
  const std::vector<double> regular_path =
      make_lambda_path(fixture, Family::kBinomial);
  const RunResult invalid = run_solver(
      fixture, Family::kBinomial, picasso::solver::MCP,
      regular_path, 2);
  ok &= require(
      invalid.number_fit == 0 && invalid.failed_lambda == 0 &&
          invalid.failed_stage == -1 &&
          invalid.path_status ==
              picasso::solver::ActNewtonLlaStatus::kNumericalFailure,
      "a nonconvex budget below three must fail before committing a model");

  const std::vector<double> failure_path =
      make_lambda_path(fixture, Family::kBinomial, true);
  const RunResult failed = run_solver(
      fixture, Family::kBinomial, picasso::solver::MCP,
      failure_path, 3, 1, 1e-7);
  ok &= require(
      failed.number_fit == 1 && failed.failed_lambda == 1 &&
          failed.failed_stage == 0 &&
          failed.path_status ==
              picasso::solver::ActNewtonLlaStatus::kSubproblemFailed,
      "a failed later subproblem must retain exactly the committed prefix");
  ok &= require(
      failed.stages[0] == 3 && failed.stages[1] == 0 &&
          failed.statuses[0] ==
              picasso::solver::ActNewtonLlaStatus::kCompleted &&
          failed.statuses[1] ==
              picasso::solver::ActNewtonLlaStatus::kSubproblemFailed &&
          std::isfinite(failed.objective[0]) &&
          failed.models.size() == 1 && failed.models[0].beta.allFinite(),
      "failure diagnostics must not present an uncommitted candidate as a "
      "solution");
  return ok;
}

bool test_fast_tall_poisson_deferred_state() {
  bool ok = true;
  const double precision = 1e-4;
  const Fixture fixture = make_fixture();
  const std::vector<double> lambdas =
      make_lambda_path(fixture, Family::kPoisson);
  const RunResult result = run_solver(
      fixture, Family::kPoisson, picasso::solver::L1,
      lambdas, 3, 1000, precision);

  int total_iterations = 0;
  for (std::size_t index = 0; index < result.iterations.size(); ++index)
    total_iterations += result.iterations[index];
  ok &= require(
      result.number_fit == static_cast<int>(lambdas.size()) &&
          result.path_status ==
              picasso::solver::ActNewtonLlaStatus::kCompleted &&
          finite_committed_diagnostics(result) && total_iterations > 0,
      "fast tall Poisson path must execute and converge");
  if (result.number_fit != static_cast<int>(lambdas.size())) return false;

  const picasso::ModelParam &model = result.models.back();
  Eigen::ArrayXd rebuilt_xb(kSampleCount);
  rebuilt_xb.setZero();
  int nonzero = 0;
  for (int feature = 0; feature < kFeatureCount; ++feature) {
    if (std::fabs(model.beta[feature]) > 1e-8) ++nonzero;
    for (int sample = 0; sample < kSampleCount; ++sample) {
      rebuilt_xb[sample] +=
          fixture.x[static_cast<std::size_t>(sample) * kFeatureCount +
                    feature] *
          model.beta[feature];
    }
  }
  ok &= require(nonzero > 0,
                "fast tall Poisson fixture must exercise active coordinates");

  picasso::PoissonObjective oracle(
      fixture.x.data(), fixture.poisson.data(), kSampleCount,
      kFeatureCount, true, true);
  oracle.set_model_param(model);
  oracle.set_model_Xb(rebuilt_xb);
  oracle.update_auxiliary();
  oracle.update_all_gradients();

  std::vector<double> reference_gradient(kFeatureCount);
  for (int feature = 0; feature < kFeatureCount; ++feature)
    reference_gradient[feature] = oracle.get_grad(feature);

  // The sample count also exercises the scalar tail of the packet kernel.
  oracle.set_fast_residual_dot(true);
  for (int feature = 0; feature < kFeatureCount; ++feature) {
    oracle.update_gradient(feature);
    const double allowance =
        5e-13 * std::max(1.0, std::fabs(reference_gradient[feature]));
    ok &= require(
        std::fabs(oracle.get_grad(feature) -
                  reference_gradient[feature]) <= allowance,
        "fast residual dot must agree with the full-gradient GEMV");
  }

  const double lambda = lambdas.back();
  const double oracle_objective =
      oracle.eval() + lambda * model.beta.abs().sum();
  double oracle_kkt = std::fabs(oracle.get_intercept_gradient());
  for (int feature = 0; feature < kFeatureCount; ++feature) {
    const double smooth_gradient = -reference_gradient[feature];
    double residual;
    if (model.beta[feature] > 1e-8)
      residual = std::fabs(smooth_gradient + lambda);
    else if (model.beta[feature] < -1e-8)
      residual = std::fabs(smooth_gradient - lambda);
    else
      residual =
          std::max(0.0, std::fabs(smooth_gradient) - lambda);
    oracle_kkt = std::max(oracle_kkt, residual);
  }

  const std::size_t last = lambdas.size() - 1;
  ok &= require(
      std::fabs(oracle_objective - result.objective[last]) <=
          5e-11 * std::max(1.0, std::fabs(oracle_objective)),
      "deferred Poisson predictor rebuild must match returned coefficients");
  ok &= require(
      oracle_kkt <= precision + 1e-12 &&
          std::fabs(oracle_kkt - result.kkt[last]) <= 5e-11 &&
          std::fabs(oracle_kkt - result.stationarity[last]) <= 5e-11,
      "fast tall Poisson result must have an independently reconstructed "
      "L1 KKT certificate");
  return ok;
}

bool test_fast_weighted_curvature_reduction() {
  bool ok = true;
  const Fixture fixture = make_fixture();
  picasso::PoissonObjective reference(
      fixture.x.data(), fixture.poisson.data(), kSampleCount,
      kFeatureCount, true, true);
  picasso::PoissonObjective candidate(
      fixture.x.data(), fixture.poisson.data(), kSampleCount,
      kFeatureCount, true, true);
  candidate.set_fast_weighted_sq_sum(true);

  picasso::RegL1 reference_regularizer;
  picasso::RegL1 candidate_regularizer;
  reference_regularizer.set_param(0.0, 0.0);
  candidate_regularizer.set_param(0.0, 0.0);
  const double reference_coefficient =
      reference.coordinate_descent(&reference_regularizer, 0);
  const double candidate_coefficient =
      candidate.coordinate_descent(&candidate_regularizer, 0);
  const double coefficient_allowance =
      5e-13 * std::max(1.0, std::fabs(reference_coefficient));
  ok &= require(
      std::isfinite(reference_coefficient) &&
          std::isfinite(candidate_coefficient) &&
          reference_coefficient != 0.0 &&
          std::fabs(reference_coefficient - candidate_coefficient) <=
              coefficient_allowance,
      "fast weighted curvature must preserve a coordinate minimizer");

  const Eigen::ArrayXd reference_xb = reference.get_model_Xb();
  const Eigen::ArrayXd candidate_xb = candidate.get_model_Xb();
  const double predictor_error =
      (reference_xb - candidate_xb).abs().maxCoeff();
  const double predictor_scale =
      std::max(1.0, reference_xb.abs().maxCoeff());
  ok &= require(
      predictor_error <= 5e-13 * predictor_scale,
      "fast weighted curvature must preserve the accepted predictor update");
  return ok;
}

bool test_sqrt_local_change_certificate() {
  bool ok = true;
  const Fixture fixture = make_fixture();
  picasso::SqrtMSEObjective objective(
      fixture.x.data(), fixture.sqrt_lasso.data(), kSampleCount,
      kFeatureCount, true, true);
  picasso::RegL1 regularizer;
  regularizer.set_param(0.08, 0.0);
  const double threshold = objective.get_deviance() * 1e-7;
  int certified_updates = 0;
  int certified_nonzero_updates = 0;

  for (int sweep = 0; sweep < 40; ++sweep) {
    for (int feature = 0; feature < kFeatureCount; ++feature) {
      const double old_coefficient = objective.get_model_coef(feature);
      const double objective_before =
          objective.eval() +
          0.08 * objective.get_model_param_ref().beta.abs().sum();
      const double updated = objective.coordinate_descent(
          &regularizer, feature);
      const double objective_after =
          objective.eval() +
          0.08 * objective.get_model_param_ref().beta.abs().sum();
      const double descent_allowance =
          128.0 * std::numeric_limits<double>::epsilon() *
          std::max(1.0, std::fabs(objective_before));
      const bool certified = objective.can_skip_local_change(
          old_coefficient, feature, threshold);
      const double exact_change = objective.get_local_change(
          old_coefficient, feature);
      ok &= require(std::isfinite(updated) && std::isfinite(exact_change),
                    "sqrt local-change fixture must remain finite");
      ok &= require(objective_after <= objective_before + descent_allowance,
                    "sqrt MM coordinate update must not raise the weighted "
                    "L1 objective");
      if (certified) {
        ++certified_updates;
        if (updated != old_coefficient) ++certified_nonzero_updates;
        ok &= require(exact_change <= threshold,
                      "sqrt local-change certificate skipped a large change");
      }
    }
    const double old_intercept = objective.get_model_coef(-1);
    objective.intercept_update();
    ok &= require(!objective.can_skip_local_change(
                      old_intercept, -1, threshold),
                  "sqrt intercept must retain its exact O(1) check");
  }

  ok &= require(certified_updates > 0 && certified_nonzero_updates > 0,
                "sqrt fixture must exercise the fast local-change certificate");
  ok &= require(!objective.can_skip_local_change(
                    objective.get_model_coef(0), 0, -1.0),
                "sqrt certificate must reject a negative threshold");

  const picasso::ModelParam &model = objective.get_model_param_ref();
  const Eigen::ArrayXd xb = objective.get_model_Xb();
  double maximum_xb_error = 0.0;
  double maximum_xb_scale = 0.0;
  for (int sample = 0; sample < kSampleCount; ++sample) {
    double recomputed = 0.0;
    for (int feature = 0; feature < kFeatureCount; ++feature) {
      recomputed +=
          fixture.x[static_cast<std::size_t>(sample) * kFeatureCount +
                    feature] *
          model.beta[feature];
    }
    maximum_xb_error =
        std::max(maximum_xb_error, std::fabs(xb[sample] - recomputed));
    maximum_xb_scale = std::max(maximum_xb_scale, std::fabs(recomputed));
  }
  ok &= require(maximum_xb_error <=
                    1e-12 * std::max(1.0, maximum_xb_scale),
                "sqrt accepted coefficients and cached linear predictor "
                "must remain consistent");
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_retained_and_direct_sink_are_identical();
  ok &= test_adaptive_lla_for_all_scalar_families();
  ok &= test_invalid_budget_and_transactional_failure();
  ok &= test_fast_tall_poisson_deferred_state();
  ok &= test_fast_weighted_curvature_reduction();
  ok &= test_sqrt_local_change_certificate();
  if (!ok) return 1;
  std::cout << "ActNewton adaptive-LLA tests passed\n";
  return 0;
}
