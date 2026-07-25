#include <picasso/actgd.hpp>
#include <picasso/actnewton.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class InspectGaussian : public picasso::GaussianNaiveUpdateObjective {
 public:
  using picasso::GaussianNaiveUpdateObjective::GaussianNaiveUpdateObjective;
  Eigen::Index owned_design_size() const { return X.size(); }
};

class InspectLogistic : public picasso::LogisticObjective {
 public:
  using picasso::LogisticObjective::LogisticObjective;
  Eigen::Index owned_design_size() const { return X.size(); }
};

class InspectPoisson : public picasso::PoissonObjective {
 public:
  using picasso::PoissonObjective::PoissonObjective;
  Eigen::Index owned_design_size() const { return X.size(); }
};

class InspectSqrtMSE : public picasso::SqrtMSEObjective {
 public:
  using picasso::SqrtMSEObjective::SqrtMSEObjective;
  Eigen::Index owned_design_size() const { return X.size(); }
};

struct PathFixture {
  int n;
  int d;
  std::vector<double> design;
  std::vector<double> gaussian;
  std::vector<double> binomial;
  std::vector<double> poisson;
  std::vector<double> offset;
};

PathFixture make_path_fixture() {
  PathFixture fixture;
  fixture.n = 180;
  fixture.d = 12;
  fixture.design.resize(
      static_cast<std::size_t>(fixture.n) * fixture.d);
  fixture.gaussian.resize(fixture.n);
  fixture.binomial.resize(fixture.n);
  fixture.poisson.resize(fixture.n);
  fixture.offset.resize(fixture.n);

  for (int feature = 0; feature < fixture.d; ++feature) {
    double mean = 0.0;
    for (int sample = 0; sample < fixture.n; ++sample) {
      const double value =
          std::sin(0.07 * (sample + 1.0) * (feature + 1.0)) +
          0.25 * std::cos(0.13 * (sample + feature + 2.0));
      fixture.design[static_cast<std::size_t>(feature) * fixture.n +
                     sample] = value;
      mean += value;
    }
    mean /= fixture.n;
    double sum_squares = 0.0;
    for (int sample = 0; sample < fixture.n; ++sample) {
      double &value =
          fixture.design[static_cast<std::size_t>(feature) * fixture.n +
                         sample];
      value -= mean;
      sum_squares += value * value;
    }
    const double scale =
        std::sqrt(sum_squares / static_cast<double>(fixture.n - 1));
    for (int sample = 0; sample < fixture.n; ++sample) {
      fixture.design[static_cast<std::size_t>(feature) * fixture.n +
                     sample] /= scale;
    }
  }

  const double beta[] = {1.0, -0.8, 0.65, -0.5};
  for (int sample = 0; sample < fixture.n; ++sample) {
    double signal = 0.0;
    for (int feature = 0; feature < 4; ++feature) {
      signal += beta[feature] *
                fixture.design[static_cast<std::size_t>(feature) *
                                   fixture.n +
                               sample];
    }
    const double draw = std::fmod(
        0.6180339887498949 * static_cast<double>(sample + 1), 1.0);
    const double probability =
        1.0 / (1.0 + std::exp(-(-0.2 + signal)));
    fixture.gaussian[sample] =
        0.4 + signal + 0.35 * std::sin(0.31 * (sample + 1.0));
    fixture.binomial[sample] = draw < probability ? 1.0 : 0.0;
    fixture.poisson[sample] =
        std::floor(std::exp(0.15 + 0.28 * signal) + draw);
    fixture.offset[sample] =
        0.12 * std::sin(0.19 * (sample + 1.0));
  }
  return fixture;
}

bool nearly_equal(double left, double right, double tolerance = 1e-13) {
  return std::fabs(left - right) <=
         tolerance * (1.0 + std::max(std::fabs(left), std::fabs(right)));
}

bool require(bool condition, const std::string &message) {
  if (!condition) std::cerr << "FAIL: " << message << '\n';
  return condition;
}

std::vector<double> lambda_path(picasso::ObjFunction *objective, int d) {
  double lambda_max = 0.0;
  for (int feature = 0; feature < d; ++feature) {
    lambda_max = std::max(lambda_max,
                          std::fabs(objective->get_grad(feature)));
  }
  return std::vector<double>{lambda_max, 0.72 * lambda_max,
                             0.52 * lambda_max, 0.37 * lambda_max};
}

picasso::solver::PicassoSolverParams solver_parameters(
    const std::vector<double> &lambdas,
    picasso::solver::RegType penalty = picasso::solver::L1) {
  picasso::solver::PicassoSolverParams parameters;
  parameters.set_lambdas(lambdas.data(), static_cast<int>(lambdas.size()));
  parameters.reg_type = penalty;
  parameters.gamma = 3.5;
  parameters.num_relaxation_round = 3;
  parameters.prec = 1e-5;
  parameters.max_iter = 1000;
  parameters.include_intercept = true;
  parameters.dfmax = -1;
  parameters.min_lambda_count = static_cast<int>(lambdas.size()) + 1;
  return parameters;
}

template <typename Solver>
bool equivalent_model_path(const Solver &owned, const Solver &borrowed,
                           const std::string &label) {
  bool ok = true;
  const int owned_count = owned.get_num_lambdas_fit();
  const int borrowed_count = borrowed.get_num_lambdas_fit();
  ok &= require(owned_count == borrowed_count,
                label + " fitted path lengths must agree");
  const int common_count = std::min(owned_count, borrowed_count);
  for (int index = 0; index < common_count; ++index) {
    const picasso::ModelParam &owned_model = owned.get_model_param(index);
    const picasso::ModelParam &borrowed_model = borrowed.get_model_param(index);
    ok &= require(
        (owned_model.beta - borrowed_model.beta).abs().maxCoeff() <= 1e-13,
        label + " coefficients must agree at lambda " +
            std::to_string(index));
    ok &= require(nearly_equal(owned_model.intercept,
                               borrowed_model.intercept),
                  label + " intercepts must agree at lambda " +
                      std::to_string(index));
  }
  ok &= require(owned.get_itercnt_path() == borrowed.get_itercnt_path(),
                label + " iteration counts must agree");
  return ok;
}

template <typename Objective>
bool equivalent_actgd_path(const std::vector<double> &design,
                           const std::vector<double> &response, int n, int d,
                           const std::string &label) {
  Objective owned(design.data(), response.data(), n, d, true, false);
  Objective borrowed(
      design.data(), response.data(), n, d, true, false,
      picasso::detail::DesignStorage::kBorrowedColumnMajor);
  const std::vector<double> lambdas = lambda_path(&owned, d);
  const picasso::solver::PicassoSolverParams parameters =
      solver_parameters(lambdas);
  picasso::solver::ActGDSolver owned_solver(&owned, parameters);
  picasso::solver::ActGDSolver borrowed_solver(&borrowed, parameters);
  owned_solver.solve();
  borrowed_solver.solve();

  bool ok = equivalent_model_path(owned_solver, borrowed_solver, label);
  ok &= require(owned_solver.get_status() == borrowed_solver.get_status(),
                label + " path statuses must agree");
  ok &= require(owned_solver.get_failed_lambda() ==
                    borrowed_solver.get_failed_lambda(),
                label + " failed-lambda diagnostics must agree");
  const std::vector<double> &owned_objective =
      owned_solver.get_smooth_objective_path();
  const std::vector<double> &borrowed_objective =
      borrowed_solver.get_smooth_objective_path();
  ok &= require(owned_objective.size() == borrowed_objective.size(),
                label + " objective path lengths must agree");
  for (std::size_t index = 0;
       index < std::min(owned_objective.size(), borrowed_objective.size());
       ++index) {
    ok &= require(nearly_equal(owned_objective[index],
                               borrowed_objective[index]),
                  label + " objectives must agree");
  }
  return ok;
}

template <typename Objective>
bool equivalent_actnewton_path(
    const std::vector<double> &design, const std::vector<double> &response,
    const std::vector<double> *offset, int n, int d,
    picasso::solver::RegType penalty, const std::string &label) {
  Objective owned(design.data(), response.data(), n, d, true, false);
  Objective borrowed(
      design.data(), response.data(), n, d, true, false,
      picasso::detail::DesignStorage::kBorrowedColumnMajor);
  bool ok = true;
  if (offset != nullptr) {
    ok &= require(owned.set_offset(offset->data(), n) &&
                      borrowed.set_offset(offset->data(), n),
                  label + " offsets must be accepted");
  }
  const std::vector<double> lambdas = lambda_path(&owned, d);
  const picasso::solver::PicassoSolverParams parameters =
      solver_parameters(lambdas, penalty);
  picasso::solver::ActNewtonSolver owned_solver(&owned, parameters);
  picasso::solver::ActNewtonSolver borrowed_solver(&borrowed, parameters);
  owned_solver.solve();
  borrowed_solver.solve();

  ok &= equivalent_model_path(owned_solver, borrowed_solver, label);
  ok &= require(owned_solver.get_lla_path_status() ==
                    borrowed_solver.get_lla_path_status(),
                label + " aggregate statuses must agree");
  ok &= require(owned_solver.get_lla_status_path() ==
                    borrowed_solver.get_lla_status_path(),
                label + " per-lambda statuses must agree");
  ok &= require(owned_solver.get_lla_stages_path() ==
                    borrowed_solver.get_lla_stages_path(),
                label + " LLA stages must agree");
  ok &= require(owned_solver.get_failed_lambda() ==
                    borrowed_solver.get_failed_lambda() &&
                    owned_solver.get_failed_stage() ==
                        borrowed_solver.get_failed_stage(),
                label + " failure diagnostics must agree");
  const std::vector<double> *owned_diagnostics[] = {
      &owned_solver.get_objective_path(),
      &owned_solver.get_smooth_objective_path(),
      &owned_solver.get_kkt_path(),
      &owned_solver.get_stationarity_path()};
  const std::vector<double> *borrowed_diagnostics[] = {
      &borrowed_solver.get_objective_path(),
      &borrowed_solver.get_smooth_objective_path(),
      &borrowed_solver.get_kkt_path(),
      &borrowed_solver.get_stationarity_path()};
  for (int diagnostic = 0; diagnostic < 4; ++diagnostic) {
    ok &= require(owned_diagnostics[diagnostic]->size() ==
                      borrowed_diagnostics[diagnostic]->size(),
                  label + " diagnostic path lengths must agree");
    const std::size_t common =
        std::min(owned_diagnostics[diagnostic]->size(),
                 borrowed_diagnostics[diagnostic]->size());
    for (std::size_t index = 0; index < common; ++index) {
      const double owned_value = (*owned_diagnostics[diagnostic])[index];
      const double borrowed_value =
          (*borrowed_diagnostics[diagnostic])[index];
      const bool matches = nearly_equal(owned_value, borrowed_value, 1e-12);
      if (!matches) {
        std::cerr << std::setprecision(17) << "DETAIL: " << label
                  << " diagnostic " << diagnostic << " lambda " << index
                  << " owned=" << owned_value
                  << " borrowed=" << borrowed_value << '\n';
      }
      ok &= require(matches, label + " diagnostics must agree");
    }
  }
  return ok;
}

template <typename Objective>
bool equivalent_initial_state(Objective *owned, Objective *borrowed, int d,
                              const std::string &label) {
  bool ok = true;
  ok &= require(owned->owned_design_size() > 0,
                label + " owning constructor must retain its design");
  ok &= require(borrowed->owned_design_size() == 0,
                label + " borrowed constructor must not copy its design");
  ok &= require(nearly_equal(owned->eval(), borrowed->eval()),
                label + " initial losses must agree");
  for (int feature = 0; feature < d; ++feature) {
    ok &= require(nearly_equal(owned->get_grad(feature),
                               borrowed->get_grad(feature)),
                  label + " initial gradients must agree");
  }
  const picasso::ModelParam owned_model = owned->get_model_param();
  const picasso::ModelParam borrowed_model = borrowed->get_model_param();
  ok &= require(nearly_equal(owned_model.intercept,
                             borrowed_model.intercept),
                label + " initial intercepts must agree");
  ok &= require((owned_model.beta == borrowed_model.beta).all(),
                label + " initial coefficients must agree");
  return ok;
}

}  // namespace

int main() {
  const int n = 8;
  const int d = 3;
  const std::vector<double> design = {
      -1.0, -0.7, -0.2, 0.1, 0.4, 0.8, 1.1, 1.5,
       0.3, -0.5,  0.9, 1.2, -1.1, 0.6, 0.2, -0.8,
       1.0,  0.4, -0.6, 0.7,  0.2, 1.3, -0.9, 0.5};
  const std::vector<double> gaussian_y =
      {-1.2, -0.4, 0.1, 0.8, 0.5, 1.4, 1.1, 1.8};
  const std::vector<double> binary_y = {0, 0, 0, 1, 0, 1, 1, 1};
  const std::vector<double> count_y = {0, 1, 0, 2, 1, 3, 2, 4};
  bool ok = true;

  std::vector<double> input = design;
  InspectGaussian gaussian_owned(
      input.data(), gaussian_y.data(), n, d, true, false);
  InspectGaussian gaussian_borrowed(
      input.data(), gaussian_y.data(), n, d, true, false,
      picasso::detail::DesignStorage::kBorrowedColumnMajor);
  ok &= equivalent_initial_state(&gaussian_owned, &gaussian_borrowed, d,
                                 "Gaussian");

  InspectLogistic logistic_owned(
      input.data(), binary_y.data(), n, d, true, false);
  InspectLogistic logistic_borrowed(
      input.data(), binary_y.data(), n, d, true, false,
      picasso::detail::DesignStorage::kBorrowedColumnMajor);
  ok &= equivalent_initial_state(&logistic_owned, &logistic_borrowed, d,
                                 "Logistic");

  InspectPoisson poisson_owned(
      input.data(), count_y.data(), n, d, true, false);
  InspectPoisson poisson_borrowed(
      input.data(), count_y.data(), n, d, true, false,
      picasso::detail::DesignStorage::kBorrowedColumnMajor);
  ok &= equivalent_initial_state(&poisson_owned, &poisson_borrowed, d,
                                 "Poisson");

  InspectSqrtMSE sqrt_owned(
      input.data(), gaussian_y.data(), n, d, true, false);
  InspectSqrtMSE sqrt_borrowed(
      input.data(), gaussian_y.data(), n, d, true, false,
      picasso::detail::DesignStorage::kBorrowedColumnMajor);
  ok &= equivalent_initial_state(&sqrt_owned, &sqrt_borrowed, d,
                                 "Square-root loss");
  ok &= require(input == design, "borrowed objectives must not mutate X");

  // Exercise complete paths, not only objective initialization. Gaussian
  // covers both backends; GLM paths cover offsets and adaptive nonconvex LLA.
  const PathFixture path_fixture = make_path_fixture();
  ok &= equivalent_actgd_path<picasso::GaussianNaiveUpdateObjective>(
      path_fixture.design, path_fixture.gaussian, path_fixture.n,
      path_fixture.d, "Gaussian naive full path");
  ok &= equivalent_actgd_path<picasso::GaussianCovUpdateObjective>(
      path_fixture.design, path_fixture.gaussian, path_fixture.n,
      path_fixture.d, "Gaussian covariance full path");
  ok &= equivalent_actnewton_path<picasso::LogisticObjective>(
      path_fixture.design, path_fixture.binomial, &path_fixture.offset,
      path_fixture.n, path_fixture.d, picasso::solver::L1,
      "Logistic offset full path");
  ok &= equivalent_actnewton_path<picasso::PoissonObjective>(
      path_fixture.design, path_fixture.poisson, &path_fixture.offset,
      path_fixture.n, path_fixture.d, picasso::solver::L1,
      "Poisson offset full path");
  ok &= equivalent_actnewton_path<picasso::SqrtMSEObjective>(
      path_fixture.design, path_fixture.gaussian, nullptr, path_fixture.n,
      path_fixture.d, picasso::solver::L1,
      "Square-root loss full path");
  ok &= equivalent_actnewton_path<picasso::LogisticObjective>(
      path_fixture.design, path_fixture.binomial, &path_fixture.offset,
      path_fixture.n, path_fixture.d, picasso::solver::MCP,
      "Logistic MCP offset full path");

  bool invalid_layout_rejected = false;
  try {
    InspectGaussian invalid(
        design.data(), gaussian_y.data(), n, d, true, true,
        picasso::detail::DesignStorage::kBorrowedColumnMajor);
  } catch (const std::invalid_argument &) {
    invalid_layout_rejected = true;
  }
  ok &= require(invalid_layout_rejected,
                "row-major input must reject borrowed column-major storage");

  // A copied borrowed objective must take ownership. Replacing the caller's
  // buffer afterwards must not affect either copy construction or assignment.
  std::vector<double> copy_input = design;
  InspectGaussian borrowed_source(
      copy_input.data(), gaussian_y.data(), n, d, true, false,
      picasso::detail::DesignStorage::kBorrowedColumnMajor);
  InspectGaussian copied(borrowed_source);
  InspectGaussian assigned(
      design.data(), gaussian_y.data(), n, d, true, false);
  assigned = borrowed_source;
  std::fill(copy_input.begin(), copy_input.end(), 1000.0);

  InspectGaussian oracle(
      design.data(), gaussian_y.data(), n, d, true, false);
  copied.update_auxiliary();
  copied.intercept_update();
  assigned.update_auxiliary();
  assigned.intercept_update();
  oracle.update_auxiliary();
  oracle.intercept_update();
  ok &= require(copied.owned_design_size() == n * d,
                "copy construction must own the borrowed design");
  ok &= require(assigned.owned_design_size() == n * d,
                "copy assignment must own the borrowed design");
  ok &= require(nearly_equal(copied.eval(), oracle.eval()) &&
                    nearly_equal(assigned.eval(), oracle.eval()),
                "copied objectives must survive caller-buffer replacement");

  // Because ObjFunction deliberately has value-preserving copy operations and
  // no pointer-transferring move operation, rvalue construction/assignment
  // must also materialize borrowed storage.
  std::vector<double> move_input = design;
  InspectGaussian move_source(
      move_input.data(), gaussian_y.data(), n, d, true, false,
      picasso::detail::DesignStorage::kBorrowedColumnMajor);
  InspectGaussian move_constructed(std::move(move_source));
  InspectGaussian move_assign_source(
      move_input.data(), gaussian_y.data(), n, d, true, false,
      picasso::detail::DesignStorage::kBorrowedColumnMajor);
  InspectGaussian move_assigned(
      design.data(), gaussian_y.data(), n, d, true, false);
  InspectGaussian move_oracle(
      design.data(), gaussian_y.data(), n, d, true, false);
  move_assigned = std::move(move_assign_source);
  std::fill(move_input.begin(), move_input.end(), -1000.0);
  move_constructed.update_auxiliary();
  move_assigned.update_auxiliary();
  move_oracle.update_auxiliary();
  ok &= require(move_constructed.owned_design_size() == n * d &&
                    move_assigned.owned_design_size() == n * d,
                "rvalue copies must own the borrowed design");
  ok &= require(nearly_equal(move_constructed.eval(), move_oracle.eval()) &&
                    nearly_equal(move_assigned.eval(), move_oracle.eval()),
                "rvalue copies must survive caller-buffer replacement");

  if (!ok) return 1;
  std::cout << "scalar_borrowed_design_test: PASS\n";
  return 0;
}
