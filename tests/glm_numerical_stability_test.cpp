#include <picasso/objective.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool require(bool condition, const std::string &message) {
  if (!condition) std::cerr << "FAIL: " << message << "\n";
  return condition;
}

bool finite_gradient(picasso::ObjFunction *objective) {
  objective->update_all_gradients();
  for (int feature = 0; feature < objective->get_dim(); ++feature) {
    if (!std::isfinite(objective->get_grad(feature))) return false;
  }
  return true;
}

bool test_extreme_logistic_loss() {
  const int n = 2;
  const int d = 1;
  const double x[n * d] = {0.0, 0.0};
  const double y[n] = {0.0, 1.0};
  picasso::LogisticObjective objective(x, y, n, d, false, true);

  bool ok = true;
  const double links[] = {-1000.0, 1000.0};
  for (int index = 0; index < 2; ++index) {
    objective.set_model_coef(links[index], -1);
    objective.update_auxiliary();
    const double loss = objective.eval();
    ok &= require(std::isfinite(loss),
                  "logistic loss must remain finite for an extreme link");
    ok &= require(std::fabs(loss - 500.0) < 1e-10,
                  "extreme balanced logistic loss must equal its softplus oracle");
    ok &= require(finite_gradient(&objective),
                  "extreme logistic gradients must remain finite");
  }
  return ok;
}

bool test_logistic_mean_does_not_overflow() {
  const int n = 8;
  const int d = 1;
  const double x[n * d] = {0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0};
  const double y[n] = {0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0};
  picasso::LogisticObjective objective(x, y, n, d, false, true);
  objective.set_model_coef(1e308, -1);
  objective.update_auxiliary();
  const double loss = objective.eval();
  return require(std::isfinite(loss),
                 "mean logistic loss overflowed while every row was finite") &&
         require(std::fabs(loss / 1e308 - 1.0) < 1e-15,
                 "extreme logistic mean disagrees with its linear-tail oracle");
}

bool test_degenerate_logistic_initialization() {
  const int n = 4;
  const int d = 1;
  const double x[n * d] = {0.0, 0.0, 0.0, 0.0};
  const double all_zero[n] = {0.0, 0.0, 0.0, 0.0};
  const double all_one[n] = {1.0, 1.0, 1.0, 1.0};
  picasso::LogisticObjective zero_objective(
      x, all_zero, n, d, true, true);
  picasso::LogisticObjective one_objective(
      x, all_one, n, d, true, true);

  return require(std::isfinite(zero_objective.get_model_coef(-1)) &&
                     std::isfinite(zero_objective.eval()),
                 "all-zero logistic initialization must be finite") &&
         require(std::isfinite(one_objective.get_model_coef(-1)) &&
                     std::isfinite(one_objective.eval()),
                 "all-one logistic initialization must be finite");
}

bool test_poisson_rejects_unsafe_link() {
  const int n = 4;
  const int d = 1;
  const double x[n * d] = {0.0, 0.0, 0.0, 0.0};
  const double y[n] = {0.0, 1.0, 2.0, 3.0};
  picasso::PoissonObjective objective(x, y, n, d, false, true);

  objective.set_model_coef(1000.0, -1);
  objective.update_auxiliary();
  return require(!std::isfinite(objective.eval()),
                 "unsafe Poisson link must produce a numerical failure") &&
         require(!finite_gradient(&objective),
                 "unsafe Poisson gradient must produce a numerical failure");
}

bool test_poisson_objective_gradient_consistency() {
  const int n = 5;
  const int d = 1;
  const double x[n * d] = {-1.5, -0.5, 0.0, 0.75, 1.25};
  const double y[n] = {0.0, 1.0, 2.0, 1.0, 3.0};
  picasso::PoissonObjective objective(x, y, n, d, true, true);
  const double beta = 0.23;
  Eigen::ArrayXd xb(n);
  for (int row = 0; row < n; ++row) xb[row] = beta * x[row];
  objective.set_model_coef(beta, 0);
  objective.set_model_Xb(xb);
  objective.update_auxiliary();
  objective.update_all_gradients();
  const double analytic = -objective.get_grad(0);

  const double step = 1e-6;
  Eigen::ArrayXd shifted_xb(n);
  objective.set_model_coef(beta + step, 0);
  for (int row = 0; row < n; ++row)
    shifted_xb[row] = (beta + step) * x[row];
  objective.set_model_Xb(shifted_xb);
  objective.update_auxiliary();
  const double upper = objective.eval();

  objective.set_model_coef(beta - step, 0);
  for (int row = 0; row < n; ++row)
    shifted_xb[row] = (beta - step) * x[row];
  objective.set_model_Xb(shifted_xb);
  objective.update_auxiliary();
  const double lower = objective.eval();
  const double numerical = (upper - lower) / (2.0 * step);
  return require(std::isfinite(analytic) && std::isfinite(numerical),
                 "ordinary Poisson derivative became non-finite") &&
         require(std::fabs(analytic - numerical) < 2e-9,
                 "Poisson objective and gradient are inconsistent");
}

bool test_tiny_coordinate_update_keeps_linear_predictor_consistent() {
  const int n = 4;
  const int d = 1;
  const double x[n * d] = {-1.0, -0.5, 0.5, 1.0};
  const double y[n] = {0.0, 0.0, 1.0, 1.0};
  picasso::LogisticObjective objective(x, y, n, d, false, true);

  // At the null model p=1/2, so a=sum(p(1-p)x^2)/n=0.15625.
  // Choose lambda so the unconstrained coordinate candidate moves by only
  // 5e-9, below the objective's accepted-update threshold.
  const double curvature = 0.15625;
  picasso::RegL1 regularizer;
  regularizer.set_param(
      objective.get_grad(0) - curvature * 5e-9, 0.0);
  const double coefficient = objective.coordinate_descent(&regularizer, 0);
  const Eigen::ArrayXd &xb = objective.get_model_Xb_ref();

  double maximum_error = 0.0;
  for (int row = 0; row < n; ++row)
    maximum_error =
        std::max(maximum_error, std::fabs(xb[row] - coefficient * x[row]));
  return require(maximum_error <= 1e-15,
                 "a rejected tiny GLM update desynchronized beta and Xb");
}

bool test_flat_zero_coordinates_select_zero() {
  const int n = 6;
  const int d = 1;
  const double x[n * d] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const double binary[n] = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
  const double counts[n] = {0.0, 1.0, 2.0, 0.0, 3.0, 1.0};
  const double continuous[n] = {-1.0, 0.5, 2.0, -0.25, 1.5, -0.75};
  picasso::RegL1 regularizer;
  regularizer.set_param(0.2, 0.0);

  picasso::LogisticObjective logistic(x, binary, n, d, true, true);
  picasso::PoissonObjective poisson(x, counts, n, d, true, true);
  picasso::PoissonObjective deferred_poisson(
      x, counts, n, d, true, true);
  picasso::SqrtMSEObjective sqrt_loss(
      x, continuous, n, d, true, true);

  return require(logistic.coordinate_descent(&regularizer, 0) == 0.0,
                 "flat logistic coordinate must select exact zero") &&
         require(poisson.coordinate_descent(&regularizer, 0) == 0.0,
                 "flat Poisson coordinate must select exact zero") &&
         require(deferred_poisson.coordinate_descent_deferred(
                     &regularizer, 0) == 0.0,
                 "flat deferred Poisson coordinate must select exact zero") &&
         require(sqrt_loss.coordinate_descent(&regularizer, 0) == 0.0,
                 "flat square-root-loss coordinate must select exact zero");
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_extreme_logistic_loss();
  ok &= test_logistic_mean_does_not_overflow();
  ok &= test_degenerate_logistic_initialization();
  ok &= test_poisson_rejects_unsafe_link();
  ok &= test_poisson_objective_gradient_consistency();
  ok &= test_tiny_coordinate_update_keeps_linear_predictor_consistent();
  ok &= test_flat_zero_coordinates_select_zero();
  return ok ? 0 : 1;
}
