#include <cassert>
#include <picasso/objective.hpp>

namespace picasso {
GaussianNaiveUpdateObjective::GaussianNaiveUpdateObjective(
    const double *xmat, const double *y, int n, int d, bool include_intercept, bool usePython)
    : ObjFunction(xmat, y, n, d, usePython) {
  XX.resize(d);
  r.resize(n);

  if (include_intercept) {
    double avr_y = Y.sum()/n;
    model_param.intercept = avr_y;
  }

  for (int j = 0; j < d; j++)
    XX[j] = (X.col(j)*X.col(j)).sum()/n;

  r = Y;
  update_auxiliary();

  // saturated fvalue = 0
  deviance = fabs(eval());
}

void GaussianNaiveUpdateObjective::intercept_update() {
  double sum_r = r.sum();
  model_param.intercept = sum_r / n;
}
double GaussianNaiveUpdateObjective::coordinate_descent(RegFunction *regfunc, int idx) {
  double beta_old = model_param.beta[idx];
  double tmp = gr[idx] + model_param.beta[idx] * XX[idx];
  model_param.beta[idx] = regfunc->threshold(tmp) / XX[idx];
  const double delta = model_param.beta[idx] - beta_old;
  if (delta != 0.0) {
    r = r - X.col(idx) * delta;
  }
  return model_param.beta[idx];
}

void GaussianNaiveUpdateObjective::update_auxiliary() {
  gr = (X.matrix().transpose() * r.matrix()).array() / n;
}

void GaussianNaiveUpdateObjective::update_gradient(int idx) {
  gr[idx] = (r*X.col(idx)).sum()/n;
}

double GaussianNaiveUpdateObjective::get_local_change(double old, int idx) {
  assert(idx >= 0);
  double tmp = old - model_param.beta[idx];
  return tmp * tmp * XX[idx];
}

double GaussianNaiveUpdateObjective::eval() {
  // r = Y - X*beta is maintained incrementally by coordinate_descent(),
  // but does not account for the intercept. The true residual is
  // Y - X*beta - intercept = r - intercept.
  return (r - model_param.intercept).square().sum() / n;
}

}  // namespace picasso
