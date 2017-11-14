#include <cassert>
#include <picasso/objective.hpp>

namespace picasso {
GaussianNaiveUpdateObjective::GaussianNaiveUpdateObjective(
    const double *xmat, const double *y, int n, int d, bool include_intercept, bool usePypthon)
    : ObjFunction(xmat, y, n, d, usePypthon) {
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

double GaussianNaiveUpdateObjective::coordinate_descent(RegFunction *regfunc,
                                                        int idx) {
  double beta_old = model_param.beta[idx];
  double tmp = gr[idx] + model_param.beta[idx] * XX[idx];
  model_param.beta[idx] = regfunc->threshold(tmp) / XX[idx];

  r = r - X.col(idx) * (model_param.beta[idx] - beta_old);
  return model_param.beta[idx];
}

void GaussianNaiveUpdateObjective::intercept_update() {
  double sum_r = r.sum();
  model_param.intercept = sum_r / n;
}
void GaussianNaiveUpdateObjective::update_auxiliary() {
  for (int idx = 0; idx < d; idx++)
    update_gradient(idx);
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
  double v = 0.0;
  for (int i = 0; i < n; i++) {
    double pred = model_param.intercept + model_param.beta.matrix().dot(X.row(i).matrix());
    v += (Y[i] - pred) * (Y[i] - pred);
  }
  v = v / n;
  return v;
}

}  // namespace picasso
