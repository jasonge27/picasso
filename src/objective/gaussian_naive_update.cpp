#include <cassert>
#include <picasso/objective.hpp>

namespace picasso {

GaussianNaiveUpdateObjective::GaussianNaiveUpdateObjective(const double *xmat,
                                                           const double *y,
                                                           int n, int d)
    : ObjFunction(xmat, y, n, d) {
  XX.resize(n);
  r.resize(n);

  for (int j = 0; j < d; j++) {
    XX[j] = 0.0;
    for (int i = 0; i < n; i++) XX[j] += X[j][i] * X[j][i];
    XX[j] = XX[j] / n;
  }

  for (int i = 0; i < n; i++) r[i] = Y[i];
  update_auxiliary();

  // saturated fvalue = 0
  deviance = fabs(eval());
}

GaussianNaiveUpdateObjective::GaussianNaiveUpdateObjective(
    const double *xmat, const double *y, int n, int d, bool include_intercept)
    : ObjFunction(xmat, y, n, d) {
  XX.resize(n);
  r.resize(n);

  if (include_intercept) {
    double avr_y = 0.0;
    for (int i = 0; i < n; i++) avr_y += Y[i];
    avr_y = avr_y / n;
    model_param.intercept = avr_y;
  }

  for (int j = 0; j < d; j++) {
    XX[j] = 0.0;
    for (int i = 0; i < n; i++) XX[j] += X[j][i] * X[j][i];
    XX[j] = XX[j] / n;
  }

  for (int i = 0; i < n; i++) r[i] = Y[i];
  update_auxiliary();

  // saturated fvalue = 0
  deviance = fabs(eval());
}

double GaussianNaiveUpdateObjective::coordinate_descent(RegFunction *regfunc,
                                                        int idx) {
  double beta_old = model_param.beta[idx];
  double tmp = gr[idx] + model_param.beta[idx] * XX[idx];
  model_param.beta[idx] = regfunc->threshold(tmp) / XX[idx];

  for (int i = 0; i < n; i++)
    r[i] = r[i] - X[idx][i] * (model_param.beta[idx] - beta_old);

  return model_param.beta[idx];
}

void GaussianNaiveUpdateObjective::intercept_update() {
  double sum_r = 0.0;
  for (int i = 0; i < n; i++) sum_r += r[i];
  model_param.intercept = sum_r / n;
}

void GaussianNaiveUpdateObjective::set_model_param(ModelParam &other_param) {
  model_param = other_param;
}

void GaussianNaiveUpdateObjective::update_auxiliary() {
  for (int idx = 0; idx < d; idx++) {
    gr[idx] = 0.0;
    for (int i = 0; i < n; i++) gr[idx] += r[i] * X[idx][i] / n;
  }
}

void GaussianNaiveUpdateObjective::update_gradient(int idx) {
  gr[idx] = 0.0;
  for (int i = 0; i < n; i++) gr[idx] += r[i] * X[idx][i] / n;
}

double GaussianNaiveUpdateObjective::get_local_change(double old, int idx) {
  assert(idx >= 0);
  double tmp = old - model_param.beta[idx];
  return tmp * tmp * XX[idx];
}

double GaussianNaiveUpdateObjective::eval() {
  double v = 0.0;
  for (int i = 0; i < n; i++) {
    double pred = model_param.intercept;
    for (int j = 0; j < d; j++) {
      pred += model_param.beta[j] * X[j][i];
    }
    v += (Y[i] - pred) * (Y[i] - pred);
  }
  v = v / n;
  return v;
}

}  // namespace picasso