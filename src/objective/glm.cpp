#include <picasso/objective.hpp>

namespace picasso {

GLMObjective::GLMObjective(const double *xmat, const double *y, int n, int d,
                           bool include_intercept, bool usePypthon)
    : ObjFunction(xmat, y, n, d, usePypthon) {
  a = 0.0;
  g = 0.0;

  p.resize(n);
  w.resize(n);
  r.resize(n);

  wXX.resize(d);

  if (include_intercept) {
    double avr_y = Y.sum() / n;

    model_param.intercept = log(avr_y / (1 - avr_y));
  }
}

double GLMObjective::coordinate_descent(RegFunction *regfunc, int idx) {
  g = 0.0;
  a = 0.0;

  // g = (<wXX, model_param.beta> + <r, X>)/n
  // a = sum(wXX)/n
  Eigen::ArrayXd wXX = w * X.col(idx) * X.col(idx);
  a = wXX.sum() / n;

  g = (model_param.beta[idx] * wXX + r * X.col(idx)).sum()/n;

  double tmp;
  tmp = model_param.beta[idx];
  model_param.beta[idx] = regfunc->threshold(g) / a;

  tmp = model_param.beta[idx] - tmp;
  if (fabs(tmp) > 1e-8) {
    // Xb += delta*X[idx*n]
    Xb = Xb + tmp * X.col(idx);

    // r -= delta*w*X
    r = r - tmp * w * X.col(idx);
  }
  return (model_param.beta[idx]);
}

void GLMObjective::intercept_update() {
  double sum_r = r.sum();
  model_param.intercept += sum_r/sum_w;
  r = r - sum_r/sum_w * w;
  sum_r = 0;
}

void GLMObjective::update_gradient(int idx) {
  Eigen::ArrayXd tmp = (Y - p) * X.col(idx) / n;
  gr[idx] = tmp.sum();
}

double GLMObjective::get_local_change(double old, int idx) {
  if (idx >= 0) {
    double tmp = old - model_param.beta[idx];
    return ((w*X.col(idx)*X.col(idx)).sum() * tmp * tmp / (2 * n));
  } else {
    double tmp = old - model_param.intercept;
    return (sum_w * tmp * tmp / (2 * n));
  }
}

LogisticObjective::LogisticObjective(const double *xmat, const double *y, int n,
                                     int d, bool include_intercept, bool usePypthon)
    : GLMObjective(xmat, y, n, d, include_intercept, usePypthon) {
  update_auxiliary();
  for (int i = 0; i < d; i++) update_gradient(i);

  model_param.intercept = 0.0;
  update_auxiliary();

  deviance = fabs(eval());
};

void LogisticObjective::update_auxiliary() {
  p = -model_param.intercept - Xb;
  p = p.exp();
  p = 1.0 / (1.0 + p);
  r = Y - p;

  w = p * (1 - p);
  sum_w = w.sum();
}

double LogisticObjective::eval() {
  double v = 0.0;
  for (int i = 0; i < n; i++) v -= Y[i] * (model_param.intercept + Xb[i]);

  for (int i = 0; i < n; i++)
    if (p[i] > 1e-8) v -= (log(p[i]) - model_param.intercept - Xb[i]);

  return (v / n);
}

PoissonObjective::PoissonObjective(const double *xmat, const double *y, int n,
                                   int d, bool include_intercept, bool usePypthon)
    : GLMObjective(xmat, y, n, d, include_intercept, usePypthon) {
  update_auxiliary();
  for (int i = 0; i < d; i++) update_gradient(i);

  model_param.intercept = 0.0;
  update_auxiliary();

  deviance = fabs(eval());
};

void PoissonObjective::update_auxiliary() {
  p = model_param.intercept + Xb;
  p = p.exp();
  r = Y - p;
  w = p;
  sum_w = w.sum();
}

double PoissonObjective::eval() {
  double v = 0.0;
  for (int i = 0; i < n; i++)
    v = v + p[i] - Y[i] * (model_param.intercept + Xb[i]);
  return (v / n);
}

}  // namespace picasso
