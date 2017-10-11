#include <picasso/objective.hpp>

namespace picasso {
SqrtMSEObjective::SqrtMSEObjective(const double *xmat, const double *y, int n,
                                   int d)
    : ObjFunction(xmat, y, n, d) {
  a = 0.0;
  g = 0.0;
  L = 0.0;

  Xb.resize(n, 0);
  r.resize(n, 0);

  update_auxiliary();

  for (int i = 0; i < d; i++) update_gradient(i);

  // saturated fvalue = 0
  deviance = fabs(eval());
};

SqrtMSEObjective::SqrtMSEObjective(const double *xmat, const double *y, int n,
                                   int d, bool include_intercept)
    : ObjFunction(xmat, y, n, d) {
  a = 0.0;
  g = 0.0;
  L = 0.0;
  Xb.resize(n, 0);
  r.resize(n, 0);

  if (include_intercept) {
    double avr_y = 0.0;
    for (int i = 0; i < n; i++) avr_y += Y[i];
    avr_y = avr_y / n;
    model_param.intercept = avr_y;
  }

  update_auxiliary();

  for (int i = 0; i < d; i++) update_gradient(i);

  deviance = fabs(eval());
};

double SqrtMSEObjective::coordinate_descent(RegFunction *regfunc, int idx) {
  g = 0.0;
  a = 0.0;

  double tmp;
  sum_r2 = 0.0;
  for (int i = 0; i < n; i++) sum_r2 += r[i] * r[i];
  L = sqrt(sum_r2 / n);

  for (int i = 0; i < n; i++) {
    tmp = (1 - r[i] * r[i] / sum_r2) * X[idx][i] * X[idx][i];
    g += tmp * model_param.beta[idx] + r[i] * X[idx][i];
    a += tmp;
  }
  g = g / (n * L);
  a = a / (n * L);

  tmp = model_param.beta[idx];
  model_param.beta[idx] = regfunc->threshold(g) / a;

  tmp = model_param.beta[idx] - tmp;
  // Xb += delta*X[idx*n]
  for (int i = 0; i < n; i++) Xb[i] = Xb[i] + tmp * X[idx][i];

  sum_r = 0.0;
  sum_r2 = 0.0;
  // r -= delta*X
  for (int i = 0; i < n; i++) {
    r[i] = r[i] - X[idx][i] * tmp;
    sum_r += r[i];
    sum_r2 += r[i] * r[i];
  }
  L = sqrt(sum_r2 / n);

  return (model_param.beta[idx]);
}

void SqrtMSEObjective::intercept_update() {
  double tmp = sum_r / n;
  model_param.intercept += tmp;

  sum_r2 = 0.0;
  for (int i = 0; i < n; i++) {
    r[i] = r[i] - tmp;
    sum_r2 += r[i] * r[i];
  }
  sum_r = 0.0;
  L = sqrt(sum_r2 / n);
}

void SqrtMSEObjective::set_model_param(ModelParam &other_param) {
  model_param = other_param;
  for (int i = 0; i < n; i++) {
    Xb[i] = 0.0;
    for (int j = 0; j < d; j++) Xb[i] += X[j][i] * model_param.beta[j];
  }
}

void SqrtMSEObjective::update_auxiliary() {
  sum_r = 0.0;
  sum_r2 = 0.0;
  for (int i = 0; i < n; i++) {
    r[i] = Y[i] - Xb[i] - model_param.intercept;
    sum_r += r[i];
    sum_r2 += r[i] * r[i];
  }
  L = sqrt(sum_r2 / n);
}

void SqrtMSEObjective::update_gradient(int idx) {
  gr[idx] = 0.0;
  for (int i = 0; i < n; i++) gr[idx] += r[i] * X[idx][i];
  gr[idx] = gr[idx] / (n * L);
}

double SqrtMSEObjective::get_local_change(double old, int idx) {
  if (idx >= 0) {
    double a = 0.0;
    double tmp = 0.0;
    for (int i = 0; i < n; i++) {
      tmp = r[i] / L;
      tmp = tmp * tmp / n;
      a += X[idx][i] * X[idx][i] * (1 - tmp);
    }

    tmp = old - model_param.beta[idx];
    a = a / (n * L);
    return (a * tmp * tmp / (2 * L * n));
  } else {
    double tmp = old - model_param.intercept;
    return (fabs(tmp));
  }
}

double SqrtMSEObjective::eval() { return (L); }

};  // namespace picasso