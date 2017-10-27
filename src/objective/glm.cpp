#include <picasso/objective.hpp>

namespace picasso {
GLMObjective::GLMObjective(const double *xmat, const double *y, int n, int d)
    : ObjFunction(xmat, y, n, d) {
  a = 0.0;
  g = 0.0;

  p.resize(n);
  w.resize(n);

  r.resize(n);
  wXX.resize(d);
}

GLMObjective::GLMObjective(const double *xmat, const double *y, int n, int d,
                           bool include_intercept)
    : ObjFunction(xmat, y, n, d) {
  a = 0.0;
  g = 0.0;

  p.resize(n);
  w.resize(n);
  r.resize(n);

  wXX.resize(d);

  if (include_intercept) {
    double avr_y = 0.0;
    for (int i = 0; i < n; i++) {
      avr_y += Y[i];
    }
    avr_y = avr_y / n;
    model_param.intercept = log(avr_y / (1 - avr_y));
  }
}

double GLMObjective::coordinate_descent(RegFunction *regfunc, int idx) {
  g = 0.0;
  a = 0.0;

  double tmp;
  // g = (<wXX, model_param.beta> + <r, X>)/n
  // a = sum(wXX)/n
  for (int i = 0; i < n; i++) {
    tmp = w[i] * X[idx][i] * X[idx][i];
    g += tmp * model_param.beta[idx] + r[i] * X[idx][i];
    a += tmp;
  }
  g = g / n;
  a = a / n;

  tmp = model_param.beta[idx];
  model_param.beta[idx] = regfunc->threshold(g) / a;

  tmp = model_param.beta[idx] - tmp;
  if (fabs(tmp) > 1e-8) {
    // Xb += delta*X[idx*n]
    for (int i = 0; i < n; i++) Xb[i] = Xb[i] + tmp * X[idx][i];
    // sum_r = 0.0;
    // r -= delta*w*X
    for (int i = 0; i < n; i++) {
      r[i] = r[i] - w[i] * X[idx][i] * tmp;
      // sum_r += r[i];
    }
  }
  return (model_param.beta[idx]);
}

void GLMObjective::intercept_update() {
  sum_r = 0.0;
  for (int i = 0; i < n; i++) sum_r += r[i];
  double tmp = sum_r / sum_w;
  model_param.intercept += tmp;

  // sum_r = 0.0;
  for (int i = 0; i < n; i++) {
    r[i] = r[i] - tmp * w[i];
    // sum_r += r[i];
  }
}

/*
void GLMObjective::set_model_param(ModelParam &other_param,
                                   const std::vector<double> &old_Xb) {
  model_param = other_param;


  for (int i = 0; i < n; i++) {
    Xb[i] = 0.0;
    for (int j = 0; j < d; j++) Xb[i] += X[j][i] * model_param.beta[j];
  }
  for (int i = 0; i < n; i++) Xb[i] = old_Xb[i];
}
*/

void GLMObjective::update_auxiliary() {
  update_key_aux();
  sum_w = 0.0;
  for (int i = 0; i < n; i++) {
    r[i] = Y[i] - p[i];
    sum_w += w[i];
  }

  /*
    for (int idx = 0; idx < d; idx++) {
      wXX[idx] = 0.0;
      for (int i = 0; i < n; i++) wXX[idx] += w[i] * X[idx][i] * X[idx][i];
    }*/
}

void GLMObjective::update_gradient(int idx) {
  gr[idx] = 0.0;
  for (int i = 0; i < n; i++) gr[idx] += (Y[i] - p[i]) * X[idx][i] / n;
}

double GLMObjective::get_local_change(double old, int idx) {
  if (idx >= 0) {
    double tmp = old - model_param.beta[idx];
    double wXX_idx = 0.0;
    for (int i = 0; i < n; i++) wXX_idx += w[i] * X[idx][i] * X[idx][i];
    return (wXX_idx * tmp * tmp / (2 * n));
  } else {
    double tmp = old - model_param.intercept;
    return (sum_w * tmp * tmp / (2 * n));
  }
}

LogisticObjective::LogisticObjective(const double *xmat, const double *y, int n,
                                     int d)
    : GLMObjective(xmat, y, n, d) {
  update_auxiliary();

  for (int i = 0; i < d; i++) update_gradient(i);
  model_param.intercept = 0.0;

  deviance = fabs(eval());
};

LogisticObjective::LogisticObjective(const double *xmat, const double *y, int n,
                                     int d, bool include_intercept)
    : GLMObjective(xmat, y, n, d, include_intercept) {
  update_auxiliary();
  for (int i = 0; i < d; i++) update_gradient(i);

  model_param.intercept = 0.0;
  update_auxiliary();

  deviance = fabs(eval());
};

void LogisticObjective::update_key_aux() {
  for (int i = 0; i < n; i++) {
    p[i] = 1.0 / (1.0 + exp(-model_param.intercept - Xb[i]));
    w[i] = p[i] * (1 - p[i]);
  }
}

double LogisticObjective::eval() {
  double v = 0.0;
  for (int i = 0; i < n; i++) v -= Y[i] * (model_param.intercept + Xb[i]);

  for (int i = 0; i < n; i++)
    if (p[i] > 1e-8) v -= (log(p[i]) - model_param.intercept - Xb[i]);

  return (v / n);
}

PoissonObjective::PoissonObjective(const double *xmat, const double *y, int n,
                                   int d)
    : GLMObjective(xmat, y, n, d) {
  update_auxiliary();

  for (int i = 0; i < d; i++) update_gradient(i);
  model_param.intercept = 0.0;

  deviance = fabs(eval());
};

PoissonObjective::PoissonObjective(const double *xmat, const double *y, int n,
                                   int d, bool include_intercept)
    : GLMObjective(xmat, y, n, d, include_intercept) {
  update_auxiliary();
  for (int i = 0; i < d; i++) update_gradient(i);

  model_param.intercept = 0.0;
  update_auxiliary();

  deviance = fabs(eval());
};

void PoissonObjective::update_key_aux() {
  for (int i = 0; i < n; i++) {
    p[i] = exp(model_param.intercept + Xb[i]);
    w[i] = p[i];
  }
}

double PoissonObjective::eval() {
  double v = 0.0;
  for (int i = 0; i < n; i++)
    v = v + p[i] - Y[i] * (model_param.intercept + Xb[i]);
  return (v / n);
}

}  // namespace picasso