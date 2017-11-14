#include <picasso/objective.hpp>

namespace picasso {
SqrtMSEObjective::SqrtMSEObjective(const double *xmat, const double *y, int n,
                                   int d, bool include_intercept, bool usePypthon)
    : ObjFunction(xmat, y, n, d, usePypthon) {
  a = 0.0;
  g = 0.0;
  L = 0.0;
  Xb.resize(n);
  Xb.setZero();

  r.resize(n);
  r.setZero();

  if (include_intercept) {
    double avr_y = Y.sum() / n;
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

  sum_r2 = r.matrix().dot(r.matrix());
  L = sqrt(sum_r2 / n);

  Eigen::ArrayXd wXX  = (1 - r*r/sum_r2) * X.col(idx) * X.col(idx);
  g = (wXX * model_param.beta[idx] + r * X.col(idx)).sum()/(n*L);
  a = wXX.sum()/(n*L);

  tmp = model_param.beta[idx];
  model_param.beta[idx] = regfunc->threshold(g) / a;

  tmp = model_param.beta[idx] - tmp;
  // Xb += delta*X[idx*n]
  Xb = Xb + tmp * X.col(idx);

  sum_r = 0.0;
  sum_r2 = 0.0;
  // r -= delta*X
  r = r - tmp * X.col(idx);
  sum_r = r.sum();

  sum_r2 = r.matrix().dot(r.matrix());
  L = sqrt(sum_r2 / n);

  return (model_param.beta[idx]);
}

void SqrtMSEObjective::intercept_update() {
  double tmp = sum_r / n;
  model_param.intercept += tmp;

  r = r - tmp;
  sum_r = 0.0;
  sum_r2 = r.matrix().dot(r.matrix());
  L = sqrt(sum_r2 / n);
}


void SqrtMSEObjective::update_auxiliary() {
  sum_r = 0.0;
  sum_r2 = 0.0;
  r = Y - Xb - model_param.intercept;
  sum_r = r.sum();
  sum_r2 = r.matrix().dot(r.matrix());
  L = sqrt(sum_r2 / n);
}

void SqrtMSEObjective::update_gradient(int idx) {
  gr[idx] = (r * X.col(idx)).sum() / (n*L);
}

double SqrtMSEObjective::get_local_change(double old, int idx) {
  if (idx >= 0) {
    double a =  (X.col(idx) * X.col(idx) * (1 - r * r/(L*L*n))).sum()/(n*L);
    double tmp = old - model_param.beta[idx];
    return (a * tmp * tmp / (2 * L * n));
  } else {
    double tmp = old - model_param.intercept;
    return (fabs(tmp));
  }
}

double SqrtMSEObjective::eval() { return (L); }

};  // namespace picasso
