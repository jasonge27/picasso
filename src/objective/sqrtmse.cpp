#include <picasso/objective.hpp>

namespace picasso {
SqrtMSEObjective::SqrtMSEObjective(const double *xmat, const double *y, int n,
                                   int d, bool include_intercept, bool usePython)
    : ObjFunction(xmat, y, n, d, usePython) {
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
  const auto xcol = X.col(idx);

  double tmp0 = (xcol * xcol).sum();
  double tmp1 = (r * xcol).sum();
  double tmp2 = (xcol * xcol * r * r).sum();

  a = (tmp0 / n - tmp2 / (n * sum_r2)) / L;
  g = tmp1 / (n * L) + a * model_param.beta[idx];

  double old_beta = model_param.beta[idx];
  model_param.beta[idx] = regfunc->threshold(g) / a;
  double delta = model_param.beta[idx] - old_beta;

  if (fabs(delta) > 1e-8) {
    // Incremental update: r_new = r_old - delta * xcol
    // sum_r_new = sum_r_old - delta * sum(xcol)
    // sum_r2_new = sum_r2_old - 2*delta*(r_old·xcol) + delta^2*(xcol·xcol)
    //            = sum_r2_old - 2*delta*tmp1 + delta^2*tmp0
    sum_r -= delta * xcol.sum();
    sum_r2 += -2.0 * delta * tmp1 + delta * delta * tmp0;
    if (sum_r2 < 0.0) sum_r2 = 0.0;  // guard against negative from rounding
    L = sqrt(sum_r2 / n);

    r = r - delta * xcol;
    Xb = Xb + delta * xcol;
  }
  return (model_param.beta[idx]);
}

void SqrtMSEObjective::intercept_update() {
  double tmp = sum_r / n;
  model_param.intercept += tmp;

  r = r - tmp;
  // sum_r2_new = sum_r2 - 2*tmp*sum_r + n*tmp^2 = sum_r2 - sum_r^2/n
  sum_r2 -= sum_r * sum_r / n;
  if (sum_r2 < 0.0) sum_r2 = 0.0;
  sum_r = 0.0;
  L = sqrt(sum_r2 / n);
}


void SqrtMSEObjective::update_auxiliary() {
  sum_r = 0.0;
  sum_r2 = 0.0;
  r = Y - Xb - model_param.intercept;
  sum_r = r.sum();
  sum_r2 = r.square().sum();
  L = sqrt(sum_r2 / n);
}

void SqrtMSEObjective::update_gradient(int idx) {
  gr[idx] = (r * X.col(idx)).sum() / (n*L);
}

double SqrtMSEObjective::get_local_change(double old, int idx) {
  if (idx >= 0) {
    const auto xcol = X.col(idx);
    double a =  (xcol * xcol * (1 - r * r/(L*L*n))).sum()/(n*L);
    double tmp = old - model_param.beta[idx];
    return (a * tmp * tmp / (2 * L * n));
  } else {
    double tmp = old - model_param.intercept;
    return (fabs(tmp));
  }
}

double SqrtMSEObjective::eval() { return (L); }

};  // namespace picasso
