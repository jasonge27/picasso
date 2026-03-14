#include <cassert>
#include <picasso/objective.hpp>

namespace picasso {

GaussianCovUpdateObjective::GaussianCovUpdateObjective(
    const double *xmat, const double *y, int n, int d, bool include_intercept,
    bool usePython)
    : ObjFunction(xmat, y, n, d, usePython) {
  XX.resize(d);
  Xy.resize(d);
  Xmean.resize(d);

  for (int j = 0; j < d; j++) Xmean[j] = X.col(j).sum() / n;

  Ymean = Y.sum() / n;

  if (include_intercept) model_param.intercept = Ymean;

  // Gram matrix X^T X / n (d x d)
  C.noalias() = X.matrix().transpose() * X.matrix();
  C /= n;

  for (int j = 0; j < d; j++) XX[j] = C(j, j);

  // X^T Y / n
  Xy = (X.matrix().transpose() * Y.matrix()).array() / n;

  // Initial gradient (beta = 0): gr = X^T Y / n
  gr = Xy;

  // saturated fvalue = 0
  deviance = fabs(eval());
}

double GaussianCovUpdateObjective::coordinate_descent(RegFunction *regfunc, int idx) {
  double beta_old = model_param.beta[idx];
  double tmp = gr[idx] + beta_old * XX[idx];
  model_param.beta[idx] = regfunc->threshold(tmp) / XX[idx];
  double delta = model_param.beta[idx] - beta_old;
  if (delta != 0.0) {
    gr -= delta * C.col(idx).array();
  }
  return model_param.beta[idx];
}

void GaussianCovUpdateObjective::intercept_update() {
  // intercept = mean(Y - X beta) = Ymean - Xmean^T beta
  model_param.intercept = Ymean - (Xmean * model_param.beta).sum();
}

void GaussianCovUpdateObjective::update_auxiliary() {
  // Full recomputation: gr = X^T(Y - X beta) / n = Xy - C beta
  gr = Xy - (C * model_param.beta.matrix()).array();
}

void GaussianCovUpdateObjective::update_gradient(int /*idx*/) {
  // No-op: gradients are maintained incrementally by coordinate_descent
}

double GaussianCovUpdateObjective::get_local_change(double old, int idx) {
  assert(idx >= 0);
  double tmp = old - model_param.beta[idx];
  return tmp * tmp * XX[idx];
}

double GaussianCovUpdateObjective::eval() {
  double v = 0.0;
  for (int i = 0; i < n; i++) {
    double pred =
        model_param.intercept + model_param.beta.matrix().dot(X.row(i).matrix());
    v += (Y[i] - pred) * (Y[i] - pred);
  }
  return v / n;
}

}  // namespace picasso
