#include <cassert>
#include <picasso/objective.hpp>

#include <stdexcept>

namespace picasso {
GaussianNaiveUpdateObjective::GaussianNaiveUpdateObjective(
    const double *xmat, const double *y, int n, int d, bool include_intercept, bool usePython)
    : GaussianNaiveUpdateObjective(
          xmat, y, n, d, include_intercept, usePython,
          detail::DesignStorage::kOwned) {}

GaussianNaiveUpdateObjective::GaussianNaiveUpdateObjective(
    const double *xmat, const double *y, int n, int d,
    bool include_intercept, bool usePython,
    detail::DesignStorage design_storage)
    : ObjFunction(xmat, y, n, d, usePython, design_storage),
      residual_mean(0.0),
      Ymean(0.0),
      m_include_intercept(include_intercept) {
  XX.resize(d);
  Xmean.resize(d);
  r.resize(n);

  Ymean = Y.sum() / n;
  if (!std::isfinite(Ymean))
    throw std::invalid_argument("Gaussian response must be finite");

  for (int j = 0; j < d; j++) {
    const auto xcol = design_column(j);
    Xmean[j] = xcol.sum() / n;
    XX[j] = m_include_intercept
                ? (xcol - Xmean[j]).square().sum() / n
                : xcol.square().sum() / n;
    if (!std::isfinite(Xmean[j]) || !std::isfinite(XX[j]))
      throw std::invalid_argument("Gaussian design must be finite");
  }

  update_auxiliary();
  intercept_update();

  // saturated fvalue = 0
  deviance = fabs(eval());
}

void GaussianNaiveUpdateObjective::intercept_update() {
  model_param.intercept = m_include_intercept ? residual_mean : 0.0;
}
double GaussianNaiveUpdateObjective::coordinate_descent(RegFunction *regfunc, int idx) {
  double beta_old = model_param.beta[idx];
  double tmp = gr[idx] + model_param.beta[idx] * XX[idx];
  model_param.beta[idx] =
      XX[idx] > 0.0 ? regfunc->coordinate_minimize(tmp, XX[idx]) : 0.0;
  const double delta = model_param.beta[idx] - beta_old;
  if (delta != 0.0) {
    const auto xcol = design_column(idx);
    if (m_include_intercept)
      r -= (xcol - Xmean[idx]) * delta;
    else
      r -= xcol * delta;
    residual_mean -= Xmean[idx] * delta;
  }
  return model_param.beta[idx];
}

void GaussianNaiveUpdateObjective::update_auxiliary() {
  if (m_include_intercept) {
    residual_mean = Ymean - (Xmean * model_param.beta).sum();
    r = Y - Ymean;
    for (int j = 0; j < d; ++j) {
      if (model_param.beta[j] != 0.0)
        r -= (design_column(j) - Xmean[j]) * model_param.beta[j];
    }
    for (int j = 0; j < d; ++j) {
      gr[j] = ((design_column(j) - Xmean[j]) * r).sum() / n;
    }
  } else {
    r = Y - (design_matrix() * model_param.beta.matrix()).array();
    residual_mean = r.sum() / n;
    gr = (design_matrix().transpose() * r.matrix()).array() / n;
  }
}

void GaussianNaiveUpdateObjective::update_gradient(int idx) {
  gr[idx] = m_include_intercept
                ? ((design_column(idx) - Xmean[idx]) * r).sum() / n
                : (r * design_column(idx)).sum() / n;
}

double GaussianNaiveUpdateObjective::get_local_change(double old, int idx) {
  assert(idx >= 0);
  double tmp = old - model_param.beta[idx];
  return tmp * tmp * XX[idx];
}

double GaussianNaiveUpdateObjective::eval() {
  if (m_include_intercept) {
    // r is the residual at the profiled intercept. Account for a caller that
    // has explicitly set a different intercept without sacrificing stability.
    return (r + residual_mean - model_param.intercept).square().sum() / n;
  }
  return (r - model_param.intercept).square().sum() / n;
}

}  // namespace picasso
