#include <cassert>
#include <picasso/objective.hpp>

#include <stdexcept>

namespace picasso {

GaussianCovUpdateObjective::GaussianCovUpdateObjective(
    const double *xmat, const double *y, int n, int d, bool include_intercept,
    bool usePython)
    : GaussianCovUpdateObjective(
          xmat, y, n, d, include_intercept, usePython,
          detail::DesignStorage::kOwned) {}

GaussianCovUpdateObjective::GaussianCovUpdateObjective(
    const double *xmat, const double *y, int n, int d, bool include_intercept,
    bool usePython, detail::DesignStorage design_storage)
    : ObjFunction(xmat, y, n, d, usePython, design_storage),
      cached_column_count(0),
      Ymean(0.0),
      m_include_intercept(include_intercept) {
  XX.resize(d);
  Xy.resize(d);
  Xmean.resize(d);
  C_columns.resize(d);
  C_valid.assign(d, 0);

  for (int j = 0; j < d; j++) {
    Xmean[j] = design_column(j).sum() / n;
    if (!std::isfinite(Xmean[j]))
      throw std::invalid_argument("Gaussian design must be finite");
  }

  Ymean = Y.sum() / n;
  if (!std::isfinite(Ymean))
    throw std::invalid_argument("Gaussian response must be finite");

  // Compute only the diagonal and X^T Y at construction. Off-diagonal Gram
  // columns are materialized only when a coordinate first changes.
  if (m_include_intercept) {
    const Eigen::ArrayXd centered_y = Y - Ymean;
    for (int j = 0; j < d; ++j) {
      const Eigen::ArrayXd centered_x = design_column(j) - Xmean[j];
      XX[j] = std::max(0.0, centered_x.square().sum() / n);
      if (!std::isfinite(XX[j]))
        throw std::invalid_argument("Gaussian design must be finite");
      Xy[j] = (centered_x * centered_y).sum() / n;
    }
  } else {
    for (int j = 0; j < d; ++j) {
      XX[j] = std::max(0.0, design_column(j).square().sum() / n);
      if (!std::isfinite(XX[j]))
        throw std::invalid_argument("Gaussian design must be finite");
    }
    Xy = (design_matrix().transpose() * Y.matrix()).array() / n;
  }

  // Initial profiled gradient (beta = 0).
  gr = Xy;
  intercept_update();

  // saturated fvalue = 0
  deviance = fabs(eval());
}

const Eigen::ArrayXd &GaussianCovUpdateObjective::covariance_column(int idx) {
  if (C_valid[idx]) return C_columns[idx];

  Eigen::ArrayXd &column = C_columns[idx];
  column.resize(d);
  Eigen::ArrayXd centered_idx = design_column(idx);
  if (m_include_intercept) centered_idx -= Xmean[idx];
  for (int j = 0; j < d; ++j) {
    if (j == idx) {
      column[j] = XX[j];
    } else if (C_valid[j]) {
      // Reuse an already-computed symmetric entry.
      column[j] = C_columns[j][idx];
    } else if (m_include_intercept) {
      column[j] =
          ((design_column(j) - Xmean[j]) * centered_idx).sum() / n;
    } else {
      column[j] = (design_column(j) * centered_idx).sum() / n;
    }
  }
  C_valid[idx] = 1;
  ++cached_column_count;
  return column;
}

double GaussianCovUpdateObjective::coordinate_descent(RegFunction *regfunc, int idx) {
  const double beta_old = model_param.beta[idx];
  const double tmp = gr[idx] + beta_old * XX[idx];
  const double beta_new =
      XX[idx] > 0.0 ? regfunc->coordinate_minimize(tmp, XX[idx]) : 0.0;
  const double delta = beta_new - beta_old;
  if (delta != 0.0) {
    const Eigen::ArrayXd &column = covariance_column(idx);
    model_param.beta[idx] = beta_new;
    gr -= delta * column;
  } else {
    model_param.beta[idx] = beta_new;
  }
  return model_param.beta[idx];
}

void GaussianCovUpdateObjective::intercept_update() {
  // intercept = mean(Y - X beta) = Ymean - Xmean^T beta
  model_param.intercept =
      m_include_intercept ? Ymean - (Xmean * model_param.beta).sum() : 0.0;
}

void GaussianCovUpdateObjective::update_auxiliary() {
  // Recompute directly from X without forcing any Gram columns into cache.
  const Eigen::ArrayXd residual =
      Y - (design_matrix() * model_param.beta.matrix()).array();
  if (m_include_intercept) {
    const Eigen::ArrayXd centered_residual = residual - residual.sum() / n;
    for (int j = 0; j < d; ++j)
      gr[j] =
          ((design_column(j) - Xmean[j]) * centered_residual).sum() / n;
  } else {
    gr = (design_matrix().transpose() * residual.matrix()).array() / n;
  }
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
  // X is column-major.  Reuse the objective's existing predictor buffer so
  // Eigen can evaluate one cache-friendly GEMV without an additional n-vector.
  Xb = (design_matrix() * model_param.beta.matrix()).array();
  return (Y - Xb - model_param.intercept).square().sum() / n;
}

double GaussianCovUpdateObjective::path_eval() {
  // For the profiled Gaussian covariance state,
  //   gr = Xy - C * beta
  // and MSE = null_MSE - beta' * (Xy + gr).
  // Neumaier compensation reduces summation error. Near interpolation the
  // subtraction is ill-conditioned, so retain the direct residual oracle.
  double sum = deviance;
  double correction = 0.0;
  double scale = std::fabs(deviance);
  for (int feature = 0; feature < d; ++feature) {
    const double term =
        -model_param.beta[feature] * (Xy[feature] + gr[feature]);
    if (!std::isfinite(term)) {
      return eval();
    }
    scale += std::fabs(term);
    const double updated = sum + term;
    if (std::fabs(sum) >= std::fabs(term))
      correction += (sum - updated) + term;
    else
      correction += (term - updated) + sum;
    sum = updated;
  }

  const double candidate = sum + correction;
  const bool cancellation =
      scale > 0.0 && candidate <= 1e-4 * scale;
  if (!std::isfinite(candidate) || candidate < 0.0 || cancellation) {
    return eval();
  }
  return candidate;
}

}  // namespace picasso
