#include <picasso/objective.hpp>

#include <limits>

namespace picasso {
namespace {

bool is_exact_null_fit(double loss, const ModelParam &model) {
  return loss == 0.0 && (model.beta == 0.0).all();
}

}  // namespace

SqrtMSEObjective::SqrtMSEObjective(const double *xmat, const double *y, int n,
                                   int d, bool include_intercept, bool usePython)
    : SqrtMSEObjective(xmat, y, n, d, include_intercept, usePython,
                       detail::DesignStorage::kOwned) {}

SqrtMSEObjective::SqrtMSEObjective(
    const double *xmat, const double *y, int n, int d,
    bool include_intercept, bool usePython,
    detail::DesignStorage design_storage)
    : ObjFunction(xmat, y, n, d, usePython, design_storage) {
  a = 0.0;
  g = 0.0;
  L = 0.0;
  Xb.resize(n);
  Xb.setZero();

  r.resize(n);
  r.setZero();

  XX.resize(d);
  Xsum.resize(d);
  for (int feature = 0; feature < d; ++feature) {
    const auto xcol = design_column(feature);
    XX[feature] = xcol.square().sum();
    Xsum[feature] = xcol.sum();
  }

  if (include_intercept) {
    // Preserve the exact null fit for a constant response.  Computing the
    // mean by summation can round away from the common value and turn this
    // degenerate, exactly solved problem into an ill-conditioned 1/L update.
    bool constant_response = true;
    for (int row = 1; row < n; ++row) {
      if (Y[row] != Y[0]) {
        constant_response = false;
        break;
      }
    }
    model_param.intercept =
        constant_response ? Y[0] : Y.sum() / n;
  }

  update_auxiliary();

  for (int i = 0; i < d; i++) update_gradient(i);

  deviance = fabs(eval());
};

double SqrtMSEObjective::coordinate_descent(RegFunction *regfunc, int idx) {
  // At an exact null fit, ||r||/sqrt(n) is nondifferentiable.  Zero is a
  // valid subgradient and the null model is already the global regularized
  // solution, so do not form the undefined 1/L quadratic model.
  if (is_exact_null_fit(L, model_param)) return model_param.beta[idx];

  g = 0.0;
  a = 0.0;
  const auto xcol = design_column(idx);

  double tmp0 = XX[idx];
  double tmp1 = (r * xcol).sum();

  // Concavity of sqrt gives the global quadratic majorizer
  //   L(r - delta*x) <= L(r) - delta*(x'r)/(nL)
  //                       + delta^2*||x||^2/(2nL).
  // Its curvature is both safer near interpolation than the raw Hessian
  // diagonal and avoids a second O(n) weighted reduction per coordinate.
  a = tmp0 / (n * L);
  g = tmp1 / (n * L) + a * model_param.beta[idx];

  double old_beta = model_param.beta[idx];
  model_param.beta[idx] =
      regfunc->thresholded_coordinate_minimize(g, a);
  double delta = model_param.beta[idx] - old_beta;

  if (fabs(delta) <= 1e-8) {
    // The legacy update threshold treats this as no coordinate change. Keep
    // beta, Xb, and the residual summaries on the same accepted model.
    model_param.beta[idx] = old_beta;
    return old_beta;
  }

  // Incremental update: r_new = r_old - delta * xcol
  // sum_r_new = sum_r_old - delta * sum(xcol)
  // sum_r2_new = sum_r2_old - 2*delta*(r_old·xcol) + delta^2*(xcol·xcol)
  //            = sum_r2_old - 2*delta*tmp1 + delta^2*tmp0
  sum_r -= delta * Xsum[idx];
  sum_r2 += -2.0 * delta * tmp1 + delta * delta * tmp0;
  if (sum_r2 < 0.0) sum_r2 = 0.0;  // guard against negative from rounding
  L = sqrt(sum_r2 / n);

  // Both maintained arrays consume the same contiguous feature column.
  // Updating them together avoids a second O(n) read of xcol for every
  // accepted coordinate without changing the accepted model or summaries.
  const double *x_values = xcol.data();
  double *residual_values = r.data();
  double *predictor_values = Xb.data();
  for (int observation = 0; observation < n; ++observation) {
    const double predictor_change = delta * x_values[observation];
    residual_values[observation] -= predictor_change;
    predictor_values[observation] += predictor_change;
  }
  return (model_param.beta[idx]);
}

void SqrtMSEObjective::intercept_update() {
  if (is_exact_null_fit(L, model_param)) return;

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
  gr[idx] = is_exact_null_fit(L, model_param)
                ? 0.0
                : (r * design_column(idx)).sum() / (n * L);
}

void SqrtMSEObjective::update_all_gradients() {
  if (is_exact_null_fit(L, model_param)) {
    gr.setZero();
    return;
  }
  gr = (design_matrix().transpose() * r.matrix()).array() / (n * L);
}

double SqrtMSEObjective::get_intercept_gradient() {
  if (is_exact_null_fit(L, model_param)) return 0.0;
  if (!(L > 0.0) || !std::isfinite(L))
    return std::numeric_limits<double>::quiet_NaN();
  return -sum_r / (n * L);
}

double SqrtMSEObjective::get_local_change(double old, int idx) {
  if (is_exact_null_fit(L, model_param)) {
    const double current = idx >= 0 ? model_param.beta[idx]
                                    : model_param.intercept;
    return old == current ? 0.0
                          : std::numeric_limits<double>::quiet_NaN();
  }
  if (idx >= 0) {
    const auto xcol = design_column(idx);
    double a =  (xcol * xcol * (1 - r * r/(L*L*n))).sum()/(n*L);
    double tmp = old - model_param.beta[idx];
    return (a * tmp * tmp / (2 * L * n));
  } else {
    double tmp = old - model_param.intercept;
    return (fabs(tmp));
  }
}

bool SqrtMSEObjective::can_skip_local_change(double old, int idx,
                                             double threshold) {
  if (!(threshold >= 0.0) || !std::isfinite(threshold)) return false;
  if (is_exact_null_fit(L, model_param)) {
    const double current = idx >= 0 ? model_param.beta[idx]
                                    : model_param.intercept;
    return old == current;
  }
  if (idx < 0 || !(L > 0.0) || !std::isfinite(L)) return false;

  const double change = old - model_param.beta[idx];
  if (!std::isfinite(change) || !(XX[idx] >= 0.0) ||
      !std::isfinite(XX[idx]))
    return false;

  const long double sample_count = static_cast<long double>(n);
  const long double loss = static_cast<long double>(L);
  const long double delta = static_cast<long double>(change);
  const long double upper_bound =
      static_cast<long double>(XX[idx]) * delta * delta /
      (2.0L * sample_count * sample_count * loss * loss);
  if (!std::isfinite(upper_bound)) return false;

  // Inflate the analytic bound for the rounding accumulated by the exact
  // nonnegative reduction.  This keeps the fast decision conservative.
  const long double rounding_margin =
      1.0L + 8.0L * (sample_count + 4.0L) *
                   std::numeric_limits<double>::epsilon();
  return upper_bound * rounding_margin <=
         static_cast<long double>(threshold);
}

double SqrtMSEObjective::eval() { return (L); }

};  // namespace picasso
