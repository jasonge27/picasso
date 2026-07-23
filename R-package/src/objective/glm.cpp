#include <picasso/objective.hpp>

#include <limits>

namespace {

constexpr double kProbabilityFloor = 1e-12;

long double stable_logistic_loss(long double eta, long double response) {
  if (eta >= 0.0)
    return (1.0 - response) * eta + std::log1p(std::exp(-eta));
  return -response * eta + std::log1p(std::exp(eta));
}

double stable_logistic_probability(double eta) {
  if (eta >= 0.0) {
    const double scaled = std::exp(-eta);
    return 1.0 / (1.0 + scaled);
  }
  const double scaled = std::exp(eta);
  return scaled / (1.0 + scaled);
}

double finite_double(long double value) {
  const long double limit = std::numeric_limits<double>::max();
  if (value > limit) return std::numeric_limits<double>::max();
  if (value < -limit) return -std::numeric_limits<double>::max();
  return static_cast<double>(value);
}

double logistic_null_intercept(const Eigen::ArrayXd &response,
                               const Eigen::ArrayXd &offset) {
  const int sample_size = static_cast<int>(response.size());
  double target = response.sum() / sample_size;
  target = std::max(kProbabilityFloor,
                    std::min(1.0 - kProbabilityFloor, target));
  const double target_link = std::log(target) - std::log1p(-target);

  const long double lower_value =
      static_cast<long double>(target_link) - offset.maxCoeff();
  const long double upper_value =
      static_cast<long double>(target_link) - offset.minCoeff();
  double lower = finite_double(lower_value);
  double upper = finite_double(upper_value);

  const auto score = [&](double intercept) {
    double mean = 0.0;
    for (int index = 0; index < sample_size; ++index)
      mean += stable_logistic_probability(intercept + offset[index]);
    return mean / sample_size - target;
  };

  double lower_score = score(lower);
  double upper_score = score(upper);
  if (lower_score >= 0.0 || lower == upper) return lower;
  if (upper_score <= 0.0) return upper;

  // The score is monotone. A fixed bracket is more reliable than an
  // unguarded Newton step when probabilities or IRLS weights are near zero.
  for (int iteration = 0; iteration < 200; ++iteration) {
    const double middle = 0.5 * lower + 0.5 * upper;
    if (middle == lower || middle == upper) break;
    const double middle_score = score(middle);
    if (middle_score < 0.0) {
      lower = middle;
      lower_score = middle_score;
    } else {
      upper = middle;
      upper_score = middle_score;
    }
  }
  return std::fabs(lower_score) <= std::fabs(upper_score) ? lower : upper;
}

double poisson_null_intercept(const Eigen::ArrayXd &response,
                              const Eigen::ArrayXd &offset) {
  const int sample_size = static_cast<int>(response.size());
  const long double maximum_offset = offset.maxCoeff();
  long double scaled_offset_sum = 0.0L;
  for (int index = 0; index < sample_size; ++index) {
    scaled_offset_sum +=
        std::exp(static_cast<long double>(offset[index]) - maximum_offset);
  }
  long double response_sum = 0.0L;
  for (int index = 0; index < sample_size; ++index)
    response_sum += response[index];
  const long double log_target_sum =
      response_sum > 0.0L
          ? std::log(response_sum)
          : std::log(static_cast<long double>(sample_size)) +
                std::log(static_cast<long double>(
                    std::numeric_limits<double>::min()));
  // Subtract the two logarithmic sums before the maximum offset. This avoids
  // overflowing max(offset) + log(sum(exp(offset - max))) when all finite
  // offsets are themselves close to DBL_MAX.
  return finite_double(log_target_sum - std::log(scaled_offset_sum) -
                       maximum_offset);
}

double maximum_poisson_link(int sample_size) {
  // Keep the sum of fitted means comfortably below DBL_MAX. Individual
  // feature products can still be rejected by the solver's finite checks.
  return std::log(std::numeric_limits<double>::max()) -
         std::log(static_cast<double>(std::max(sample_size, 1))) -
         std::log(16.0);
}

}  // namespace

namespace picasso {

GLMObjective::GLMObjective(const double *xmat, const double *y, int n, int d,
                           bool include_intercept, bool usePython)
    : GLMObjective(xmat, y, n, d, include_intercept, usePython,
                   detail::DesignStorage::kOwned) {}

GLMObjective::GLMObjective(const double *xmat, const double *y, int n, int d,
                           bool include_intercept, bool usePython,
                           detail::DesignStorage design_storage)
    : ObjFunction(xmat, y, n, d, usePython, design_storage),
      m_include_intercept(include_intercept),
      m_use_fused_coordinate_updates(false),
      m_use_fast_residual_dot(false),
      m_use_fast_weighted_sq_sum(false) {
  a = 0.0;
  g = 0.0;

  p.resize(n);
  w.resize(n);
  r.resize(n);

  wXX.resize(d);
  wXX.setZero();
  wXX_valid.assign(d, 0);
  // Intercept initialization is done in each subclass with the correct link function.
}

double GLMObjective::get_weighted_sq_sum(int idx) {
  if (!wXX_valid[idx]) {
    const auto xcol = design_column(idx);
    if (!m_use_fast_weighted_sq_sum || xcol.innerStride() != 1) {
      wXX[idx] = (w * xcol * xcol).sum();
    } else {
      typedef Eigen::internal::packet_traits<double>::type Packet;
      const int packet_size = Eigen::internal::packet_traits<double>::size;
      const int block_size = 4 * packet_size;
      Packet sum0 = Eigen::internal::pset1<Packet>(0.0);
      Packet sum1 = Eigen::internal::pset1<Packet>(0.0);
      Packet sum2 = Eigen::internal::pset1<Packet>(0.0);
      Packet sum3 = Eigen::internal::pset1<Packet>(0.0);
      const double *x_values = xcol.data();
      const double *weights = w.data();
      int observation = 0;
      for (; observation <= n - block_size; observation += block_size) {
        const Packet x0 =
            Eigen::internal::ploadu<Packet>(x_values + observation);
        const Packet x1 = Eigen::internal::ploadu<Packet>(
            x_values + observation + packet_size);
        const Packet x2 = Eigen::internal::ploadu<Packet>(
            x_values + observation + 2 * packet_size);
        const Packet x3 = Eigen::internal::ploadu<Packet>(
            x_values + observation + 3 * packet_size);
        const Packet w0 =
            Eigen::internal::ploadu<Packet>(weights + observation);
        const Packet w1 = Eigen::internal::ploadu<Packet>(
            weights + observation + packet_size);
        const Packet w2 = Eigen::internal::ploadu<Packet>(
            weights + observation + 2 * packet_size);
        const Packet w3 = Eigen::internal::ploadu<Packet>(
            weights + observation + 3 * packet_size);
        sum0 = Eigen::internal::pmadd(
            Eigen::internal::pmul(w0, x0), x0, sum0);
        sum1 = Eigen::internal::pmadd(
            Eigen::internal::pmul(w1, x1), x1, sum1);
        sum2 = Eigen::internal::pmadd(
            Eigen::internal::pmul(w2, x2), x2, sum2);
        sum3 = Eigen::internal::pmadd(
            Eigen::internal::pmul(w3, x3), x3, sum3);
      }
      const Packet sum = Eigen::internal::padd(
          Eigen::internal::padd(sum0, sum1),
          Eigen::internal::padd(sum2, sum3));
      double result = Eigen::internal::predux(sum);
      for (; observation < n; ++observation) {
        result += (weights[observation] * x_values[observation]) *
                  x_values[observation];
      }
      wXX[idx] = result;
    }
    wXX_valid[idx] = 1;
  }
  return wXX[idx];
}

double GLMObjective::residual_dot_product(int idx) {
  const auto xcol = design_column(idx);
  if (!m_use_fast_residual_dot || xcol.innerStride() != 1)
    return (r * xcol).sum();

  // Eigen's generic redux carries one packet dependency chain. Four
  // independent accumulators expose enough instruction-level parallelism for
  // the repeated coordinate and active-KKT dot products in fast mode.
  typedef Eigen::internal::packet_traits<double>::type Packet;
  const int packet_size = Eigen::internal::packet_traits<double>::size;
  const int block_size = 4 * packet_size;
  Packet sum0 = Eigen::internal::pset1<Packet>(0.0);
  Packet sum1 = Eigen::internal::pset1<Packet>(0.0);
  Packet sum2 = Eigen::internal::pset1<Packet>(0.0);
  Packet sum3 = Eigen::internal::pset1<Packet>(0.0);
  const double *x_values = xcol.data();
  const double *residual_values = r.data();
  int observation = 0;
  for (; observation <= n - block_size; observation += block_size) {
    const Packet x0 =
        Eigen::internal::ploadu<Packet>(x_values + observation);
    const Packet x1 = Eigen::internal::ploadu<Packet>(
        x_values + observation + packet_size);
    const Packet x2 = Eigen::internal::ploadu<Packet>(
        x_values + observation + 2 * packet_size);
    const Packet x3 = Eigen::internal::ploadu<Packet>(
        x_values + observation + 3 * packet_size);
    const Packet r0 =
        Eigen::internal::ploadu<Packet>(residual_values + observation);
    const Packet r1 = Eigen::internal::ploadu<Packet>(
        residual_values + observation + packet_size);
    const Packet r2 = Eigen::internal::ploadu<Packet>(
        residual_values + observation + 2 * packet_size);
    const Packet r3 = Eigen::internal::ploadu<Packet>(
        residual_values + observation + 3 * packet_size);
    sum0 = Eigen::internal::pmadd(x0, r0, sum0);
    sum1 = Eigen::internal::pmadd(x1, r1, sum1);
    sum2 = Eigen::internal::pmadd(x2, r2, sum2);
    sum3 = Eigen::internal::pmadd(x3, r3, sum3);
  }
  const Packet sum = Eigen::internal::padd(
      Eigen::internal::padd(sum0, sum1),
      Eigen::internal::padd(sum2, sum3));
  double result = Eigen::internal::predux(sum);
  for (; observation < n; ++observation)
    result += x_values[observation] * residual_values[observation];
  return result;
}

double GLMObjective::coordinate_descent(RegFunction *regfunc, int idx) {
  return coordinate_descent_impl(regfunc, idx, true);
}

double GLMObjective::coordinate_descent_deferred(
    RegFunction *regfunc, int idx) {
  return coordinate_descent_impl(regfunc, idx, false);
}

double GLMObjective::coordinate_descent_impl(
    RegFunction *regfunc, int idx, bool update_linear_predictor) {
  g = 0.0;
  a = 0.0;
  const auto xcol = design_column(idx);
  const double weighted_sq_sum = get_weighted_sq_sum(idx);
  a = weighted_sq_sum / n;
  g = (model_param.beta[idx] * weighted_sq_sum +
       residual_dot_product(idx)) / n;

  const double old_beta = model_param.beta[idx];
  model_param.beta[idx] =
      regfunc->thresholded_coordinate_minimize(g, a);
  const double delta = model_param.beta[idx] - old_beta;

  if (fabs(delta) <= 1e-8) {
    model_param.beta[idx] = old_beta;
    return old_beta;
  }

  if (!update_linear_predictor) {
    const double *x_values = xcol.data();
    double *residual = r.data();
    const double *weights = w.data();
    for (int observation = 0; observation < n; ++observation) {
      const double feature_value = x_values[observation];
      residual[observation] -=
          (delta * weights[observation]) * feature_value;
    }
  } else if (m_use_fused_coordinate_updates) {
    // X is column-major, so xcol is contiguous. Fusing the maintained linear
    // predictor and IRLS residual updates avoids reading the same feature
    // column twice per accepted coordinate update.
    const double *x_values = xcol.data();
    double *linear_predictor = Xb.data();
    double *residual = r.data();
    const double *weights = w.data();
    for (int observation = 0; observation < n; ++observation) {
      const double feature_value = x_values[observation];
      linear_predictor[observation] += delta * feature_value;
      residual[observation] -=
          (delta * weights[observation]) * feature_value;
    }
  } else {
    Xb = Xb + delta * xcol;
    r = r - delta * w * xcol;
  }
  return (model_param.beta[idx]);
}

bool GLMObjective::coordinate_state_all_finite() const {
  return model_param.beta.allFinite() &&
         std::isfinite(model_param.intercept) && r.allFinite();
}

bool GLMObjective::rebuild_linear_predictor(
    const std::vector<int> &features) {
  Xb.setZero();
  for (std::size_t index = 0; index < features.size(); ++index) {
    const int feature = features[index];
    if (feature < 0 || feature >= d) return false;
    const double coefficient = model_param.beta[feature];
    if (!std::isfinite(coefficient)) return false;
    if (coefficient != 0.0) Xb += coefficient * design_column(feature);
  }
  return Xb.allFinite();
}

void GLMObjective::intercept_update() {
  if (!(sum_w > 0.0) || !std::isfinite(sum_w)) {
    model_param.intercept = std::numeric_limits<double>::quiet_NaN();
    return;
  }
  double sum_r = r.sum();
  if (!std::isfinite(sum_r)) {
    model_param.intercept = std::numeric_limits<double>::quiet_NaN();
    return;
  }
  model_param.intercept += sum_r/sum_w;
  r = r - sum_r/sum_w * w;
}

void GLMObjective::update_gradient(int idx) {
  gr[idx] = residual_dot_product(idx) / n;
}

void GLMObjective::update_all_gradients() {
  gr.matrix().noalias() =
      (design_matrix().transpose() * r.matrix()) / static_cast<double>(n);
}

double GLMObjective::get_intercept_gradient() { return -r.sum() / n; }

double GLMObjective::get_local_change(double old, int idx) {
  if (idx >= 0) {
    double tmp = old - model_param.beta[idx];
    return (get_weighted_sq_sum(idx) * tmp * tmp / (2 * n));
  } else {
    double tmp = old - model_param.intercept;
    return (sum_w * tmp * tmp / (2 * n));
  }
}

LogisticObjective::LogisticObjective(const double *xmat, const double *y, int n,
                                     int d, bool include_intercept, bool usePython)
    : LogisticObjective(xmat, y, n, d, include_intercept, usePython,
                        detail::DesignStorage::kOwned) {}

LogisticObjective::LogisticObjective(
    const double *xmat, const double *y, int n, int d,
    bool include_intercept, bool usePython,
    detail::DesignStorage design_storage)
    : GLMObjective(xmat, y, n, d, include_intercept, usePython,
                   design_storage) {
  initialize_null_model();
};

void LogisticObjective::initialize_null_model() {
  model_param.beta.setZero();
  Xb.setZero();
  model_param.intercept =
      m_include_intercept ? logistic_null_intercept(Y, m_offset) : 0.0;
  update_auxiliary();
  update_all_gradients();
  deviance = fabs(eval());
}

bool LogisticObjective::set_offset(const double *off, int len) {
  if (!ObjFunction::set_offset(off, len)) return false;
  initialize_null_model();
  return true;
}

void LogisticObjective::update_auxiliary() {
  // This Eigen expression is branch-free and already stable at both tails:
  // exp(-eta) may overflow only when the resulting probability is exactly 0.
  p = -(model_param.intercept + Xb + m_offset);
  p = p.exp();
  p = 1.0 / (1.0 + p);
  r = Y - p;

  w = p * (1 - p);
  sum_w = w.sum();
  std::fill(wXX_valid.begin(), wXX_valid.end(), 0);
}

double LogisticObjective::eval() {
  long double mean_loss = 0.0L;
  for (int i = 0; i < n; ++i) {
    const long double eta =
        static_cast<long double>(model_param.intercept) + Xb[i] + m_offset[i];
    const long double loss = stable_logistic_loss(eta, Y[i]);
    // An online mean cannot overflow merely because several individually
    // finite, same-scale losses are added. This also works on platforms where
    // long double has the same range as double.
    mean_loss += (loss - mean_loss) / static_cast<long double>(i + 1);
  }

  return finite_double(mean_loss);
}

PoissonObjective::PoissonObjective(const double *xmat, const double *y, int n,
                                   int d, bool include_intercept, bool usePython)
    : PoissonObjective(xmat, y, n, d, include_intercept, usePython,
                       detail::DesignStorage::kOwned) {}

PoissonObjective::PoissonObjective(
    const double *xmat, const double *y, int n, int d,
    bool include_intercept, bool usePython,
    detail::DesignStorage design_storage)
    : GLMObjective(xmat, y, n, d, include_intercept, usePython,
                   design_storage) {
  initialize_null_model();
};

void PoissonObjective::initialize_null_model() {
  model_param.beta.setZero();
  Xb.setZero();
  model_param.intercept =
      m_include_intercept ? poisson_null_intercept(Y, m_offset) : 0.0;
  update_auxiliary();
  update_all_gradients();
  deviance = fabs(eval());
}

bool PoissonObjective::set_offset(const double *off, int len) {
  if (!ObjFunction::set_offset(off, len)) return false;
  initialize_null_model();
  return true;
}

void PoissonObjective::update_auxiliary() {
  const double maximum_link = maximum_poisson_link(n);
  const Eigen::ArrayXd link = model_param.intercept + Xb + m_offset;
  for (int index = 0; index < n; ++index) {
    if (!std::isfinite(link[index]) || link[index] > maximum_link) {
      const double invalid = std::numeric_limits<double>::quiet_NaN();
      p.setConstant(invalid);
      r.setConstant(invalid);
      w.setConstant(invalid);
      sum_w = invalid;
      std::fill(wXX_valid.begin(), wXX_valid.end(), 0);
      return;
    }
  }
  p = link.exp();
  r = Y - p;
  w = p;
  sum_w = w.sum();
  std::fill(wXX_valid.begin(), wXX_valid.end(), 0);
}

double PoissonObjective::eval() {
  long double mean_loss = 0.0L;
  for (int i = 0; i < n; i++) {
    const long double link =
        static_cast<long double>(model_param.intercept) + Xb[i] + m_offset[i];
    if (!std::isfinite(p[i]) || !std::isfinite(link))
      return std::numeric_limits<double>::quiet_NaN();
    const long double loss =
        p[i] - static_cast<long double>(Y[i]) * link;
    mean_loss += (loss - mean_loss) / static_cast<long double>(i + 1);
  }
  return finite_double(mean_loss);
}

}  // namespace picasso
