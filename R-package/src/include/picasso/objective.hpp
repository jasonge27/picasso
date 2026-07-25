#ifndef PICASSO_OBJECTIVE_H
#define PICASSO_OBJECTIVE_H

#include <algorithm>
#include <cmath>
#if defined(__clang__) && defined(__has_warning)
#  if __has_warning("-Wdeprecated-anon-enum-enum-conversion")
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#    define PICASSO_OBJECTIVE_RESTORE_EIGEN_DIAGNOSTICS
#  endif
#endif

#include <Eigen/Dense>

#if defined(PICASSO_OBJECTIVE_RESTORE_EIGEN_DIAGNOSTICS)
#  pragma clang diagnostic pop
#  undef PICASSO_OBJECTIVE_RESTORE_EIGEN_DIAGNOSTICS
#endif
#include <limits>
#include <stdexcept>
#include <vector>

#include <ctime>

namespace picasso {

class ModelParam {
 public:
  int d;
  Eigen::ArrayXd beta;
  double intercept;

  ModelParam(int dim) {
    d = dim;
    beta.resize(d);
    beta.setZero();
    intercept = 0.0;
  }
};

class RegFunction {
 public:
  virtual double threshold(double x) = 0;
  virtual void set_param(double lambda, double gamma) = 0;
  virtual double get_lambda() = 0;

  // Apply a threshold-based coordinate update while handling a flat
  // quadratic safely.  If both curvature and linear term are exactly zero,
  // beta = 0 is the minimum-penalty representative of the flat set.  A
  // nonzero linear term is deliberately rejected: capped nonconvex penalties
  // would make that zero-curvature subproblem unbounded even when their local
  // threshold happens to return zero.
  // Keeping this helper nonvirtual preserves the single threshold dispatch
  // used by the GLM and square-root-loss hot loops.
  double thresholded_coordinate_minimize(double linear_term,
                                         double curvature) {
    if (!std::isfinite(curvature) || !std::isfinite(linear_term))
      return std::numeric_limits<double>::quiet_NaN();
    if (curvature > 0.0) {
      const double thresholded = threshold(linear_term);
      return std::isfinite(thresholded)
                 ? thresholded / curvature
                 : std::numeric_limits<double>::quiet_NaN();
    }
    if (curvature == 0.0 && linear_term == 0.0 && threshold(0.0) == 0.0)
      return 0.0;
    return std::numeric_limits<double>::quiet_NaN();
  }

  // Minimize 0.5 * curvature * beta^2 - linear_term * beta + penalty(beta).
  // The default covers convex L1 and preserves the historical update exactly.
  virtual double coordinate_minimize(double linear_term, double curvature) {
    return thresholded_coordinate_minimize(linear_term, curvature);
  }

  virtual ~RegFunction(){};

  double threshold_l1(double x, double thr) {
    if (x > thr)
      return x - thr;
    else if (x < -thr)
      return x + thr;
    else
      return 0;
  }
};

class RegL1 : public RegFunction {
 private:
  double m_lambda;

 public:
  void set_param(double lambda, double gamma) { m_lambda = lambda; }
  double get_lambda() { return m_lambda; };
  double threshold(double x) { return threshold_l1(x, m_lambda); }
};

class RegSCAD : public RegFunction {
 private:
  double m_lambda;
  double m_gamma;

 public:
  void set_param(double lambda, double gamma) {
    m_lambda = lambda;
    m_gamma = gamma;
  };
  double get_lambda() { return m_lambda; };

  double threshold(double x) {
    if (fabs(x) > fabs(m_gamma * m_lambda)) {
      return x;
    } else {
      if (fabs(x) > fabs(2 * m_lambda)) {
        return threshold_l1(x, m_gamma * m_lambda / (m_gamma - 1)) /
               (1 - 1 / (m_gamma - 1));
      } else {
        return threshold_l1(x, m_lambda);
      }
    }
  };

  double coordinate_minimize(double linear_term, double curvature) {
    if (!(curvature > 0.0) || !std::isfinite(curvature) ||
        !std::isfinite(linear_term) || !(m_lambda >= 0.0) ||
        !std::isfinite(m_lambda) || !(m_gamma > 2.0) ||
        !std::isfinite(m_gamma)) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    const double abs_linear = std::fabs(linear_term);
    if (m_lambda == 0.0) return linear_term / curvature;

    const double first_knot = m_lambda;
    const double second_knot = m_gamma * m_lambda;
    double best = 0.0;
    long double best_value = 0.0L;

    const auto consider = [&](double candidate) {
      if (!(candidate >= 0.0) || !std::isfinite(candidate)) return;

      long double penalty;
      if (candidate <= first_knot) {
        penalty = static_cast<long double>(m_lambda) * candidate;
      } else if (candidate <= second_knot) {
        const long double t = candidate;
        const long double lambda = m_lambda;
        const long double gamma = m_gamma;
        penalty = (-t * t + 2.0L * gamma * lambda * t -
                   lambda * lambda) /
                  (2.0L * (gamma - 1.0L));
      } else {
        penalty = 0.5L * (m_gamma + 1.0L) * m_lambda * m_lambda;
      }

      const long double t = candidate;
      const long double value =
          0.5L * curvature * t * t - abs_linear * t + penalty;
      if (value < best_value) {
        best_value = value;
        best = candidate;
      }
    };

    // Every global minimizer lies at a knot or a stationary point of one of
    // the three smooth pieces. This also covers a concave or linear middle
    // piece because both endpoints are always compared.
    consider(first_knot);
    consider(second_knot);

    const double first_stationary = (abs_linear - m_lambda) / curvature;
    if (first_stationary >= 0.0 && first_stationary <= first_knot)
      consider(first_stationary);

    const double middle_curvature = curvature - 1.0 / (m_gamma - 1.0);
    if (middle_curvature != 0.0) {
      const double middle_stationary =
          (abs_linear - m_gamma * m_lambda / (m_gamma - 1.0)) /
          middle_curvature;
      if (middle_stationary >= first_knot &&
          middle_stationary <= second_knot)
        consider(middle_stationary);
    }

    const double last_stationary = abs_linear / curvature;
    if (last_stationary >= second_knot) consider(last_stationary);

    return best == 0.0 ? 0.0 : std::copysign(best, linear_term);
  }
};

class RegMCP : public RegFunction {
 private:
  double m_lambda;
  double m_gamma;

 public:
  void set_param(double lambda, double gamma) {
    m_lambda = lambda;
    m_gamma = gamma;
  }
  double get_lambda() { return m_lambda; };

  double threshold(double x) {
    if (fabs(x) > fabs(m_gamma * m_lambda)) {
      return x;
    } else {
      return threshold_l1(x, m_lambda)/(1-1/m_gamma);
    }
  }

  double coordinate_minimize(double linear_term, double curvature) {
    if (!(curvature > 0.0) || !std::isfinite(curvature) ||
        !std::isfinite(linear_term) || !(m_lambda >= 0.0) ||
        !std::isfinite(m_lambda) || !(m_gamma > 1.0) ||
        !std::isfinite(m_gamma)) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    const double abs_linear = std::fabs(linear_term);
    if (m_lambda == 0.0) return linear_term / curvature;

    const double knot = m_gamma * m_lambda;
    double best = 0.0;
    long double best_value = 0.0L;

    const auto consider = [&](double candidate) {
      if (!(candidate >= 0.0) || !std::isfinite(candidate)) return;

      const long double t = candidate;
      long double penalty;
      if (candidate <= knot) {
        penalty = static_cast<long double>(m_lambda) * t -
                  t * t / (2.0L * m_gamma);
      } else {
        penalty = 0.5L * m_gamma * m_lambda * m_lambda;
      }

      const long double value =
          0.5L * curvature * t * t - abs_linear * t + penalty;
      if (value < best_value) {
        best_value = value;
        best = candidate;
      }
    };

    // When curvature <= 1/gamma the inner piece is nonconvex. Its stationary
    // point can be a maximum, but comparing it with both piece boundaries and
    // the outer stationary point still gives the global coordinate minimizer.
    consider(knot);

    const double inner_curvature = curvature - 1.0 / m_gamma;
    if (inner_curvature != 0.0) {
      const double inner_stationary =
          (abs_linear - m_lambda) / inner_curvature;
      if (inner_stationary >= 0.0 && inner_stationary <= knot)
        consider(inner_stationary);
    }

    const double outer_stationary = abs_linear / curvature;
    if (outer_stationary >= knot) consider(outer_stationary);

    return best == 0.0 ? 0.0 : std::copysign(best, linear_term);
  }
};

namespace detail {

// The public objective constructors retain their historical owning behavior.
// Synchronous C API entry points may opt into a read-only view when their
// caller already supplies a stable column-major buffer (notably R's .Call).
enum class DesignStorage { kOwned, kBorrowedColumnMajor };

}  // namespace detail

class ObjFunction {
 protected:
  typedef Eigen::Map<const Eigen::ArrayXd, Eigen::Unaligned>
      ConstDesignColumn;
  typedef Eigen::Map<
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::ColMajor>,
      Eigen::Unaligned>
      ConstDesignMatrix;

  int n;  // sample number
  int d;  // sample dimension

  Eigen::ArrayXXd X;
  const double *m_borrowed_x;
  Eigen::ArrayXd Y;

  Eigen::ArrayXd gr;
  Eigen::ArrayXd Xb;

  Eigen::ArrayXd m_offset;  // per-observation offset (default zeros)

  ModelParam model_param;

  double deviance;

  const double *design_data() const {
    return m_borrowed_x == nullptr ? X.data() : m_borrowed_x;
  }

  ConstDesignColumn design_column(int index) const {
    return ConstDesignColumn(
        design_data() + static_cast<std::size_t>(index) * n, n);
  }

  ConstDesignMatrix design_matrix() const {
    return ConstDesignMatrix(design_data(), n, d);
  }

 public:
  ObjFunction(const double *xmat, const double *y, int n, int d,
              bool usePython = false)
      : ObjFunction(xmat, y, n, d, usePython,
                    detail::DesignStorage::kOwned) {}

  ObjFunction(const double *xmat, const double *y, int n, int d,
              bool usePython, detail::DesignStorage design_storage)
      : n(n),
        d(d),
        m_borrowed_x(
            design_storage == detail::DesignStorage::kBorrowedColumnMajor
                ? xmat
                : nullptr),
        model_param(d) {
    if (usePython &&
        design_storage == detail::DesignStorage::kBorrowedColumnMajor) {
      throw std::invalid_argument(
          "row-major Python input cannot use borrowed column-major storage");
    }
    Y.resize(n);
    gr.resize(d);

    Xb.resize(n);
    Xb.setZero();

    m_offset.resize(n);
    m_offset.setZero();

    std::copy(y, y + n, Y.data());

    if (design_storage == detail::DesignStorage::kOwned) {
      X.resize(n, d);
      if (!usePython) {
        std::copy(xmat, xmat + static_cast<std::size_t>(n) * d, X.data());
      } else {
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < d; j++) X(i, j) = xmat[i * d + j];
        }
      }
    }
  };

  // Copies always own their design, even when the source borrows one. This
  // preserves the historical value semantics and prevents a copied objective
  // from retaining a caller-owned pointer beyond the source call.
  ObjFunction(const ObjFunction &other)
      : n(other.n),
        d(other.d),
        X(other.design_matrix().array()),
        m_borrowed_x(nullptr),
        Y(other.Y),
        gr(other.gr),
        Xb(other.Xb),
        m_offset(other.m_offset),
        model_param(other.model_param),
        deviance(other.deviance) {}

  ObjFunction &operator=(const ObjFunction &other) {
    if (this == &other) return *this;
    // Materialize before mutating this object. This also makes assignment
    // robust when a public C++ caller supplies a borrowed view into this
    // objective's current owned buffer.
    Eigen::ArrayXXd owned_design = other.design_matrix().array();
    n = other.n;
    d = other.d;
    X = owned_design;
    m_borrowed_x = nullptr;
    Y = other.Y;
    gr = other.gr;
    Xb = other.Xb;
    m_offset = other.m_offset;
    model_param = other.model_param;
    deviance = other.deviance;
    return *this;
  }

  virtual bool set_offset(const double *off, int len) {
    if (off == nullptr || len != n) return false;
    for (int index = 0; index < len; ++index) {
      if (!std::isfinite(off[index])) return false;
    }
    std::copy(off, off + len, m_offset.data());
    return true;
  }

  int get_dim() { return d; }
  int get_sample_num() { return n; }

  double get_grad(int idx) { return gr[idx]; };

  // initial |eval()| used as a scale reference for convergence thresholds
  double get_deviance() { return (deviance); };

  double get_model_coef(int idx) {
    return ((idx < 0) ? model_param.intercept : model_param.beta[idx]);
  }
  void set_model_coef(double value, int idx) {
    if (idx >= 0)
      model_param.beta[idx] = value;
    else
      model_param.intercept = value;
  }

  ModelParam get_model_param() { return model_param; };
  Eigen::ArrayXd get_model_Xb() const { return Xb; };

  const ModelParam &get_model_param_ref() { return model_param; };
  const Eigen::ArrayXd &get_model_Xb_ref() const { return Xb; };

  // reset model param and also update related aux vars
  void set_model_param(const ModelParam &other_param) {
    model_param.d = other_param.d;
    model_param.beta = other_param.beta;
    model_param.intercept = other_param.intercept;
  };

  void set_model_Xb(const Eigen::ArrayXd &other_Xb) { Xb = other_Xb; };

  // coordinate descent
  virtual double coordinate_descent(RegFunction *regfun, int idx) = 0;

  // update intercept term
  virtual void intercept_update() = 0;

  // update gradient and other aux vars
  virtual void update_auxiliary() = 0;
  virtual void update_gradient(int idx){};
  virtual void update_all_gradients() {
    for (int idx = 0; idx < d; ++idx) update_gradient(idx);
  }

  // Signed gradient of the smooth loss with respect to the intercept. The
  // default is sufficient for objectives that do not use ActNewton.
  virtual double get_intercept_gradient() { return 0.0; }

  // compute quadratic change of fvalue on the idx dimension
  virtual double get_local_change(double old, int idx) = 0;

  // Return true only when an O(1) bound proves that get_local_change() cannot
  // exceed threshold.  Objectives without such a certificate retain the
  // exact calculation.
  virtual bool can_skip_local_change(double, int, double) { return false; }

  // unpenalized function value
  virtual double eval() = 0;

  virtual ~ObjFunction(){};
};

class GLMObjective : public ObjFunction {
 protected:
  Eigen::ArrayXd p, w, r;

  // wXX[j] = sum(w*X[j]*X[j])
  Eigen::ArrayXd wXX;
  std::vector<unsigned char> wXX_valid;

  // quadratic approx coefs for each coordinate
  // a*x^2 + g*x + constant
  double a, g;
  double sum_r;
  double sum_w;
  bool m_include_intercept;
  bool m_use_fused_coordinate_updates;
  bool m_use_fast_residual_dot;
  bool m_use_fast_weighted_sq_sum;

  double get_weighted_sq_sum(int idx);
  double residual_dot_product(int idx);
  double coordinate_descent_impl(RegFunction *regfunc, int idx,
                                 bool update_linear_predictor);

 public:
  GLMObjective(const double *xmat, const double *y, int n, int d,
               bool include_intercept = false, bool usePython = false);
  GLMObjective(const double *xmat, const double *y, int n, int d,
               bool include_intercept, bool usePython,
               detail::DesignStorage design_storage);

  double coordinate_descent(RegFunction *regfunc, int idx);
  double coordinate_descent_deferred(RegFunction *regfunc, int idx);
  bool coordinate_state_all_finite() const;
  bool rebuild_linear_predictor(const std::vector<int> &features);

  void intercept_update();
  void update_gradient(int);
  void update_all_gradients();
  double get_intercept_gradient();

  double get_local_change(double old, int idx);

  // The fused state update is beneficial for L1 and fast-tolerance paths.
  // Strict nonconvex paths retain the legacy Eigen update schedule.
  void set_fused_coordinate_updates(bool enabled) {
    m_use_fused_coordinate_updates = enabled;
  }
  void set_fast_residual_dot(bool enabled) {
    m_use_fast_residual_dot = enabled;
  }
  void set_fast_weighted_sq_sum(bool enabled) {
    m_use_fast_weighted_sq_sum = enabled;
  }
};

class LogisticObjective : public GLMObjective {
 private:
  void initialize_null_model();

 public:
  LogisticObjective(const double *xmat, const double *y, int n, int d,
                    bool include_intercept = false, bool usePython = false);
  LogisticObjective(const double *xmat, const double *y, int n, int d,
                    bool include_intercept, bool usePython,
                    detail::DesignStorage design_storage);

  bool set_offset(const double *off, int len) override;

  void update_auxiliary() override;

  double eval() override;
};

class PoissonObjective : public GLMObjective {
 private:
  void initialize_null_model();

 public:
  PoissonObjective(const double *xmat, const double *y, int n, int d,
                   bool include_intercept = false, bool usePython = false);
  PoissonObjective(const double *xmat, const double *y, int n, int d,
                   bool include_intercept, bool usePython,
                   detail::DesignStorage design_storage);

  bool set_offset(const double *off, int len) override;

  void update_auxiliary() override;

  double eval() override;
};

class SqrtMSEObjective : public ObjFunction {
 private:
  Eigen::ArrayXd r;
  Eigen::ArrayXd XX;
  Eigen::ArrayXd Xsum;

  // quadratic approx coefs for each coordinate
  // a*x^2 + g*x + constant
  double a, g;
  double L;  // sqrt(MSE)
  double sum_r;
  double sum_r2;

 public:
  SqrtMSEObjective(const double *xmat, const double *y, int n, int d,
                   bool include_intercept = false, bool usePython = false);
  SqrtMSEObjective(const double *xmat, const double *y, int n, int d,
                   bool include_intercept, bool usePython,
                   detail::DesignStorage design_storage);

  double coordinate_descent(RegFunction *regfunc, int idx);

  void intercept_update();

  void update_auxiliary();
  void update_gradient(int idx);
  void update_all_gradients();
  double get_intercept_gradient();

  double get_local_change(double old, int idx);
  bool can_skip_local_change(double old, int idx, double threshold);

  double eval();
};

class GaussianNaiveUpdateObjective: public ObjFunction {
 private:
  Eigen::ArrayXd r, XX, Xmean;
  double residual_mean;
  double Ymean;
  bool m_include_intercept;

 public:
  GaussianNaiveUpdateObjective(const double *xmat, const double *y, int n,
                               int d, bool include_intercept = false,
                               bool usePython = false);
  GaussianNaiveUpdateObjective(const double *xmat, const double *y, int n,
                               int d, bool include_intercept, bool usePython,
                               detail::DesignStorage design_storage);
  double coordinate_descent(RegFunction *regfunc, int idx);

  void intercept_update();
  void update_auxiliary();
  void update_gradient(int idx);

  double get_local_change(double old, int idx);

  double eval();
};

class GaussianCovUpdateObjective : public ObjFunction {
 private:
  Eigen::ArrayXd XX;      // d: diagonal covariance/second moment
  Eigen::ArrayXd Xy;      // d: centered/raw X^T Y / n
  Eigen::ArrayXd Xmean;   // d: column means of X
  std::vector<Eigen::ArrayXd> C_columns;
  std::vector<unsigned char> C_valid;
  int cached_column_count;
  double Ymean;
  bool m_include_intercept;

  const Eigen::ArrayXd &covariance_column(int idx);

 public:
  GaussianCovUpdateObjective(const double *xmat, const double *y, int n, int d,
                             bool include_intercept = false,
                             bool usePython = false);
  GaussianCovUpdateObjective(const double *xmat, const double *y, int n, int d,
                             bool include_intercept, bool usePython,
                             detail::DesignStorage design_storage);
  double coordinate_descent(RegFunction *regfunc, int idx);

  void intercept_update();
  void update_auxiliary();
  void update_gradient(int idx);

  double get_local_change(double old, int idx);

  double eval();
  double path_eval();

  int get_cached_covariance_column_count() const {
    return cached_column_count;
  }
};

}  // namespace picasso

#endif  // PICASSO_OBJECTIVE_H
