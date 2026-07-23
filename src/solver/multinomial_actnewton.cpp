#include <picasso/multinomial_actnewton.hpp>

#include "../internal/multinomial_solver_view.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace picasso {
namespace solver {

namespace detail {

struct MultinomialPathSmoothCache {
  const void *problem_identity;
  Eigen::MatrixXd beta_snapshot;
  Eigen::VectorXd intercept_snapshot;
  Eigen::MatrixXd logits;
  Eigen::MatrixXd probabilities;
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  double smooth_negative_log_likelihood;
};

}  // namespace detail

namespace {

typedef ::picasso::detail::MultinomialProblemView MultinomialProblemView;

MultinomialProblemView actnewton_problem_view(
    const MultinomialObjective &objective) {
  return MultinomialProblemView(
      objective.design_matrix(), objective.labels(), objective.class_num(),
      &objective);
}

const double kInexactNewtonForcingFactor = 0.1;
const double kFeatureCompactionFastTolerance = 1e-4;
// Incremental logits are algebraically equivalent to rebuilding X * beta.
// Keep only ultra-strict custom tolerances on the legacy rebuild path; the
// default high-accuracy 1e-7 mode is covered by the exact final KKT refresh.
const double kIncrementalLogitsMinimumTolerance = 1e-7;
const int kFeatureCompactionMaximumFeatureNum = 96;
const int kFeatureCompactionMaximumClassNum = 8;
const int kActiveGradientMinimumFeatureNum = 128;
const int kActiveGradientTallAspectRatio = 8;

double soft_threshold(double value, double threshold) {
  if (value > threshold) return value - threshold;
  if (value < -threshold) return value + threshold;
  return 0.0;
}

double l1_norm(const Eigen::MatrixXd &value) {
  return value.cwiseAbs().sum();
}

// Every median minimizes sum_k |beta_jk - shift|.  Odd K has a unique median;
// for even K use the midpoint of the two central values.  The midpoint avoids
// choosing an artificial L1 kink at either end of the flat median interval and
// is equivariant to sign changes and class permutations.
bool canonicalize_feature_l1_gauge(Eigen::MatrixXd *beta) {
  const Eigen::Index num_classes = beta->cols();
  const Eigen::Index lower_median_index = (num_classes - 1) / 2;
  const Eigen::Index upper_median_index = num_classes / 2;
  std::vector<double> values(static_cast<std::size_t>(num_classes));
  for (Eigen::Index feature = 0; feature < beta->rows(); ++feature) {
    for (Eigen::Index klass = 0; klass < num_classes; ++klass)
      values[static_cast<std::size_t>(klass)] = (*beta)(feature, klass);
    std::nth_element(values.begin(),
                     values.begin() + lower_median_index,
                     values.end());
    const double lower_median =
        values[static_cast<std::size_t>(lower_median_index)];
    double shift = lower_median;
    if (upper_median_index != lower_median_index) {
      std::nth_element(values.begin(),
                       values.begin() + upper_median_index,
                       values.end());
      const double upper_median =
          values[static_cast<std::size_t>(upper_median_index)];
      shift = 0.5 * lower_median + 0.5 * upper_median;
    }
    for (Eigen::Index klass = 0; klass < num_classes; ++klass)
      (*beta)(feature, klass) -= shift;
  }
  return beta->allFinite();
}

struct WeightedMedianEntry {
  double value;
  double weight;
};

struct WeightedMedianEntryLess {
  bool operator()(const WeightedMedianEntry &left,
                  const WeightedMedianEntry &right) const {
    return left.value < right.value;
  }
};

// A common feature shift leaves multinomial probabilities unchanged.  With
// coordinate-specific penalties, the minimum-penalty representative subtracts
// a weighted median of the class coefficients.  The midpoint of a flat median
// interval retains the scalar solver's symmetric gauge convention.
bool canonicalize_weighted_feature_l1_gauge(
    const Eigen::MatrixXd &l1_penalties, Eigen::MatrixXd *beta) {
  const Eigen::Index num_classes = beta->cols();
  const Eigen::Index lower_median_index = (num_classes - 1) / 2;
  const Eigen::Index upper_median_index = num_classes / 2;
  std::vector<WeightedMedianEntry> entries;
  entries.reserve(static_cast<std::size_t>(num_classes));
  std::vector<double> unweighted_values(
      static_cast<std::size_t>(num_classes));

  for (Eigen::Index feature = 0; feature < beta->rows(); ++feature) {
    entries.clear();
    long double total_weight = 0.0L;
    for (Eigen::Index klass = 0; klass < num_classes; ++klass) {
      const double weight = l1_penalties(feature, klass);
      if (weight > 0.0) {
        WeightedMedianEntry entry;
        entry.value = (*beta)(feature, klass);
        entry.weight = weight;
        entries.push_back(entry);
        total_weight += static_cast<long double>(weight);
      }
    }

    double shift = 0.0;
    if (entries.empty()) {
      // With a completely unpenalized row every common shift is optimal.
      // Retain the scalar solver's deterministic symmetric representative.
      for (Eigen::Index klass = 0; klass < num_classes; ++klass) {
        unweighted_values[static_cast<std::size_t>(klass)] =
            (*beta)(feature, klass);
      }
      std::nth_element(unweighted_values.begin(),
                       unweighted_values.begin() + lower_median_index,
                       unweighted_values.end());
      const double lower_median =
          unweighted_values[static_cast<std::size_t>(lower_median_index)];
      shift = lower_median;
      if (upper_median_index != lower_median_index) {
        std::nth_element(unweighted_values.begin(),
                         unweighted_values.begin() + upper_median_index,
                         unweighted_values.end());
        const double upper_median =
            unweighted_values[static_cast<std::size_t>(upper_median_index)];
        shift = 0.5 * lower_median + 0.5 * upper_median;
      }
    } else {
      if (!std::isfinite(total_weight)) return false;
      std::sort(entries.begin(), entries.end(), WeightedMedianEntryLess());
      const long double half_weight = 0.5L * total_weight;
      long double cumulative_weight = 0.0L;
      shift = entries.back().value;
      for (std::size_t index = 0; index < entries.size(); ++index) {
        cumulative_weight +=
            static_cast<long double>(entries[index].weight);
        if (cumulative_weight > half_weight) {
          shift = entries[index].value;
          break;
        }
        if (cumulative_weight == half_weight) {
          const double lower_median = entries[index].value;
          const double upper_median =
              index + 1 < entries.size()
                  ? entries[index + 1].value
                  : lower_median;
          shift = 0.5 * lower_median + 0.5 * upper_median;
          break;
        }
      }
    }

    for (Eigen::Index klass = 0; klass < num_classes; ++klass)
      (*beta)(feature, klass) -= shift;
  }
  return beta->allFinite();
}

struct ScalarPenalty {
  explicit ScalarPenalty(double value) : penalty(value) {}

  double coefficient(Eigen::Index, Eigen::Index) const {
    return penalty;
  }

  double value(const Eigen::MatrixXd &beta) const {
    // Keep the scalar reduction order unchanged for numerical compatibility.
    return penalty * l1_norm(beta);
  }

  double difference(const Eigen::MatrixXd &next_beta,
                    const Eigen::MatrixXd &current_beta) const {
    return penalty *
           (l1_norm(next_beta) - l1_norm(current_beta));
  }

  bool canonicalize(Eigen::MatrixXd *beta) const {
    return canonicalize_feature_l1_gauge(beta);
  }

  double penalty;
};

struct MatrixPenalty {
  explicit MatrixPenalty(const Eigen::MatrixXd &values)
      : penalties(values) {}

  double coefficient(Eigen::Index feature, Eigen::Index klass) const {
    return penalties(feature, klass);
  }

  double value(const Eigen::MatrixXd &beta) const {
    return (penalties.array() * beta.array().abs()).sum();
  }

  double difference(const Eigen::MatrixXd &next_beta,
                    const Eigen::MatrixXd &current_beta) const {
    return value(next_beta) - value(current_beta);
  }

  bool canonicalize(Eigen::MatrixXd *beta) const {
    return canonicalize_weighted_feature_l1_gauge(penalties, beta);
  }

  const Eigen::MatrixXd &penalties;
};

bool prefer_feature_resolution_compaction(
    const ScalarPenalty &,
    const MultinomialActNewtonOptions &options,
    int feature_num,
    int class_num) {
  return options.outer_kkt_tolerance >= kFeatureCompactionFastTolerance &&
         feature_num < kFeatureCompactionMaximumFeatureNum &&
         class_num < kFeatureCompactionMaximumClassNum;
}

bool prefer_feature_resolution_compaction(
    const MatrixPenalty &,
    const MultinomialActNewtonOptions &,
    int,
    int) {
  return false;
}

Eigen::VectorXd empirical_null_intercept(
    const MultinomialProblemView &objective) {
  Eigen::VectorXd intercept =
      Eigen::VectorXd::Zero(objective.class_num());
  const Eigen::VectorXi &labels = objective.labels();
  for (Eigen::Index i = 0; i < labels.size(); ++i)
    intercept[labels[i]] += 1.0;
  intercept.array() /= static_cast<double>(objective.sample_num());
  // A class absent from a low-level K-class call has no finite unpenalized
  // intercept MLE.  Match the legacy solver's finite initialization floor;
  // observed-class paths are unaffected by it.
  for (Eigen::Index klass = 0; klass < intercept.size(); ++klass)
    intercept[klass] = std::log(std::max(intercept[klass], 1e-8));
  intercept.array() -= intercept.mean();
  return intercept;
}

template <typename Penalty>
double total_objective(const MultinomialProblemView &objective,
                       const Eigen::MatrixXd &beta,
                       const Eigen::VectorXd &intercept,
                       const Penalty &penalty,
                       Eigen::MatrixXd *probabilities) {
  return objective.negative_log_likelihood(beta, intercept, probabilities) +
         penalty.value(beta);
}

double coefficient_kkt_residual(double coefficient, double gradient,
                                double penalty, double zero_tolerance) {
  if (coefficient > zero_tolerance)
    return std::fabs(gradient + penalty);
  if (coefficient < -zero_tolerance)
    return std::fabs(gradient - penalty);
  return std::max(0.0, std::fabs(gradient) - penalty);
}

template <typename Penalty>
double outer_kkt_residual(const Eigen::MatrixXd &beta,
                          const Eigen::MatrixXd &beta_gradient,
                          const Eigen::VectorXd &intercept_gradient,
                          const Penalty &penalty,
                          const MultinomialActNewtonOptions &options) {
  double residual = 0.0;
  for (Eigen::Index j = 0; j < beta.rows(); ++j) {
    for (Eigen::Index k = 0; k < beta.cols(); ++k) {
      residual = std::max(
          residual, coefficient_kkt_residual(
                        beta(j, k), beta_gradient(j, k),
                        penalty.coefficient(j, k),
                        options.zero_tolerance));
    }
  }
  if (options.include_intercept)
    residual = std::max(residual, intercept_gradient.cwiseAbs().maxCoeff());
  return residual;
}

typedef std::vector<unsigned char> MultinomialActiveSet;

template <typename Penalty>
double restricted_outer_kkt_residual(
    const Eigen::MatrixXd &beta,
    const Eigen::MatrixXd &beta_gradient,
    const Eigen::VectorXd &intercept_gradient,
    const Penalty &penalty,
    const MultinomialActNewtonOptions &options,
    const MultinomialActiveSet &active_set) {
  double residual = 0.0;
  for (Eigen::Index j = 0; j < beta.rows(); ++j) {
    if (active_set[static_cast<std::size_t>(j)] == 0) continue;
    for (Eigen::Index k = 0; k < beta.cols(); ++k) {
      residual = std::max(
          residual, coefficient_kkt_residual(
                        beta(j, k), beta_gradient(j, k),
                        penalty.coefficient(j, k),
                        options.zero_tolerance));
    }
  }
  if (options.include_intercept)
    residual = std::max(residual, intercept_gradient.cwiseAbs().maxCoeff());
  return residual;
}

int count_active_features(const MultinomialActiveSet &active_set) {
  int count = 0;
  for (std::size_t index = 0; index < active_set.size(); ++index)
    count += active_set[index] != 0 ? 1 : 0;
  return count;
}

void activate_nonzero_features(const Eigen::MatrixXd &beta,
                               double zero_tolerance,
                               MultinomialActiveSet *active_set) {
  for (Eigen::Index feature = 0; feature < beta.rows(); ++feature) {
    for (Eigen::Index klass = 0; klass < beta.cols(); ++klass) {
      if (std::fabs(beta(feature, klass)) > zero_tolerance) {
        (*active_set)[static_cast<std::size_t>(feature)] = 1;
        break;
      }
    }
  }
}

template <typename Penalty>
int activate_outer_kkt_features(
    const Eigen::MatrixXd &beta, const Eigen::MatrixXd &beta_gradient,
    const Penalty &penalty,
    const MultinomialActNewtonOptions &options,
    MultinomialActiveSet *active_set,
    double activation_tolerance = -1.0) {
  const double tolerance =
      activation_tolerance >= 0.0
          ? activation_tolerance
          : options.inner_kkt_tolerance;
  int newly_activated = 0;
  for (Eigen::Index feature = 0; feature < beta.rows(); ++feature) {
    if ((*active_set)[static_cast<std::size_t>(feature)] != 0) continue;
    bool activate = false;
    for (Eigen::Index klass = 0; klass < beta.cols(); ++klass) {
      const double residual = coefficient_kkt_residual(
          beta(feature, klass), beta_gradient(feature, klass),
          penalty.coefficient(feature, klass),
          options.zero_tolerance);
      if (std::fabs(beta(feature, klass)) > options.zero_tolerance ||
          residual > tolerance) {
        activate = true;
        break;
      }
    }
    if (activate) {
      (*active_set)[static_cast<std::size_t>(feature)] = 1;
      ++newly_activated;
    }
  }
  return newly_activated;
}

// Smooth gradient of the fixed IRLS quadratic at one direction coordinate.
// probability_dot_direction caches p_i^T Deta_i when nonnull.  A null pointer
// retains the O(nK)-per-coordinate implementation as a numerical A/B baseline.
template <typename Design>
double direction_coordinate_gradient(
    const Eigen::MatrixXd &probabilities,
    const Eigen::MatrixXd &linear_direction,
    const Eigen::VectorXd *probability_dot_direction,
    const Design &x, int feature, int klass, double base_gradient,
    double direction_coordinate, double damping,
    bool use_vectorized_kernel) {
  const int n = static_cast<int>(probabilities.rows());
  const int num_classes = static_cast<int>(probabilities.cols());
  const double inverse_n = 1.0 / static_cast<double>(n);
  double value = base_gradient + damping * direction_coordinate;
  if (use_vectorized_kernel && probability_dot_direction != 0) {
    if (feature < 0) {
      value +=
          (probabilities.col(klass).array() *
           (linear_direction.col(klass).array() -
            probability_dot_direction->array()) * inverse_n)
              .sum();
    } else {
      value +=
          (x.col(feature).array() * probabilities.col(klass).array() *
           (linear_direction.col(klass).array() -
            probability_dot_direction->array()) * inverse_n)
              .sum();
    }
    return value;
  }
  for (int i = 0; i < n; ++i) {
    double probability_dot_direction_value = 0.0;
    if (probability_dot_direction) {
      probability_dot_direction_value = (*probability_dot_direction)[i];
    } else {
      for (int other_class = 0; other_class < num_classes; ++other_class) {
        probability_dot_direction_value +=
            probabilities(i, other_class) *
            linear_direction(i, other_class);
      }
    }
    const double design_value = feature < 0 ? 1.0 : x(i, feature);
    value += design_value * probabilities(i, klass) *
             (linear_direction(i, klass) - probability_dot_direction_value) /
             static_cast<double>(n);
  }
  return value;
}

template <typename Design>
void update_linear_direction_coordinate(
    const Eigen::MatrixXd &probabilities, const Design &x,
    int feature, int klass, double change,
    Eigen::MatrixXd *linear_direction,
    Eigen::VectorXd *probability_dot_direction,
    bool use_vectorized_kernel) {
  const int n = static_cast<int>(probabilities.rows());
  if (use_vectorized_kernel) {
    if (feature < 0) {
      linear_direction->col(klass).array() += change;
      if (probability_dot_direction)
        probability_dot_direction->array() +=
            change * probabilities.col(klass).array();
    } else {
      linear_direction->col(klass).noalias() += change * x.col(feature);
      if (probability_dot_direction)
        probability_dot_direction->array() +=
            change * probabilities.col(klass).array() *
            x.col(feature).array();
    }
    return;
  }
  for (int i = 0; i < n; ++i) {
    const double design_value = feature < 0 ? 1.0 : x(i, feature);
    const double linear_change = change * design_value;
    (*linear_direction)(i, klass) += linear_change;
    if (probability_dot_direction)
      (*probability_dot_direction)[i] +=
          probabilities(i, klass) * linear_change;
  }
}

void center_intercept_direction(
    const Eigen::MatrixXd &probabilities, double common_shift,
    Eigen::MatrixXd *linear_direction,
    Eigen::VectorXd *probability_dot_direction) {
  linear_direction->array() -= common_shift;
  if (!probability_dot_direction) return;

  // Use the represented probability row sum instead of assuming exact unit
  // sum, so the cache undergoes precisely the same full-K gauge shift.
  for (Eigen::Index i = 0; i < probabilities.rows(); ++i)
    (*probability_dot_direction)[i] -=
        common_shift * probabilities.row(i).sum();
}

template <typename Design>
void compute_coordinate_curvatures(
    const Eigen::MatrixXd &probabilities, const Design &x,
    double damping, const MultinomialActiveSet *active_features,
    Eigen::MatrixXd *beta_curvature,
    Eigen::VectorXd *intercept_curvature) {
  const int n = static_cast<int>(probabilities.rows());
  const int d = static_cast<int>(x.cols());
  const int num_classes = static_cast<int>(probabilities.cols());
  beta_curvature->resize(d, num_classes);
  intercept_curvature->resize(num_classes);
  beta_curvature->setConstant(damping);
  intercept_curvature->setConstant(damping);
  const double inverse_n = 1.0 / static_cast<double>(n);

  // Unrestricted solves may activate a feature inside this fixed quadratic,
  // so retain the full precomputation and its original reduction order.
  if (active_features == 0) {
    for (int i = 0; i < n; ++i) {
      for (int klass = 0; klass < num_classes; ++klass) {
        const double variance = probabilities(i, klass) *
                                (1.0 - probabilities(i, klass)) * inverse_n;
        (*intercept_curvature)[klass] += variance;
        for (int j = 0; j < d; ++j)
          (*beta_curvature)(j, klass) +=
              variance * x(i, j) * x(i, j);
      }
    }
    return;
  }

  // Restricted path solves cannot grow the feature set until the next outer
  // IRLS iteration.  Compute only the rows that the quadratic solver can
  // visit.  Keeping each coordinate's observation reduction in the original
  // order preserves the scalar curvature result without storing X squared.
  for (int i = 0; i < n; ++i) {
    for (int klass = 0; klass < num_classes; ++klass) {
      const double variance = probabilities(i, klass) *
                              (1.0 - probabilities(i, klass)) * inverse_n;
      (*intercept_curvature)[klass] += variance;
    }
  }
  for (int j = 0; j < d; ++j) {
    if ((*active_features)[static_cast<std::size_t>(j)] == 0) continue;
    for (int klass = 0; klass < num_classes; ++klass) {
      for (int i = 0; i < n; ++i) {
        const double variance = probabilities(i, klass) *
                                (1.0 - probabilities(i, klass)) * inverse_n;
        (*beta_curvature)(j, klass) +=
            variance * x(i, j) * x(i, j);
      }
    }
  }
}

struct SubproblemKktScan {
  double residual;
  int newly_activated_features;
};

template <typename Design, typename Penalty>
SubproblemKktScan scan_subproblem_kkt(
    const Eigen::MatrixXd &base_beta,
    const Eigen::MatrixXd &beta_direction,
    const Eigen::VectorXd &intercept_direction,
    const Eigen::MatrixXd &linear_direction,
    const Eigen::VectorXd *probability_dot_direction,
    const Eigen::MatrixXd &probabilities,
    const Eigen::MatrixXd &beta_gradient,
    const Eigen::VectorXd &intercept_gradient,
    const Design &x, const Penalty &penalty,
    const MultinomialActNewtonOptions &options,
    bool full_scan, MultinomialActiveSet *active_set) {
  SubproblemKktScan scan;
  scan.residual = 0.0;
  scan.newly_activated_features = 0;
  for (Eigen::Index j = 0; j < base_beta.rows(); ++j) {
    const bool feature_was_active =
        (*active_set)[static_cast<std::size_t>(j)] != 0;
    if (!full_scan && !feature_was_active) continue;
    bool activate_feature = false;
    for (Eigen::Index k = 0; k < base_beta.cols(); ++k) {
      const double gradient = direction_coordinate_gradient(
          probabilities, linear_direction, probability_dot_direction, x,
          static_cast<int>(j),
          static_cast<int>(k), beta_gradient(j, k), beta_direction(j, k),
          options.hessian_damping,
          options.use_vectorized_coordinate_kernels);
      const double coordinate_residual = coefficient_kkt_residual(
          base_beta(j, k) + beta_direction(j, k), gradient,
          penalty.coefficient(j, k),
          options.zero_tolerance);
      scan.residual = std::max(scan.residual, coordinate_residual);
      if (full_scan && !feature_was_active &&
          coordinate_residual > options.inner_kkt_tolerance)
        activate_feature = true;
    }
    if (activate_feature) {
      (*active_set)[static_cast<std::size_t>(j)] = 1;
      ++scan.newly_activated_features;
    }
  }
  if (options.include_intercept) {
    for (Eigen::Index k = 0; k < intercept_direction.size(); ++k) {
      const double gradient = direction_coordinate_gradient(
          probabilities, linear_direction, probability_dot_direction, x, -1,
          static_cast<int>(k),
          intercept_gradient[k], intercept_direction[k],
          options.hessian_damping,
          options.use_vectorized_coordinate_kernels);
      scan.residual = std::max(scan.residual, std::fabs(gradient));
    }
  }
  return scan;
}

struct QuadraticSubproblemResult {
  Eigen::MatrixXd beta_direction;
  Eigen::VectorXd intercept_direction;
  Eigen::MatrixXd linear_direction;
  int sweeps;
  long long coordinate_updates;
  double kkt_residual;
  int initial_active_features;
  int final_active_features;
  int reactivated_features;
  int full_kkt_scans;
  bool converged;
  bool finite;
};

template <bool use_coefficient_compaction, typename Design,
          typename Penalty>
QuadraticSubproblemResult solve_quadratic_subproblem_impl(
    const MultinomialProblemView &objective,
    const Design &x,
    const Eigen::MatrixXd &base_beta,
    const Eigen::MatrixXd &probabilities,
    const Eigen::MatrixXd &beta_gradient,
    const Eigen::VectorXd &intercept_gradient,
    const Penalty &penalty,
    const MultinomialActNewtonOptions &options,
    MultinomialActiveSet *active_set,
    bool allow_inactive_reactivation) {
  const int n = objective.sample_num();
  const int d = objective.feature_num();
  const int num_classes = objective.class_num();

  QuadraticSubproblemResult result;
  result.beta_direction = Eigen::MatrixXd::Zero(d, num_classes);
  result.intercept_direction = Eigen::VectorXd::Zero(num_classes);
  result.sweeps = 0;
  result.coordinate_updates = 0;
  result.kkt_residual = std::numeric_limits<double>::infinity();
  result.initial_active_features = count_active_features(*active_set);
  result.final_active_features = result.initial_active_features;
  result.reactivated_features = 0;
  result.full_kkt_scans = 0;
  result.converged = false;
  result.finite = true;

  result.linear_direction = Eigen::MatrixXd::Zero(n, num_classes);
  Eigen::MatrixXd &linear_direction = result.linear_direction;
  Eigen::VectorXd probability_dot_direction;
  Eigen::VectorXd *probability_dot_direction_ptr = 0;
  if (options.use_probability_dot_direction_cache) {
    probability_dot_direction = Eigen::VectorXd::Zero(n);
    probability_dot_direction_ptr = &probability_dot_direction;
  }
  Eigen::MatrixXd beta_curvature;
  Eigen::VectorXd intercept_curvature;
  const MultinomialActiveSet *curvature_active_features =
      !allow_inactive_reactivation &&
              result.initial_active_features <= d / 2
          ? active_set
          : 0;
  compute_coordinate_curvatures(probabilities, x, options.hessian_damping,
                                curvature_active_features,
                                &beta_curvature, &intercept_curvature);

  // Keep the feature-level strong set authoritative, as in glmnet.  Small,
  // low-class scalar-L1 fits at fast precision use the lower-overhead original
  // feature list; strict, wide, high-class, or coordinate-weighted fits visit
  // only coefficients that have moved.  Both schedules require a full
  // candidate-feature by class sweep before exact KKT certification, so
  // cross-class Hessian coupling cannot hide a violation.
  const bool use_compact_sweeps =
      options.use_active_set && options.use_compact_inner_active_set &&
      !allow_inactive_reactivation;
  bool use_compact_coordinates =
      use_compact_sweeps && use_coefficient_compaction;
  bool use_compact_features =
      use_compact_sweeps && !use_coefficient_compaction;
  MultinomialActiveSet compact_feature_mask;
  std::vector<int> compact_features;
  std::vector<int> candidate_features;
  MultinomialActiveSet compact_coordinate_mask;
  std::vector<int> compact_coordinates;
  std::size_t candidate_coordinate_count = 0;
  std::size_t compact_coordinate_limit = 0;
  bool compact_coordinate_limit_reached = false;
  if (use_compact_features) {
    compact_feature_mask.assign(static_cast<std::size_t>(d), 0);
    compact_features.reserve(
        static_cast<std::size_t>(result.initial_active_features));
    for (int feature = 0; feature < d; ++feature) {
      if ((*active_set)[static_cast<std::size_t>(feature)] == 0) continue;
      for (int klass = 0; klass < num_classes; ++klass) {
        if (std::fabs(base_beta(feature, klass)) > options.zero_tolerance) {
          compact_feature_mask[static_cast<std::size_t>(feature)] = 1;
          compact_features.push_back(feature);
          break;
        }
      }
    }
  }
  if (use_compact_coordinates) {
    candidate_features.reserve(
        static_cast<std::size_t>(result.initial_active_features));
    for (int feature = 0; feature < d; ++feature) {
      if ((*active_set)[static_cast<std::size_t>(feature)] == 0) continue;
      candidate_features.push_back(feature);
    }
    const std::size_t class_count =
        static_cast<std::size_t>(num_classes);
    const std::size_t maximum_coordinate_count =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (candidate_features.size() > maximum_coordinate_count / class_count) {
      use_compact_coordinates = false;
      std::vector<int>().swap(candidate_features);
    } else {
      candidate_coordinate_count = candidate_features.size() * class_count;
    }
  }
  if (use_compact_coordinates) {
    compact_coordinate_mask.assign(candidate_coordinate_count, 0);
    compact_coordinate_limit =
        candidate_coordinate_count - candidate_coordinate_count / 4;
    const auto register_coordinate = [&](std::size_t coordinate) {
      if (compact_coordinate_mask[coordinate] != 0) return;
      if (compact_coordinates.size() + 1 >= compact_coordinate_limit) {
        compact_coordinate_limit_reached = true;
        return;
      }
      compact_coordinate_mask[coordinate] = 1;
      compact_coordinates.push_back(static_cast<int>(coordinate));
    };
    for (std::size_t candidate = 0;
         candidate < candidate_features.size(); ++candidate) {
      const int feature = candidate_features[candidate];
      for (int klass = 0; klass < num_classes; ++klass) {
        if (std::fabs(base_beta(feature, klass)) > options.zero_tolerance) {
          const std::size_t coordinate =
              candidate * static_cast<std::size_t>(num_classes) +
              static_cast<std::size_t>(klass);
          register_coordinate(coordinate);
        }
      }
    }
  }
  bool sweep_all_compact_candidates = use_compact_sweeps;

  for (int sweep = 1; sweep <= options.max_inner_sweeps; ++sweep) {
    double maximum_coordinate_change_proxy = 0.0;
    if (options.include_intercept) {
      for (int klass = 0; klass < num_classes; ++klass) {
        const double curvature = intercept_curvature[klass];
        if (!(curvature > 0.0) || !std::isfinite(curvature)) {
          result.finite = false;
          return result;
        }
        const double old_direction = result.intercept_direction[klass];
        const double gradient = direction_coordinate_gradient(
            probabilities, linear_direction, probability_dot_direction_ptr,
            x, -1, klass,
            intercept_gradient[klass], old_direction,
            options.hessian_damping,
            options.use_vectorized_coordinate_kernels);
        const double new_direction = old_direction - gradient / curvature;
        const double change = new_direction - old_direction;
        if (!std::isfinite(new_direction)) {
          result.finite = false;
          return result;
        }
        maximum_coordinate_change_proxy = std::max(
            maximum_coordinate_change_proxy,
            curvature * std::fabs(change));
        if (change != 0.0) {
          result.intercept_direction[klass] = new_direction;
          update_linear_direction_coordinate(
              probabilities, x, -1, klass, change, &linear_direction,
              probability_dot_direction_ptr,
              options.use_vectorized_coordinate_kernels);
        }
        ++result.coordinate_updates;
      }
      // The multinomial likelihood is invariant to a common intercept shift.
      // Damping selects the zero-mean representative; enforce it explicitly
      // so the redundant full-K parameterization cannot drift numerically.
      const double common_shift = result.intercept_direction.mean();
      maximum_coordinate_change_proxy = std::max(
          maximum_coordinate_change_proxy,
          intercept_curvature.maxCoeff() * std::fabs(common_shift));
      result.intercept_direction.array() -= common_shift;
      center_intercept_direction(
          probabilities, common_shift, &linear_direction,
          probability_dot_direction_ptr);
    }

    const bool compact_sweep_mode =
        use_compact_features || use_compact_coordinates;
    bool performed_candidate_sweep = true;
    if (!compact_sweep_mode) {
      for (int j = 0; j < d; ++j) {
        if ((*active_set)[static_cast<std::size_t>(j)] == 0) continue;
        for (int klass = 0; klass < num_classes; ++klass) {
          const double curvature = beta_curvature(j, klass);
          if (!(curvature > 0.0) || !std::isfinite(curvature)) {
            result.finite = false;
            return result;
          }
          const double old_direction = result.beta_direction(j, klass);
          const double gradient = direction_coordinate_gradient(
              probabilities, linear_direction,
              probability_dot_direction_ptr, x, j, klass,
              beta_gradient(j, klass), old_direction,
              options.hessian_damping,
              options.use_vectorized_coordinate_kernels);
          const double current_coefficient =
              base_beta(j, klass) + old_direction;
          const double new_coefficient = soft_threshold(
              current_coefficient - gradient / curvature,
              penalty.coefficient(j, klass) / curvature);
          const double new_direction =
              new_coefficient - base_beta(j, klass);
          const double change = new_direction - old_direction;
          if (!std::isfinite(new_direction)) {
            result.finite = false;
            return result;
          }
          maximum_coordinate_change_proxy = std::max(
              maximum_coordinate_change_proxy,
              curvature * std::fabs(change));
          if (change != 0.0) {
            result.beta_direction(j, klass) = new_direction;
            update_linear_direction_coordinate(
                probabilities, x, j, klass, change, &linear_direction,
                probability_dot_direction_ptr,
                options.use_vectorized_coordinate_kernels);
          }
          ++result.coordinate_updates;
        }
      }
    } else if (use_compact_features) {
      performed_candidate_sweep =
          sweep_all_compact_candidates || sweep == options.max_inner_sweeps;
      const auto update_compact_feature = [&](int j) -> bool {
        bool feature_changed = false;
        for (int klass = 0; klass < num_classes; ++klass) {
          const double curvature = beta_curvature(j, klass);
          if (!(curvature > 0.0) || !std::isfinite(curvature)) {
            result.finite = false;
            return false;
          }
          const double old_direction = result.beta_direction(j, klass);
          const double gradient = direction_coordinate_gradient(
              probabilities, linear_direction,
              probability_dot_direction_ptr, x, j, klass,
              beta_gradient(j, klass), old_direction,
              options.hessian_damping,
              options.use_vectorized_coordinate_kernels);
          const double current_coefficient =
              base_beta(j, klass) + old_direction;
          const double new_coefficient = soft_threshold(
              current_coefficient - gradient / curvature,
              penalty.coefficient(j, klass) / curvature);
          const double new_direction =
              new_coefficient - base_beta(j, klass);
          const double change = new_direction - old_direction;
          if (!std::isfinite(new_direction)) {
            result.finite = false;
            return false;
          }
          maximum_coordinate_change_proxy = std::max(
              maximum_coordinate_change_proxy,
              curvature * std::fabs(change));
          if (change != 0.0) {
            feature_changed = true;
            result.beta_direction(j, klass) = new_direction;
            update_linear_direction_coordinate(
                probabilities, x, j, klass, change, &linear_direction,
                probability_dot_direction_ptr,
                options.use_vectorized_coordinate_kernels);
          }
          ++result.coordinate_updates;
        }
        if (feature_changed &&
            compact_feature_mask[static_cast<std::size_t>(j)] == 0) {
          compact_feature_mask[static_cast<std::size_t>(j)] = 1;
          compact_features.push_back(j);
        }
        return true;
      };
      if (performed_candidate_sweep) {
        for (int j = 0; j < d; ++j) {
          if ((*active_set)[static_cast<std::size_t>(j)] == 0) continue;
          if (!update_compact_feature(j)) return result;
        }
      } else {
        for (std::size_t index = 0; index < compact_features.size(); ++index) {
          if (!update_compact_feature(compact_features[index])) return result;
        }
      }
    } else {
      performed_candidate_sweep =
          sweep_all_compact_candidates ||
          sweep == options.max_inner_sweeps;
      const auto update_compact_coordinate =
          [&](int candidate, int klass, bool register_change) -> bool {
          const int j = candidate_features[static_cast<std::size_t>(candidate)];
          const double curvature = beta_curvature(j, klass);
          if (!(curvature > 0.0) || !std::isfinite(curvature)) {
            result.finite = false;
            return false;
          }
          const double old_direction = result.beta_direction(j, klass);
          const double gradient = direction_coordinate_gradient(
              probabilities, linear_direction,
              probability_dot_direction_ptr, x, j, klass,
              beta_gradient(j, klass), old_direction,
              options.hessian_damping,
              options.use_vectorized_coordinate_kernels);
          const double current_coefficient =
              base_beta(j, klass) + old_direction;
          const double new_coefficient = soft_threshold(
              current_coefficient - gradient / curvature,
              penalty.coefficient(j, klass) / curvature);
          const double new_direction =
              new_coefficient - base_beta(j, klass);
          const double change = new_direction - old_direction;
          if (!std::isfinite(new_direction)) {
            result.finite = false;
            return false;
          }
          maximum_coordinate_change_proxy = std::max(
              maximum_coordinate_change_proxy,
              curvature * std::fabs(change));
          if (change != 0.0) {
            result.beta_direction(j, klass) = new_direction;
            update_linear_direction_coordinate(
                probabilities, x, j, klass, change, &linear_direction,
                probability_dot_direction_ptr,
                options.use_vectorized_coordinate_kernels);
            const std::size_t coordinate =
                static_cast<std::size_t>(candidate) *
                    static_cast<std::size_t>(num_classes) +
                static_cast<std::size_t>(klass);
            if (register_change &&
                compact_coordinate_mask[coordinate] == 0) {
              if (compact_coordinates.size() + 1 >=
                  compact_coordinate_limit) {
                compact_coordinate_limit_reached = true;
              } else {
                compact_coordinate_mask[coordinate] = 1;
                compact_coordinates.push_back(
                    static_cast<int>(coordinate));
              }
            }
          }
          ++result.coordinate_updates;
          return true;
      };
      if (performed_candidate_sweep) {
        for (std::size_t candidate = 0;
             candidate < candidate_features.size(); ++candidate) {
          for (int klass = 0; klass < num_classes; ++klass) {
            if (!update_compact_coordinate(
                    static_cast<int>(candidate), klass, true))
              return result;
          }
        }
      } else {
        for (std::size_t index = 0; index < compact_coordinates.size();
             ++index) {
          const int coordinate = compact_coordinates[index];
          const int candidate = coordinate / num_classes;
          const int klass = coordinate % num_classes;
          if (!update_compact_coordinate(candidate, klass, false))
            return result;
        }
      }
    }

    // A second tier only pays when it is materially smaller than the strong
    // candidate set.  Dense quadratics otherwise add compact/full KKT
    // transitions without saving vector work, so fall back to full candidate
    // sweeps after the first pass.  Stop registering at the 75% gate and
    // release the temporary lists immediately to bound dense-case memory.
    if (use_compact_features && performed_candidate_sweep &&
        result.initial_active_features > 0) {
      const std::size_t compact_feature_limit =
          static_cast<std::size_t>(result.initial_active_features) -
          static_cast<std::size_t>(result.initial_active_features) / 4;
      if (compact_features.size() >= compact_feature_limit) {
        use_compact_features = false;
        sweep_all_compact_candidates = false;
        MultinomialActiveSet().swap(compact_feature_mask);
        std::vector<int>().swap(compact_features);
      }
    }
    if (use_compact_coordinates && performed_candidate_sweep &&
        result.initial_active_features > 0 &&
        (compact_coordinate_limit_reached ||
         compact_coordinates.size() >= compact_coordinate_limit)) {
      use_compact_coordinates = false;
      sweep_all_compact_candidates = false;
      std::vector<int>().swap(candidate_features);
      MultinomialActiveSet().swap(compact_coordinate_mask);
      std::vector<int>().swap(compact_coordinates);
    }

    result.sweeps = sweep;
    // Match glmnet's partial/full WLS sweep schedule: partial sweeps use the
    // cheap maximum-coordinate-change proxy, then a full candidate sweep is
    // required before an exact KKT certificate can stop the subproblem.
    if (use_compact_features || use_compact_coordinates) {
      if (!performed_candidate_sweep) {
        if (maximum_coordinate_change_proxy <=
            options.inner_kkt_tolerance)
          sweep_all_compact_candidates = true;
        continue;
      }
      if (maximum_coordinate_change_proxy > options.inner_kkt_tolerance &&
          sweep < options.max_inner_sweeps) {
        sweep_all_compact_candidates = false;
        continue;
      }
    }
    const bool exact_scan_due =
        (compact_sweep_mode && performed_candidate_sweep) || sweep == 1 ||
        sweep == options.max_inner_sweeps ||
        sweep % options.exact_kkt_scan_interval == 0 ||
        maximum_coordinate_change_proxy <= options.inner_kkt_tolerance;
    if (!exact_scan_due) continue;

    const SubproblemKktScan active_scan = scan_subproblem_kkt(
        base_beta, result.beta_direction, result.intercept_direction,
        linear_direction, probability_dot_direction_ptr, probabilities,
        beta_gradient, intercept_gradient, x, penalty, options, false,
        active_set);
    result.kkt_residual = active_scan.residual;
    if (!std::isfinite(result.kkt_residual)) {
      result.finite = false;
      return result;
    }
    if (result.kkt_residual <= options.inner_kkt_tolerance) {
      if (!options.use_active_set || !allow_inactive_reactivation) {
        result.converged = true;
        return result;
      }
      const SubproblemKktScan full_scan = scan_subproblem_kkt(
          base_beta, result.beta_direction, result.intercept_direction,
          linear_direction, probability_dot_direction_ptr, probabilities,
          beta_gradient, intercept_gradient, x, penalty, options, true,
          active_set);
      ++result.full_kkt_scans;
      result.reactivated_features += full_scan.newly_activated_features;
      result.final_active_features = count_active_features(*active_set);
      result.kkt_residual = full_scan.residual;
      if (!std::isfinite(result.kkt_residual)) {
        result.finite = false;
        return result;
      }
      if (result.kkt_residual <= options.inner_kkt_tolerance) {
        result.converged = true;
        return result;
      }
    }
    if ((use_compact_features || use_compact_coordinates) &&
        performed_candidate_sweep)
      sweep_all_compact_candidates = false;
  }
  if (!options.use_active_set || !allow_inactive_reactivation) return result;
  const SubproblemKktScan final_scan = scan_subproblem_kkt(
      base_beta, result.beta_direction, result.intercept_direction,
      linear_direction, probability_dot_direction_ptr, probabilities,
      beta_gradient, intercept_gradient, x, penalty, options, true,
      active_set);
  ++result.full_kkt_scans;
  result.reactivated_features += final_scan.newly_activated_features;
  result.final_active_features = count_active_features(*active_set);
  result.kkt_residual = final_scan.residual;
  if (!std::isfinite(result.kkt_residual)) result.finite = false;
  return result;
}

template <typename Design, typename Penalty>
QuadraticSubproblemResult solve_quadratic_subproblem_with_design(
    const MultinomialProblemView &objective,
    const Design &x,
    const Eigen::MatrixXd &base_beta,
    const Eigen::MatrixXd &probabilities,
    const Eigen::MatrixXd &beta_gradient,
    const Eigen::VectorXd &intercept_gradient,
    const Penalty &penalty,
    const MultinomialActNewtonOptions &options,
    MultinomialActiveSet *active_set,
    bool allow_inactive_reactivation) {
  // Dispatch once so each hot coordinate loop is compiled without the other
  // compact schedule in its instruction stream.  Feature-resolution lists
  // only win for small, low-class scalar-L1 problems at fast precision.
  // Strict solves and coordinate-specific LLA surrogates retain coefficient
  // resolution, where their class supports can diverge materially.
  const bool use_coefficient_compaction =
      !prefer_feature_resolution_compaction(
          penalty, options, objective.feature_num(), objective.class_num());
  if (use_coefficient_compaction) {
    return solve_quadratic_subproblem_impl<true>(
        objective, x, base_beta, probabilities, beta_gradient,
        intercept_gradient, penalty, options, active_set,
        allow_inactive_reactivation);
  }
  return solve_quadratic_subproblem_impl<false>(
      objective, x, base_beta, probabilities, beta_gradient,
      intercept_gradient, penalty, options, active_set,
      allow_inactive_reactivation);
}

template <typename Penalty>
QuadraticSubproblemResult solve_quadratic_subproblem(
    const MultinomialProblemView &objective,
    const Eigen::MatrixXd &base_beta,
    const Eigen::MatrixXd &probabilities,
    const Eigen::MatrixXd &beta_gradient,
    const Eigen::VectorXd &intercept_gradient,
    const Penalty &penalty,
    const MultinomialActNewtonOptions &options,
    MultinomialActiveSet *active_set,
    bool allow_inactive_reactivation) {
  const MultinomialProblemView::ConstDesignMap mapped_design =
      objective.mapped_design_matrix();
  return solve_quadratic_subproblem_with_design(
      objective, mapped_design, base_beta, probabilities, beta_gradient,
      intercept_gradient, penalty, options, active_set,
      allow_inactive_reactivation);
}

MultinomialIterationRecord initial_record(double objective,
                                          double kkt_residual) {
  MultinomialIterationRecord record;
  record.outer_iteration = 0;
  record.inner_sweeps = 0;
  record.line_search_steps = 0;
  record.objective = objective;
  record.kkt_residual = kkt_residual;
  record.inner_kkt_residual = 0.0;
  record.step_size = 0.0;
  record.direction_norm = 0.0;
  record.composite_directional_derivative = 0.0;
  record.inner_converged = true;
  record.active_features = 0;
  record.newly_activated_features = 0;
  record.subproblem_reactivated_features = 0;
  record.outer_reactivated_features = 0;
  record.full_subproblem_kkt_scans = 0;
  return record;
}

bool valid_iterate_cache(
    const detail::MultinomialPathSmoothCache &cache,
    const MultinomialProblemView &objective,
    const Eigen::MatrixXd &beta,
    const Eigen::VectorXd &intercept,
    bool require_logits) {
  const bool valid_logits =
      !require_logits ||
      (cache.logits.rows() == objective.sample_num() &&
       cache.logits.cols() == objective.class_num() &&
       cache.logits.allFinite());
  return valid_logits && cache.problem_identity == objective.identity() &&
         cache.beta_snapshot.rows() == beta.rows() &&
         cache.beta_snapshot.cols() == beta.cols() &&
         cache.intercept_snapshot.size() == intercept.size() &&
         cache.probabilities.rows() == objective.sample_num() &&
         cache.probabilities.cols() == objective.class_num() &&
         cache.beta_gradient.rows() == objective.feature_num() &&
         cache.beta_gradient.cols() == objective.class_num() &&
         cache.intercept_gradient.size() == objective.class_num() &&
         cache.probabilities.allFinite() &&
         cache.beta_gradient.allFinite() &&
         cache.intercept_gradient.allFinite() &&
         cache.beta_snapshot.allFinite() &&
         cache.intercept_snapshot.allFinite() &&
         (cache.beta_snapshot.array() == beta.array()).all() &&
         (cache.intercept_snapshot.array() == intercept.array()).all() &&
         std::isfinite(cache.smooth_negative_log_likelihood);
}

void save_iterate_cache(
    double smooth_negative_log_likelihood,
    const MultinomialProblemView &objective,
    const Eigen::MatrixXd &beta,
    const Eigen::VectorXd &intercept,
    Eigen::MatrixXd *logits,
    Eigen::MatrixXd *probabilities,
    Eigen::MatrixXd *beta_gradient,
    Eigen::VectorXd *intercept_gradient,
    std::shared_ptr<const detail::MultinomialPathSmoothCache> *cache) {
  if (cache == 0) return;
  std::shared_ptr<detail::MultinomialPathSmoothCache> candidate(
      new detail::MultinomialPathSmoothCache());
  candidate->problem_identity = objective.identity();
  candidate->beta_snapshot = beta;
  candidate->intercept_snapshot = intercept;
  if (logits != 0) {
    // A common per-row shift leaves the softmax unchanged.  Remove it before
    // carrying logits across lambda values so gauge canonicalization cannot
    // accumulate a large, numerically harmful offset along a long path.
    for (Eigen::Index row = 0; row < logits->rows(); ++row)
      logits->row(row).array() -= logits->row(row).maxCoeff();
    candidate->logits.swap(*logits);
  }
  candidate->probabilities.swap(*probabilities);
  candidate->beta_gradient.swap(*beta_gradient);
  candidate->intercept_gradient.swap(*intercept_gradient);
  candidate->smooth_negative_log_likelihood =
      smooth_negative_log_likelihood;
  *cache = candidate;
}

template <typename Penalty>
MultinomialActNewtonResult solve_impl(
    const MultinomialProblemView &objective,
    const MultinomialActNewtonOptions &options,
    const Penalty &penalty,
    const Eigen::MatrixXd &initial_beta,
    const Eigen::VectorXd &initial_intercept,
    const MultinomialActiveSet *initial_active_features = 0,
    bool eagerly_identify_active_features = false,
    Eigen::VectorXd *final_feature_gradient_max = 0,
    const detail::MultinomialPathSmoothCache *initial_iterate_cache = 0,
    std::shared_ptr<const detail::MultinomialPathSmoothCache>
        *final_iterate_cache = 0,
    bool *reused_initial_iterate_cache = 0) {
  if (initial_beta.rows() != objective.feature_num() ||
      initial_beta.cols() != objective.class_num())
    throw std::invalid_argument(
        "initial multinomial beta has the wrong shape");
  if (initial_intercept.size() != objective.class_num())
    throw std::invalid_argument(
        "initial multinomial intercept has the wrong length");
  if (!initial_beta.allFinite() || !initial_intercept.allFinite())
    throw std::invalid_argument(
        "initial multinomial parameters must be finite");
  if (initial_active_features != 0 &&
      initial_active_features->size() !=
          static_cast<std::size_t>(objective.feature_num()))
    throw std::invalid_argument(
        "initial multinomial active set has the wrong length");

  const bool restricted_path_mode =
      options.use_active_set && initial_active_features != 0;
  const bool use_incremental_logits =
      options.reuse_line_search_probabilities &&
      options.outer_kkt_tolerance >= kIncrementalLogitsMinimumTolerance;

  MultinomialActNewtonResult result;
  result.beta = initial_beta;
  result.intercept = initial_intercept;
  const bool intercept_reset_preserves_smooth_state =
      options.include_intercept ||
      result.intercept.maxCoeff() == result.intercept.minCoeff();
  const bool reuse_initial_iterate =
      initial_iterate_cache != 0 &&
      intercept_reset_preserves_smooth_state &&
      valid_iterate_cache(*initial_iterate_cache, objective, result.beta,
                          result.intercept, use_incremental_logits);
  if (reused_initial_iterate_cache != 0)
    *reused_initial_iterate_cache = reuse_initial_iterate;
  if (options.include_intercept)
    result.intercept.array() -= result.intercept.mean();
  else
    result.intercept.setZero();
  result.status = MultinomialSolverStatus::kOuterIterationLimit;
  result.outer_iterations = 0;
  result.total_inner_sweeps = 0;
  result.total_coordinate_updates = 0;
  result.final_objective = std::numeric_limits<double>::quiet_NaN();
  result.final_kkt_residual = std::numeric_limits<double>::infinity();
  result.final_active_features = 0;
  result.initial_active_features = 0;
  result.total_reactivated_features = 0;
  result.total_subproblem_reactivated_features = 0;
  result.total_outer_reactivated_features = 0;
  result.total_full_subproblem_kkt_scans = 0;
  result.active_features.assign(
      static_cast<std::size_t>(objective.feature_num()), 0);
  if (options.canonicalize_feature_l1_gauge &&
      !penalty.canonicalize(&result.beta)) {
    result.status = MultinomialSolverStatus::kNumericalFailure;
    return result;
  }
  Eigen::MatrixXd current_logits;
  Eigen::MatrixXd probabilities;
  Eigen::MatrixXd beta_gradient;
  Eigen::VectorXd intercept_gradient;
  // In restricted path mode inactive beta-gradient rows may deliberately
  // retain the last full certificate between accepted outer iterates.
  bool gradient_is_full = true;
  double current_negative_log_likelihood =
      std::numeric_limits<double>::quiet_NaN();
  if (reuse_initial_iterate) {
    if (use_incremental_logits)
      current_logits = initial_iterate_cache->logits;
    probabilities = initial_iterate_cache->probabilities;
    beta_gradient = initial_iterate_cache->beta_gradient;
    intercept_gradient = initial_iterate_cache->intercept_gradient;
    current_negative_log_likelihood =
        initial_iterate_cache->smooth_negative_log_likelihood;
    result.final_objective =
        current_negative_log_likelihood + penalty.value(result.beta);
  } else {
    try {
      if (use_incremental_logits) {
        objective.linear_predictor(result.beta, result.intercept,
                                   &current_logits);
        current_negative_log_likelihood =
            objective.negative_log_likelihood_from_logits(
                current_logits, &probabilities);
      } else {
        current_negative_log_likelihood = objective.negative_log_likelihood(
            result.beta, result.intercept, &probabilities);
      }
      result.final_objective =
          current_negative_log_likelihood + penalty.value(result.beta);
    } catch (const std::invalid_argument &) {
      result.status = MultinomialSolverStatus::kNumericalFailure;
      return result;
    }
    objective.smooth_gradient_from_probabilities(
        probabilities, &beta_gradient, &intercept_gradient);
  }
  if (!probabilities.allFinite() || !beta_gradient.allFinite() ||
      !intercept_gradient.allFinite() ||
      !std::isfinite(current_negative_log_likelihood)) {
    result.status = MultinomialSolverStatus::kNumericalFailure;
    return result;
  }
  result.final_kkt_residual = outer_kkt_residual(
      result.beta, beta_gradient, intercept_gradient, penalty, options);
  MultinomialActiveSet active_set(
      static_cast<std::size_t>(objective.feature_num()),
      options.use_active_set ? 0 : 1);
  int initially_activated = 0;
  if (restricted_path_mode) {
    active_set = *initial_active_features;
    for (std::size_t index = 0; index < active_set.size(); ++index)
      active_set[index] = active_set[index] == 0 ? 0 : 1;
    activate_nonzero_features(result.beta, options.zero_tolerance,
                              &active_set);
    if (eagerly_identify_active_features)
      activate_outer_kkt_features(
          result.beta, beta_gradient, penalty, options, &active_set);
    initially_activated = count_active_features(active_set);
  } else if (options.use_active_set) {
    initially_activated = activate_outer_kkt_features(
        result.beta, beta_gradient, penalty, options, &active_set);
  } else {
    initially_activated = objective.feature_num();
  }
  result.initial_active_features = count_active_features(active_set);
  result.final_active_features = count_active_features(active_set);
  result.active_features = active_set;
  result.history.push_back(
      initial_record(result.final_objective, result.final_kkt_residual));
  result.history.back().active_features = result.final_active_features;
  result.history.back().newly_activated_features = initially_activated;

  if (!std::isfinite(result.final_objective) ||
      !std::isfinite(result.final_kkt_residual)) {
    result.status = MultinomialSolverStatus::kNumericalFailure;
    return result;
  }
  if (result.final_kkt_residual <= options.outer_kkt_tolerance) {
    if (final_feature_gradient_max != 0) {
      final_feature_gradient_max->resize(objective.feature_num());
      for (int feature = 0; feature < objective.feature_num(); ++feature) {
        (*final_feature_gradient_max)[feature] =
            beta_gradient.row(feature).cwiseAbs().maxCoeff();
      }
    }
    save_iterate_cache(current_negative_log_likelihood, objective,
                       result.beta, result.intercept,
                       use_incremental_logits ? &current_logits : 0,
                       &probabilities,
                       &beta_gradient, &intercept_gradient,
                       final_iterate_cache);
    result.status = MultinomialSolverStatus::kConverged;
    return result;
  }
  if (restricted_path_mode &&
      restricted_outer_kkt_residual(
          result.beta, beta_gradient, intercept_gradient, penalty, options,
          active_set) <= options.outer_kkt_tolerance) {
    const int reactivated = activate_outer_kkt_features(
        result.beta, beta_gradient, penalty, options, &active_set,
        options.outer_kkt_tolerance);
    result.total_reactivated_features += reactivated;
    result.total_outer_reactivated_features += reactivated;
    result.final_active_features = count_active_features(active_set);
    result.active_features = active_set;
    result.history.back().active_features = result.final_active_features;
    result.history.back().newly_activated_features += reactivated;
    result.history.back().outer_reactivated_features += reactivated;
    if (reactivated == 0) {
      result.status = MultinomialSolverStatus::kNumericalFailure;
      return result;
    }
  }

  // Public failed-point diagnostics have always described the committed
  // iterate with an exact full KKT scan.  Restore that contract before every
  // terminal return after a partial active-gradient refresh.  Recompute the
  // probabilities as well: a failed line search may have overwritten the
  // shared trial-probability buffer.
  const auto ensure_exact_terminal_gradient = [&]() -> bool {
    if (!gradient_is_full) {
      try {
        if (use_incremental_logits) {
          current_negative_log_likelihood =
              objective.negative_log_likelihood_from_logits(
                  current_logits, &probabilities);
        } else {
          current_negative_log_likelihood = objective.negative_log_likelihood(
              result.beta, result.intercept, &probabilities);
        }
        result.final_objective =
            current_negative_log_likelihood + penalty.value(result.beta);
        objective.smooth_gradient_from_probabilities(
            probabilities, &beta_gradient, &intercept_gradient);
        gradient_is_full = true;
      } catch (const std::invalid_argument &) {
        return false;
      }
    }
    if (!probabilities.allFinite() || !beta_gradient.allFinite() ||
        !intercept_gradient.allFinite() ||
        !std::isfinite(result.final_objective))
      return false;
    result.final_kkt_residual = outer_kkt_residual(
        result.beta, beta_gradient, intercept_gradient, penalty, options);
    if (!result.history.empty())
      result.history.back().kkt_residual = result.final_kkt_residual;
    return std::isfinite(result.final_kkt_residual);
  };
  const auto terminal_failure =
      [&](MultinomialSolverStatus status) -> MultinomialActNewtonResult {
    result.status = status;
    if (!ensure_exact_terminal_gradient())
      result.status = MultinomialSolverStatus::kNumericalFailure;
    return result;
  };

  for (int outer_iteration = 1;
       outer_iteration <= options.max_outer_iterations;
       ++outer_iteration) {
    MultinomialActNewtonOptions subproblem_options = options;
    if (options.use_adaptive_inner_tolerance) {
      const double restricted_kkt = restricted_outer_kkt_residual(
          result.beta, beta_gradient, intercept_gradient, penalty, options,
          active_set);
      if (!std::isfinite(restricted_kkt)) {
        return terminal_failure(MultinomialSolverStatus::kNumericalFailure);
      }
      subproblem_options.inner_kkt_tolerance = std::max(
          options.inner_kkt_tolerance,
          kInexactNewtonForcingFactor * restricted_kkt);
    }
    const bool adaptive_subproblem_was_relaxed =
        subproblem_options.inner_kkt_tolerance >
        options.inner_kkt_tolerance;
    QuadraticSubproblemResult subproblem =
        solve_quadratic_subproblem(
            objective, result.beta, probabilities, beta_gradient,
            intercept_gradient, penalty, subproblem_options, &active_set,
            !restricted_path_mode);
    int iteration_inner_sweeps = 0;
    int iteration_reactivated_features = 0;
    int iteration_full_kkt_scans = 0;
    const auto accumulate_subproblem_work =
        [&](const QuadraticSubproblemResult &attempt) {
          result.total_inner_sweeps += attempt.sweeps;
          result.total_coordinate_updates += attempt.coordinate_updates;
          result.total_reactivated_features +=
              attempt.reactivated_features;
          result.total_subproblem_reactivated_features +=
              attempt.reactivated_features;
          result.total_full_subproblem_kkt_scans += attempt.full_kkt_scans;
          iteration_inner_sweeps += attempt.sweeps;
          iteration_reactivated_features += attempt.reactivated_features;
          iteration_full_kkt_scans += attempt.full_kkt_scans;
          result.final_active_features = attempt.final_active_features;
          result.active_features = active_set;
        };
    accumulate_subproblem_work(subproblem);
    if (!subproblem.finite) {
      return terminal_failure(MultinomialSolverStatus::kNumericalFailure);
    }
    if (!subproblem.converged) {
      return terminal_failure(MultinomialSolverStatus::kInnerIterationLimit);
    }

    double composite_directional_derivative =
        (beta_gradient.array() * subproblem.beta_direction.array()).sum() +
        intercept_gradient.dot(subproblem.intercept_direction) +
        penalty.difference(result.beta + subproblem.beta_direction,
                           result.beta);
    double direction_norm = std::sqrt(
        subproblem.beta_direction.squaredNorm() +
        subproblem.intercept_direction.squaredNorm());
    if (!std::isfinite(composite_directional_derivative) ||
        !std::isfinite(direction_norm)) {
      return terminal_failure(MultinomialSolverStatus::kNumericalFailure);
    }
    if (!(composite_directional_derivative < 0.0) &&
        adaptive_subproblem_was_relaxed) {
      subproblem = solve_quadratic_subproblem(
          objective, result.beta, probabilities, beta_gradient,
          intercept_gradient, penalty, options, &active_set,
          !restricted_path_mode);
      accumulate_subproblem_work(subproblem);
      if (!subproblem.finite) {
        return terminal_failure(MultinomialSolverStatus::kNumericalFailure);
      }
      if (!subproblem.converged) {
        return terminal_failure(
            MultinomialSolverStatus::kInnerIterationLimit);
      }
      composite_directional_derivative =
          (beta_gradient.array() * subproblem.beta_direction.array()).sum() +
          intercept_gradient.dot(subproblem.intercept_direction) +
          penalty.difference(result.beta + subproblem.beta_direction,
                             result.beta);
      direction_norm = std::sqrt(
          subproblem.beta_direction.squaredNorm() +
          subproblem.intercept_direction.squaredNorm());
      if (!std::isfinite(composite_directional_derivative) ||
          !std::isfinite(direction_norm)) {
        return terminal_failure(MultinomialSolverStatus::kNumericalFailure);
      }
    }
    if (!(composite_directional_derivative < 0.0)) {
      return terminal_failure(MultinomialSolverStatus::kNoDescentDirection);
    }

    bool accepted = false;
    double step_size = 1.0;
    int line_search_steps = 0;
    Eigen::MatrixXd candidate_beta;
    Eigen::VectorXd candidate_intercept;
    Eigen::MatrixXd candidate_logits;
    double candidate_objective = result.final_objective;
    double candidate_negative_log_likelihood =
        std::numeric_limits<double>::quiet_NaN();
    for (int line_step = 1;
         line_step <= options.max_line_search_steps;
         ++line_step) {
      line_search_steps = line_step;
      candidate_beta =
          result.beta + step_size * subproblem.beta_direction;
      candidate_intercept =
          result.intercept + step_size * subproblem.intercept_direction;
      if (options.include_intercept)
        candidate_intercept.array() -= candidate_intercept.mean();
      else
        candidate_intercept.setZero();
      if (candidate_beta.allFinite() &&
          candidate_intercept.allFinite()) {
        try {
          if (options.reuse_line_search_probabilities) {
            if (use_incremental_logits) {
              candidate_logits.noalias() =
                  current_logits +
                  step_size * subproblem.linear_direction;
              candidate_negative_log_likelihood =
                  objective.negative_log_likelihood_from_logits(
                      candidate_logits, &probabilities);
            } else {
              candidate_negative_log_likelihood =
                  objective.negative_log_likelihood(
                      candidate_beta, candidate_intercept, &probabilities);
            }
            candidate_objective =
                candidate_negative_log_likelihood +
                penalty.value(candidate_beta);
          } else {
            candidate_objective = total_objective(
                objective, candidate_beta, candidate_intercept, penalty, 0);
          }
          const double armijo_bound =
              result.final_objective +
              options.armijo_constant * step_size *
                  composite_directional_derivative;
          const double roundoff_allowance =
              10.0 * std::numeric_limits<double>::epsilon() *
              std::max(1.0, std::fabs(result.final_objective));
          if (std::isfinite(candidate_objective) &&
              candidate_objective <= armijo_bound + roundoff_allowance) {
            accepted = true;
            break;
          }
        } catch (const std::invalid_argument &) {
          // A nonfinite trial point is rejected and backtracked below.
        }
      }
      step_size *= options.backtracking_factor;
      if (step_size < options.minimum_step_size) break;
    }
    if (!accepted) {
      return terminal_failure(MultinomialSolverStatus::kLineSearchFailed);
    }
    if (options.canonicalize_feature_l1_gauge &&
        !penalty.canonicalize(&candidate_beta)) {
      return terminal_failure(MultinomialSolverStatus::kNumericalFailure);
    }

    result.beta.swap(candidate_beta);
    result.intercept.swap(candidate_intercept);
    gradient_is_full = false;
    if (options.reuse_line_search_probabilities) {
      if (use_incremental_logits)
        current_logits.swap(candidate_logits);
      current_negative_log_likelihood =
          candidate_negative_log_likelihood;
      result.final_objective =
          current_negative_log_likelihood + penalty.value(result.beta);
    } else {
      try {
        current_negative_log_likelihood = objective.negative_log_likelihood(
            result.beta, result.intercept, &probabilities);
        result.final_objective =
            current_negative_log_likelihood + penalty.value(result.beta);
      } catch (const std::invalid_argument &) {
        return terminal_failure(MultinomialSolverStatus::kNumericalFailure);
      }
    }
    // Mirror scalar ActNewton's level-1 gate.  While the restricted active
    // KKT is not yet small, the next quadratic solve reads only active rows,
    // so avoid the O(n*d*K) full GEMM.  Row-wise products are worthwhile only
    // for a genuinely sparse working set; dense sets retain the full GEMM.
    const int active_feature_count = count_active_features(active_set);
    const bool favorable_active_gradient_shape =
        objective.feature_num() >= kActiveGradientMinimumFeatureNum ||
        static_cast<long long>(objective.sample_num()) >=
            static_cast<long long>(kActiveGradientTallAspectRatio) *
                static_cast<long long>(objective.feature_num());
    const bool use_active_gradient_gate =
        restricted_path_mode && favorable_active_gradient_shape &&
        4LL * static_cast<long long>(active_feature_count) <=
            static_cast<long long>(objective.feature_num());
    if (use_active_gradient_gate) {
      objective.smooth_gradient_from_probabilities_on_active_features(
          probabilities, active_set, &beta_gradient, &intercept_gradient);
      result.final_kkt_residual = restricted_outer_kkt_residual(
          result.beta, beta_gradient, intercept_gradient, penalty, options,
          active_set);
      if (result.final_kkt_residual <= options.outer_kkt_tolerance) {
        objective.smooth_gradient_from_probabilities(
            probabilities, &beta_gradient, &intercept_gradient);
        gradient_is_full = true;
        result.final_kkt_residual = outer_kkt_residual(
            result.beta, beta_gradient, intercept_gradient, penalty, options);
      }
    } else {
      objective.smooth_gradient_from_probabilities(
          probabilities, &beta_gradient, &intercept_gradient);
      gradient_is_full = true;
      result.final_kkt_residual = outer_kkt_residual(
          result.beta, beta_gradient, intercept_gradient, penalty, options);
    }
    if (!probabilities.allFinite() || !beta_gradient.allFinite() ||
        !intercept_gradient.allFinite()) {
      return terminal_failure(MultinomialSolverStatus::kNumericalFailure);
    }
    int outer_reactivated = 0;
    if (options.use_active_set && gradient_is_full) {
      if (!restricted_path_mode) {
        outer_reactivated = activate_outer_kkt_features(
            result.beta, beta_gradient, penalty, options, &active_set);
      } else {
        // Match Logistic ActNewton's level-1 loop: after each converged
        // fixed-IRLS solve and auxiliary update, check every coordinate
        // outside the current set before taking another PN/IRLS step.  The
        // already available true gradient makes this O(dK); the expensive
        // O(ndK) full scan remains absent from the inner quadratic solve.
        outer_reactivated = activate_outer_kkt_features(
            result.beta, beta_gradient, penalty, options, &active_set,
            options.outer_kkt_tolerance);
      }
    }
    result.total_reactivated_features += outer_reactivated;
    result.total_outer_reactivated_features += outer_reactivated;
    result.final_active_features = count_active_features(active_set);
    result.active_features = active_set;
    result.outer_iterations = outer_iteration;

    MultinomialIterationRecord record;
    record.outer_iteration = outer_iteration;
    record.inner_sweeps = iteration_inner_sweeps;
    record.line_search_steps = line_search_steps;
    record.objective = result.final_objective;
    record.kkt_residual = result.final_kkt_residual;
    record.inner_kkt_residual = subproblem.kkt_residual;
    record.step_size = step_size;
    record.direction_norm = direction_norm;
    record.composite_directional_derivative =
        composite_directional_derivative;
    record.inner_converged = subproblem.converged;
    record.active_features = result.final_active_features;
    record.newly_activated_features =
        iteration_reactivated_features + outer_reactivated;
    record.subproblem_reactivated_features =
        iteration_reactivated_features;
    record.outer_reactivated_features = outer_reactivated;
    record.full_subproblem_kkt_scans = iteration_full_kkt_scans;
    result.history.push_back(record);

    if (!std::isfinite(result.final_objective) ||
        !std::isfinite(result.final_kkt_residual)) {
      return terminal_failure(MultinomialSolverStatus::kNumericalFailure);
    }
    if (gradient_is_full &&
        result.final_kkt_residual <= options.outer_kkt_tolerance) {
      if (final_feature_gradient_max != 0) {
        final_feature_gradient_max->resize(objective.feature_num());
        for (int feature = 0; feature < objective.feature_num(); ++feature) {
          (*final_feature_gradient_max)[feature] =
              beta_gradient.row(feature).cwiseAbs().maxCoeff();
        }
      }
      save_iterate_cache(current_negative_log_likelihood, objective,
                         result.beta, result.intercept,
                         use_incremental_logits ? &current_logits : 0,
                         &probabilities,
                         &beta_gradient, &intercept_gradient,
                         final_iterate_cache);
      result.status = MultinomialSolverStatus::kConverged;
      return result;
    }
  }

  result.status = MultinomialSolverStatus::kOuterIterationLimit;
  if (!ensure_exact_terminal_gradient())
    result.status = MultinomialSolverStatus::kNumericalFailure;
  return result;
}

void validate_actnewton_options(
    const MultinomialActNewtonOptions &options) {
  if (options.max_outer_iterations <= 0 ||
      options.max_inner_sweeps <= 0 ||
      options.max_line_search_steps <= 0 ||
      options.exact_kkt_scan_interval <= 0)
    throw std::invalid_argument(
        "multinomial iteration limits must be positive");
  if (!std::isfinite(options.outer_kkt_tolerance) ||
      !std::isfinite(options.inner_kkt_tolerance) ||
      !std::isfinite(options.armijo_constant) ||
      !std::isfinite(options.backtracking_factor) ||
      !std::isfinite(options.minimum_step_size) ||
      !std::isfinite(options.hessian_damping) ||
      !std::isfinite(options.zero_tolerance) ||
      !(options.outer_kkt_tolerance > 0.0) ||
      !(options.inner_kkt_tolerance > 0.0) ||
      (options.use_adaptive_inner_tolerance &&
       options.inner_kkt_tolerance > options.outer_kkt_tolerance) ||
      !(options.armijo_constant > 0.0) ||
      !(options.armijo_constant < 1.0) ||
      !(options.backtracking_factor > 0.0) ||
      !(options.backtracking_factor < 1.0) ||
      !(options.minimum_step_size > 0.0) ||
      !(options.minimum_step_size < 1.0) ||
      !(options.hessian_damping > 0.0) ||
      !(options.zero_tolerance >= 0.0))
    throw std::invalid_argument("invalid multinomial solver option");
}

MultinomialActNewtonPathResult solve_path_view_impl(
    const MultinomialProblemView &objective,
    const MultinomialActNewtonOptions &options, double lambda,
    Eigen::MatrixXd *state_beta, Eigen::VectorXd *state_intercept,
    Eigen::VectorXd *state_feature_gradient_max,
    MultinomialActiveSet *state_strong_set,
    std::shared_ptr<const detail::MultinomialPathSmoothCache>
        *state_smooth_cache,
    double *state_previous_lambda, bool *state_initialized) {
  if (state_beta == 0 || state_intercept == 0 ||
      state_feature_gradient_max == 0 || state_strong_set == 0 ||
      state_smooth_cache == 0 || state_previous_lambda == 0 ||
      state_initialized == 0)
    throw std::invalid_argument("multinomial path state must not be null");
  if (!(lambda >= 0.0) || !std::isfinite(lambda))
    throw std::invalid_argument(
        "multinomial lambda must be finite and nonnegative");
  validate_actnewton_options(options);

  const int d = objective.feature_num();
  const int num_classes = objective.class_num();
  Eigen::MatrixXd initial_beta;
  Eigen::VectorXd initial_intercept;
  Eigen::VectorXd previous_gradient_max;
  MultinomialActiveSet strong_set;

  if (*state_initialized) {
    if (state_beta->rows() != d || state_beta->cols() != num_classes ||
        state_intercept->size() != num_classes ||
        state_feature_gradient_max->size() != d ||
        state_strong_set->size() != static_cast<std::size_t>(d) ||
        !state_beta->allFinite() || !state_intercept->allFinite() ||
        !state_feature_gradient_max->allFinite() ||
        !(*state_previous_lambda >= 0.0) ||
        !std::isfinite(*state_previous_lambda))
      throw std::invalid_argument("invalid multinomial path state");
    initial_beta = *state_beta;
    initial_intercept = *state_intercept;
    previous_gradient_max = *state_feature_gradient_max;
    strong_set = *state_strong_set;
    for (std::size_t index = 0; index < strong_set.size(); ++index)
      strong_set[index] = strong_set[index] == 0 ? 0 : 1;
  } else {
    initial_beta = Eigen::MatrixXd::Zero(d, num_classes);
    initial_intercept = Eigen::VectorXd::Zero(num_classes);
    if (options.include_intercept)
      initial_intercept = empirical_null_intercept(objective);
    Eigen::MatrixXd gradient;
    Eigen::VectorXd intercept_gradient;
    objective.smooth_gradient(initial_beta, initial_intercept, &gradient,
                              &intercept_gradient);
    if (!gradient.allFinite() || !intercept_gradient.allFinite())
      throw std::runtime_error(
          "nonfinite multinomial null gradient in path initialization");
    previous_gradient_max.resize(d);
    for (int feature = 0; feature < d; ++feature)
      previous_gradient_max[feature] =
          gradient.row(feature).cwiseAbs().maxCoeff();
    strong_set.assign(static_cast<std::size_t>(d),
                      options.use_active_set ? 0 : 1);
  }

  MultinomialActNewtonPathResult path_result;
  path_result.initial_strong_features = count_active_features(strong_set);
  path_result.strong_rule_activated_features = 0;
  path_result.full_kkt_reactivated_features = 0;
  path_result.final_strong_features = path_result.initial_strong_features;
  path_result.used_strong_rule = false;
  path_result.reused_initial_smooth_state = false;

  if (!options.use_active_set) {
    std::fill(strong_set.begin(), strong_set.end(), 1);
  } else if (*state_initialized && lambda > *state_previous_lambda) {
    // Sequential strong screening assumes a nonincreasing path.  Arbitrary
    // increasing steps remain correct by disabling screening for that step.
    std::fill(strong_set.begin(), strong_set.end(), 1);
  } else {
    path_result.used_strong_rule = true;
    const double strong_threshold =
        *state_initialized ? 2.0 * lambda - *state_previous_lambda
                           : 2.0 * lambda;
    for (int feature = 0; feature < d; ++feature) {
      if (strong_set[static_cast<std::size_t>(feature)] == 0 &&
          previous_gradient_max[feature] > strong_threshold) {
        strong_set[static_cast<std::size_t>(feature)] = 1;
        ++path_result.strong_rule_activated_features;
      }
    }
  }
  activate_nonzero_features(initial_beta, options.zero_tolerance,
                            &strong_set);

  Eigen::VectorXd final_feature_gradient_max;
  std::shared_ptr<const detail::MultinomialPathSmoothCache>
      final_iterate_cache;
  const detail::MultinomialPathSmoothCache *initial_iterate_cache =
      *state_initialized ? state_smooth_cache->get() : 0;
  path_result.solution = solve_impl(
      objective, options, ScalarPenalty(lambda), initial_beta,
      initial_intercept, &strong_set, false, &final_feature_gradient_max,
      initial_iterate_cache, &final_iterate_cache,
      &path_result.reused_initial_smooth_state);
  path_result.full_kkt_reactivated_features =
      path_result.solution.total_outer_reactivated_features;
  path_result.final_strong_features =
      path_result.solution.final_active_features;

  if (!path_result.solution.converged()) return path_result;

  if (final_feature_gradient_max.size() != d ||
      !final_feature_gradient_max.allFinite() ||
      final_iterate_cache.get() == 0 ||
      !valid_iterate_cache(*final_iterate_cache, objective,
                           path_result.solution.beta,
                           path_result.solution.intercept,
                           options.reuse_line_search_probabilities &&
                               options.outer_kkt_tolerance >=
                                   kIncrementalLogitsMinimumTolerance))
    throw std::runtime_error(
        "nonfinite multinomial gradient after path solve");

  // Commit only after every path-state invariant has been certified.
  *state_beta = path_result.solution.beta;
  *state_intercept = path_result.solution.intercept;
  *state_feature_gradient_max = std::move(final_feature_gradient_max);
  *state_strong_set = path_result.solution.active_features;
  *state_smooth_cache = std::move(final_iterate_cache);
  *state_previous_lambda = lambda;
  *state_initialized = true;
  return path_result;
}

}  // namespace

const char *multinomial_solver_status_string(MultinomialSolverStatus status) {
  switch (status) {
    case MultinomialSolverStatus::kConverged:
      return "converged";
    case MultinomialSolverStatus::kOuterIterationLimit:
      return "outer_iteration_limit";
    case MultinomialSolverStatus::kInnerIterationLimit:
      return "inner_iteration_limit";
    case MultinomialSolverStatus::kLineSearchFailed:
      return "line_search_failed";
    case MultinomialSolverStatus::kNoDescentDirection:
      return "no_descent_direction";
    case MultinomialSolverStatus::kNumericalFailure:
      return "numerical_failure";
  }
  return "unknown";
}

MultinomialActNewtonOptions::MultinomialActNewtonOptions()
    : max_outer_iterations(100),
      max_inner_sweeps(1000),
      max_line_search_steps(50),
      exact_kkt_scan_interval(4),
      outer_kkt_tolerance(1e-6),
      inner_kkt_tolerance(1e-8),
      armijo_constant(1e-4),
      backtracking_factor(0.5),
      minimum_step_size(1e-12),
      hessian_damping(1e-10),
      zero_tolerance(1e-12),
      include_intercept(true),
      use_probability_dot_direction_cache(true),
      use_active_set(true),
      canonicalize_feature_l1_gauge(true),
      use_adaptive_inner_tolerance(false),
      use_vectorized_coordinate_kernels(false),
      reuse_line_search_probabilities(false),
      use_compact_inner_active_set(false) {}

MultinomialActNewtonSolver::MultinomialActNewtonSolver(
    const MultinomialObjective &objective,
    const MultinomialActNewtonOptions &options)
    : m_objective(objective), m_options(options) {
  validate_actnewton_options(m_options);
}

MultinomialActNewtonResult MultinomialActNewtonSolver::solve(
    double lambda) const {
  Eigen::VectorXd initial_intercept =
      Eigen::VectorXd::Zero(m_objective.class_num());
  if (m_options.include_intercept)
    initial_intercept =
        empirical_null_intercept(actnewton_problem_view(m_objective));
  return solve(lambda,
               Eigen::MatrixXd::Zero(m_objective.feature_num(),
                                     m_objective.class_num()),
               initial_intercept);
}

MultinomialActNewtonResult MultinomialActNewtonSolver::solve(
    double lambda, const Eigen::MatrixXd &initial_beta,
    const Eigen::VectorXd &initial_intercept) const {
  if (!(lambda >= 0.0) || !std::isfinite(lambda))
    throw std::invalid_argument(
        "multinomial lambda must be finite and nonnegative");
  const MultinomialProblemView objective =
      actnewton_problem_view(m_objective);
  return solve_impl(objective, m_options, ScalarPenalty(lambda),
                    initial_beta, initial_intercept);
}

MultinomialActNewtonResult MultinomialActNewtonSolver::solve(
    double lambda, const Eigen::MatrixXd &initial_beta,
    const Eigen::VectorXd &initial_intercept,
    const std::vector<unsigned char> &initial_active_features) const {
  if (!(lambda >= 0.0) || !std::isfinite(lambda))
    throw std::invalid_argument(
        "multinomial lambda must be finite and nonnegative");
  const MultinomialProblemView objective =
      actnewton_problem_view(m_objective);
  return solve_impl(objective, m_options, ScalarPenalty(lambda),
                    initial_beta, initial_intercept,
                    &initial_active_features);
}

MultinomialActNewtonResult MultinomialActNewtonSolver::solve(
    const Eigen::MatrixXd &l1_penalties) const {
  Eigen::VectorXd initial_intercept =
      Eigen::VectorXd::Zero(m_objective.class_num());
  if (m_options.include_intercept)
    initial_intercept =
        empirical_null_intercept(actnewton_problem_view(m_objective));
  return solve(l1_penalties,
               Eigen::MatrixXd::Zero(m_objective.feature_num(),
                                     m_objective.class_num()),
               initial_intercept);
}

MultinomialActNewtonResult MultinomialActNewtonSolver::solve(
    const Eigen::MatrixXd &l1_penalties,
    const Eigen::MatrixXd &initial_beta,
    const Eigen::VectorXd &initial_intercept) const {
  if (l1_penalties.rows() != m_objective.feature_num() ||
      l1_penalties.cols() != m_objective.class_num())
    throw std::invalid_argument(
        "multinomial L1 penalties must have shape d-by-K");

  const double first_penalty = l1_penalties(0, 0);
  bool uniform = true;
  for (Eigen::Index feature = 0;
       feature < l1_penalties.rows(); ++feature) {
    for (Eigen::Index klass = 0;
         klass < l1_penalties.cols(); ++klass) {
      const double value = l1_penalties(feature, klass);
      if (!(value >= 0.0) || !std::isfinite(value))
        throw std::invalid_argument(
            "multinomial L1 penalties must be finite and nonnegative");
      if (value != first_penalty) uniform = false;
    }
  }
  if (uniform)
    return solve(first_penalty, initial_beta, initial_intercept);

  const MultinomialProblemView objective =
      actnewton_problem_view(m_objective);
  return solve_impl(objective, m_options,
                    MatrixPenalty(l1_penalties),
                    initial_beta, initial_intercept);
}

MultinomialActNewtonResult MultinomialActNewtonSolver::solve(
    const Eigen::MatrixXd &l1_penalties,
    const Eigen::MatrixXd &initial_beta,
    const Eigen::VectorXd &initial_intercept,
    const std::vector<unsigned char> &initial_active_features) const {
  if (l1_penalties.rows() != m_objective.feature_num() ||
      l1_penalties.cols() != m_objective.class_num())
    throw std::invalid_argument(
        "multinomial L1 penalties must have shape d-by-K");

  const double first_penalty = l1_penalties(0, 0);
  bool uniform = true;
  for (Eigen::Index feature = 0;
       feature < l1_penalties.rows(); ++feature) {
    for (Eigen::Index klass = 0;
         klass < l1_penalties.cols(); ++klass) {
      const double value = l1_penalties(feature, klass);
      if (!(value >= 0.0) || !std::isfinite(value))
        throw std::invalid_argument(
            "multinomial L1 penalties must be finite and nonnegative");
      if (value != first_penalty) uniform = false;
    }
  }
  if (uniform)
    return solve(first_penalty, initial_beta, initial_intercept,
                 initial_active_features);

  const MultinomialProblemView objective =
      actnewton_problem_view(m_objective);
  return solve_impl(objective, m_options,
                    MatrixPenalty(l1_penalties), initial_beta,
                    initial_intercept, &initial_active_features, true);
}

MultinomialActNewtonPathState::MultinomialActNewtonPathState()
    : previous_lambda(std::numeric_limits<double>::quiet_NaN()),
      initialized(false) {}

void MultinomialActNewtonPathState::reset() {
  beta.resize(0, 0);
  intercept.resize(0);
  feature_gradient_max.resize(0);
  strong_set.clear();
  m_smooth_cache.reset();
  previous_lambda = std::numeric_limits<double>::quiet_NaN();
  initialized = false;
}

namespace internal {

MultinomialPathViewState::MultinomialPathViewState()
    : previous_lambda(std::numeric_limits<double>::quiet_NaN()),
      initialized(false) {}

void MultinomialPathViewState::reset() {
  beta.resize(0, 0);
  intercept.resize(0);
  feature_gradient_max.resize(0);
  strong_set.clear();
  smooth_cache.reset();
  previous_lambda = std::numeric_limits<double>::quiet_NaN();
  initialized = false;
}

MultinomialActNewtonPathResult solve_multinomial_actnewton_path_view(
    const ::picasso::detail::MultinomialProblemView &problem,
    const MultinomialActNewtonOptions &options, double lambda,
    MultinomialPathViewState *state) {
  if (state == 0)
    throw std::invalid_argument("multinomial path state must not be null");
  return solve_path_view_impl(
      problem, options, lambda, &state->beta, &state->intercept,
      &state->feature_gradient_max, &state->strong_set,
      &state->smooth_cache, &state->previous_lambda, &state->initialized);
}

MultinomialActNewtonResult solve_multinomial_actnewton_weighted_view(
    const ::picasso::detail::MultinomialProblemView &problem,
    const MultinomialActNewtonOptions &options,
    const Eigen::MatrixXd &l1_penalties,
    const Eigen::MatrixXd &initial_beta,
    const Eigen::VectorXd &initial_intercept,
    const std::vector<unsigned char> &initial_active_features) {
  validate_actnewton_options(options);
  if (l1_penalties.rows() != problem.feature_num() ||
      l1_penalties.cols() != problem.class_num())
    throw std::invalid_argument(
        "multinomial L1 penalties must have shape d-by-K");

  const double first_penalty = l1_penalties(0, 0);
  bool uniform = true;
  for (Eigen::Index feature = 0; feature < l1_penalties.rows(); ++feature) {
    for (Eigen::Index klass = 0; klass < l1_penalties.cols(); ++klass) {
      const double value = l1_penalties(feature, klass);
      if (!(value >= 0.0) || !std::isfinite(value))
        throw std::invalid_argument(
            "multinomial L1 penalties must be finite and nonnegative");
      if (value != first_penalty) uniform = false;
    }
  }
  if (uniform) {
    return solve_impl(problem, options, ScalarPenalty(first_penalty),
                      initial_beta, initial_intercept,
                      &initial_active_features);
  }
  return solve_impl(problem, options, MatrixPenalty(l1_penalties),
                    initial_beta, initial_intercept,
                    &initial_active_features, true);
}

}  // namespace internal

MultinomialActNewtonPathSolver::MultinomialActNewtonPathSolver(
    const MultinomialObjective &objective,
    const MultinomialActNewtonOptions &options)
    : m_objective(objective), m_options(options), m_solver(objective, options) {}

MultinomialActNewtonPathResult MultinomialActNewtonPathSolver::solve(
    double lambda, MultinomialActNewtonPathState *state) const {
  if (state == 0)
    throw std::invalid_argument("multinomial path state must not be null");
  const MultinomialProblemView objective =
      actnewton_problem_view(m_objective);
  return solve_path_view_impl(
      objective, m_options, lambda, &state->beta, &state->intercept,
      &state->feature_gradient_max, &state->strong_set,
      &state->m_smooth_cache, &state->previous_lambda,
      &state->initialized);
}

}  // namespace solver
}  // namespace picasso
