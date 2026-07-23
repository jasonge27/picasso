#ifndef PICASSO_MULTINOMIAL_ACTNEWTON_HPP
#define PICASSO_MULTINOMIAL_ACTNEWTON_HPP

#include <Eigen/Dense>

#include <memory>
#include <vector>

#include <picasso/multinomial_objective.hpp>

namespace picasso {
namespace solver {

namespace detail {
struct MultinomialPathSmoothCache;
}

enum class MultinomialSolverStatus {
  kConverged,
  kOuterIterationLimit,
  kInnerIterationLimit,
  kLineSearchFailed,
  kNoDescentDirection,
  kNumericalFailure
};

const char *multinomial_solver_status_string(MultinomialSolverStatus status);

struct MultinomialActNewtonOptions {
  int max_outer_iterations;
  int max_inner_sweeps;
  int max_line_search_steps;
  // Run an exact quadratic KKT scan on the first sweep, at least once every
  // this many sweeps, when the cheap coordinate-change proxy is small, and on
  // the final allowed sweep.  Only an exact scan may certify convergence or
  // trigger a full inactive-feature scan.  The production default is four;
  // set to one for the legacy per-sweep schedule.
  int exact_kkt_scan_interval;
  double outer_kkt_tolerance;
  double inner_kkt_tolerance;
  double armijo_constant;
  double backtracking_factor;
  double minimum_step_size;
  double hessian_damping;
  double zero_tolerance;
  bool include_intercept;
  // Cache t_i = p_i^T Deta_i during the quadratic coordinate solve.  Disable
  // only to retain the uncached numerical/performance A/B baseline.
  bool use_probability_dot_direction_cache;
  // Restrict each quadratic solve to feature blocks that are nonzero or
  // violate KKT conditions.  Full quadratic and outer-model KKT scans safely
  // reactivate screened blocks; every active block updates all K classes.
  bool use_active_set;
  // Select a symmetric minimum-L1 representative of each feature's full-K
  // softmax gauge: an ordinary median for scalar penalties and a weighted
  // median for coordinate-specific penalties, using the midpoint of a flat
  // median interval.
  bool canonicalize_feature_l1_gauge;
  // These production-kernel switches occupy tail padding in the original
  // options layout, preserving the size and every pre-existing field offset.
  // New direct C++ callers may opt in; legacy callers retain old behavior.
  bool use_adaptive_inner_tolerance;
  bool use_vectorized_coordinate_kernels;
  bool reuse_line_search_probabilities;
  // Extend glmnet's second working-set tier inside a restricted fixed-IRLS
  // quadratic.  The first sweep visits every strong feature/class candidate;
  // later sweeps use a shape- and precision-adaptive compact list.  Small
  // low-class scalar-L1 fits use feature resolution at fast precision; strict,
  // wide, high-class, and coordinate-weighted fits use coefficient resolution.
  // A full candidate KKT check remains mandatory before convergence.  This
  // flag stays off for direct C++ callers and is enabled by the production C
  // API.
  bool use_compact_inner_active_set;
  MultinomialActNewtonOptions();
};

// One accepted outer iterate.  The first record describes the starting point
// and therefore has zero inner sweeps, line-search steps, and step size.
struct MultinomialIterationRecord {
  int outer_iteration;
  int inner_sweeps;
  int line_search_steps;
  double objective;
  // During a restricted path solve, a nonterminal record may contain the
  // exact KKT residual on the current active set.  The terminal record and
  // MultinomialActNewtonResult::final_kkt_residual are always full-model KKT
  // residuals, including on iteration-limit and other failed returns.
  double kkt_residual;
  double inner_kkt_residual;
  double step_size;
  double direction_norm;
  double composite_directional_derivative;
  bool inner_converged;
  int active_features;
  int newly_activated_features;
  int subproblem_reactivated_features;
  int outer_reactivated_features;
  int full_subproblem_kkt_scans;
};

struct MultinomialActNewtonResult {
  Eigen::MatrixXd beta;
  Eigen::VectorXd intercept;
  MultinomialSolverStatus status;
  int outer_iterations;
  int total_inner_sweeps;
  long long total_coordinate_updates;
  double final_objective;
  double final_kkt_residual;
  int final_active_features;
  int initial_active_features;
  int total_reactivated_features;
  int total_subproblem_reactivated_features;
  int total_outer_reactivated_features;
  int total_full_subproblem_kkt_scans;
  // Feature-block working set accepted at termination.  Keeping the mask in
  // the result lets a path driver reuse the active set exactly as the legacy
  // Logistic ActNewton solver reuses actset_indcat along a lambda path.
  std::vector<unsigned char> active_features;
  std::vector<MultinomialIterationRecord> history;

  bool converged() const {
    return status == MultinomialSolverStatus::kConverged;
  }
};

// Solves one full-variable, L1-penalized multinomial problem with a coupled
// proximal Newton/IRLS method.  The solver borrows the objective and never
// copies its design matrix or labels.
class MultinomialActNewtonSolver {
 public:
  explicit MultinomialActNewtonSolver(
      const MultinomialObjective &objective,
      const MultinomialActNewtonOptions &options =
          MultinomialActNewtonOptions());

  MultinomialActNewtonResult solve(double lambda) const;

  MultinomialActNewtonResult solve(
      double lambda, const Eigen::MatrixXd &initial_beta,
      const Eigen::VectorXd &initial_intercept) const;

  // Logistic-style restricted solve.  Each fixed-IRLS quadratic is solved to
  // convergence on the supplied feature-block set.  After the accepted
  // proximal-Newton step updates probabilities/weights, inactive features are
  // checked against the true multinomial KKT conditions; violations are
  // activated before the next restricted IRLS solve.
  MultinomialActNewtonResult solve(
      double lambda, const Eigen::MatrixXd &initial_beta,
      const Eigen::VectorXd &initial_intercept,
      const std::vector<unsigned char> &initial_active_features) const;

  // Element (j, k) is the absolute nonnegative L1 penalty applied to
  // beta(j, k).  Intercepts remain unpenalized.
  MultinomialActNewtonResult solve(
      const Eigen::MatrixXd &l1_penalties) const;

  MultinomialActNewtonResult solve(
      const Eigen::MatrixXd &l1_penalties,
      const Eigen::MatrixXd &initial_beta,
      const Eigen::VectorXd &initial_intercept) const;

  MultinomialActNewtonResult solve(
      const Eigen::MatrixXd &l1_penalties,
      const Eigen::MatrixXd &initial_beta,
      const Eigen::VectorXd &initial_intercept,
      const std::vector<unsigned char> &initial_active_features) const;

 private:
  const MultinomialObjective &m_objective;
  MultinomialActNewtonOptions m_options;
};

// Mutable master-path state.  It contains only the last successfully
// committed L1 point; a failed solve leaves it unchanged.
struct MultinomialActNewtonPathState {
  Eigen::MatrixXd beta;
  Eigen::VectorXd intercept;
  // max_k |gradient(j, k)| at the committed master point; the sequential
  // strong rule needs only one scalar per feature, not a d-by-K copy.
  Eigen::VectorXd feature_gradient_max;
  std::vector<unsigned char> strong_set;
  double previous_lambda;
  bool initialized;

  MultinomialActNewtonPathState();
  void reset();

 private:
  std::shared_ptr<const detail::MultinomialPathSmoothCache> m_smooth_cache;

  friend class MultinomialActNewtonPathSolver;
};

struct MultinomialActNewtonPathResult {
  MultinomialActNewtonResult solution;
  int initial_strong_features;
  int strong_rule_activated_features;
  int full_kkt_reactivated_features;
  int final_strong_features;
  bool used_strong_rule;
  bool reused_initial_smooth_state;
};

// Path-level analogue of the legacy Logistic ActNewton driver: retain the L1
// master and its strong set, add sequential-strong-rule candidates, solve the
// constrained proximal-Newton/IRLS problem, then certify with a full KKT scan.
class MultinomialActNewtonPathSolver {
 public:
  explicit MultinomialActNewtonPathSolver(
      const MultinomialObjective &objective,
      const MultinomialActNewtonOptions &options =
          MultinomialActNewtonOptions());

  MultinomialActNewtonPathResult solve(
      double lambda, MultinomialActNewtonPathState *state) const;

 private:
  const MultinomialObjective &m_objective;
  MultinomialActNewtonOptions m_options;
  MultinomialActNewtonSolver m_solver;
};

}  // namespace solver
}  // namespace picasso

#endif  // PICASSO_MULTINOMIAL_ACTNEWTON_HPP
