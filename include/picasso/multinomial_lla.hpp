#ifndef PICASSO_MULTINOMIAL_LLA_HPP
#define PICASSO_MULTINOMIAL_LLA_HPP

#include <Eigen/Dense>

#include <vector>

#include <picasso/multinomial_actnewton.hpp>
#include <picasso/multinomial_objective.hpp>

namespace picasso {
namespace solver {

enum class MultinomialLlaPenalty {
  kMCP,
  kSCAD
};

enum class MultinomialLlaStatus {
  kCompleted,
  kStationarityLimit,
  kSubproblemFailed,
  kMajorizationFailed,
  kNumericalFailure
};

const char *multinomial_lla_status_string(MultinomialLlaStatus status);

// absolute_value is |beta| and must be finite and nonnegative.  The helpers
// validate lambda and gamma with the same rules as MultinomialLlaSolver.
double multinomial_lla_penalty_value(
    MultinomialLlaPenalty penalty, double absolute_value,
    double lambda, double gamma);

double multinomial_lla_penalty_derivative(
    MultinomialLlaPenalty penalty, double absolute_value,
    double lambda, double gamma);

enum class MultinomialLlaStoppingRule {
  // Run at least minimum_stages and stop only after the target nonconvex
  // stationarity residual reaches stationarity_tolerance.
  kTargetStationarity,
  // Run exactly minimum_stages. This preserves the historical fixed-stage
  // multi-stage convex-relaxation result, even when it is not a target
  // stationary point. In this mode maximum_stages must equal minimum_stages.
  kFixedStages
};

struct MultinomialLlaOptions {
  MultinomialLlaStoppingRule stopping_rule;
  // Both counts include the exact scalar-L1 master as stage zero. Production
  // uses an adaptive stationarity check with a default three-stage budget:
  // one L1 master plus two weighted-L1 updates. Raise maximum_stages when a
  // stricter nonconvex stationarity certificate is required.
  int minimum_stages;
  int maximum_stages;
  // A value of zero inherits the proximal-Newton outer KKT tolerance.
  double stationarity_tolerance;
  double majorization_tolerance;

  MultinomialLlaOptions();

  // Explicit opt-in to the historical one-L1-plus-two-LLA algorithm.
  static MultinomialLlaOptions fixed_stage_compatibility(
      int stage_count = 3);
};

struct MultinomialLlaStageRecord {
  int stage;
  bool is_l1_master;
  MultinomialSolverStatus subproblem_status;
  int outer_iterations;
  int inner_sweeps;
  long long coordinate_updates;
  double surrogate_objective;
  double subproblem_kkt_residual;
  double target_objective;
  double tangent_constant;
  double majorizer_at_anchor;
  double majorizer_at_solution;
  double target_stationarity;
};

struct MultinomialLlaResult {
  Eigen::MatrixXd beta;
  Eigen::VectorXd intercept;
  Eigen::MatrixXd l1_master_beta;
  Eigen::VectorXd l1_master_intercept;
  MultinomialLlaStatus status;
  int failed_stage;
  int completed_stages;
  int total_outer_iterations;
  int total_inner_sweeps;
  long long total_coordinate_updates;
  double final_target_objective;
  double final_target_stationarity;
  std::vector<MultinomialLlaStageRecord> stages;

  bool completed() const {
    return status == MultinomialLlaStatus::kCompleted;
  }

  // Reaching the adaptive stage budget is not a subproblem failure: beta and
  // intercept still contain the last fully validated majorization step.
  bool has_valid_solution() const {
    return completed() ||
           status == MultinomialLlaStatus::kStationarityLimit;
  }
};

// Runs one exact scalar-L1 master followed by local-linear approximation
// stages. Every weighted-L1 stage retains an outer-KKT certificate; adaptive
// mode avoids over-solving its inner quadratic beyond the tighter of that
// outer tolerance and the requested target-stationarity tolerance. A candidate
// is committed only after all majorization checks. The production default
// checks target stationarity adaptively and caps work at three total stages.
// A stationarity-limit result remains usable but is not convergence-certified.
// Fixed-stage compatibility is available only through an explicit option.
class MultinomialLlaSolver {
 public:
  explicit MultinomialLlaSolver(
      const MultinomialObjective &objective,
      const MultinomialActNewtonOptions &proximal_newton_options =
          MultinomialActNewtonOptions(),
      const MultinomialLlaOptions &lla_options =
          MultinomialLlaOptions());

  MultinomialLlaResult solve(
      MultinomialLlaPenalty penalty, double lambda, double gamma) const;

  MultinomialLlaResult solve(
      MultinomialLlaPenalty penalty, double lambda, double gamma,
      const Eigen::MatrixXd &initial_beta,
      const Eigen::VectorXd &initial_intercept) const;

  // Complete MCP/SCAD LLA from an already solved L1 master.  Production path
  // code uses this entry so the L1 master can come from the Logistic-style
  // strong-set path driver and remain the sole state committed across lambda.
  MultinomialLlaResult solve_from_l1_master(
      MultinomialLlaPenalty penalty, double lambda, double gamma,
      MultinomialActNewtonResult master) const;

 private:
  const MultinomialObjective &m_objective;
  MultinomialActNewtonOptions m_proximal_newton_options;
  MultinomialLlaOptions m_lla_options;
  MultinomialActNewtonSolver m_l1_master_solver;
  MultinomialActNewtonSolver m_proximal_newton_solver;
};

}  // namespace solver
}  // namespace picasso

#endif  // PICASSO_MULTINOMIAL_LLA_HPP
