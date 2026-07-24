#ifndef PICASSO_ACTNEWTON_H
#define PICASSO_ACTNEWTON_H

#include <cmath>
#include <string>

#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>

namespace picasso {
namespace solver {

// Per-lambda and aggregate termination state for scalar ActNewton paths.
// kStationarityLimit retains a fully solved/majorization-checked model; the
// three failure states do not commit the failing lambda.  kInterrupted is an
// aggregate-only cooperative host stop at a lambda boundary; the committed
// prefix remains usable.
enum class ActNewtonLlaStatus {
  kNotRun,
  kCompleted,
  kStationarityLimit,
  kSubproblemFailed,
  kMajorizationFailed,
  kNumericalFailure,
  kInterrupted
};

class ActNewtonSolver {
 private:
  struct CommitSink;

  PicassoSolverParams m_param;
  ObjFunction *m_obj;

  std::vector<int> itercnt_path;
  std::vector<double> runtime_path;
  std::vector<ModelParam> solution_path;
  std::vector<ActNewtonLlaStatus> lla_status_path;
  std::vector<int> lla_stages_path;
  std::vector<double> objective_path;
  std::vector<double> smooth_objective_path;
  std::vector<double> kkt_path;
  std::vector<double> stationarity_path;
  ActNewtonLlaStatus lla_path_status;
  int failed_lambda;
  int failed_stage;

  // Preserve the pre-output-sink private symbol for binary compatibility.
  // It retains the path exactly like the historical implementation.
  void solve_impl(bool objective_state_preinitialized);
  int solve_impl(bool objective_state_preinitialized, CommitSink *sink);

 public:
  ActNewtonSolver(ObjFunction *obj, PicassoSolverParams param);

  void solve();

  // C API objectives are solved immediately after construction (and optional
  // offset initialization), so their auxiliary state and full gradient are
  // already current. Ordinary C++ callers should keep using solve().
  void solve_preinitialized();

  // C API adapters: commit each validated final LLA model directly to
  // caller-owned buffers without retaining the coefficient path.  The
  // committed-count output is advanced only after an entire model has been
  // written, so callers can preserve a successful prefix if a later solve
  // throws.  last_nonzero_count is computed even when size_act is null.
  int solve_to_buffers(double *beta, double *intcpt, int *ite_lamb,
                       int *size_act, double *runt,
                       double *smooth_objective, int *committed_count,
                       int *last_nonzero_count);
  int solve_preinitialized_to_buffers(
      double *beta, double *intcpt, int *ite_lamb, int *size_act,
      double *runt, double *smooth_objective, int *committed_count,
      int *last_nonzero_count);

  const std::vector<int> &get_itercnt_path() const { return itercnt_path; };
  const std::vector<double> &get_runtime_path() const { return runtime_path; };
  const ModelParam &get_model_param(int i) const { return solution_path[i]; };
  int get_num_lambdas_fit() const { return static_cast<int>(solution_path.size()); };
  const std::vector<ActNewtonLlaStatus> &get_lla_status_path() const {
    return lla_status_path;
  }
  const std::vector<int> &get_lla_stages_path() const {
    return lla_stages_path;
  }
  const std::vector<double> &get_objective_path() const {
    return objective_path;
  }
  const std::vector<double> &get_smooth_objective_path() const {
    return smooth_objective_path;
  }
  const std::vector<double> &get_kkt_path() const { return kkt_path; }
  const std::vector<double> &get_stationarity_path() const {
    return stationarity_path;
  }
  ActNewtonLlaStatus get_lla_path_status() const { return lla_path_status; }
  int get_failed_lambda() const { return failed_lambda; }
  int get_failed_stage() const { return failed_stage; }

  // ObjFunction lifetime is owned by the caller.
  ~ActNewtonSolver() = default;
};

}  // namespace solver
}  // namespace picasso

#endif  // PICASSO_ACTNEWTON_H
