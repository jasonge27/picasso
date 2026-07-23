#ifndef PICASSO_ACTGD_H
#define PICASSO_ACTGD_H

#include <cmath>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>
#include <string>

namespace picasso {
namespace solver {

enum class ActGDPathStatus {
  kCompleted,
  kDfmaxReached,
  kIterationLimit
};

class ActGDSolver {
 private:
  struct CommitSink;

  PicassoSolverParams m_param;
  ObjFunction *m_obj;
  ActGDPathStatus m_status;
  int m_failed_lambda;

  std::vector<int> itercnt_path;
  std::vector<double> runtime_path;
  std::vector<double> smooth_objective_path;
  std::vector<ModelParam> solution_path;

  int solve_impl(CommitSink *sink);

 public:
  ActGDSolver(ObjFunction *obj, PicassoSolverParams param);

  void solve();

  // C API adapter: commit each fitted model directly to caller-owned buffers
  // without retaining the coefficient path in this solver.
  int solve_to_buffers(double *beta, double *intcpt, int *ite_lamb,
                       int *size_act, double *runt,
                       double *smooth_objective);

  const std::vector<int> &get_itercnt_path() const { return itercnt_path; };
  const std::vector<double> &get_runtime_path() const { return runtime_path; };
  const std::vector<double> &get_smooth_objective_path() const {
    return smooth_objective_path;
  }
  ActGDPathStatus get_status() const { return m_status; }
  int get_failed_lambda() const { return m_failed_lambda; }
  const ModelParam &get_model_param(int i) const { return solution_path[i]; };
  int get_num_lambdas_fit() const { return static_cast<int>(solution_path.size()); };

  // ObjFunction lifetime is owned by the caller.
  ~ActGDSolver() = default;
};

}  // namespace solver
}  // namespace picasso
#endif
