#ifndef PICASSO_ACTGD_H
#define PICASSO_ACTGD_H

#include <cmath>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>
#include <string>

namespace picasso {
namespace solver {
class ActGDSolver {
 private:
  PicassoSolverParams m_param;
  ObjFunction *m_obj;

  std::vector<int> itercnt_path;
  std::vector<double> runtime_path;
  std::vector<ModelParam> solution_path;

 public:
  ActGDSolver(ObjFunction *obj, PicassoSolverParams param);

  void solve();

  const std::vector<int> &get_itercnt_path() const { return itercnt_path; };
  const std::vector<double> &get_runtime_path() const { return runtime_path; };
  const ModelParam &get_model_param(int i) const { return solution_path[i]; };
  int get_num_lambdas_fit() const { return static_cast<int>(solution_path.size()); };

  // ObjFunction lifetime is owned by the caller.
  ~ActGDSolver() = default;
};

}  // namespace solver
}  // namespace picasso
#endif
