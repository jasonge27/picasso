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
  std::vector<ModelParam> solution_path;

 public:
  ActGDSolver(ObjFunction *obj, PicassoSolverParams param);

  void solve();

  const std::vector<int> &get_itercnt_path() const { return itercnt_path; };
  const ModelParam &get_model_param(int i) const { return solution_path[i]; };

  ~ActGDSolver() {
    delete m_obj;
    m_obj = nullptr;
  }
};

}  // namespace solver
}  // namespace picasso
#endif