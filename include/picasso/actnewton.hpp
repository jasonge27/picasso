#ifndef PICASSO_ACTNEWTON_H
#define PICASSO_ACTNEWTON_H

#include <cmath>
#include <string>

#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>

namespace picasso {
namespace solver {
class ActNewtonSolver {
 private:
  PicassoSolverParams m_param;
  ObjFunction *m_obj;

  std::vector<int> itercnt_path;
  std::vector<ModelParam> solution_path;

 public:
  ActNewtonSolver(ObjFunction *obj, PicassoSolverParams param);

  void solve();

  const std::vector<int> &get_itercnt_path() const { return itercnt_path; };
  const ModelParam &get_model_param(int i) const { return solution_path[i]; };

  ~ActNewtonSolver() {
    delete m_obj;
    m_obj = nullptr;
  }
};

}  // namespace solver
}  // namespace picasso

#endif  // PICASSO_ACTNEWTON_H