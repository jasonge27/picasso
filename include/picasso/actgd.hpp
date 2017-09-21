#ifndef PICASSO_ACTGD_H
#define PICASSO_ACTGD_H

#include <cmath>
#include <picasso/solver_params.h>
#include <string>

namespace picasso {
namespace solver {
class ActGDSolver {
private:
  const PicassoSolverParams m_param;
  const ObjFunction *m_obj;

  std::vector<int> itercnt_path;
  std::vector<ModelParam> solution_path;

public:
  ActGDSolver(ObjFunction *obj, PicassoSolverParams param)
      : m_param(param), m_obj(obj);

  void solve(ObjFunction *obj);
};

} // namespace solver
} // namespace picasso
#endif