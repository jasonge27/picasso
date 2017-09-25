#include <picasso/actgd.hpp>
#include <picasso/objective.hpp>

namespace picasso {
namespace solver {
ActGDSolver::ActGDSolver(ObjFunction *obj, PicassoSolverParams param)
    : m_obj(obj), m_param(param) {
  itercnt_path.clear();
  solution_path.clear();
}

void ActGDSolver::solve() {
  unsigned int d = m_obj->get_dim();

  const std::vector<double> &lambdas = m_param.get_lambda_path();
  itercnt_path.resize(lambdas.size(), 0);

  double dev_thr = m_obj->get_deviance() * m_param.prec;

  std::vector<int> actset_indcat(d, 0);
  std::vector<int> actset_indcat_aux(d, 0);
  std::vector<int> actset_idx;

  std::vector<double> old_coef(d);

  double tmp = 0.0;
  bool new_active_idx;
  RegFunction *regfunc = nullptr;
  if (m_param.reg_type == SCAD)
    regfunc = new RegSCAD();
  else if (m_param.reg_type == MCP)
    regfunc = new RegMCP();
  else
    regfunc = new RegL1();

  for (int i = 0; i < lambdas.size(); i++) {
    m_obj->update_auxiliary();
    regfunc->set_param(lambdas[i], m_param.gamma);

    for (int j = 0; j < d; j++)
      if (actset_indcat[j] == 0) {
        // tmp = regfunc->threshold
        // tmp = soft_thresh_l1(m_obj->get_grad(j), lambdas[i], m_param.gamma,
        //                 m_param.reg_type);
        if (fabs(tmp) > 1e-8) actset_indcat[j] = 1;
      }

    int loopcnt_level_0 = 0;
    bool terminate_loop_level_0 = true;
    bool new_active_idx = false;
    while (loopcnt_level_0 < m_param.max_iter) {
      loopcnt_level_0 += 1;

      // Step 1: First pass constructing active set
      terminate_loop_level_0 = true;
      new_active_idx = false;
      for (int j = 0; j < d; j++)
        if (actset_indcat[j] == 1) {
          double beta_old = m_obj->get_model_coef(j);

          double updated_coord = m_obj->coordinate_descent(regfunc, j);

          if (updated_coord == beta_old) continue;

          if (actset_indcat_aux[j] == 0) {
            actset_idx.push_back(j);
            actset_indcat_aux[j] = 1;
            new_active_idx = true;
          }

          if (m_obj->get_local_change(beta_old, j) > dev_thr)
            terminate_loop_level_0 = false;
        }

      if (!new_active_idx) terminate_loop_level_0 = true;

      if (terminate_loop_level_0) break;

      // Step 2 : active set minimization
      // on the active coordinates
      int loopcnt_level_1 = 0;
      bool terminate_loop_level_1 = true;
      while (loopcnt_level_1 < m_param.max_iter) {
        loopcnt_level_1 += 1;

        terminate_loop_level_1 = true;
        for (int j = 0; j < actset_idx.size(); j++) {
          int idx = actset_idx[j];
          double beta_old = m_obj->get_model_coef(idx);

          // compute thresholded coordinate
          double updated_coord = m_obj->coordinate_descent(regfunc, j);

          if (m_obj->get_local_change(beta_old, idx) > dev_thr)
            terminate_loop_level_1 = false;
          // update gradient for idx
        }

        if (terminate_loop_level_1) break;
      }

      // update gradient
      for (int j = 0; j < d; j++)
        if (actset_idx[j]) {
          // update gradient here
        }

      // update intercept
    }

    solution_path.push_back(m_obj->get_model_param());
  }

  delete regfunc;
}

}  // namespace solver
}  // namespace picasso