#include <picasso/actgd.hpp>
#include <picasso/objective.hpp>

namespace picasso {
namespace solver {
ActGDSolver::ActGDSolver(ObjFunction *obj, PicassoSolverParams param)
    : m_param(param), m_obj(obj) {
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
  actset_idx.clear();

  std::vector<double> grad(d, 0);
  for (int i = 0; i < d; i++) grad[i] = m_obj->get_grad(i);

  std::vector<double> old_coef(d);

  double tmp = 0.0;
  RegFunction *regfunc = nullptr;
  if (m_param.reg_type == SCAD)
    regfunc = new RegSCAD();
  else if (m_param.reg_type == MCP)
    regfunc = new RegMCP();
  else
    regfunc = new RegL1();

  int flag1 = 0;
  int flag2 = 1;
  for (int i = 0; i < lambdas.size(); i++) {
    // m_obj->update_auxiliary();
    regfunc->set_param(lambdas[i], m_param.gamma);

    for (int j = 0; j < d; j++)
      if (actset_indcat[j] == 0) {
        tmp = regfunc->threshold(fabs(grad[j]));
        if (fabs(tmp) > 1e-8) actset_indcat[j] = 1;
      }

    int loopcnt_level_0 = 0;
    flag2 = 1;
    while (loopcnt_level_0 < m_param.max_iter) {
      loopcnt_level_0 += 1;

      // Step 1: First pass constructing active set
      bool new_active_idx = true;
      int loopcnt_level_1 = 0;
      if (flag1 * flag2 != 0) {
        loopcnt_level_1 = m_param.max_iter + 1;
        new_active_idx = true;
      }

      while (loopcnt_level_1 < m_param.max_iter) {
        loopcnt_level_1 += 1;
        bool terminate_loop_level_1 = false;

        for (int j = 0; j < d; j++) {
          if (actset_indcat[j] == 0) continue;

          double beta_old = m_obj->get_model_coef(j);

          // compute thresholded coordinate
          m_obj->update_gradient(j);

          double updated_coord = m_obj->coordinate_descent(regfunc, j);

          if (updated_coord == beta_old) continue;

          if (actset_indcat_aux[j] == 0) {
            actset_idx.push_back(j);
            actset_indcat_aux[j] = 1;
          }

          if (m_obj->get_local_change(beta_old, j) > dev_thr)
            terminate_loop_level_1 = true;
        }

        if (terminate_loop_level_1) {
          new_active_idx = true;
          break;
        }

        new_active_idx = false;
        for (int j = 0; j < d; j++)
          if (actset_indcat[j] == 0) {
            m_obj->update_gradient(j);
            grad[j] = fabs(m_obj->get_grad(j));
            double tmp = regfunc->threshold(grad[j]);
            if (fabs(tmp) > 1e-8) {
              actset_indcat[j] = 1;
              new_active_idx = true;
            }
          }

        if (!new_active_idx) break;
      }

      flag1 = 1;

      if (!new_active_idx) break;

      // Step 2 : active set minimization
      // on the active coordinates
      loopcnt_level_1 = 0;
      while (loopcnt_level_1 < m_param.max_iter) {
        loopcnt_level_1 += 1;

        bool terminate_loop_level_1 = true;
        for (int j = 0; j < actset_idx.size(); j++) {
          int idx = actset_idx[j];
          double beta_old = m_obj->get_model_coef(idx);

          // compute thresholded coordinate
          m_obj->update_gradient(idx);
          double updated_coord = m_obj->coordinate_descent(regfunc, idx);

          if (beta_old == updated_coord) continue;

          if (m_obj->get_local_change(beta_old, idx) > dev_thr)
            terminate_loop_level_1 = false;
          // update gradient for idx
        }

        if (terminate_loop_level_1) {
          flag2 = 0;
          break;
        }
      }
    }
    m_obj->intercept_update();

    solution_path.push_back(m_obj->get_model_param());
  }

  delete regfunc;
}

}  // namespace solver
}  // namespace picasso