#include <picasso/actnewton.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>

namespace picasso {
namespace solver {
ActNewtonSolver::ActNewtonSolver(ObjFunction *obj, PicassoSolverParams param)
    : m_param(param), m_obj(obj) {
  itercnt_path.clear();
  solution_path.clear();
}

void ActNewtonSolver::solve() {
  unsigned int d = m_obj->get_dim();

  const std::vector<double> &lambdas = m_param.get_lambda_path();
  itercnt_path.resize(lambdas.size(), 0);

  double dev_thr = m_obj->get_deviance() * m_param.prec;

  // actset_indcat[i] == 1 if i is in the active set
  std::vector<int> actset_indcat(d, 0);
  // actset_idx <- which(actset_indcat==1)
  std::vector<int> actset_idx;

  std::vector<double> old_coef(d);

  // model parameters on the master path
  // each master parameter is relaxed into SCAD/MCP parameter
  ModelParam model_master = m_obj->get_model_param();

  std::vector<double> stage_lambdas(d, 0);
  RegFunction *regfunc = new RegL1();
  for (int i = 0; i < lambdas.size(); i++) {
    // start with the previous solution on the master path
    m_obj->set_model_param(model_master);

    // calculating gradients and other auxiliary vars such as r
    m_obj->update_auxiliary();

    // init the active set
    double threshold = 2 * lambdas[i];
    if (i > 0) threshold -= lambdas[i - 1];
    for (int j = 0; j < d; ++j) {
      stage_lambdas[j] = lambdas[i];

      if (m_obj->get_grad(j) > threshold) actset_indcat[j] = 1;
    }

    // loop level 0: multistage convex relaxation
    int loopcnt_level_0 = 0;
    while (loopcnt_level_0 < m_param.num_relaxation_round) {
      loopcnt_level_0++;

      // loop level 1: active set update
      int loopcnt_level_1 = 0;
      bool terminate_loop_level_1 = true;
      while (loopcnt_level_1 < m_param.max_iter) {
        loopcnt_level_1++;
        terminate_loop_level_1 = true;

        double old_intcpt = m_obj->get_model_coef(-1);
        for (int j = 0; j < d; j++) old_coef[j] = m_obj->get_model_coef(j);

        // initialize actset_idx
        actset_idx.clear();
        for (int j = 0; j < d; j++)
          if (actset_indcat[j]) {
            regfunc->set_param(stage_lambdas[j], 0.0);
            double updated_coord = m_obj->coordinate_descent(regfunc, j);

            if (fabs(updated_coord) > 0) actset_idx.push_back(j);
          }

        // loop level 2: proximal newton on active set
        int loopcnt_level_2 = 0;
        bool terminate_loop_level_2 = true;
        while (loopcnt_level_2 < m_param.max_iter) {
          loopcnt_level_2++;
          terminate_loop_level_2 = true;

          for (int k = 0; k < actset_idx.size(); k++) {
            int idx = actset_idx[k];

            double old_beta = m_obj->get_model_coef(idx);
            regfunc->set_param(stage_lambdas[idx], 0.0);
            m_obj->coordinate_descent(regfunc, idx);

            if (m_obj->get_local_change(old_beta, idx) > dev_thr)
              terminate_loop_level_2 = false;
          }

          if (m_param.include_intercept) {
            double old_intcpt = m_obj->get_model_coef(-1);
            m_obj->intercept_update();
            if (m_obj->get_local_change(old_intcpt, -1) > dev_thr)
              terminate_loop_level_2 = false;
          }

          if (terminate_loop_level_2) break;
        }

        itercnt_path[i] += loopcnt_level_2;

        // check stopping criterion 1: fvalue change
        for (int k = 0; k < actset_idx.size(); ++k) {
          int idx = actset_idx[k];
          if (m_obj->get_local_change(old_coef[idx], idx) > dev_thr)
            terminate_loop_level_1 = false;
        }
        if ((m_param.include_intercept) &&
            (m_obj->get_local_change(old_intcpt, -1) > dev_thr))
          terminate_loop_level_1 = false;

        // recompute grad, second order coef w jand other aux vars
        m_obj->update_auxiliary();

        // check stopping criterion 2: active set change
        for (int k = 0; k < d; k++)
          if (actset_indcat[k] == 0) {
            if (fabs(m_obj->get_grad(k)) > stage_lambdas[k]) {
              actset_indcat[k] = 1;
              terminate_loop_level_1 = false;
            }
          }

        if (terminate_loop_level_1) break;
      }

      if (loopcnt_level_0 == 1) model_master = m_obj->get_model_param();

      if (m_param.reg_type == L1) break;

      // update stage lambda
      for (int j = 0; j < d; j++) {
        double beta = m_obj->get_model_coef(j);

        switch (m_param.reg_type) {
          case MCP:
            stage_lambdas[j] = (fabs(beta) > lambdas[i] * m_param.gamma)
                                   ? 0.0
                                   : lambdas[i] - fabs(beta) / m_param.gamma;
          case SCAD:
            stage_lambdas[j] =
                (fabs(beta) > lambdas[i] * m_param.gamma)
                    ? 0.0
                    : ((fabs(beta) > lambdas[i])
                           ? ((lambdas[i] * m_param.gamma - fabs(beta)) /
                              (m_param.gamma - 1))
                           : lambdas[i]);
          default:
            stage_lambdas[j] = lambdas[i];
        }
      }
    }

    solution_path.push_back(m_obj->get_model_param());
  }

  delete regfunc;
}

}  // namespace solver
}  // namespace picasso
