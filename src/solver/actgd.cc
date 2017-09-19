#include <picasso/actgd.h>

namespace picasso {
namespace solver {
void ActGDSolver : solve(ObjFunction *obj) {
  unsigned int d = obj->get_dim();

  const std::vector<double> &lambdas = m_param.get_lambda_path();
  itercnt_path.resize(lambdas.size(), 0);

  double dev_thr = obj->get_deviance() * m_param.pred;

  std::vector<int> actset_indcat(d, 0);
  std::vector<int> actset_indcat_aux(d, 0);
  std::vector<int> actset_idx;

  std::vector<double> old_coef(d);

  double tmp = 0.0;
  bool new_active_idx;
  for (int i = 0; i < lambdas.size(); i++) {
    obj->update_auxiliary();

    for (int j = 0; j < d; j++)
      if (actset_indcat[j] == 0) {
        tmp = soft_thresh(obj->get_grad(j), lambdas[i], m_param.gamma,
                          m_param.reg_type);
        if (fabs(tmp) > 1e-8)
          actset_indcat[j] = 1;
      }

    int loopcnt_level_0 = 0;
    bool terminate_loop_level_0 = true;
    bool new_active_idx = false;
    while (loopcnt_level_0 < m_param.max_iter) {
      loopcnt_level_0 += 1;

      // Step 1: First pass constructing active set
      terminate_loop_level_0 = true;
      new_active_idx = false;
      for (int j = 0; j < d; j++) {
        if (actset_indcat[j] == 1) {
          double beta_old = obj->get_model_coef(j);

          // compute gradient
          // thresholding

          if (obj->get_grad(j) == beta_old)
            continue;

          if (actset_indcat_aux[j] == 0) {
            actset_idx.push_back(j);
            actset_indcat_aux[j] = 1;
            new_active_idx = true;
          }

          if (obj->get_local_change(beta_old, j) > dev_thr)
            terminate_loop_level_0 = false;
        }
      }

      if (!new_active_idx)
        terminate_loop_level_0 = true;

      if (terminate_loop_level_0)
        break;

      // Step 2 : active set minimization
      // on the active coordinates
      int loopcnt_level_1 = 0;
      bool terminate_loop_level_1 = true;
      while (loopcnt_level_1 < m_param.max_iter) {
        loopcnt_level_1 += 1;

        terminate_loop_level_1 = true;
        for (int j = 0; j < actset_idx.size(); j++) {
          int idx = actset_idx[j];

          double beta_old = obj->get_model_coef(idx);

          // compute thresholded coordinate

          if (obj->get_local_change(beta_old, idx) > dev_thr)
            terminate_loop_level_1 = false;
          // update gradient for idx
        }

        if (terminte_loop_level_1)
          break;
      }
    }

    // Second pass
  }
}
}
} // namespace solver
} // naemspace picasso