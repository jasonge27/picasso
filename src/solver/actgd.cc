#include <picasso/actgd.h>

namespace picasso {
namespace solver {
void ActGDSolver : solve(ObjFunction *obj) {
  unsigned int d = obj->get_dim();

  const std::vector<double> &lambdas = m_param.get_lambda_path();
  itercnt_path.resize(lambdas.size(), 0);

  double dev_thr = obj->get_deviance() * m_param.pred;

  std::vector<int> actset_indcat(d, 0);
  std::vector<int> actset_idx;

  std::vector<double> old_coef(d);

  double tmp = 0.0;
  int flag1 = 0;
  int flag2 = 1;
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
    flag2 = 1;
    while (loopcnt_level_0 < m_param.max_iter) {
      loopcnt_level_0 += 1;

      if (flag1 * flag2 != 0) {
        loopcnt_level_1 = m_param.max_iter + 1;
        new_active_idx = true;
      }

      // First pass, constructing active set
      int loopcnt_level_1 = 0;
      bool terminate_loop_level_1 = false;
      while (loopcnt_level_1 < m_param.max_iter) {
        loopcnt_level_1 += 1;

        for (int j = 0; j < d; j++) {
          if (actset_indcat[j] == 1) {
          }
          continue;
        }
      }
    }
  }
}
} // namespace solver
} // naemspace picasso