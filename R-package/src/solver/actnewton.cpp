#include <picasso/actnewton.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>

namespace picasso {
namespace solver {
ActNewtonSolver::ActNewtonSolver(ObjFunction *obj, PicassoSolverParams param)
    : m_param(param), m_obj(obj) {
  itercnt_path.clear();
  runtime_path.clear();
  solution_path.clear();
}

void ActNewtonSolver::solve() {
  int d = m_obj->get_dim();
  int n = m_obj->get_sample_num();

  const std::vector<double> &lambdas = m_param.get_lambda_path();
  itercnt_path.resize(lambdas.size(), 0);
  runtime_path.resize(lambdas.size(), 0.0);
  solution_path.clear();
  solution_path.reserve(lambdas.size());

  double dev_thr = m_obj->get_deviance() * m_param.prec;

  // actset_indcat[i] == 1 if i is in the active set
  std::vector<int> actset_indcat(d, 0);
  std::vector<int> actset_indcat_master(d, 0);
  // actset_idx <- which(actset_indcat==1)
  std::vector<int> actset_idx;

  std::vector<double> old_coef(d);
  std::vector<double> grad(d);
  std::vector<double> grad_master(d);
  std::vector<double> dev_path;

  // std::vector<double> Xb_master(n);
  Eigen::ArrayXd Xb_master(n);

  for (int i = 0; i < d; i++) grad[i] = fabs(m_obj->get_grad(i));

  // model parameters on the master path
  // each master parameter is relaxed into SCAD/MCP parameter
  ModelParam model_master = m_obj->get_model_param();
  Xb_master = m_obj->get_model_Xb();

  std::copy(grad.begin(), grad.end(), grad_master.begin());

  std::vector<double> stage_lambdas(d, 0);
  RegFunction *regfunc = new RegL1();
  for (int i = 0; i < lambdas.size(); i++) {
    // start with the previous solution on the master path
    m_obj->set_model_param(model_master);
    m_obj->set_model_Xb(Xb_master);

    std::copy(grad_master.begin(), grad_master.end(), grad.begin());
    std::copy(actset_indcat_master.begin(), actset_indcat_master.end(),
              actset_indcat.begin());

    // init the active set
    double threshold;
    if (i > 0)
      threshold = 2 * lambdas[i] - lambdas[i - 1];
    else
      threshold = 2 * lambdas[i];

    for (int j = 0; j < d; ++j) {
      stage_lambdas[j] = lambdas[i];

      if (grad[j] > threshold) actset_indcat[j] = 1;
    }

    m_obj->update_auxiliary();
    // loop level 0: multistage convex relaxation
    int loopcnt_level_0 = 0;
    int idx;
    double old_beta, old_intcpt, updated_coord, beta;
    while (loopcnt_level_0 < m_param.num_relaxation_round) {
      loopcnt_level_0++;

      // loop level 1: active set update
      int loopcnt_level_1 = 0;
      bool terminate_loop_level_1 = true;
      while (loopcnt_level_1 < m_param.max_iter) {
        loopcnt_level_1++;
        terminate_loop_level_1 = true;

        old_intcpt = m_obj->get_model_coef(-1);
        for (int j = 0; j < d; j++) old_coef[j] = m_obj->get_model_coef(j);

        // initialize actset_idx
        actset_idx.clear();
        for (int j = 0; j < d; j++)
          if (actset_indcat[j]) {
            regfunc->set_param(stage_lambdas[j], 0.0);
            updated_coord = m_obj->coordinate_descent(regfunc, j);

            if (fabs(updated_coord) > 0) actset_idx.push_back(j);
          }

        // loop level 2: proximal newton on active set
        int loopcnt_level_2 = 0;
        bool terminate_loop_level_2 = true;
        while (loopcnt_level_2 < m_param.max_iter) {
          loopcnt_level_2++;
          terminate_loop_level_2 = true;

          for (int k = 0; k < actset_idx.size(); k++) {
            idx = actset_idx[k];

            old_beta = m_obj->get_model_coef(idx);
            regfunc->set_param(stage_lambdas[idx], 0.0);

            m_obj->coordinate_descent(regfunc, idx);

            if (m_obj->get_local_change(old_beta, idx) > dev_thr)
              terminate_loop_level_2 = false;
          }

          if (m_param.include_intercept) {
            old_intcpt = m_obj->get_model_coef(-1);
            m_obj->intercept_update();
            if (m_obj->get_local_change(old_intcpt, -1) > dev_thr)
              terminate_loop_level_2 = false;
          }

          if (terminate_loop_level_2) break;
        }

        itercnt_path[i] += loopcnt_level_2;

        terminate_loop_level_1 = true;
        // check stopping criterion 1: fvalue change
        for (int k = 0; k < actset_idx.size(); ++k) {
          idx = actset_idx[k];
          if (m_obj->get_local_change(old_coef[idx], idx) > dev_thr)
            terminate_loop_level_1 = false;
        }
        if ((m_param.include_intercept) &&
            (m_obj->get_local_change(old_intcpt, -1) > dev_thr))
          terminate_loop_level_1 = false;

        // update p and w
        m_obj->update_auxiliary();

        // check for new active variables (must run even when active set
        // is empty or converged, otherwise variables never get activated)
        bool new_active_idx = false;
        for (int k = 0; k < d; k++)
          if (actset_indcat[k] == 0) {
            m_obj->update_gradient(k);
            grad[k] = fabs(m_obj->get_grad(k));
            if (grad[k] > stage_lambdas[k]) {
              actset_indcat[k] = 1;
              new_active_idx = true;
            }
          }

        if (!new_active_idx && terminate_loop_level_1) break;
      }

      if (loopcnt_level_0 == 1) {
        const ModelParam &model_master_ref = m_obj->get_model_param_ref();
        const Eigen::ArrayXd &Xb_master_ref = m_obj->get_model_Xb_ref();

        model_master.intercept = model_master_ref.intercept;
        model_master.beta = model_master_ref.beta;

        std::copy(grad.begin(), grad.end(), grad_master.begin());
        std::copy(actset_indcat.begin(), actset_indcat.end(),
                  actset_indcat_master.begin());

        Xb_master = Xb_master_ref;
      }

      if (m_param.reg_type == L1) break;

      m_obj->update_auxiliary();

      // update stage lambda
      for (int j = 0; j < d; j++) {
        beta = m_obj->get_model_coef(j);

        if (m_param.reg_type == MCP) {
          stage_lambdas[j] = (fabs(beta) > lambdas[i] * m_param.gamma)
                                 ? 0.0
                                 : lambdas[i] - fabs(beta) / m_param.gamma;

        } else if (m_param.reg_type == SCAD)
          stage_lambdas[j] =
              (fabs(beta) > lambdas[i] * m_param.gamma)
                  ? 0.0
                  : ((fabs(beta) > lambdas[i])
                         ? ((lambdas[i] * m_param.gamma - fabs(beta)) /
                            (m_param.gamma - 1))
                         : lambdas[i]);
        else
          stage_lambdas[j] = lambdas[i];
      }
    }

    solution_path.push_back(m_obj->get_model_param_ref());
    runtime_path[i] = 0.0;

    // track deviance for early stopping
    double cur_obj = fabs(m_obj->eval());
    dev_path.push_back(cur_obj);

    // early stopping checks (only after min_lambda_count lambdas)
    int num_fit = static_cast<int>(solution_path.size());
    if (num_fit >= m_param.min_lambda_count) {
      int nnz = 0;
      for (int j = 0; j < d; j++)
        if (fabs(m_obj->get_model_coef(j)) > 1e-8) nnz++;

      // 1. dfmax: too many nonzero coefficients
      if (m_param.dfmax >= 0 && nnz > m_param.dfmax) break;

      // deviance checks only when model has started fitting (nnz > 0)
      if (nnz > 0) {
        double null_dev = m_obj->get_deviance();
        if (null_dev > 0) {
          // 2. deviance ratio saturation
          double dev_ratio = 1.0 - cur_obj / null_dev;
          if (dev_ratio > m_param.dev_ratio_max) break;

          // 3. small relative deviance change over last min_lambda_count steps
          int prev_idx = num_fit - 1 - m_param.min_lambda_count;
          if (prev_idx >= 0) {
            double prev_obj = dev_path[prev_idx];
            double change = fabs(prev_obj - cur_obj);
            if (cur_obj > 0 && change / cur_obj < m_param.dev_change_min)
              break;
          }
        }
      }
    }
  }
  delete regfunc;
}

}  // namespace solver
}  // namespace picasso
