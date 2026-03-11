#include <picasso/actgd.hpp>
#include <picasso/objective.hpp>

namespace picasso {
namespace solver {
ActGDSolver::ActGDSolver(ObjFunction *obj, PicassoSolverParams param)
    : m_param(param), m_obj(obj) {
  itercnt_path.clear();
  runtime_path.clear();
  solution_path.clear();
}

void ActGDSolver::solve() {
  unsigned int d = m_obj->get_dim();

  const std::vector<double> &lambdas = m_param.get_lambda_path();
  itercnt_path.resize(lambdas.size(), 0);
  runtime_path.resize(lambdas.size(), 0.0);
  solution_path.clear();
  solution_path.reserve(lambdas.size());

  double dev_thr = m_obj->get_deviance() * m_param.prec;

  // strong_set[j] == 1: variable j passed the strong rule screen
  // ever_active[j] == 1: variable j has been nonzero at some point
  std::vector<int> strong_set(d, 0);
  std::vector<int> ever_active(d, 0);
  std::vector<int> actset_idx;

  std::vector<double> grad(d, 0);
  for (unsigned int i = 0; i < d; i++) grad[i] = fabs(m_obj->get_grad(i));

  std::vector<double> dev_path;

  RegFunction *regfunc = NULL;
  if (m_param.reg_type == SCAD)
    regfunc = new RegSCAD();
  else if (m_param.reg_type == MCP)
    regfunc = new RegMCP();
  else
    regfunc = new RegL1();

  for (unsigned int i = 0; i < lambdas.size(); i++) {
    regfunc->set_param(lambdas[i], m_param.gamma);

    // Step 1: Strong rule screening
    // Variables already active stay in. New variables screened by strong rule.
    double strong_thr;
    if (i > 0)
      strong_thr = 2.0 * lambdas[i] - lambdas[i - 1];
    else
      strong_thr = 2.0 * lambdas[i];

    for (unsigned int j = 0; j < d; j++) {
      if (strong_set[j] == 0 && grad[j] > strong_thr) strong_set[j] = 1;
    }

    // Outer loop: solve on strong set, then check KKT on the rest
    for (int outer = 0; outer < m_param.max_iter; outer++) {

      // Step 2: Coordinate descent on strong set until convergence
      int cd_iter = 0;
      while (cd_iter < m_param.max_iter) {
        cd_iter++;
        bool converged = true;

        for (unsigned int j = 0; j < d; j++) {
          if (strong_set[j] == 0) continue;

          double beta_old = m_obj->get_model_coef(j);
          m_obj->update_gradient(j);
          double updated = m_obj->coordinate_descent(regfunc, j);

          if (updated != beta_old) {
            // track which variables have ever been active
            if (ever_active[j] == 0 && fabs(updated) > 1e-8) {
              actset_idx.push_back(j);
              ever_active[j] = 1;
            }
            if (m_obj->get_local_change(beta_old, j) > dev_thr)
              converged = false;
          }
        }

        if (converged) break;
      }

      // Step 3: KKT check on variables outside strong set
      bool kkt_violated = false;
      for (unsigned int j = 0; j < d; j++) {
        if (strong_set[j] == 1) continue;

        m_obj->update_gradient(j);
        grad[j] = fabs(m_obj->get_grad(j));

        // Check if this variable should be active (KKT violation)
        double tmp = regfunc->threshold(grad[j]);
        if (fabs(tmp) > 1e-8) {
          strong_set[j] = 1;
          kkt_violated = true;
        }
      }

      if (!kkt_violated) break;
      // If violations found, re-solve with expanded strong set
    }

    // Update gradients for strong set variables (for next lambda's screening)
    for (unsigned int j = 0; j < d; j++) {
      if (strong_set[j] == 1) {
        m_obj->update_gradient(j);
        grad[j] = fabs(m_obj->get_grad(j));
      }
    }

    if (m_param.include_intercept) {
      m_obj->intercept_update();
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
      for (unsigned int j = 0; j < d; j++)
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

          // 3. small relative deviance change
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
