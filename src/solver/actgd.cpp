#include <picasso/actgd.hpp>
#include <picasso/objective.hpp>

#include <algorithm>
#include <limits>
#include <memory>

namespace picasso {
namespace solver {
ActGDSolver::ActGDSolver(ObjFunction *obj, PicassoSolverParams param)
    : m_param(param),
      m_obj(obj),
      m_status(ActGDPathStatus::kCompleted),
      m_failed_lambda(-1) {
  itercnt_path.clear();
  runtime_path.clear();
  smooth_objective_path.clear();
  solution_path.clear();
}

struct ActGDSolver::CommitSink {
  CommitSink(double *beta_out, double *intercept_out, int *iterations_out,
             int *active_size_out, double *runtime_out,
             double *smooth_objective_out)
      : beta(beta_out),
        intercept(intercept_out),
        iterations(iterations_out),
        active_size(active_size_out),
        runtime(runtime_out),
        smooth_objective(smooth_objective_out) {}

  void commit(int path_index, const ModelParam &model, int dimension,
              int iteration_count, double runtime_value,
              double smooth_objective_value) {
    int nonzero_count = 0;
    for (int feature = 0; feature < dimension; ++feature) {
      const double coefficient = model.beta[feature];
      if (beta != nullptr)
        beta[static_cast<std::size_t>(path_index) * dimension + feature] =
            coefficient;
      if (fabs(coefficient) > 1e-8) ++nonzero_count;
    }
    if (intercept != nullptr) intercept[path_index] = model.intercept;
    if (iterations != nullptr) iterations[path_index] = iteration_count;
    if (active_size != nullptr) active_size[path_index] = nonzero_count;
    if (runtime != nullptr) runtime[path_index] = runtime_value;
    if (smooth_objective != nullptr)
      smooth_objective[path_index] = smooth_objective_value;
  }

  double *beta;
  double *intercept;
  int *iterations;
  int *active_size;
  double *runtime;
  double *smooth_objective;
};

void ActGDSolver::solve() { (void)solve_impl(nullptr); }

int ActGDSolver::solve_to_buffers(double *beta, double *intcpt,
                                  int *ite_lamb, int *size_act,
                                  double *runt, double *smooth_objective) {
  CommitSink sink(beta, intcpt, ite_lamb, size_act, runt, smooth_objective);
  return solve_impl(&sink);
}

int ActGDSolver::solve_impl(CommitSink *sink) {
  m_status = ActGDPathStatus::kCompleted;
  m_failed_lambda = -1;
  const int d = m_obj->get_dim();
  GaussianCovUpdateObjective *covariance_objective =
      dynamic_cast<GaussianCovUpdateObjective *>(m_obj);

  const std::vector<double> &lambdas = m_param.get_lambda_path();
  itercnt_path.resize(lambdas.size(), 0);
  runtime_path.resize(lambdas.size(), 0.0);
  smooth_objective_path.clear();
  smooth_objective_path.reserve(lambdas.size());
  solution_path.clear();
  if (sink == nullptr) solution_path.reserve(lambdas.size());

  int committed_count = 0;

  double dev_thr = m_obj->get_deviance() * m_param.prec;

  // strong_set[j] == 1: variable j passed the strong rule screen
  // ever_active[j] == 1: variable j has been nonzero at some point
  std::vector<unsigned char> strong_set(d, 0);
  std::vector<unsigned char> ever_active(d, 0);
  std::vector<int> strong_idx;
  std::vector<int> actset_idx;
  strong_idx.reserve(d);

  std::vector<double> grad(d, 0);
  for (int i = 0; i < d; i++) grad[i] = fabs(m_obj->get_grad(i));

  std::unique_ptr<RegFunction> regfunc;
  if (m_param.reg_type == SCAD)
    regfunc.reset(new RegSCAD());
  else if (m_param.reg_type == MCP)
    regfunc.reset(new RegMCP());
  else
    regfunc.reset(new RegL1());

  for (std::size_t i = 0; i < lambdas.size(); i++) {
    regfunc->set_param(lambdas[i], m_param.gamma);

    // Step 1: Strong rule screening
    // Variables already active stay in. New variables screened by strong rule.
    double strong_thr;
    if (i > 0)
      strong_thr = 2.0 * lambdas[i] - lambdas[i - 1];
    else
      strong_thr = 2.0 * lambdas[i];

    bool strong_added = false;
    for (int j = 0; j < d; j++) {
      if (strong_set[j] == 0 && grad[j] > strong_thr) {
        strong_set[j] = 1;
        strong_idx.push_back(j);
        strong_added = true;
      }
    }
    if (strong_added && m_param.reg_type != L1)
      std::sort(strong_idx.begin(), strong_idx.end());

    long long coordinate_passes = 0;
    bool lambda_converged = false;

    // Outer loop: converge on the compact strong set, then certify the full
    // KKT conditions. Newly violating variables expand strong_idx once.
    for (int outer = 0; outer < m_param.max_iter; outer++) {
      int strong_passes = 0;
      bool strong_problem_converged = false;
      while (strong_passes < m_param.max_iter) {
        // MCP and SCAD always sweep the sorted compact strong set: changing
        // their coordinate order can select a different local minimum. L1
        // uses the same low-overhead pass unless strong-inactive work is large
        // enough for the smaller active-first loop to amortize its extra scan.
        const std::size_t inactive_count =
            strong_idx.size() - actset_idx.size();
        const std::size_t active_first_threshold =
            std::max<std::size_t>(64, 2 * actset_idx.size());
        const bool use_active_first =
            m_param.reg_type == L1 &&
            inactive_count > active_first_threshold;
        if (!use_active_first) {
          bool strong_converged = true;
          for (std::size_t position = 0; position < strong_idx.size();
               position++) {
            const int j = strong_idx[position];
            const double beta_old = m_obj->get_model_coef(j);
            m_obj->update_gradient(j);
            const double updated =
                m_obj->coordinate_descent(regfunc.get(), j);
            if (ever_active[j] == 0 && fabs(updated) > 1e-8) {
              ever_active[j] = 1;
              actset_idx.push_back(j);
            }
            if (updated != beta_old &&
                m_obj->get_local_change(beta_old, j) > dev_thr)
              strong_converged = false;
          }
          if (m_param.include_intercept) m_obj->intercept_update();
          if (!strong_idx.empty()) {
            strong_passes++;
            coordinate_passes++;
          }
          if (strong_converged || strong_idx.empty()) {
            strong_problem_converged = true;
            break;
          }
          continue;
        }

        // First reconverge variables that have ever been nonzero. This is the
        // small hot loop on sparse paths and avoids scanning all d features.
        bool active_converged = true;
        if (!actset_idx.empty()) {
          for (std::size_t position = 0; position < actset_idx.size();
               position++) {
            const int j = actset_idx[position];
            const double beta_old = m_obj->get_model_coef(j);
            m_obj->update_gradient(j);
            const double updated =
                m_obj->coordinate_descent(regfunc.get(), j);
            if (updated != beta_old &&
                m_obj->get_local_change(beta_old, j) > dev_thr)
              active_converged = false;
          }
          if (m_param.include_intercept) m_obj->intercept_update();
          strong_passes++;
          coordinate_passes++;
        } else if (m_param.include_intercept) {
          m_obj->intercept_update();
        }

        if (!active_converged) continue;

        // The active set is conditionally converged. Scan only strong-rule
        // candidates that have never entered it. Any material update expands
        // actset_idx and sends control back to the compact active loop.
        bool inactive_changed = false;
        bool visited_inactive = false;
        for (std::size_t position = 0; position < strong_idx.size();
             position++) {
          const int j = strong_idx[position];
          if (ever_active[j] != 0) continue;
          visited_inactive = true;

          m_obj->update_gradient(j);
          grad[j] = fabs(m_obj->get_grad(j));
          if (fabs(regfunc->threshold(grad[j])) <= 1e-8) continue;

          const double beta_old = m_obj->get_model_coef(j);
          const double updated =
              m_obj->coordinate_descent(regfunc.get(), j);
          if (fabs(updated) > 1e-8) {
            ever_active[j] = 1;
            actset_idx.push_back(j);
            // Even a tolerance-sized first update changes the gradients seen
            // by variables scanned earlier. Reconverge the expanded active
            // set before certifying strong-inactive or global KKT conditions.
            inactive_changed = true;
          }
          if (updated != beta_old &&
              m_obj->get_local_change(beta_old, j) > dev_thr)
            inactive_changed = true;
        }
        if (m_param.include_intercept) m_obj->intercept_update();
        if (visited_inactive) {
          strong_passes++;
          coordinate_passes++;
        }

        if (!inactive_changed) {
          strong_problem_converged = true;
          break;
        }
      }

      // Do not certify or commit a quadratic subproblem that exhausted its
      // coordinate-sweep budget.  The caller retains only earlier lambdas.
      if (!strong_problem_converged) break;

      // The compact strong problem is converged.
      // Only now pay for a full KKT scan of variables outside the strong set.
      bool kkt_violated = false;
      for (int j = 0; j < d; j++) {
        if (strong_set[j] == 1) continue;

        m_obj->update_gradient(j);
        grad[j] = fabs(m_obj->get_grad(j));

        // L1 activation is characterized by the usual first-order KKT check.
        // For MCP/SCAD with a low-curvature, unstandardized column, beta = 0
        // can satisfy that check while a distant nonzero point has a smaller
        // one-coordinate objective. Probe the exact coordinate minimizer in
        // that case so screening stays consistent with coordinate_descent().
        bool activates = fabs(regfunc->threshold(grad[j])) > 1e-8;
        if (!activates && m_param.reg_type != L1) {
          const double updated =
              m_obj->coordinate_descent(regfunc.get(), j);
          activates = std::isfinite(updated) && fabs(updated) > 1e-8;
          if (activates && ever_active[j] == 0) {
            ever_active[j] = 1;
            actset_idx.push_back(j);
          }
        }
        if (activates) {
          strong_set[j] = 1;
          strong_idx.push_back(j);
          kkt_violated = true;
        }
      }
      if (kkt_violated && m_param.reg_type != L1)
        std::sort(strong_idx.begin(), strong_idx.end());

      if (!kkt_violated) {
        lambda_converged = true;
        break;
      }
      // If violations found, re-solve with expanded strong set
    }

    if (!lambda_converged) {
      m_status = ActGDPathStatus::kIterationLimit;
      m_failed_lambda = static_cast<int>(i);
      return committed_count;
    }

    // Update only compact strong-set gradients for the next sequential rule.
    for (std::size_t position = 0; position < strong_idx.size(); position++) {
      const int j = strong_idx[position];
      m_obj->update_gradient(j);
      grad[j] = fabs(m_obj->get_grad(j));
    }

    if (m_param.include_intercept) {
      m_obj->intercept_update();
    }

    const ModelParam &committed_model = m_obj->get_model_param_ref();
    if (sink == nullptr) solution_path.push_back(committed_model);
    itercnt_path[i] = static_cast<int>(std::min<long long>(
        coordinate_passes, std::numeric_limits<int>::max()));
    runtime_path[i] = 0.0;

    // track deviance for early stopping
    double cur_obj = fabs(covariance_objective != NULL
                              ? covariance_objective->path_eval()
                              : m_obj->eval());
    smooth_objective_path.push_back(cur_obj);

    if (sink != nullptr)
      sink->commit(committed_count, committed_model, d, itercnt_path[i],
                   runtime_path[i], cur_obj);
    ++committed_count;

    // early stopping checks (only after min_lambda_count lambdas)
    const int num_fit = committed_count;
    if (num_fit >= m_param.min_lambda_count) {
      int nnz = 0;
      for (std::size_t position = 0; position < actset_idx.size(); position++)
        if (fabs(committed_model.beta[actset_idx[position]]) > 1e-8)
          nnz++;

      // 1. dfmax: too many nonzero coefficients
      if (m_param.dfmax >= 0 && nnz > m_param.dfmax) {
        m_status = ActGDPathStatus::kDfmaxReached;
        return committed_count;
      }

      // deviance checks only when model has started fitting (nnz > 0)
      if (nnz > 0) {
        double null_dev = m_obj->get_deviance();
        if (null_dev > 0) {
          // 2. deviance ratio saturation
          double dev_ratio = 1.0 - cur_obj / null_dev;
          if (dev_ratio > m_param.dev_ratio_max) return committed_count;

          // 3. small relative deviance change
          int prev_idx = num_fit - 1 - m_param.min_lambda_count;
          if (prev_idx >= 0) {
            double prev_obj = smooth_objective_path[prev_idx];
            double change = fabs(prev_obj - cur_obj);
            if (cur_obj > 0 && change / cur_obj < m_param.dev_change_min)
              return committed_count;
          }
        }
      }
    }
  }
  return committed_count;
}

}  // namespace solver
}  // namespace picasso
