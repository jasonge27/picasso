#include <picasso/solver_params.hpp>

#include <atomic>

namespace picasso {
namespace solver {

namespace {
std::atomic<int (*)(void)> g_interrupt_callback(nullptr);
}  // namespace

void set_interrupt_callback(int (*callback)(void)) {
  g_interrupt_callback.store(callback, std::memory_order_relaxed);
}

bool interrupt_requested() {
  int (*callback)(void) =
      g_interrupt_callback.load(std::memory_order_relaxed);
  return callback != nullptr && callback() != 0;
}

// training parameters
PicassoSolverParams::PicassoSolverParams() {
  num_lambda = 100;
  target_lambda = 1e-6;
  reg_type = L1;
  gamma = 3.0;
  num_relaxation_round = 3;
  prec = 1e-4;
  max_iter = 1000;
  include_intercept = true;
  dfmax = -1;
  dev_ratio_max = 0.999;
  dev_change_min = 1e-5;
  min_lambda_count = 5;
  lambdas.clear();
}

void PicassoSolverParams::set_lambdas(const double *lambda_path, int n) {
  if (lambda_path == nullptr || n <= 0) {
    lambdas.clear();
    num_lambda = 0;
    target_lambda = 0.0;
    return;
  }

  lambdas.resize(n);
  for (int i = 0; i < n; i++) lambdas[i] = lambda_path[i];
  num_lambda = lambdas.size();
  target_lambda = lambdas[num_lambda - 1];
}

const std::vector<double> &PicassoSolverParams::get_lambda_path() const {
  return lambdas;
}

}  // namespace solver
}  // namespace picasso
