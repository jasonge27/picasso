#include <picasso/solver_params.h>

namespace picasso {
namespace solver {
// training parameters
PicassoSolverParams::PicassoSolverParams() {
  num_lambda = 100;
  target_lambda = 1e-6;
  reg_type = L1;
  reg_gamma = 3.0;
  num_relaxation_round = 3;
  prec = 1e-4;
  max_iter = 1000;
  include_intercept = true;
  lambdas.clear();
}

void PicassoSolverParams::configure(
    const std::vector<std::pair<std::string, std::string>> &cfg) {
  for (auto iter = cfg.begin(); iter != cfg.end(); iter++) {
    if (iter->first == "nlambda")
      num_lambda = stoi(iter->second);
    else if (iter->first == "target_lambda")
      target_lambda = stof(iter->second);
    else if (iter->first == "reg_type") {
      if (iter->second == "L1")
        reg_type = L1;
      else if (iter->second == "SCAD")
        reg_type = SCAD;
      else if (iter->second == "MCP")
        reg_type = MCP;
      else { /* throw exception */
      }
    } else {
      /* TODO */
    }
  }
}

void PicassoSolverParams::set_lambdas(const double *lambda_path, int n) {
  lambdas.resize(n);
  for (int i = 0; i < n; i++)
    lambdas[i] = lambda_path[i];
  num_lambda = lambdas.size();
  target_lambda = lambdas[num_lambda - 1];
}

std::vector<double> PicassoSolverParams::get_lambda_path() const {
  // TODO
  std::vector<double> emptyvec;
  return emptyvec;
}

} // namespace solver
} // namespace picasso