#include <picasso/actgd.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

struct Scenario {
  const char *name;
  int n;
  int d;
  bool dump_path;
  bool dense_path;
};

Scenario parse_scenario(int argc, char **argv) {
  const std::string name = argc > 1 ? argv[1] : "equivalence";
  if (name == "equivalence")
    return {"equivalence", 160, 24, true, false};
  if (name == "n_gt_p") return {"n_gt_p", 20000, 120, false, false};
  if (name == "p_large") return {"p_large", 128, 3000, false, false};
  if (name == "dense_worst")
    return {"dense_worst", 1000, 600, false, true};
  std::cerr << "unknown scenario: " << name << std::endl;
  std::exit(2);
}

double uniform_signed(std::uint64_t &state) {
  state = state * UINT64_C(6364136223846793005) +
          UINT64_C(1442695040888963407);
  const double unit = static_cast<double>(state >> 11) *
                      (1.0 / 9007199254740992.0);
  return 2.0 * unit - 1.0;
}

std::size_t peak_rss_bytes() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
  return static_cast<std::size_t>(usage.ru_maxrss);
#else
  return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
}

double elapsed_ms(const std::chrono::steady_clock::time_point &start,
                  const std::chrono::steady_clock::time_point &end) {
  return std::chrono::duration_cast<
             std::chrono::duration<double, std::milli> >(end - start)
      .count();
}

}  // namespace

int main(int argc, char **argv) {
  Scenario scenario = parse_scenario(argc, argv);
  if (argc > 2 && std::string(argv[2]) == "dump")
    scenario.dump_path = true;
  const int n = scenario.n;
  const int d = scenario.d;

  std::vector<double> x(static_cast<std::size_t>(n) * d);
  std::vector<double> y(n, 0.35);
  std::vector<double> shared(n);
  std::uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
  for (int i = 0; i < n; ++i) {
    shared[i] = uniform_signed(state);
    y[i] += 0.08 * uniform_signed(state);
  }

  const double signal[] = {1.4, -1.1, 0.8, -0.65, 0.5, -0.4};
  const int signal_size =
      static_cast<int>(sizeof(signal) / sizeof(signal[0]));
  for (int j = 0; j < d; ++j) {
    for (int i = 0; i < n; ++i) {
      const double value =
          0.92 * uniform_signed(state) + 0.08 * shared[i];
      x[static_cast<std::size_t>(j) * n + i] = value;
      if (j < signal_size) y[i] += signal[j] * value;
    }
  }

  const auto construct_start = std::chrono::steady_clock::now();
  picasso::GaussianCovUpdateObjective objective(
      x.data(), y.data(), n, d, true, false);
  const auto construct_end = std::chrono::steady_clock::now();

  double lambda_max = 0.0;
  for (int j = 0; j < d; ++j)
    lambda_max = std::max(lambda_max, std::fabs(objective.get_grad(j)));
  const double sparse_ratios[] = {1.0, 0.82, 0.67, 0.54, 0.43, 0.34};
  const double dense_ratios[] = {1.0, 0.5, 0.1, 0.01, 0.001, 0.0};
  const double *ratios = scenario.dense_path ? dense_ratios : sparse_ratios;
  const int nlambda = 6;
  std::vector<double> lambdas(nlambda);
  for (int i = 0; i < nlambda; ++i)
    lambdas[i] = lambda_max * ratios[i];

  picasso::solver::PicassoSolverParams params;
  params.set_lambdas(lambdas.data(), nlambda);
  params.reg_type = picasso::solver::L1;
  params.include_intercept = true;
  params.prec = 1e-9;
  params.max_iter = 1000;
  params.min_lambda_count = nlambda + 1;

  const auto solve_start = std::chrono::steady_clock::now();
  picasso::solver::ActGDSolver solver(&objective, params);
  solver.solve();
  const auto solve_end = std::chrono::steady_clock::now();

  const int num_fit = solver.get_num_lambdas_fit();
  long double checksum = 0.0L;
  int last_nnz = 0;
  for (int path_index = 0; path_index < num_fit; ++path_index) {
    const picasso::ModelParam &model = solver.get_model_param(path_index);
    checksum += static_cast<long double>(path_index + 1) * model.intercept;
    if (scenario.dump_path) {
      std::cout << std::setprecision(17) << "PATH " << path_index << ' '
                << model.intercept;
    }
    for (int j = 0; j < d; ++j) {
      checksum += static_cast<long double>((path_index + 1) * (j + 1)) *
                  model.beta[j];
      if (path_index + 1 == num_fit && std::fabs(model.beta[j]) > 1e-8)
        ++last_nnz;
      if (scenario.dump_path) std::cout << ' ' << model.beta[j];
    }
    if (scenario.dump_path) std::cout << '\n';
  }

  std::cout << std::setprecision(17)
            << "RESULT scenario=" << scenario.name << " n=" << n
            << " d=" << d << " num_fit=" << num_fit
            << " last_nnz=" << last_nnz
            << " construct_ms=" << elapsed_ms(construct_start, construct_end)
            << " solve_ms=" << elapsed_ms(solve_start, solve_end)
            << " peak_rss_bytes=" << peak_rss_bytes()
            << " checksum=" << static_cast<double>(checksum) << '\n';
  return 0;
}
