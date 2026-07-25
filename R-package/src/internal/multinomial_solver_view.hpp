#ifndef PICASSO_INTERNAL_MULTINOMIAL_SOLVER_VIEW_HPP
#define PICASSO_INTERNAL_MULTINOMIAL_SOLVER_VIEW_HPP

#include "multinomial_problem_view.hpp"

#include <picasso/multinomial_actnewton.hpp>
#include <picasso/multinomial_lla.hpp>

#include <memory>
#include <vector>

namespace picasso {
namespace solver {
namespace internal {

#if defined(__GNUC__) || defined(__clang__)
#  define PICASSO_MULTINOMIAL_SOLVER_PRIVATE \
    __attribute__((visibility("hidden")))
#else
#  define PICASSO_MULTINOMIAL_SOLVER_PRIVATE
#endif

// Private path state used by synchronous view-based callers.  It deliberately
// mirrors the public path state without changing that public type or exposing
// the private smooth-state cache in an installed header.
struct PICASSO_MULTINOMIAL_SOLVER_PRIVATE MultinomialPathViewState {
  Eigen::MatrixXd beta;
  Eigen::VectorXd intercept;
  Eigen::VectorXd feature_gradient_max;
  std::vector<unsigned char> strong_set;
  std::shared_ptr<const detail::MultinomialPathSmoothCache> smooth_cache;
  double previous_lambda;
  bool initialized;

  MultinomialPathViewState();
  void reset();
};

PICASSO_MULTINOMIAL_SOLVER_PRIVATE
MultinomialActNewtonPathResult solve_multinomial_actnewton_path_view(
    const ::picasso::detail::MultinomialProblemView &problem,
    const MultinomialActNewtonOptions &options, double lambda,
    MultinomialPathViewState *state);

PICASSO_MULTINOMIAL_SOLVER_PRIVATE
MultinomialActNewtonResult solve_multinomial_actnewton_weighted_view(
    const ::picasso::detail::MultinomialProblemView &problem,
    const MultinomialActNewtonOptions &options,
    const Eigen::MatrixXd &l1_penalties,
    const Eigen::MatrixXd &initial_beta,
    const Eigen::VectorXd &initial_intercept,
    const std::vector<unsigned char> &initial_active_features);

PICASSO_MULTINOMIAL_SOLVER_PRIVATE
MultinomialLlaResult solve_multinomial_lla_from_l1_master_view(
    const ::picasso::detail::MultinomialProblemView &problem,
    const MultinomialActNewtonOptions &proximal_newton_options,
    const MultinomialLlaOptions &lla_options,
    MultinomialLlaPenalty penalty, double lambda, double gamma,
    MultinomialActNewtonResult master);

#undef PICASSO_MULTINOMIAL_SOLVER_PRIVATE

}  // namespace internal
}  // namespace solver
}  // namespace picasso

#endif  // PICASSO_INTERNAL_MULTINOMIAL_SOLVER_VIEW_HPP
