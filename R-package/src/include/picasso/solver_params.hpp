#ifndef PICASSO_SOLVER_PARAMS_H
#define PICASSO_SOLVER_PARAMS_H

#include <vector>

namespace picasso {
namespace solver {
enum RegType { L1, SCAD, MCP };

// Optional cooperative-interrupt hook shared by every native path loop.
// The callback must be cheap, must not throw or longjmp, and returns nonzero
// when the caller wants the current path to stop after the committed prefix.
// A null callback (the default) disables polling entirely.
void set_interrupt_callback(int (*callback)(void));
bool interrupt_requested();

// training parameters
class PicassoSolverParams {
 public:
  /*! number of regularization parameters */
  unsigned num_lambda;

  /*! the last paramter on the regularization path */
  double target_lambda;

  /*! type of regularization terms */
  RegType reg_type;

  /*! gamma param for SCAD and MCP regularization */
  double gamma;

  /*! Maximum total adaptive-LLA stages for SCAD and MCP. The count includes
   *  the initial L1 master and must be at least three for nonconvex fits. */
  unsigned num_relaxation_round;

  /*! precision of optimization */
  double prec;

  /*! max number of iteration for innner loop */
  int max_iter;

  /*! whether or not to add intercept term */
  bool include_intercept;

  /*! max number of nonzero coefficients for early stopping (-1 = no limit) */
  int dfmax;

  /*! max deviance ratio for early stopping (default 0.999) */
  double dev_ratio_max;

  /*! min relative deviance change for early stopping (default 1e-5) */
  double dev_change_min;

  /*! min number of lambdas before checking early stopping (default 5) */
  int min_lambda_count;

  std::vector<double> lambdas;

  PicassoSolverParams();

  void set_lambdas(const double *lambda_path, int n);

  const std::vector<double> &get_lambda_path() const;
};

}  // namespace solver
}  // namespace picasso

#endif
