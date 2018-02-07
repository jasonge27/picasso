#include <picasso/actgd.hpp>
#include <picasso/actnewton.hpp>
#include <picasso/c_api.hpp>
#include <picasso/objective.hpp>
#include <picasso/solver_params.hpp>
#include <vector>

void picasso_actnewton_solver(
    picasso::ObjFunction *obj,
    double *Y,       // input: 0/1 model response
    double *X,       // input: model covariates
    int n,           // input: number of samples
    int d,           // input: dimension
    double *lambda,  // input: regularization parameter
    int nlambda,     // input: number of lambda on the regularization path
    double gamma,    // input: gamma for SCAD or MCP penalty
    int mmax_ite,    // input: max number of interations
    double pprec,    // input: optimization precision
    int reg_type,    // input: type of regularization
    bool intercept,  // input: to have intercept term or not
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt     // output: runtime
) {
  picasso::solver::PicassoSolverParams param;
  param.set_lambdas(lambda, nlambda);
  param.gamma = gamma;
  if (reg_type == 1)
    param.reg_type = picasso::solver::L1;
  else if (reg_type == 2)
    param.reg_type = picasso::solver::MCP;
  else
    param.reg_type = picasso::solver::SCAD;

  param.include_intercept = intercept;
  param.prec = pprec;
  param.max_iter = mmax_ite;
  param.num_relaxation_round = 3;

  picasso::solver::ActNewtonSolver actnewton_solver(obj, param);
  actnewton_solver.solve();

  const std::vector<int> &itercnt_path = actnewton_solver.get_itercnt_path();

  for (int i = 0; i < nlambda; i++) {
    const picasso::ModelParam &model_param =
        actnewton_solver.get_model_param(i);
    ite_lamb[i] = itercnt_path[i];
    size_act[i] = 0;
    for (int j = 0; j < d; j++) {
      beta[i * d + j] = model_param.beta[j];
      if (fabs(beta[i * d + j]) > 1e-8) size_act[i]++;
    }
    intcpt[i] = model_param.intercept;
    runt[i] = 0.0;
  }
}

void picasso_actgd_solver(
    picasso::ObjFunction *obj,
    double *Y,       // input: 0/1 model response
    double *X,       // input: model covariates
    int n,           // input: number of samples
    int d,           // input: dimension
    double *lambda,  // input: regularization parameter
    int nlambda,     // input: number of lambda on the regularization path
    double gamma,    // input: gamma for SCAD or MCP penalty
    int max_ite,     // input: max number of interations
    double prec,     // input: optimization precision
    int reg_type,    // input: type of regularization
    bool intercept,  // input: to have intercept term or not
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt     // output: runtime
) {
  picasso::solver::PicassoSolverParams param;
  param.set_lambdas(lambda, nlambda);
  param.gamma = gamma;
  if (reg_type == 1)
    param.reg_type = picasso::solver::L1;
  else if (reg_type == 2)
    param.reg_type = picasso::solver::MCP;
  else
    param.reg_type = picasso::solver::SCAD;

  param.include_intercept = intercept;
  param.prec = prec;
  param.max_iter = max_ite;
  param.num_relaxation_round = 3;

  picasso::solver::ActGDSolver actgd_solver(obj, param);
  actgd_solver.solve();
  const std::vector<int> &itercnt_path = actgd_solver.get_itercnt_path();
  for (int i = 0; i < nlambda; i++) {
    const picasso::ModelParam &model_param = actgd_solver.get_model_param(i);
    ite_lamb[i] = itercnt_path[i];
    size_act[i] = 0;
    for (int j = 0; j < d; j++) {
      beta[i * d + j] = model_param.beta[j];
      if (fabs(beta[i * d + j]) > 1e-8) size_act[i]++;
    }
    intcpt[i] = model_param.intercept;
    runt[i] = 0.0;
  }
}

extern "C" void SolveLogisticRegression(
    double *Y,       // input: 0/1 model response
    double *X,       // input: model covariates
    int n,           // input: number of samples
    int d,           // input: dimension
    double *lambda,  // input: regularization parameter
    int nlambda,     // input: number of lambda on the regularization path
    double gamma,    // input: gamma for SCAD or MCP penalty
    int max_ite,     // input: max number of interations
    double pprec,    // input: optimization precision
    int reg_type,    // input: type of regularization
    bool intercept,  // input: to have intercept term or not
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    // default settings
    bool usePypthon
) {
  picasso::ObjFunction *obj =
      new picasso::LogisticObjective(X, Y, n, d, intercept, usePypthon);
  picasso_actnewton_solver(obj, Y, X, n, d, lambda, nlambda, gamma, max_ite,
                           pprec, reg_type, intercept, beta, intcpt, ite_lamb,
                           size_act, runt);
}

extern "C" void SolvePoissonRegression(
    double *Y,       // input: count model response
    double *X,       // input: model covariates
    int nn,          // input: number of samples
    int dd,          // input: dimension
    double *lambda,  // input: regularization parameter
    int nnlambda,    // input: number of lambda on the regularization path
    double gamma,    // input: gamma for SCAD or MCP penalty
    int mmax_ite,    // input: max number of interations
    double pprec,    // input: optimization precision
    int reg_type,    // input: type of regularization
    bool intercept,  // input: to have intercept term or not
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    // default settings
    bool usePypthon
) {
  picasso::ObjFunction *obj =
      new picasso::PoissonObjective(X, Y, nn, dd, intercept, usePypthon);
  picasso_actnewton_solver(obj, Y, X, nn, dd, lambda, nnlambda, gamma, mmax_ite,
                           pprec, reg_type, intercept, beta, intcpt, ite_lamb,
                           size_act, runt);
}

extern "C" void SolveSqrtLinearRegression(
    double *Y,       // input: model response
    double *X,       // input: model covariates
    int nn,          // input: number of samples
    int dd,          // input: dimension
    double *lambda,  // input: regularization parameter
    int nnlambda,    // input: number of lambda on the regularization path
    double gamma,    // input: gamma for SCAD or MCP penalty
    int mmax_ite,    // input: max number of interations
    double pprec,    // input: optimization precision
    int reg_type,    // input: type of regularization
    bool intercept,  // input: to have intercept term or not
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    // default settings
    bool usePypthon
) {
  picasso::ObjFunction *obj =
      new picasso::SqrtMSEObjective(X, Y, nn, dd, intercept, usePypthon);

  picasso_actnewton_solver(obj, Y, X, nn, dd, lambda, nnlambda, gamma, mmax_ite,
                           pprec, reg_type, intercept, beta, intcpt, ite_lamb,
                           size_act, runt);
}

extern "C" void SolveLinearRegressionNaiveUpdate(
    double *Y,       // input: model response
    double *X,       // input: model covariates
    int nn,          // input: number of samples
    int dd,          // input: dimension
    double *lambda,  // input: regularization parameter
    int nnlambda,    // input: number of lambda on the regularization path
    double gamma,    // input: gamma for SCAD or MCP penalty
    int mmax_ite,    // input: max number of interations
    double pprec,    // input: optimization precision
    int reg_type,    // input: type of regularization
    bool intercept,  // input: to have intercept term or not
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    // default settings
    bool usePypthon
) {
  picasso::ObjFunction *obj =
      new picasso::GaussianNaiveUpdateObjective(X, Y, nn, dd, intercept, usePypthon);
  picasso_actgd_solver(obj, Y, X, nn, dd, lambda, nnlambda, gamma, mmax_ite,
                       pprec, reg_type, intercept, beta, intcpt, ite_lamb,
                       size_act, runt);
}

extern "C" void SolveLinearRegressionCovUpdate(
    double *Y,       // input: model response
    double *X,       // input: model covariates
    int nn,          // input: number of samples
    int dd,          // input: dimension
    double *lambda,  // input: regularization parameter
    int nnlambda,    // input: number of lambda on the regularization path
    double gamma,    // input: gamma for SCAD or MCP penalty
    int mmax_ite,    // input: max number of interations
    double pprec,    // input: optimization precision
    int reg_type,    // input: type of regularization
    bool intercept,  // input: to have intercept term or not
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    // default settings
    bool usePypthon
) {
  picasso::ObjFunction *obj =
      new picasso::GaussianNaiveUpdateObjective(X, Y, nn, dd, intercept, usePypthon);
  picasso_actgd_solver(obj, Y, X, nn, dd, lambda, nnlambda, gamma, mmax_ite,
                       pprec, reg_type, intercept, beta, intcpt, ite_lamb,
                       size_act, runt);
}
