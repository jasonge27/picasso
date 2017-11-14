#ifndef PICASSO_C_API_H
#define PICASSO_C_API_H

extern "C" void SolveLogisticRegression(
    double *Y,       // input: 0/1 model response
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
    bool usePypthon = false
    );

extern "C" void SolvePoissonRegression(
    double *Y,       // input: 0/1 model response
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
    bool usePypthon = false
    );

extern "C" void SolveSqrtLinearRegression(
    double *Y,       // input: 0/1 model response
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
    bool usePypthon = false
    );

extern "C" void SolveLinearRegressionNaiveUpdate(
    double *Y,       // input: 0/1 model response
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
    bool usePypthon = false
    );

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
    bool usePypthon = false
    );

#endif  // PICASSO_C_API_H
