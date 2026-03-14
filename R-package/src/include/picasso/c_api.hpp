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
    int dfmax,       // input: max nonzero coefficients for early stopping (-1 = no limit)
    double *offset,  // input: per-observation offset (length nn, may be nullptr)
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    int *num_fit,    // output: number of lambdas actually fit
    // default settings
    bool usePython = false
    );

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
    int dfmax,       // input: max nonzero coefficients for early stopping (-1 = no limit)
    double *offset,  // input: per-observation offset (length nn, may be nullptr)
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    int *num_fit,    // output: number of lambdas actually fit
    // default settings
    bool usePython = false
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
    int dfmax,       // input: max nonzero coefficients for early stopping (-1 = no limit)
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    int *num_fit,    // output: number of lambdas actually fit
    // default settings
    bool usePython = false
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
    int dfmax,       // input: max nonzero coefficients for early stopping (-1 = no limit)
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    int *num_fit,    // output: number of lambdas actually fit
    // default settings
    bool usePython = false
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
    int dfmax,       // input: max nonzero coefficients for early stopping (-1 = no limit)
    double *beta,    // output: an nlambda * d dim matrix
                     //         saving the coefficients for each lambda
    double *intcpt,  // output: an nlambda dim array
                     //         saving the model intercept for each lambda
    int *ite_lamb,   // output: number of iterations for each lambda
    int *size_act,   // output: an array of solution sparsity (model df)
    double *runt,    // output: runtime
    int *num_fit,    // output: number of lambdas actually fit
    // default settings
    bool usePython = false
    );


extern "C" void SolveMultinomialRegression(
    double *Y_int,   // input: class labels 0..K-1, length n
    double *X,       // input: model covariates (n x d, col-major)
    int nn,          // input: number of samples
    int dd,          // input: dimension
    int num_classes, // input: number of classes K
    double *lambda,  // input: regularization parameter
    int nnlambda,    // input: number of lambdas
    double gamma,    // input: gamma for SCAD or MCP
    int mmax_ite,    // input: max iterations
    double pprec,    // input: precision
    int reg_type,    // input: 1=L1, 2=MCP, 3=SCAD
    bool intercept,  // input: include intercept
    int dfmax,       // input: max nonzero (-1 = no limit)
    double *beta,    // output: d * K * nlambda (for each lambda: K*d, col-major per class)
    double *intcpt,  // output: K * nlambda
    int *ite_lamb,   // output: iterations per lambda
    int *size_act,   // output: total nonzero per lambda
    double *runt,    // output: runtime per lambda
    int *num_fit,    // output: lambdas actually fit
    bool usePython = false
    );

#endif  // PICASSO_C_API_H
