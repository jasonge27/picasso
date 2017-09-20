#ifndef PICASSO_C_API_H
#define PICASSO_C_API_H

extern "C" void SolveLogisticRegression(
    double *Y,      // input: 0/1 model response 
    double *X,      // input: model covariates
    double *beta,   // output: an nlambda * d dim matrix 
                    // saving the coefficients for each lambda
    double *intcpt, // output: an nlambda dim array
                    // saving the model intercept for each lambda
    int nn,        // number of samples
    int dd,        // dimension
    int *ite_lamb,  // 
    int *ite_cyc,   // 
    int *size_act,  // output: an array of solution sparsity (model df)
    double *obj,    // output: objective function value
    double *runt,   // output: runtime
    double *lambda, // input: regularization parameter
    int nnlambda,  // input: number of lambda on the regularization path
    double ggamma, // input: 
    int mmax_ite,  //
    double pprec,  //
    int reg_type,    // type of regularization
    bool intercept   // to have intercept term or not 
){}

extern "C" void SolvePoissonRegression(
    double *Y,      // input: 0/1 model response 
    double *X,      // input: model covariates
    double *beta,   // output: an nlambda * d dim matrix 
                    // saving the coefficients for each lambda
    double *intcpt, // output: an nlambda dim array
                    // saving the model intercept for each lambda
    int nn,        // number of samples
    int dd,        // dimension
    int *ite_lamb,  // 
    int *ite_cyc,   // 
    int *size_act,  // output: an array of solution sparsity (model df)
    double *obj,    // output: objective function value
    double *runt,   // output: runtime
    double *lambda, // input: regularization parameter
    int nnlambda,  // input: number of lambda on the regularization path
    double ggamma, // input: 
    int mmax_ite,  //
    double pprec,  //
    int reg_type,    // type of regularization
    bool intercept   // to have intercept term or not
){}


extern "C" void SolveSqrtLinearRegression(
    double *Y,      // input: 0/1 model response 
    double *X,      // input: model covariates
    double *beta,   // output: an nlambda * d dim matrix 
                    // saving the coefficients for each lambda
    double *intcpt, // output: an nlambda dim array
                    // saving the model intercept for each lambda
    int nn,        // number of samples
    int dd,        // dimension
    int *ite_lamb,  // 
    int *ite_cyc,   // 
    int *size_act,  // output: an array of solution sparsity (model df)
    double *obj,    // output: objective function value
    double *runt,   // output: runtime
    double *lambda, // input: regularization parameter
    int nnlambda,  // input: number of lambda on the regularization path
    double ggamma, // input: 
    int mmax_ite,  //
    double pprec,  //
    int reg_type,    // type of regularization
    bool intercept   // to have intercept term or not
){}

extern "C" void SolveLinearRegressionNaiveUpdate(
    double *Y,      // input: 0/1 model response 
    double *X,      // input: model covariates
    double *beta,   // output: an nlambda * d dim matrix 
                    // saving the coefficients for each lambda
    double *intcpt, // output: an nlambda dim array
                    // saving the model intercept for each lambda
    int nn,        // number of samples
    int dd,        // dimension
    int *ite_lamb,  // 
    int *ite_cyc,   // 
    int *size_act,  // output: an array of solution sparsity (model df)
    double *obj,    // output: objective function value
    double *runt,   // output: runtime
    double *lambda, // input: regularization parameter
    int nnlambda,  // input: number of lambda on the regularization path
    double ggamma, // input: 
    int mmax_ite,  //
    double pprec,  //
    int reg_type,    // type of regularization
    bool intercept   // to have intercept term or not
){}

extern "C" void SolveLinearRegressionCovUpdate(
    double *Y,      // input: 0/1 model response 
    double *X,      // input: model covariates
    double *beta,   // output: an nlambda * d dim matrix 
                    // saving the coefficients for each lambda
    double *intcpt, // output: an nlambda dim array
                    // saving the model intercept for each lambda
    int nn,        // number of samples
    int dd,        // dimension
    int *ite_lamb,  // 
    int *ite_cyc,   // 
    int *size_act,  // output: an array of solution sparsity (model df)
    double *obj,    // output: objective function value
    double *runt,   // output: runtime
    double *lambda, // input: regularization parameter
    int nnlambda,  // input: number of lambda on the regularization path
    double ggamma, // input: 
    int mmax_ite,  //
    double pprec,  //
    int reg_type,    // type of regularization
    bool intercept   // to have intercept term or not
){}

#endif // PICASSO_C_API_H