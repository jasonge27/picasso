#include "mymath.h"

void solve_lasso_with_cov_update(const double* X, 
    double * y, // the response
    const double* S, // S[j] = X[,j] * X[,j]
    const double* lambda, 
    const int n, const int d, const int df,
    const int max_ite, const double prec, const double dev_null,
    double* beta, 
    double* Xy, 
    double** XX, double * XX_act_idx, int * XX_act_idx_size,
    double* grad, // d-dim, gradient of square loss w.r.t beta
    double* intcpt, int* set_act, int* act_size, 
    double* runt, 
    int* inner_loop_count){
    int i, j;
    int d4 = d*4;
    int df1 = df+1;

    clock_t start, stop;
    start = clock();


    int loop_cnt = 0;
    while (loop_cnt < max_ite){
        for(i = 0; i < d; i++){
            grad[i] = Xy[i];
            for (k = 0; k < act_size; k++){
                j = set_act[k];
                XX_idx_j = XX_act_idx[j];
                XX_idx_i = XX_act_idx[i]; 
                grad[i] = grad[i] - XX[XX_idx_i][XX_idx_j] * beta[j];
            }
            grad[i] = grad[i] / n; 
        }


    }
}

void picasso_gaussian_solver(double *Y, double * X, double * XY, 
    double * beta, double * intcpt, int * beta_idx, int * cnzz, 
    int * col_cnz, int * ite_lamb, int * ite_cyc, double *obj, 
    double *runt, int * err, double *lambda, int *nnlambda, 
    double * ggamma, int *mmax_ite, double *pprec, int *fflag, 
    double *ttrunc, int * nn, int * dd, int * ddf, int *mmax_act_in, 
    int * aalg, double *LL){
    
    int i, j, idx, n, d, d4, df, df1, max_ite;
    int nlambda, ite1, ite2, flag, act_in, hybrid, cnz, act_size;
    int act_size1, act_size_all, max_act_in, alg, total_df;
    double gamma, prec2, ilambda, ilambda1, ilambda2, dif2, dbn, lamb_max, cutoff, trunc, intcpt_tmp, L;
    clock_t start, stop;
    
    n = *nn;
    d = *dd;
    df = *ddf;
    max_ite = *mmax_ite;

    prec2 = *pprec;
    nlambda = *nnlambda;
    gamma = *ggamma;
    flag = *fflag;
    L = *LL;
    alg = *aalg; 
    max_act_in = *mmax_act_in;
    trunc = *ttrunc;
    total_df = min_int(d,n)*nlambda;
    
    start = clock();

    for (i= 0; i < n; i++){
        if(i == 0) {
            stage_intcpt = 0;
            for (j = 0; j < d; j++){
                beta1[j] = 0.0;
            }
        } 

        // initialize lambda
        if(method_flag != 1){   // for nonconvex penalty
            for (j = 0; j < d; j++)
                stage_lambda[j] = lambda[i] * 
                            penalty_derivative(method_flag, fabs(beta1[j]), lambda[i], *ggamma);  
        } else {                // for convex penalty
            for (j = 0; j < d; j++)
                stage_lambda[j] = lambda[i];
        }

        function_value_old = get_penalized_gaussian_loss(method_flag, p, Y, 
                                                Xb, beta1, stage_intcpt, 
                                                n, d, lambda[i], *ggamma);

        int stage_count = 0;
        while (stage_count < max_ite){
            stage_count += 1;

            for (j = 0; j < d; j++){
                stage_beta_old[j] = beta1[j];
            }
            stage_intcpt_old = stage_intcpt;


            solve_weighted_lasso_with_cov_update(
                Y, X,
                beta_stage, &intcpt_stage,
                lambda_stage)

            // for convex penalty, simply break out of the loop
            // no need to run multistage convex relaxation
            if (method_flag == 1){
                ite_lamb[i] = outer_loop_count;
                break;  
            }

            // for nonconvex penalty
            // 1. check stopping criterion
            // 2. update stage_lambda


            function_value = get_penalized_gaussian_loss(method_flag, p, Y, Xb, 
                                                beta1, stage_intcpt, n, d,
                                                lambda[i], *ggamma);

            // only for R
            if (verbose){
                Rprintf("Stage:%d, for lambda:%f, fvalue:%f, pre:%f\n", 
                    stage_count, lambda[i], function_value, function_value_old);
            }

            if (fabs(function_value- function_value_old) < 0.01 * fabs(function_value_old)){
                break;
            }
            function_value_old = function_value;

            // update lambdas using the multistage convex relaxation scheme
            for (j = 0; j < d; j++){
                stage_lambda[j] = lambda[i] * 
                    penalty_derivative(method_flag, fabs(beta1[j]), lambda[i], *ggamma);
            }
        }
        intcpt[i] = stage_intcpt;     
        vec_copy(beta1, beta+i*d, set_act, size_act[i]);
    }
    



}
