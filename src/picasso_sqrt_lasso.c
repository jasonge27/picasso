#include "mathutils.h"
#include "IRLS_solver.h"
#include <R.h>

void picasso_sqrt_lasso_solver(
    double *Y,      // input: 0/1 model response 
    double *X,      // input: model covariates
    double *beta,   // output: an nlambda * d dim matrix 
                    // saving the coefficients for each lambda
    double *intcpt, // output: an nlambda dim array
                    // saving the model intercept for each lambda
    int *nn,        // number of samples
    int *dd,        // dimension
    int *ite_lamb,  // 
    int *ite_cyc,   // 
    int *size_act,  // output: an array of solution sparsity (model df)
    double *obj,    // output: objective function value
    double *runt,   // output: runtime
    double *lambda, // input: regularization parameter
    int *nnlambda,  // input: number of lambda on the regularization path
    double *ggamma, // input: 
    int *mmax_ite,  //
    double *pprec,  //
    int *fflag,      //
    int *vverbose   // input: 1 for verbose mode
    ){
    int i, j, k, s, n, d, nlambda;
    double tmp;
     
    n = *nn;
    d = *dd;
    int max_ite1 = *mmax_ite;
    int max_ite2 = *mmax_ite;
    double prec1 = *pprec;
    double prec2 = *pprec;
    nlambda = *nnlambda;

    int verbose = *vverbose;

    int *set_act = (int *) Calloc(d, int);
    
    double *beta_old = (double *) Calloc(d, double);
    double *stage_beta_old = (double *) Calloc(d, double);
    double *beta1 = (double *) Calloc(d, double);

    double *p = (double *) Calloc(n, double);

    double *Xb = (double *) Calloc(n, double);

    double *w = (double *) Calloc(n, double);
    double *r = (double *) Calloc(n, double);
    double *r_old = (double *) Calloc(n, double);

    int method_flag = *fflag;

    double dev_null = 0.0;

    double beta0_null = 0.0;
    double avr_y = 0.0;
    for (i = 0; i < n; i++){
        avr_y += Y[i];
    }

    int outer_loop_count;
    double dev_local;

    int stage_count;
    double * stage_lambda = (double *) Calloc(d, double);

    // cached intermediate vars on L1 path  
    double * beta_L1 = (double *) Calloc(d, double);
    int * active_set_L1 = (int *) Calloc(d, int);
    double * Xb_L1 = (double *) Calloc(n, double);
    double * p_L1 = (double *) Calloc(n, double);
    double * gr_L1 = (double *) Calloc(d, double);
    double intcpt_L1;

    int * active_set = (int *) Calloc(d, int);
    double * gr = (double *)Calloc(d, double);

    double L0 = 0;
    for (i = 0; i < n; i++)
        L0 += Y[i]*Y[i];
    L0 = sqrt(L0/n);
    dev_null = L0;

    for (i = 0; i < d; i++){
        active_set[i] = 0;
        gr[i] = 0;
        for (j = 0; j < n; j++)
            gr[i] += X[i*n+j] * Y[j];
        
        gr[i] = fabs(gr[i])/(n*L0);
    }
  
    double stage_intcpt;
    double intcpt_old; 
    
    double sum_w;
    double function_value, function_value_old;
    int new_active_coord;
    int terminate_loop;
    double thr;


    for (i=0; i<nlambda; i++) {
        if (verbose)
        Rprintf("lambda %d :%f\n", i, lambda[i]);

        if(i == 0) {
            stage_intcpt = 0;
            for (j = 0; j < d; j++){
                beta1[j] = 0.0;
            }
            for (j = 0; j < n; j++){
                Xb[j] = 0.0;
            }   
        } 

        if (i > 0){
            for (j = 0; j < d; j++){
                active_set[j] = active_set_L1[j];
                gr[j] = gr_L1[j];
                beta1[j] = beta_L1[j];
            }
            for (j = 0; j < n; j++){
                Xb[j] = Xb_L1[j];
                p[j] = p_L1[j];
            }
            stage_intcpt = intcpt_L1;
        }


        if (i > 0){
            for (j = 0; j < d; j++)
                if (active_set[j] == 0){
                    if (gr[j] > 2*lambda[i] - lambda[i-1]) active_set[j] = 1;
                }
        } else if (i == 0){
            for (j = 0; j < d; j++)
                if (active_set[j] == 0){
                    if (gr[j] > 2*lambda[i]) active_set[j] = 1;
                }
        } 

        prec1 = prec2;
        
        stage_count = 0;
 
        // initialize lambda
        function_value_old = 0.0;
        if(method_flag != 1){   // nonconvex penalty
            for (j = 0; j < d; j++)
           //     stage_lambda[j] = lambda[i] * 
            //                penalty_derivative(method_flag, fabs(beta1[j]), lambda[i], *ggamma); 
            stage_lambda[j] = lambda[i];

            //function_value_old = get_penalized_sqrt_mse_loss(method_flag, Y, 
            //                                    Xb, n, d, lambda[i], *ggamma); 

        } else {                // for convex penalty
            for (j = 0; j < d; j++)
                stage_lambda[j] = lambda[i];
        }

       
        int max_stage_ite = 3;

        while (stage_count < max_stage_ite){
            stage_count++;

            for (j = 0; j < d; j++){
                stage_beta_old[j] = beta1[j];
            }

            outer_loop_count = 0;
            while (outer_loop_count < max_ite1) {
                outer_loop_count++;
                
                double sqrtLoss = get_sqrt_mse_loss(Y, Xb, stage_intcpt, n, d);
                // to construct the coefficients for iterative reweighted LS
                for (j = 0; j < n; j++){
                    r[j] = Y[j] - Xb[j]-stage_intcpt;
                    r_old[j] = r[j];
                }

                // backup the old coefficients
                for (j = 0; j < d; j++)
                    beta_old[j] = beta1[j];
                
                intcpt_old = stage_intcpt;

                // to solve the iterative reweighted LS
                solve_weighted_sqrt_lasso_with_naive_update(X, Y,
                    r,
                    Xb,
                    beta1,
                    stage_lambda,
                    n, d,
                    max_ite2,  
                    prec2, dev_null,
                    active_set,
                    &stage_intcpt, 
                    set_act, 
                    &size_act[i], // active set size
                    &runt[i],  // total run time
                    &ite_cyc[i], // innner loop counter
                    verbose
                ); 
                
                // compute the change in LS function value
                // and check stopping criterion
                terminate_loop = 1;
                thr = prec1 * dev_null;
                for (s = 0; s < size_act[i]; s++){
                    k = set_act[s];
                    dev_local = sqrt_mse_obj_change(r_old, X, sqrtLoss, 
                                    k, n, beta1[k], beta_old[k]);
                    if (fabs(dev_local) > thr) {
                        terminate_loop = 0;
                        break;
                    }
                }

                if (terminate_loop){
                    break;
                }
                
                //update sqrtLoss 
                sqrtLoss = get_sqrt_mse_loss(Y, Xb, stage_intcpt, n, d);
                new_active_coord = 0;
                for (s = 0; s < d; s++)
                    if (active_set[s] == 0){
                        gr[s] = 0.0;

                        for (j = 0; j < n; j++)
                            gr[s] += r[j] * X[s*n+j];
                        gr[s] = fabs(gr[s])/(n*sqrtLoss);

                        if (gr[s] > stage_lambda[s]){
                            new_active_coord = 1;
                            active_set[s] = 1;
                        }
                    }
                if (!new_active_coord){
                    break;
                } 
            }
             // only for R
            if (verbose){
                Rprintf("-outer loop: %d\n", 
                        outer_loop_count);
            } 

            // for lambda = lambda[i]
            if (stage_count == 1){
                for (j = 0; j < d; j++){
                    beta_L1[j] = beta1[j];
                    active_set_L1[j] = active_set[j];
                    gr_L1[j] = gr[j];
                }
                for (j = 0; j < n; j++){
                    Xb_L1[j] = Xb[j];
                    p_L1[j] = p[j];
                }
                intcpt_L1 = stage_intcpt;
            }

            // for convex penalty, we jump out of the loop.
            // no need to run multistage convex relaxation
            if (method_flag == 1){
                ite_lamb[i] = outer_loop_count;
                break;  
            }

            // for nonconvex penalty
            // 1. check stopping criterion
            // 2. update stage_lambda

            function_value = get_penalized_sqrt_mse_loss(method_flag, Y, Xb, 
                                                beta1,  n, d,
                                                stage_lambda, *ggamma);

            // only for R
            if (verbose && (stage_count > 1)){
                Rprintf("Stage:%d, for lambda:%f, fvalue:%f, pre:%f\n", 
                    stage_count, lambda[i], function_value, function_value_old);
            }

            function_value_old = function_value;

            // update lambdas using the multistage convex relaxation scheme
            for (j = 0; j < d; j++){
                stage_lambda[j] = penalty_derivative(method_flag, beta1[j], lambda[i], *ggamma);
            }
        }           
        intcpt[i] = stage_intcpt;     
        vec_copy(beta1, beta+i*d, set_act, size_act[i]);
    }
    
    Free(beta_old);
    Free(beta1);
    Free(stage_beta_old);
    Free(set_act);
    Free(active_set);
    Free(stage_lambda);
    Free(gr);
    Free(p);
    Free(Xb);
    Free(w);
    Free(r);
    Free(beta_L1);
    Free(active_set_L1);
    Free(Xb_L1);
    Free(p_L1);
    Free(gr_L1);
    Free(r_old);
}
