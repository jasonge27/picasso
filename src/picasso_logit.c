#include "mathutils.h"
#include "IRLS_solver.h"
#include <R.h>

void picasso_logit_solver(
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

    int method_flag = *fflag;

    double dev_null = 0.0;
    double dev_sat = 0.0;

    double beta0_null = 0.0;
    double avr_y = 0.0;
    for (i = 0; i < n; i++){
        avr_y += Y[i];
    }

    avr_y = avr_y / n;
    beta0_null = log(avr_y /(1-avr_y));
    dev_null = -(avr_y * beta0_null + log(1 - avr_y)); // dev_null > 0

    dev_sat = dev_null; 

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
    double q0 = 1/(1 + exp(-beta0_null));
    for (i = 0; i < d; i++){
        active_set[i] = 0;
        gr[i] = 0;
        for (j = 0; j < n; j++){
            gr[i] += X[i*n+j] * (Y[j] - q0);
        }
        gr[i] = fabs(gr[i])/n;
    }
  
    double stage_intcpt;
    double stage_intcpt_old;
    double intcpt_old; 
    
    double sum_w;
    double function_value, function_value_old;
    int new_active_coord;
    int terminate_loop;
    double thr;


    for (i=0; i<nlambda; i++) {
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
        } else if (i ==0){
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

            function_value_old = get_penalized_logistic_loss(method_flag, p, Y, 
                                                Xb, beta1, stage_intcpt, 
                                                n, d, lambda[i], *ggamma); 
        } else {                // for convex penalty
            for (j = 0; j < d; j++)
                stage_lambda[j] = lambda[i];
        }


        int max_stage_ite = 3;
        // p[i] = 1/(1+exp(-intcpt-Xb[i]))
        p_update(p, Xb, stage_intcpt, n); 
        while (stage_count < max_stage_ite){
            stage_count++;

            for (j = 0; j < d; j++){
                stage_beta_old[j] = beta1[j];
            }
            stage_intcpt_old = stage_intcpt;

            outer_loop_count = 0;
            while (outer_loop_count < max_ite1) {
                outer_loop_count++;

                // to construct the coefficients for iterative reweighted LS
                sum_w = 0.0;
                for (j = 0; j < n; j++){
                    w[j] = p[j] * (1 - p[j]);
                    sum_w += w[j];
                    r[j] = Y[j] - p[j];
                }

                // backup the old coefficients
                for (j = 0; j < d; j++){
                    beta_old[j] = beta1[j];
                }
                intcpt_old = stage_intcpt;

                // to solve the iterative reweighted LS
                solve_weighted_lasso_with_naive_update(X, w, stage_lambda, 
                    n, d,
                    max_ite2,  
                    prec2, dev_null,
                    beta1, Xb, active_set, r, 
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
                    tmp = (beta1[k] - beta_old[k]);
                    tmp = tmp * tmp;
                    dev_local = 0.0;
                    for (j = 0; j < n; j++){
                        dev_local += w[i]*X[k*n+j]*X[k*n+j]*tmp;
                    }
                    dev_local = dev_local / (2*n);
                    if (dev_local > thr) {
                        terminate_loop = 0;
                        break;
                    }
                }

                tmp = (stage_intcpt - intcpt_old);
                dev_local = sum_w * tmp*tmp/ (2*n);
                if (dev_local > thr){
                    terminate_loop = 0;
                } 

                if (terminate_loop){
                    break;
                }
                
                // p[i] = 1/(1+exp(-intcpt-Xb[i]))
                p_update(p, Xb, stage_intcpt, n); 
                new_active_coord = 0;
                for (s = 0; s < d; s++)
                    if (active_set[s] == 0){
                        gr[s] = 0.0;
                        for (j = 0; j < n; j++)
                            gr[s] += (Y[j] - p[j]) * X[s*n+j];
                        gr[s] = fabs(gr[s])/n;
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
                Rprintf("--outer loop: %d, dev_null:%f\n", 
                        outer_loop_count, dev_null);
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

            // check stopping criterion
            // p[i] = 1/(1+exp(-intcpt-Xb[i]))
            p_update(p, Xb, stage_intcpt, n); 
          
            function_value = get_penalized_logistic_loss(method_flag, p, Y, Xb, 
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
    
    Free(beta_old);
    Free(beta1);
    Free(stage_beta_old);
    Free(set_act);
    Free(active_set);
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
}
