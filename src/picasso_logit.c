#include "mymath.h"
#include <R.h>

void calc_IRLS_coef(const double *  w, const double *  X, 
     double *  r,  double *  beta, 
     int k,  int n, 
    double * g, double * a){
    (*g) = 0.0;
    (*a) = 0.0;

    int i;
    double tmp;
    for (i = 0; i < n; i++){
        tmp = w[i]*X[k*n+i]*X[k*n+i];
        (*g) += tmp*beta[k] + r[i]*X[k*n+i];
        (*a) += tmp;
    }
    (*g) = (*g) / (2*n);
    (*a) = (*a) / (2*n);
}

void update_residual(double * r, const double *  w, 
    const double *  X, const double delta, 
    const int k, const int n){
    int i;
    for (i = 0; i < n; i++){
        r[i] = r[i] - w[i] * X[k*n+i] * delta; // delta = beta_new(k) - beta_old(k)
    }
}

void solve_weighted_lasso_with_naive_update(const double* X, 
    const double* w, // w = p * (1-p)
    const double* lambda, 
    const int n, const int d,
    const int max_ite, const double prec, const double dev_null,
    double* beta, double* Xb, 
    double * r, // r = y - p 
    double* intcpt, int* set_act, int* act_size, 
    double* runt, 
    int* inner_loop_count){

    int i, j, k, m, size_a;
    int c_idx;

    double g, tmp;
    clock_t start, stop;
    
    start = clock();

    int loopcnt = 0;

    double a =0.0;
    size_a = 0;
  
    double sum_w = 0.0;
    for (i = 0; i < n; i++){
        sum_w += w[i];
    }

    for (k = 0; k < d; k++){
        calc_IRLS_coef(w, X, r, beta, k, n, &g, &a);

        tmp  = beta[k];
        if (2*fabs(g) > lambda[k]){
            set_act[size_a] = k;
            size_a += 1;

            beta[k] = soft_thresh_l1(2*g, lambda[k]) / (2*a);
        } else {
            beta[k] = 0.0;
        }

        if (tmp == beta[k])
            continue;

        X_beta_update(Xb, X+k*n, -tmp, n);
        X_beta_update(Xb, X+k*n, beta[k], n);

        update_residual(r, w, X, beta[k]-tmp, k, n);
    }



    double dev_local, dev_change;   
    double sum_r; 
    while (loopcnt < max_ite) {  
        loopcnt ++;
        dev_change = 0;
        for (m = 0; m < size_a; m++) {
            c_idx = set_act[m];
                
            calc_IRLS_coef(w, X, r, beta, c_idx, n, &g, &a);

            tmp  = beta[c_idx];

            if (2*fabs(g) > lambda[c_idx]){
                beta[c_idx] = soft_thresh_l1(2*g, lambda[c_idx]) / (2*a);                
            } else{
                beta[c_idx] = 0.0;
            }

            if (tmp == beta[c_idx])
                continue;
            
            update_residual(r, w, X, beta[c_idx]-tmp, c_idx, n);

            X_beta_update(Xb, X+c_idx*n, -tmp, n);
            X_beta_update(Xb, X+c_idx*n, beta[c_idx], n);

            dev_local = 0.0;
            tmp = (beta[c_idx]-tmp)*(beta[c_idx]-tmp);
            for (i = 0; i < n; i++){
                dev_local += w[i]*X[c_idx*n+i]*X[c_idx*n+i];
            }
            dev_local = dev_local * tmp / (2*n);
            if (dev_local > dev_change){
                dev_change = dev_local;
            }
        }
       
        //update intercept
        sum_r = 0.0;
        for (i = 0; i < n; i++){
            sum_r += r[i];
        } 
        tmp = sum_r / sum_w; 
        if (sum_r != 0){
            for (i = 0; i < n; i++){
                r[i] = r[i] - tmp * w[i];
            }
            (*intcpt) += tmp;
            dev_local = sum_w * tmp*tmp/ (2*n);
            if (dev_local > dev_change){
                dev_change = dev_local;
            }
        }
        
        if (dev_change < prec * dev_null){
            break;
        }           
    }

    *inner_loop_count = loopcnt;
       

    stop = clock();
    *runt = (double)(stop - start)/CLOCKS_PER_SEC;
    *act_size = size_a;
}

double penalty_derivative(int method_flag, double x, double lambda, double gamma){
    // mcp
    if (method_flag == 2){
        if (x > lambda * gamma){
            return(0);
        } else{
            return(1 - x/(lambda*gamma));
        }
    }
    // scad
    if (method_flag == 3){
        if (x > lambda * gamma){
            return(0);
        } else if ( x > lambda){
            return((lambda*gamma-x)/(gamma-1));
        } else {
            return(1.0);
        }
    }

    return(0);
}

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
    double *Xb_previous_lambda = (double *) Calloc(n, double);

    double *w = (double *) Calloc(n, double);
    double *r = (double *) Calloc(n, double);
    int * inactive_set = (int *) Calloc(n, int); // inactive_set[i] = 1 -- i is inactive

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
    double dev_local, dev_change;

    int stage_count;
    double * stage_lambda = (double *) Calloc(d, double);
    double * beta_previous_lambda = (double *) Calloc(d, double);
  
    double stage_intcpt;
    double stage_intcpt_old;
    double intcpt_old; 
    double intcpt_previous_lambda;
    double sum_w;
    double function_value, function_value_old;

    for (i = 0; i < d; i++) inactive_set[i] = 0;

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

        prec1 = prec2;
        
        stage_count = 0;
 
        // initialize lambda
        function_value_old = 0.0;
        if(method_flag != 1){   // nonconvex penalty
            for (j = 0; j < d; j++)
                stage_lambda[j] = lambda[i] * 
                            penalty_derivative(method_flag, fabs(beta1[j]), lambda[i], *ggamma); 

            function_value_old = get_penalized_logistic_loss(method_flag, p, Y, 
                                                Xb, beta1, stage_intcpt, 
                                                n, d, lambda[i], *ggamma); 
        } else {                // for convex penalty
            for (j = 0; j < d; j++)
                stage_lambda[j] = lambda[i];

/*
            if (i == 0){
                tmp = lambda[i];
            } else {
                tmp = 2*lambda[i] - lambda[i-1];
            }

            for (j = 0; j < d; j++){
                if (rX[j] < tmp){}
            }
*/
        }
          


  

        int max_stage_ite = 1000;
        while (stage_count < max_stage_ite){
            stage_count++;

            for (j = 0; j < d; j++){
                stage_beta_old[j] = beta1[j];
            }
            stage_intcpt_old = stage_intcpt;

            outer_loop_count = 0;
            while (outer_loop_count < max_ite1) {
                outer_loop_count++;

                // to construct an iterative reweighted LS
                p_update(p, Xb, stage_intcpt, n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
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
                    beta1, Xb, r, 
                    &stage_intcpt, 
                    set_act, 
                    &size_act[i], // active set size
                    &runt[i],  // total run time
                    &ite_cyc[i] // innner loop counter
                ); 
                
                // compute the change in LS function value
                // and check stopping criterion
                dev_change = -1.0;
                for (s = 0; s < size_act[i]; s++){
                    k = set_act[s];
                    tmp = (beta1[k] - beta_old[k]);
                    tmp = tmp * tmp;
                    dev_local = 0.0;
                    for (j = 0; j < n; j++){
                        dev_local += w[i]*X[k*n+j]*X[k*n+j]*tmp;
                    }
                    dev_local = dev_local / (2*n);
                    if (dev_local > dev_change){
                        dev_change = dev_local;
                    }        
                }

                tmp = (stage_intcpt - intcpt_old);
                dev_local = sum_w * tmp*tmp/ (2*n);
                if (dev_local > dev_change){
                    dev_change = dev_local;
                } 
                
                // only for R
                if (verbose){
                    Rprintf("--outer loop: %d, dev_change:%f, dev_null:%f\n", 
                        outer_loop_count, dev_change, dev_null);
                } 

                if ((dev_change >= 0) && (dev_change < prec1 * dev_null)){
                    break;
                }

                // update inactive set

            }

            // for lambda = lambda[i]
            if (stage_count == 1){
                for (j = 0; j < d; j++){
                    beta_previous_lambda[j] = beta1[j];
                }
                for (j = 0; j < n; j++){
                    Xb_previous_lambda[j] = Xb[j];
                }
                intcpt_previous_lambda = stage_intcpt;
            }

            // for convex penalty, simply break out of the loop
            // no need to run multistage convex relaxation
            if (method_flag == 1){
                ite_lamb[i] = outer_loop_count;
                break;  
            }

            // for nonconvex penalty
            // 1. check stopping criterion
            // 2. update stage_lambda

            // check stopping criterion
            p_update(p, Xb, stage_intcpt, n); // p[i] = 1/(1+exp(-intcpt-Xb[i]))
          
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
    Free(p);
    Free(Xb);
    Free(w);
    Free(r);
}
