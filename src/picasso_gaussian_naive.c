#include "mathutils.h"

void picasso_gaussian_naive(double *Y, double * X, double * beta, double * intcpt,
    int * beta_idx, int * cnzz, int * col_cnz, int * ite_lamb, int * ite_cyc, double *obj,
    double *runt, int * err, double *lambda, int *nnlambda, double * ggamma, int *mmax_ite,
    double *pprec, int *fflag, int * nn, int * dd,  int * ddf, 
      int* vverbose, int * sstandardized){
    int i, j, k, n, s, d, nlambda;
    int outer_loop_count = 0;
    int inner_loop_count = 0;
    int cnz = 0;
    double prec =  *pprec;
    clock_t start, stop;
    int verbose = (*vverbose);
    int standardized = (*sstandardized);
    
    n = *nn;
    d = *dd;
    int max_ite1 = *mmax_ite;
    int max_ite2 = *mmax_ite;

    nlambda = *nnlambda;
    int method_flag = *fflag;

    int total_df = min_int(d,n)*nlambda;
    
    start = clock();
    double *beta1 = (double *) Calloc(d, double);
    int *set_idx = (int *) Calloc(d, int);
    
    int *set_act = (int *) Calloc(d, int);
    int act_size = 0;

    int *active_set = (int *) Calloc(d, int);

    double *res = (double *) Calloc(n, double);
    double *grad = (double *) Calloc(d, double);
    double *S = (double *) Calloc(d, double);

    double *beta_L1 = (double *) Calloc(d, double);
    int *active_set_L1 = (int *) Calloc(d, int);
    int *set_idx_L1 = (int *) Calloc(d, int);
    double *grad_L1 = (double *) Calloc(d, double);
    double *res_L1 = (double *) Calloc(n, double);

    double *stage_lambda = (double *) Calloc(d, double);
    int act_size_L1 = 0;

    double gr, tmp;
    int terminate_loop = 0;
    int new_active_idx = 0;

    // grad[j] = <res, X[,j]> 
    for (i=0; i<n; i++){
        res[i] = Y[i];
    }

    for (i = 0; i < d; i++)
        grad[i] = vec_inprod(res, X+i*n, n)/n;

    for (i=0; i<d; i++){
        set_act[i] = 0;
        beta1[i] = 0;
        active_set[i] = 0;
        set_idx[i] = 0;

        // S[i] = <X[,i], X[,i]>/n
        if (standardized == 0)
            S[i] = vec_inprod(X+i*n, X+i*n, n)/n; 
        else 
            S[i] = 1.0;
    }
    
    cnz = 0;
    double tmp_change = 0.0;
    double beta_cached = 0.0;
    act_size = -1;
    int flag1 = 0;
    int flag2 = 1;
    for (i=0; i<nlambda; i++) {
        if (verbose)
            Rprintf("lambda i:%f \n", lambda[i]);

         if (i > 0){
            for (j = 0; j < d; j++)
                if (active_set[j] == 0){
                    if (grad[j] > 2*lambda[i] - lambda[i-1]) {
                        active_set[j] = 1;
                        set_act[act_size] = j;
                        act_size += 1;
                    }
                }
        } else if (i == 0){
            for (j = 0; j < d; j++)
                if (active_set[j] == 0){
                    if (grad[j] > 2*lambda[i]) {
                        active_set[j] = 1;
                        set_act[act_size] = j;
                        act_size += 1;
                    }
                }
        } 

        if (i > 0){
            for (j = 0; j < d; j++){
                active_set[j] = active_set_L1[j];
                set_idx[j] = set_idx_L1[j];
                grad[j] = grad_L1[j];
                beta1[j] = beta_L1[j];
            }
            for (j = 0; j < n; j++){
                res[j] = res_L1[j];
            }
            act_size = act_size_L1;
        }

        for (j = 0; j < d; j++)
            stage_lambda[j] = lambda[i];


        outer_loop_count = 0;
        flag2 = 1;
        
        int stage_count = 0;
        int dc_loop_max = 10;
        while (stage_count < dc_loop_max){
            stage_count += 1;

            while (outer_loop_count < max_ite1) {
                outer_loop_count += 1;

                // STEP1: constructing support set for active set minimization
                inner_loop_count = 0;
                if (flag1 * flag2 != 0)
                {
                    inner_loop_count = max_ite2+1;
                    new_active_idx = 1;
                }
                while (inner_loop_count < max_ite2){
                    inner_loop_count += 1;
                    terminate_loop = 0;

                    for (j = 0; j < d; j++){
                        if (active_set[j] == 0)
                            continue;
                        beta_cached = beta1[j];
                        // gr = <res, X[,j]> / n
                        gr = vec_inprod(res, X+j*n, n)/n;
                        coordinate_update(&beta1[j], gr, S[j], 
                            standardized, stage_lambda[j]); 
                    
                        if (beta1[j] == beta_cached)
                            continue;

                        if (set_idx[j] == 0){
                            act_size += 1;
                            set_act[act_size] = j;
                            set_idx[j] = 1;
                        }
                        

                        tmp = beta1[j] - beta_cached;
                        // res = res - tmp * X[,j] 
                        for (k = 0; k < n; k++)
                            res[k] -= tmp * X[j*n+k]; 

                        tmp_change = tmp*tmp;
                        if (standardized == 0)
                            tmp_change = tmp_change * S[j];

                        if (tmp_change > prec){
                            terminate_loop = 1;
                        }
                    }

                    // begin inner loop
                    if (terminate_loop){ 
                        new_active_idx = 1; 
                        break;
                    }

                    new_active_idx = 0;
                    for (j = 0; j < d; j++)
                        if (active_set[j] == 0){
                            grad[j] = fabs(vec_inprod(res, X+j*n, n))/n;
                        
                            tmp = soft_thresh_l1(grad[j], stage_lambda[j]);
           
                            if (fabs(tmp) > 1e-8){
                            active_set[j] = 1; 
                            new_active_idx = 1;
                            }
                        }

                    // break if there is no change in active set
                    if (new_active_idx == 0){ 
                        break;
                    }     
                }
                if (verbose == 1)
                    Rprintf("---act set selection, ite=%d, new_act=%d\n", inner_loop_count, new_active_idx);
                ite_cyc[i] += inner_loop_count;

                flag1 = 1;
            
                if (new_active_idx == 0)
                    break;
                
                inner_loop_count = 0;
                
                // STEP2: begin active set minimization
                // update the active coordinate
                while ( inner_loop_count < max_ite2) {
                    inner_loop_count += 1;
                            
                    terminate_loop = 1;
                    for (k=0; k<act_size; k++) {
                        j = set_act[k];
                        
                        beta_cached = beta1[j];
                        gr = vec_inprod(res, X+j*n, n)/n;
                        coordinate_update(&beta1[j], gr, S[j], standardized, stage_lambda[j]); 
                            
                        if (beta1[j] == beta_cached)
                            continue;
                                    
                        tmp = beta1[j] - beta_cached;
                        
                        for (s = 0; s < n; s++)
                            res[s] -= tmp * X[j*n+s]; // res = res - tmp * X[,j] 
                
                        tmp_change = tmp*tmp;
                        if (standardized == 0)
                            tmp_change = tmp_change * S[j];

                        if (tmp_change > prec){
                            terminate_loop = 0;
                        }
                    }
                            
                    if (terminate_loop){
                        flag2 = 0;
                        break;
                    }
                }
                if (verbose == 1)
                    Rprintf("---inner_loop_count=%d\n", inner_loop_count);
                ite_cyc[i] += inner_loop_count;
            }

            // for lambda = lambda[i]
            if (stage_count == 1){
                for (j = 0; j < d; j++){
                    beta_L1[j] = beta1[j];
                    active_set_L1[j] = active_set[j];
                    set_idx_L1[j] = set_idx[j];
                    grad_L1[j] = grad[j];
                }
                for (j = 0; j < n; j++){
                    res_L1[j] = res[j];
                }
                act_size_L1 = act_size;
            }

            if (method_flag == 1){
                ite_lamb[i] = outer_loop_count;
                if (verbose)
                    Rprintf("-outer_loop_count=%d\n", outer_loop_count);
                break;  
            }


            // update lambdas using the multistage convex relaxation scheme
            for (j = 0; j < d; j++)
                stage_lambda[j] = penalty_derivative(method_flag, beta1[j], lambda[i], *ggamma);
     
        }
            
        
   

        intcpt[i] = 0.0;
        for (j = 0; j < n; j++)
            intcpt[i] += res[j];
        intcpt[i] = intcpt[i] / n;

        stop = clock();
        runt[i] = (double)(stop - start)/CLOCKS_PER_SEC;
        for (j = 0; j < d;  j++){
            if ((set_idx[j] != 0) && (fabs(beta1[j])>1e-6)){
                if (cnz == total_df){
                    *err = 1;
                    break;
                }
                beta[cnz] = beta1[j];
                beta_idx[cnz] = j;
                cnz++;
            }
        }
        col_cnz[i+1] = cnz;
        if (*err==1) break;
    }
    *cnzz = cnz;
    
    Free(beta1);
    Free(active_set);

    //Free(old_beta);
    Free(set_idx);
    Free(set_act);
    
    Free(beta_L1);
    Free(active_set_L1);
    Free(set_idx_L1);
    Free(grad_L1);
    Free(res_L1);
    Free(stage_lambda);

    Free(res);
    Free(grad);
    Free(S);
}
