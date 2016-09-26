#include "mymath.h"

void picasso_gaussian_naive(double *Y, double * X, double * beta, double * intcpt,
    int * beta_idx, int * cnzz, int * col_cnz, int * ite_lamb, int * ite_cyc, double *obj,
    double *runt, int * err, double *lambda, int *nnlambda, double * ggamma, int *mmax_ite,
    double *pprec, int *fflag, int * nn, int * dd,  int * ddf, int *mmax_act_in,
      int* vverbose, int * sstandardized){
    int i, j, k, n, s, d, df, nlambda;
    int max_ite1, max_ite2, ite1, ite2, flag;
    int act_in, hybrid, cnz, max_act_in, alg, total_df;
    double gamma, prec, ilambda, ilambda1, ilambda2, lamb_max, trunc;
    clock_t start, stop;
    int verbose = (*vverbose);
    int standardized = (*sstandardized);
    
    n = *nn;
    d = *dd;
    df = *ddf;
    max_ite1 = *mmax_ite;
    max_ite2 = *mmax_ite;
    prec = *pprec;
    nlambda = *nnlambda;
    gamma = *ggamma;
    flag = *fflag;

    max_act_in = *mmax_act_in;
    total_df = min_int(d,n)*nlambda;
    
    start = clock();
    double *beta1 = (double *) Calloc(d, double);
    //double *old_beta = (double *) Calloc(d, double);

    int *set_idx = (int *) Calloc(d, int);
    
    int *set_act = (int *) Calloc(d, int);
    int act_size = 0;

    int *active_set = (int *) Calloc(d, int);

    double *res = (double *) Calloc(n, double);
    double *grad = (double *) Calloc(d, double);
    double *S = (double *) Calloc(d, double);


    double r2 = 0;
    double gr, tmp;
    int terminate_loop, new_active_idx;

    // grad[j] = <res, X[,j]> 
    for (i=0; i<n; i++){
        res[i] = Y[i];
    }
    vec_mat_prod(grad, res, X, n, d); 

    for (i = 0; i < d; i++)
        grad[i] = grad[i]/n;

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
        
        for (j=0; j<d; j++)
            if (active_set[j] == 0){
                if (flag == 1)
                    tmp = soft_thresh_l1(grad[j], lambda[i]);
                if (flag == 2)
                    tmp = soft_thresh_mcp(grad[j], lambda[i], gamma);
                if (flag == 3)
                    tmp = soft_thresh_scad(grad[j], lambda[i], gamma);

                if (fabs(tmp) > 1e-8) {
                    active_set[j] = 1;
                }
            }
        ite1 = 0;
        flag2 = 1;
        

        while (ite1 < max_ite1) {
            ite1 += 1;

            // STEP1: constructing support set for active set minimization
            ite2 = 0;
            if (flag1 * flag2 != 0)
            {
                ite2 = max_ite2+1;
                new_active_idx = 1;
            }
            while (ite2 < max_ite2){
                ite2 += 1;
                terminate_loop = 0;

                
                for (j = 0; j < d; j++){
                    if (active_set[j] == 0)
                        continue;

                    // gr = <res, X[,j]> / n
                    gr = 0;
                    for (k = 0; k < n; k++)
                        gr += res[k] * X[j*n+k];
                    gr = gr / n;
                    if (standardized)
                        tmp = gr + beta1[j];
                    else
                        tmp = gr + beta1[j] * S[j];

                    beta_cached = beta1[j];
                    if (flag==1)
                        beta1[j] = soft_thresh_l1(tmp, lambda[i]);
                    if (flag==2)
                        beta1[j] = soft_thresh_mcp(tmp, lambda[i], gamma);
                    if (flag==3)
                        beta1[j] = soft_thresh_scad(tmp, lambda[i], gamma);
                    
                    if (standardized == 0)
                        beta1[j] = beta1[j] / S[j];

                    if (beta1[j] == beta_cached)
                        continue;

                    if (set_idx[j] == 0){
                        act_size += 1;
                        set_act[act_size] = j;
                        set_idx[j] = 1;
                    }
                    

                    tmp = beta1[j] - beta_cached;
                    r2 += tmp*(2*gr-tmp); 
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
                        grad[j] = 0;
                        for (k = 0; k < n; k++)
                            grad[j] += res[k] * X[j*n+k];
                        grad[j] = fabs(grad[j])/n;
                        if (flag == 1)
                            tmp = soft_thresh_l1(grad[j], lambda[i]);
                        if (flag == 2)
                            tmp = soft_thresh_mcp(grad[j], lambda[i], gamma);
                        if (flag == 3)
                            tmp = soft_thresh_scad(grad[j], lambda[i], gamma);


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
                Rprintf("---act set selection, ite=%d, new_act=%d\n", ite2, new_active_idx);
            ite_cyc[i] += ite2;

            flag1 = 1;
        
            if (new_active_idx == 0)
                break;
            
            ite2 = 0;
            
            // STEP2: begin active set minimization
            // update the active coordinate
            while ( ite2 < max_ite2) {
                ite2 += 1;
                        
                terminate_loop = 1;
                for (k=0; k<act_size; k++) {
                    j = set_act[k];
                           
                     gr = 0;
                    for (s = 0; s < n; s++)
                        gr += res[s] * X[j*n+s]; // gr = <res, X[,j]> / n
                    gr = gr / n;
                    if (standardized)
                        tmp = gr + beta1[j];
                    else
                        tmp = gr + beta1[j] * S[j];
                            
                    beta_cached = beta1[j];
                    if (flag==1)
                        beta1[j] = soft_thresh_l1(tmp, lambda[i]);
                    if (flag==2) 
                        beta1[j] = soft_thresh_mcp(tmp, lambda[i], gamma);
                    if (flag==3) 
                        beta1[j] = soft_thresh_scad(tmp, lambda[i], gamma);
                            
                    if (standardized == 0)
                        beta1[j] = beta1[j] / S[j];

                    if (beta1[j] == beta_cached)
                        continue;
                                
                    tmp = beta1[j] - beta_cached;
                    r2 += tmp*(2*gr-tmp); 
                    
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
                Rprintf("---ite2=%d\n", ite2);
            ite_cyc[i] += ite2;
        }
        
        if (verbose)
            Rprintf("-ite1=%d\n", ite1);
        ite_lamb[i] = ite1;

        stop = clock();
        runt[i] = (double)(stop - start)/CLOCKS_PER_SEC;
        for(j=0;j<d;j++){
            if (set_act[j]!=0){
                if(cnz==total_df){
                    *err = 1;
                    break;
                }
                beta[cnz] = beta1[j];
                beta_idx[cnz] = j;
                cnz++;
            }
        }
        col_cnz[i+1] = cnz;
        if(*err==1) break;
    }
    *cnzz = cnz;
    
    Free(beta1);
    Free(active_set);

    //Free(old_beta);
    Free(set_idx);
    Free(set_act);

    Free(res);
    Free(grad);
}
