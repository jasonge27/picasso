#include "mathutils.h"

void picasso_gaussian_cov(double *Y, double * X, double * beta,
    double * intcpt, int * beta_idx, int * cnzz, int * col_cnz,
    int * ite_lamb, int * ite_cyc, double *obj, double *runt, int * err,
    double *lambda, int *nnlambda, double * ggamma, int *mmax_ite, double *pprec,
    int *fflag, int * nn, int * dd, int * ddf, 
    int *vverbose, int * sstandardized){
    int i, j, k, l, s, idx, n, d, df, max_ite, nlambda;
    int ite1, ite2, flag, act_in, cnz, total_df;
    double gamma, prec;
    clock_t start, stop;
    int verbose = (*vverbose);
    int standardized = (*sstandardized);
   
    n = *nn;
    d = *dd;
    df = *ddf;
    max_ite = *mmax_ite;
    prec = *pprec;
  
    nlambda = *nnlambda;
    gamma = *ggamma;
    flag = *fflag;

    total_df = min_int(d,n)*nlambda;

    start = clock();
    double *beta1 = (double *) Calloc(d, double);
    double *beta_old = (double *) Calloc(d, double);

    int *set_act = (int *) Calloc(d, int); 
    int *set_idx = (int *) Calloc(d, int); 
    int *set_idx_covmat = (int *) Calloc(d, int); 
    int act_size = 0;
    int act_size_covmat = 0;

    double *res = (double *) Calloc(n, double);
    double *gr = (double *) Calloc(d, double);
    double **covmat = (double **) Calloc(df, double *);
    double *S = (double *) Calloc(d, double);


    for (i = 0; i < d; i++){
        set_idx[i] = 0; // i is not in the active set yet
        beta1[i] = 0;
        set_idx_covmat[i] = -1; // j = act_idx_covmat[i]>0 is the position of i in covmat
        if (!standardized)
            S[i] = vec_inprod(X+i*n, X+i*n, n)/n;
        else
            S[i] = 1.0;
        // gr[i] = <Y-ymean, X[,i]>/n
        gr[i] = vec_inprod(Y, X+i*n,n)/n;
    }

    for (i = 0; i < n; i++)
        res[i] = Y[i];
    
    cnz = 0;
    act_size = 0;
    act_size_covmat = 0;

    double beta_cached = 0.0;
    int terminate_loop;
    double tmp, tmp_change;
    int flag1 = 0;
    int flag2 = 1;
    for (i = 0; i < nlambda; i++) {
        ite1 = 0;
        flag2 = 1;
        while (ite1 < max_ite) {
            ite1 += 1;

            // STEP1: one pass through the coordinates 
            // and select the active sets
            act_in=0;
            terminate_loop = 1;

            if (flag1 * flag2 == 0)
            {
                for (j = 0; j < d; j++){
                    beta_cached = beta1[j];

                    coordinate_update(&beta1[j], gr[j], S[j], 
                                    standardized, lambda[i], gamma, flag);

                    if (fabs(beta1[j] - beta_cached) < 1e-6)
                        continue;

                    if (set_idx[j] == 0){ 
                        set_idx[j] = 1;
                        set_act[act_size] = j;
                        act_size += 1;
                        act_in += 1;

                        // update the XX matrix if needed
                        if (set_idx_covmat[j] < 0){ // the j-th coord was not in XX before
                            set_idx_covmat[j] = act_size_covmat;
                           
                            covmat[act_size_covmat] = (double *) Calloc(d, double);

                            for (k = 0; k < d; k++)
                                covmat[act_size_covmat][k] = vec_inprod(X+j*n, X+k*n, n)/n; 

                            act_size_covmat += 1;
                        }
                    }

                    tmp = beta1[j] - beta_cached;
                    tmp_change = tmp*tmp;
                    if (standardized == 0)
                        tmp_change = tmp_change * S[j];

                    if (tmp_change > prec){
                        terminate_loop = 0;
                    }

                    idx = set_idx_covmat[j];
                    for (k = 0; k < d; k++)
                        gr[k] -= covmat[idx][k]*tmp;
                    

                    for (k = 0; k < n; k++)
                        res[k] = res[k] - tmp*X[j*n+k];
                }
            } else {
                terminate_loop = 0;
                act_in = 1;
            }
            
            flag1 = 1;

            if (terminate_loop)
                break;

            if (act_in == 0) 
                break;
           
            // STEP2: begin active set minimization
            // update the active coordinate
            ite2 = 0;
            terminate_loop = 1;

            for (k = 0; k < d; k++)
                beta_old[k] = beta1[k];

            while ( ite2 < max_ite) {
                ite2 += 1;
                        
                terminate_loop = 1;
                for (k = 0; k < act_size; k++) {
                    j = set_act[k];
                    
                    beta_cached = beta1[j];
                    coordinate_update(&beta1[j], gr[j], S[j], 
                                standardized, lambda[i], gamma, flag); 
                           
                    if (fabs(beta1[j]-beta_cached)< 1e-8)
                        continue;
                                
                    tmp = beta1[j] - beta_cached;

                    tmp_change = tmp*tmp;
                    if (standardized == 0)
                        tmp_change = tmp_change * S[j];

                    if (tmp_change > prec){
                        terminate_loop = 0;
                    }

                    idx = set_idx_covmat[j];
                    for (l = 0; l < act_size; l++)
                        gr[set_act[l]] -= tmp * covmat[idx][set_act[l]];
                    

                    for (l = 0; l < n; l++)
                        res[l] = res[l] - tmp*X[j*n+l];
                }
                        
                if (terminate_loop){
                    flag2 = 0;
                    break;
                }
            }

            for (k = 0; k < d; k++){
                if (set_idx[k])
                    continue;

                for (l = 0; l < act_size; l++){
                    s = set_act[l];
                    idx = set_idx_covmat[s];
                    gr[k] -= (beta1[s]-beta_old[s])*covmat[idx][k]; 
                }
            }
            
            if (verbose)
                Rprintf("---ite2=%d\n", ite2);
            ite_cyc[i] += ite2; 
        }

        ite_lamb[i] = ite1;
        stop = clock();
        runt[i] = (double)(stop - start)/CLOCKS_PER_SEC;

        intcpt[i] = 0.0;
        for (j = 0; j < n; j++)
            intcpt[i] += res[j];
        intcpt[i] = intcpt[i] / n;


        
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
        if(*err==1) break;
    }

    if (verbose)
        Rprintf("-ite1=%d\n", ite1);
    *cnzz = cnz;
    
    Free(beta1);
    Free(beta_old);
    Free(set_idx);
    Free(set_act);
    Free(set_idx_covmat);
    Free(S);
    Free(res);
    Free(gr);

    for (i = 0; i < df; i++){
        Free(covmat[i]);
    }
    Free(covmat);
}
